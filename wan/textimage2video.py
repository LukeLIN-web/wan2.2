# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_2 import Wan2_2_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.utils import best_output_size, masks_like


class WanTI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_2_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model = self._configure_model(
            model=self.model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def generate(self,
                 input_prompt,
                 img=None,
                 size=(1280, 704),
                 max_area=704 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 state_embedding=None,
                 intershot_kv_cache=None,
                 intershot_layers=None,
                 cache_strip_rope=False,
                 cache_first_frame_only=False,
                 cache_tcrope_shift=0,
                 intershot_gate_threshold=0.0,
                 noise_blend_latent=None,
                 noise_blend_alpha=0.0,
                 style_modulation=None,
                 predx0_config=None,
                 ref_attn_config=None):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            size (`tuple[int]`, *optional*, defaults to (1280,704)):
                Controls video resolution, (width,height).
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            state_embedding (Tensor, optional):
                State embedding tokens [B, K, dim] from StateEmbedder
            intershot_kv_cache (dict, optional):
                KV cache from previous shot for inter-shot attention
            intershot_layers (set, optional):
                Set of DiT layer indices for inter-shot attention
            cache_strip_rope (bool):
                If True, cache K without RoPE (Plan B)
            cache_tcrope_shift (int):
                Temporal phase offset for TcRoPE on cached K (0 = no shift)
            intershot_gate_threshold (float):
                Timestep ratio threshold for gating intershot injection.
                0.0 = inject at all steps (no gating).
                E.g. 0.5 = only inject when t_ratio > 0.5 (early/high-noise steps).
            predx0_config (dict, optional):
                Pred-x0 feature injection config. See WanModel.forward docstring.
            ref_attn_config (dict, optional):
                Reference attention config for cross-shot identity preservation.
                See WanModel.forward docstring.

        Returns:
            torch.Tensor or (torch.Tensor, dict):
                Generated video frames. If intershot_layers is set, returns (video, kv_cache).
        """
        # i2v
        if img is not None:
            return self.i2v(
                input_prompt=input_prompt,
                img=img,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model,
                state_embedding=state_embedding,
                intershot_kv_cache=intershot_kv_cache,
                intershot_layers=intershot_layers,
                cache_strip_rope=cache_strip_rope,
                cache_first_frame_only=cache_first_frame_only,
                cache_tcrope_shift=cache_tcrope_shift,
                intershot_gate_threshold=intershot_gate_threshold,
                noise_blend_latent=noise_blend_latent,
                noise_blend_alpha=noise_blend_alpha,
                style_modulation=style_modulation,
                predx0_config=predx0_config,
                ref_attn_config=ref_attn_config)
        # t2v
        return self.t2v(
            input_prompt=input_prompt,
            size=size,
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model)

    def t2v(self,
            input_prompt,
            size=(1280, 704),
            frame_num=121,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=50,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True,
            state_embedding=None):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (`tuple[int]`, *optional*, defaults to (1280,704)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 121):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync(),
        ):

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise
            mask1, mask2 = masks_like(noise, zero=False)

            arg_c = {'context': context, 'seq_len': seq_len}
            if state_embedding is not None:
                arg_c['state_embedding'] = state_embedding
            arg_null = {'context': context_null, 'seq_len': seq_len}

            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ])
                timestep = temp_ts.unsqueeze(0)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]
            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def i2v(self,
            input_prompt,
            img,
            max_area=704 * 1280,
            frame_num=121,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=40,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True,
            state_embedding=None,
            intershot_kv_cache=None,
            intershot_layers=None,
            cache_strip_rope=False,
            cache_first_frame_only=False,
            cache_tcrope_shift=0,
            intershot_gate_threshold=0.0,
            noise_blend_latent=None,
            noise_blend_alpha=0.0,
            style_modulation=None,
            predx0_config=None,
            ref_attn_config=None):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 121):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            state_embedding (Tensor, optional):
                State embedding tokens [B, K, dim] from StateEmbedder
            intershot_kv_cache (dict, optional):
                KV cache from previous shot for inter-shot attention
            intershot_layers (set, optional):
                Set of DiT layer indices for inter-shot attention
            cache_strip_rope (bool):
                If True, cache K without RoPE (Plan B)
            cache_tcrope_shift (int):
                Temporal phase offset for TcRoPE on cached K (0 = no shift).
                When > 0, cached K (pre-RoPE) gets RoPE applied with shifted
                temporal frequencies before injection.
            intershot_gate_threshold (float):
                Timestep ratio threshold for gating intershot KV injection.
                0.0 = inject at all steps (no gating).
                E.g. 0.5 = only inject when t_ratio > 0.5 (early/high-noise steps).
            ref_attn_config (dict, optional):
                Reference attention config for cross-shot identity preservation.
                See WanModel.forward docstring.
            predx0_config (dict, optional):
                Pred-x0 feature injection config with keys:
                - mode: 'cache' or 'inject'
                - layers: set of DiT layer indices
                - cache_indices/inject_indices/inject_weights: token masks
                - alpha: blend weight (inject mode)
                - cache_stride: cache every N steps (cache mode)
                - _all_cache: populated by cache mode, used by inject mode

        Returns:
            torch.Tensor or (torch.Tensor, dict):
                Generated video frames. If intershot_layers is set, returns (video, kv_cache).
        """
        # preprocess
        ih, iw = img.height, img.width
        dh, dw = self.patch_size[1] * self.vae_stride[1], self.patch_size[
            2] * self.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)

        scale = max(ow / iw, oh / ih)
        img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)

        # center-crop
        x1 = (img.width - ow) // 2
        y1 = (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        assert img.width == ow and img.height == oh

        # to tensor
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)

        F = frame_num
        seq_len = ((F - 1) // self.vae_stride[0] + 1) * (
            oh // self.vae_stride[1]) * (ow // self.vae_stride[2]) // (
                self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
            oh // self.vae_stride[1],
            ow // self.vae_stride[2],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        # Noise blending: mix low-freq of previous shot's last-frame latent
        if noise_blend_latent is not None and noise_blend_alpha > 0:
            nbl = noise_blend_latent  # [C, 1, H_lat, W_lat]
            nbl = nbl.squeeze(1).unsqueeze(0).float()  # [1, C, H, W]
            k = 4
            nbl = torch.nn.functional.avg_pool2d(nbl, kernel_size=k, stride=k)
            nbl = torch.nn.functional.interpolate(
                nbl, size=(noise.shape[2], noise.shape[3]), mode='nearest')
            nbl = nbl.squeeze(0).unsqueeze(1)  # [C, 1, H, W]
            nbl = nbl.expand_as(noise)  # [C, T, H, W]
            noise = (1 - noise_blend_alpha) * noise + noise_blend_alpha * nbl
            del nbl

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        z = self.vae.encode([img])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync(),
        ):

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise
            mask1, mask2 = masks_like([noise], zero=True)
            latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

            arg_c = {
                'context': [context[0]],
                'seq_len': seq_len,
            }
            if state_embedding is not None:
                arg_c['state_embedding'] = state_embedding
            if style_modulation is not None:
                arg_c['style_modulation'] = style_modulation

            arg_null = {
                'context': context_null,
                'seq_len': seq_len,
            }

            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            # Inter-shot attention setup
            use_intershot = intershot_layers is not None
            new_kv_cache = None
            if use_intershot:
                F_patches = (F - 1) // self.vae_stride[0] + 1
                if cache_first_frame_only:
                    cache_frame_indices = [0]  # Phase 2b: first frame only for cumulative KV_global
                else:
                    cache_frame_indices = [0, F_patches - 1]  # Phase 2: first + last frame

            # Pred-x0 feature injection setup
            use_predx0 = predx0_config is not None
            predx0_mode = predx0_config.get('mode') if use_predx0 else None
            predx0_all_cache = {}
            if use_predx0 and predx0_mode == 'cache':
                predx0_config['_all_cache'] = predx0_all_cache
            predx0_cache_stride = predx0_config.get('cache_stride', 5) if use_predx0 else 5

            for step_idx, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ])
                timestep = temp_ts.unsqueeze(0)

                is_last_step = (step_idx == len(timesteps) - 1)

                # Pred-x0: build per-step config for conditional forward
                step_predx0 = None
                if use_predx0:
                    if predx0_mode == 'cache':
                        # Cache at sparse steps + last step
                        if step_idx % predx0_cache_stride == 0 or is_last_step:
                            step_cache = {}
                            step_predx0 = {
                                'mode': 'cache',
                                'layers': predx0_config['layers'],
                                'cache_indices': predx0_config['cache_indices'],
                                '_step_cache': step_cache,
                            }
                    elif predx0_mode == 'inject':
                        all_cache = predx0_config.get('_all_cache', {})
                        if all_cache:
                            # Find nearest cached step
                            cached_steps = sorted(all_cache.keys())
                            nearest = min(cached_steps, key=lambda s: abs(s - step_idx))
                            step_predx0 = {
                                'mode': 'inject',
                                'layers': predx0_config['layers'],
                                'cache_indices': predx0_config['cache_indices'],
                                'step_cache': all_cache[nearest],
                                'inject_indices': predx0_config['inject_indices'],
                                'inject_weights': predx0_config['inject_weights'],
                                'alpha': predx0_config.get('alpha', 0.05),
                            }

                if use_intershot:
                    do_cache = is_last_step
                    # Timestep-gated injection: only inject at early (high-noise) steps
                    t_ratio = t.item() / 1000.0
                    inject_this_step = (
                        intershot_gate_threshold <= 0
                        or t_ratio > intershot_gate_threshold
                    )
                    kv_to_inject = intershot_kv_cache if inject_this_step else None
                    result_c = self.model(
                        latent_model_input, t=timestep, **arg_c,
                        intershot_kv_cache=kv_to_inject,
                        intershot_layers=intershot_layers,
                        cache_kv=do_cache,
                        cache_frame_indices=(
                            cache_frame_indices if do_cache else None),
                        cache_strip_rope=cache_strip_rope,
                        cache_tcrope_shift=cache_tcrope_shift,
                        predx0_config=step_predx0,
                        ref_attn_config=ref_attn_config)
                    if do_cache:
                        noise_pred_cond = result_c[0][0]
                        new_kv_cache = result_c[1]
                    else:
                        noise_pred_cond = result_c[0]
                else:
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c,
                        predx0_config=step_predx0,
                        ref_attn_config=ref_attn_config)[0]

                # Pred-x0: collect cached features
                if step_predx0 is not None and predx0_mode == 'cache' and '_step_cache' in step_predx0:
                    if step_predx0['_step_cache']:
                        predx0_all_cache[step_idx] = step_predx0['_step_cache']

                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)
                latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

                x0 = [latent]
                del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent, x0
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        video_output = videos[0] if self.rank == 0 else None
        if use_intershot:
            return video_output, new_kv_cache
        return video_output
