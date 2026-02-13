# Modified: multi-image latent frame replacement (replace first N latent frames)
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
from .utils.utils import best_output_size


class WanTI2V_Mul:

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
                 imgs=None,
                 max_area=704 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        """
        Generates video with multiple reference images anchoring the first N
        latent frames.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            imgs (`list[PIL.Image.Image]`):
                List of reference images. Each image becomes one latent frame.
                e.g. 4 images -> replace first 4 latent frames (~16 pixel frames).
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for output resolution.
            frame_num (`int`, *optional*, defaults to 81):
                Number of video frames to generate (should be 4n+1).
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver: 'unipc' or 'dpm++'.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps.
            guide_scale (`float`, *optional*, defaults to 5.0):
                Classifier-free guidance scale.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt. If empty, uses config default.
            seed (`int`, *optional*, defaults to -1):
                Random seed. -1 for random.
            offload_model (`bool`, *optional*, defaults to True):
                Offload models to CPU to save VRAM.

        Returns:
            torch.Tensor: Generated video (C, N, H, W) or None if not rank 0.
        """
        if imgs is not None and len(imgs) > 0:
            return self.mi2v(
                input_prompt=input_prompt,
                imgs=imgs,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model)

        raise ValueError("imgs must be a non-empty list of PIL Images.")

    def mi2v(self,
             input_prompt,
             imgs,
             max_area=704 * 1280,
             frame_num=121,
             shift=5.0,
             sample_solver='unipc',
             sampling_steps=40,
             guide_scale=5.0,
             n_prompt="",
             seed=-1,
             offload_model=True):
        """
        Multi-image to video: encode each image independently into one latent
        frame, replace the first N latent frames of the noise tensor.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            imgs (`list[PIL.Image.Image]`):
                List of reference images (1 to N). Each becomes one latent frame.
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for output resolution.
            frame_num (`int`, *optional*, defaults to 121):
                Number of video frames (should be 4n+1).
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver: 'unipc' or 'dpm++'.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps.
            guide_scale (`float`, *optional*, defaults to 5.0):
                Classifier-free guidance scale.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt.
            seed (`int`, *optional*, defaults to -1):
                Random seed.
            offload_model (`bool`, *optional*, defaults to True):
                Offload models to CPU to save VRAM.

        Returns:
            torch.Tensor: Generated video (C, N, H, W) or None if not rank 0.
        """
        N = len(imgs)
        T_latent = (frame_num - 1) // self.vae_stride[0] + 1
        assert N <= T_latent, (
            f"Number of images ({N}) exceeds total latent frames ({T_latent}). "
            f"Reduce images or increase frame_num.")

        # --- preprocess: use first image to determine output size ---
        ih, iw = imgs[0].height, imgs[0].width
        dh = self.patch_size[1] * self.vae_stride[1]
        dw = self.patch_size[2] * self.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)

        # resize and center-crop all images to the same (ow, oh)
        img_tensors = []
        for img in imgs:
            scale = max(ow / img.width, oh / img.height)
            img_resized = img.resize(
                (round(img.width * scale), round(img.height * scale)),
                Image.LANCZOS)
            x1 = (img_resized.width - ow) // 2
            y1 = (img_resized.height - oh) // 2
            img_cropped = img_resized.crop((x1, y1, x1 + ow, y1 + oh))
            assert img_cropped.width == ow and img_cropped.height == oh
            t = TF.to_tensor(img_cropped).sub_(0.5).div_(0.5).to(
                self.device).unsqueeze(1)  # [3, 1, H, W]
            img_tensors.append(t)

        F = frame_num
        seq_len = ((F - 1) // self.vae_stride[0] + 1) * (
            oh // self.vae_stride[1]) * (ow // self.vae_stride[2]) // (
                self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            self.vae.model.z_dim, T_latent,
            oh // self.vae_stride[1],
            ow // self.vae_stride[2],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # --- text encoding ---
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

        # --- encode each image independently, concatenate along temporal dim ---
        z_list = []
        for t in img_tensors:
            z_i = self.vae.encode([t])  # z_i[0] shape: [z_dim, 1, H_lat, W_lat]
            z_list.append(z_i[0])
        z_combined = torch.cat(z_list, dim=1)  # [z_dim, N, H_lat, W_lat]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # --- diffusion sampling ---
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

            # build mask: zero out first N latent frames
            mask = torch.ones_like(noise)   # [z_dim, T_latent, H_lat, W_lat]
            mask[:, :N] = 0.0

            # inject image latents into first N frames
            latent = mask * noise + (1.0 - mask) * z_combined

            arg_c = {
                'context': [context[0]],
                'seq_len': seq_len,
            }

            arg_null = {
                'context': context_null,
                'seq_len': seq_len,
            }

            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                # timestep encoding: mask=0 frames get timestep=0 (clean)
                temp_ts = (mask[0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ])
                timestep = temp_ts.unsqueeze(0)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
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
                # re-anchor: force first N frames back to image latents
                latent = mask * latent + (1.0 - mask) * z_combined

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

        return videos[0] if self.rank == 0 else None
