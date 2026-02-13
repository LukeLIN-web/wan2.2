"""Qwen3-VL unified reasoner: replaces GPT-4 (text) + Tarsier-34B (video understanding)."""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info


class Qwen3VLReasoner:

    def __init__(self, model_path="./models/Qwen3-VL-8B-Instruct", device="cuda:0"):
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device,
        )

    def chat(self, system_prompt, user_prompt, max_new_tokens=1024):
        """Pure text reasoning (replaces GPT-4)."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    def caption_video(self, video_path, instruction, max_new_tokens=512):
        """Video understanding (replaces Tarsier-34B)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 640,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    def unload(self):
        """Free GPU memory."""
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        torch.cuda.empty_cache()
