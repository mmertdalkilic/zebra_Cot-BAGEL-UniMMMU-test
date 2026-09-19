#!/usr/bin/env python3
"""
Native UniCoT-v0.2 backend for the Uni-MMMU sample-code harness.

Important:
- This imports Fr0zenCrane/UniCoT/inference_unicot_v0.2.py directly.
- It recreates the model-loading block from that script, then assigns the
  resulting objects back to that module so its official generation/editing
  functions are used unchanged.
- It does NOT instantiate TorchUMM or TorchUMM's BAGEL adapter.
- UniCoT upstream does not ship a Uni-MMMU understanding runner. Therefore the
  small context bridge in this file is necessarily an evaluation adapter.
  Model weights, native transforms, sampling (temperature 0.3), CFG settings,
  50 diffusion steps, timestep shift, sequential breakdown and self-reflection
  functions come from the released v0.2 code.
"""
from __future__ import annotations

import copy
import importlib.util
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw


@dataclass
class ContextItem:
    kind: str
    value: Any


def add_text(ctx, text):
    ctx.append(ContextItem("text", str(text)))


def add_image_path(ctx, path):
    ctx.append(ContextItem("image", str(path)))


def _rgb(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    return Image.open(x).convert("RGB")


def _decode_generated_text(tokenizer, tokens) -> str:
    raw = tokenizer.decode(tokens[:, 0])
    # Match the released v0.2 extraction when possible, but do not throw away
    # useful output if the chat delimiters are absent.
    if "<|im_start|>" in raw:
        raw = raw.split("<|im_start|>", 1)[1]
    if "<|im_end|>" in raw:
        raw = raw.split("<|im_end|>", 1)[0]
    return raw.strip()


class NativeUniCoTV02:
    def __init__(
        self,
        repo: Path,
        checkpoint: Path,
        *,
        max_gpu_memory: str = "90GiB",
        seed: int = 42,
        max_reflections: int = 20,
        resolution: int = 1024,
    ):
        self.repo = Path(repo).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.seed = seed
        self.max_reflections = max_reflections
        self.resolution = resolution

        if not (self.repo / "inference_unicot_v0.2.py").is_file():
            raise FileNotFoundError(self.repo / "inference_unicot_v0.2.py")
        for fn in ("llm_config.json", "vit_config.json", "ae.safetensors", "ema.safetensors"):
            if not (self.checkpoint / fn).is_file():
                raise FileNotFoundError(self.checkpoint / fn)

        sys.path.insert(0, str(self.repo))
        spec = importlib.util.spec_from_file_location(
            "unicot_v02_official",
            self.repo / "inference_unicot_v0.2.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not import inference_unicot_v0.2.py")
        u = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = u
        spec.loader.exec_module(u)
        self.u = u

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        device = "cuda:0"
        llm_config = u.Qwen2Config.from_json_file(str(self.checkpoint / "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = u.SiglipVisionConfig.from_json_file(str(self.checkpoint / "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        vae_model, vae_config = u.load_ae(local_path=str(self.checkpoint / "ae.safetensors"))

        config = u.BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=64,
        )

        language_model = u.Qwen2ForCausalLM(llm_config)
        vit_model = u.SiglipVisionModel(vit_config)
        model = u.Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

        tokenizer = u.Qwen2Tokenizer.from_pretrained(str(self.checkpoint))
        tokenizer, new_token_ids, _ = u.add_special_tokens(tokenizer)

        # Same official auto-device-map logic, with the memory ceiling adapted
        # from the source's A100-80GB value to the available RTX PRO 6000 96GB.
        device_map = u.infer_auto_device_map(
            model,
            max_memory={0: max_gpu_memory},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )

        same_device_modules = [
            "language_model.model.embed_tokens",
            "time_embedder",
            "latent_pos_embed",
            "vae2llm",
            "llm2vae",
            "connector",
            "vit_pos_embed",
        ]
        first_device = device_map.get(same_device_modules[0], device)
        for key in same_device_modules:
            device_map[key] = first_device if key in device_map else device

        offload = self.checkpoint.parent / "native_v02_offload"
        offload.mkdir(parents=True, exist_ok=True)

        model = u.load_checkpoint_and_dispatch(
            model,
            checkpoint=str(self.checkpoint / "ema.safetensors"),
            device_map=device_map,
            offload_buffers=False,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder=str(offload),
        )
        vae_model = vae_model.cuda().eval()
        model = model.eval()

        # Populate the globals expected by the released v0.2 helper functions.
        u.model = model
        u.gen_model = model
        u.vae_model = vae_model
        u.tokenizer = tokenizer
        u.new_token_ids = new_token_ids

        self.model = model
        self.vae = vae_model
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids
        self.device = device

        # These are exactly the released v0.2 defaults from the main block.
        self.gen_hyper = dict(
            cfg_scale=4.0,
            cfg_interval=[0.4, 1.0],
            timestep_shift=3.0,
            num_timesteps=50,
            cfg_renorm_min=0.0,
        )
        self.edit_hyper = dict(
            cfg_text_scale=4.0,
            cfg_img_scale=1.5,
            cfg_interval=[0.0, 1.0],
            timestep_shift=3.0,
            num_timesteps=50,
            cfg_renorm_min=0.0,
        )

    def _feed_context(self, ctx: Iterable[ContextItem], prompt_suffix: str = ""):
        """
        Native image-conditioned text decode using the same v0.2 cache,
        transforms, tokenizer and generation settings.

        This bridge is needed because upstream v0.2 does not provide a
        Uni-MMMU understanding entry point.
        """
        u = self.u
        past = u.NaiveCache(self.model.config.llm_config.num_hidden_layers)
        lens, rope = [0], [0]

        # Preserve the released v0.2 system prompt.
        items = [ContextItem("text", u.SYSTEM_PROMPT)] + list(ctx)
        if prompt_suffix:
            items.append(ContextItem("text", prompt_suffix))

        for item in items:
            if item.kind == "text":
                gi, lens, rope = self.model.prepare_prompts(
                    curr_kvlens=lens,
                    curr_rope=rope,
                    prompts=[str(item.value)],
                    tokenizer=self.tokenizer,
                    new_token_ids=self.new_token_ids,
                )
                gi = u.move_generation_input_to_device(gi, None)
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    past = self.model.forward_cache_update_text(past, **gi)
            elif item.kind == "image":
                image = _rgb(item.value)
                gi, lens, rope = self.model.prepare_vit_images(
                    curr_kvlens=lens,
                    curr_rope=rope,
                    images=[image],
                    transforms=u.vit_transform_und,
                    new_token_ids=self.new_token_ids,
                )
                gi = u.move_generation_input_to_device(gi, None)
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    past = self.model.forward_cache_update_vit(past, **gi)
            else:
                raise ValueError(f"Unknown context item kind: {item.kind}")

        gi = self.model.prepare_start_tokens(lens, rope, self.new_token_ids)
        gi = u.move_generation_input_to_device(gi, None)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            tokens = self.model.generate_text(
                past_key_values=copy.deepcopy(past),
                max_length=2048,
                do_sample=True,
                temperature=0.3,
                end_token_id=self.new_token_ids["eos_token_id"],
                **gi,
            )
        return _decode_generated_text(self.tokenizer, tokens)

    def generate_text_from_context(self, ctx, prompt_suffix=""):
        return self._feed_context(ctx, prompt_suffix=prompt_suffix)

    def _prompt_from_context(self, ctx, prompt_suffix=""):
        parts = []
        for it in ctx:
            if it.kind == "text":
                parts.append(str(it.value))
        if prompt_suffix:
            parts.append(str(prompt_suffix))
        return "\n".join(parts).strip()

    def _images_from_context(self, ctx):
        return [_rgb(it.value) for it in ctx if it.kind == "image"]

    def _montage(self, images):
        """
        Only used when Uni-MMMU supplies >1 source image (mainly Jigsaw).
        v0.2 has no released multi-source image-edit wrapper, so this keeps all
        source pixels visible in one native editing canvas. This is adapter glue,
        not an upstream UniCoT function.
        """
        imgs = [im.convert("RGB") for im in images]
        if len(imgs) == 1:
            return imgs[0]
        target_h = min(768, max(im.height for im in imgs))
        scaled = []
        for im in imgs:
            w = max(1, round(im.width * target_h / im.height))
            scaled.append(im.resize((w, target_h), Image.Resampling.LANCZOS))
        gap = 8
        canvas = Image.new("RGB", (sum(i.width for i in scaled) + gap*(len(scaled)-1), target_h), "white")
        x = 0
        for im in scaled:
            canvas.paste(im, (x, 0))
            x += im.width + gap
        return canvas

    def _text_to_image_breakdown(self, prompt: str):
        u = self.u
        think, breakdown = u.generate_think_and_breakdown(
            prompt=prompt,
            resolution=self.resolution,
            device=None,
        )
        subtasks = u.split_prompts(breakdown, split_key="subtask")
        if not subtasks:
            fallback = u.preprocess_prompt(think, split_key="think")
            subtasks = [fallback if fallback else prompt]

        image = u.generate_image(
            prompt=subtasks[0],
            resolution=self.resolution,
            device=self.device,
            **self.gen_hyper,
        )
        for subtask in subtasks[1:]:
            image = u.editing_image(
                image=u.pil_img2rgb(image),
                prompt=subtask,
                device=self.device,
                **self.edit_hyper,
            )
        return image

    def _image_to_image_self_reflect(self, image: Image.Image, prompt: str):
        u = self.u
        think, _ = u.generate_think_and_breakdown(
            prompt=prompt,
            resolution=self.resolution,
            device=None,
        )
        think = think.strip()
        current = u.pil_img2rgb(image)

        for _ in range(self.max_reflections):
            eval_text, refine_text, modify_flag = u.generate_evaluation_and_editing(
                image=current,
                prompt=prompt,
                think_text=think,
                resolution=self.resolution,
                device=self.device,
            )
            if not modify_flag:
                break
            edit_prompt = u.preprocess_prompt(refine_text, split_key="sugg")
            if not edit_prompt:
                break
            current = u.editing_image(
                image=u.pil_img2rgb(current),
                prompt=edit_prompt,
                device=self.device,
                **self.edit_hyper,
            )
        return current

    def generate_image_from_context(self, ctx, out_path, prompt_suffix=""):
        prompt = self._prompt_from_context(ctx, prompt_suffix=prompt_suffix)
        images = self._images_from_context(ctx)
        if images:
            # Uni-MMMU task-aware source policy
            if os.environ.get("UNIMMMU_TASK") == "jigsaw":
                source = self._montage(images)
            else:
                source = images[-1]
            output = self._image_to_image_self_reflect(source, prompt)
        else:
            output = self._text_to_image_breakdown(prompt)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output.save(out_path)
        return str(out_path)
