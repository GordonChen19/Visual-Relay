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

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import free_model, shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.attention import (begin_value_rms_diagnostics,
                                end_value_rms_diagnostics)
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanI2V:

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
        Initializes the image-to-video generation model components.

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
        self.dit_fsdp = dit_fsdp
        self._paired_high_noise_model_released = False

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
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
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.low_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.low_noise_checkpoint)
        self.low_noise_model = self._configure_model(
            model=self.low_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)

        self.high_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.high_noise_checkpoint)
        self.high_noise_model = self._configure_model(
            model=self.high_noise_model,
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

    def _prepare_model_for_timestep(self,
                                    t,
                                    boundary,
                                    offload_model,
                                    release_high_noise_model=False):
        r"""
        Prepares and returns the required model for the current timestep.

        Args:
            t (torch.Tensor):
                current timestep.
            boundary (`int`):
                The timestep threshold. If `t` is at or above this value,
                the `high_noise_model` is considered as the required model.
            offload_model (`bool`):
                A flag intended to control the offloading behavior.

        Returns:
            torch.nn.Module:
                The active model on the target device for the current timestep.
        """
        if t.item() >= boundary:
            required_model_name = 'high_noise_model'
            offload_model_name = 'low_noise_model'
        else:
            required_model_name = 'low_noise_model'
            offload_model_name = 'high_noise_model'
        if offload_model or self.init_on_cpu:
            if (release_high_noise_model
                    and required_model_name == 'low_noise_model'
                    and self.dit_fsdp
                    and not self._paired_high_noise_model_released):
                # The schedule crosses from high noise to low noise only once.
                # FSDP ``to('cpu')`` may retain flat-parameter/all-gather
                # storage long enough to overlap the incoming model transfer.
                # Explicitly free the completed expert before loading the next
                # one. This is enabled only for paired-anchor generation.
                if dist.is_initialized():
                    dist.barrier()
                torch.cuda.synchronize(self.device)
                outgoing_model = self.high_noise_model
                free_model(outgoing_model)
                self.high_noise_model = None
                self._paired_high_noise_model_released = True
                del outgoing_model
                gc.collect()
                torch.cuda.empty_cache()
                if dist.is_initialized():
                    dist.barrier()
                logging.info(
                    "Permanently released the completed high-noise FSDP "
                    "model before loading the low-noise model")

            offload_model_instance = getattr(self, offload_model_name)
            if (offload_model_instance is not None and next(
                    offload_model_instance.parameters()).device.type ==
                    'cuda'):
                offload_model_instance.to('cpu')
                gc.collect()
                torch.cuda.empty_cache()
            if next(getattr(
                    self,
                    required_model_name).parameters()).device.type == 'cpu':
                getattr(self, required_model_name).to(self.device)
        return getattr(self, required_model_name)

    def generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 image_latent_indices=None,
                 image_condition_paired_anchor_attention=False,
                 image_condition_paired_stream_weight=0.25,
                 image_condition_paired_prefix_future_eta=0.0,
                 image_condition_paired_fade_start_step=None,
                 image_condition_paired_fade_end_step=None,
                 image_condition_paired_suffix_fade_start_step=None,
                 image_condition_paired_suffix_fade_end_step=None,
                 image_condition_paired_prefix_route_switch_step=None,
                 image_condition_self_attention_guidance=False,
                 image_condition_self_attention_guidance_scale=0.5,
                 image_condition_self_attention_guidance_img2_weight=0.25,
                 image_condition_self_attention_guidance_rms_cap=None,
                 image_condition_self_attention_guidance_stop_step=None):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image or sequence[PIL.Image.Image]):
                One image for standard I2V, or exactly two images for paired
                anchor attention.
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            image_latent_indices (sequence[int], *optional*):
                In paired mode, use ``[0, anchor]`` to align Img2 with the
                requested latent frame in the full Img1 stream.
            image_condition_paired_anchor_attention (`bool`, *optional*):
                Enable Img1-only C, fresh-cache Img2-first B, and a one-frame
                joint-conditioned proxy H for C's first retained suffix
                frame. Prefix queries use H in place of regular C at that
                frame. H never supplies a scheduler prediction; the retained
                frame's final velocity always comes from regular C.
            image_condition_paired_stream_weight (`float`, *optional*):
                Fixed weight assigned to the full Img2 suffix-attention output
                when paired mode fuses it with C's suffix continuity output.
                The fused update is applied to C's retained suffix.
            image_condition_paired_prefix_future_eta (`float`, *optional*):
                Framewise future-attention adjustment in [0, 1]. Zero uses
                ordinary attention; positive eta adjusts future mass w to
                w + eta*w*(1-w), preserving within-group probabilities.
            image_condition_paired_fade_start_step (`int`, *optional*):
                Sampling-step index at which the Stream-B suffix fusion weight
                begins a cosine fade.
            image_condition_paired_fade_end_step (`int`, *optional*):
                Sampling-step index at which the suffix fusion weight reaches
                zero.
            image_condition_paired_suffix_fade_start_step (`int`, *optional*):
                Stream-B-suffix-only fade start. Overrides the shared fade for
                the suffix when supplied with its matching end step.
            image_condition_paired_suffix_fade_end_step (`int`, *optional*):
                Stream-B-suffix-only fade end.
            image_condition_paired_prefix_route_switch_step (`int`, *optional*):
                Legacy compatibility option; currently ignored.
            image_condition_self_attention_guidance (`bool`, *optional*):
                Run expanded Img1-only M, joint Img1/Img2 J, standard
                Img2-first suffix B. J provides the single anchor used by
                both its prefix and suffix routes. M's
                temporary slot is excluded from attention. Removing J's anchor
                aligns its ordinary attention output with M's retained output
                for uniform M/J guidance. B0 supplies the temporary anchor
                velocity used by the internal scheduler state.
            image_condition_self_attention_guidance_scale (`float`, *optional*):
                Per-layer weight of J's completed attention output.
            image_condition_self_attention_guidance_img2_weight (`float`, *optional*):
                Weight of aligned B1: outputs in M's suffix fusion.
            image_condition_self_attention_guidance_rms_cap (`float`, *optional*):
                Optional per-frame, per-head J-M attention residual RMS cap,
                expressed relative to M's attention-output RMS.
            image_condition_self_attention_guidance_stop_step (`int`, *optional*):
                Zero-based sampling step at which J is removed. At and after
                this step, B continues guiding M's suffix and supplying the
                anchor velocity while M's temporary slot stays isolated.

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        # preprocess
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale
        if (image_condition_paired_anchor_attention
                and image_condition_self_attention_guidance):
            raise ValueError(
                "paired-anchor attention and self-attention guidance are "
                "separate ablations")
        if not 0 <= image_condition_self_attention_guidance_scale <= 1:
            raise ValueError(
                "self-attention guidance scale must be in [0, 1]")
        if not 0 <= image_condition_self_attention_guidance_img2_weight <= 1:
            raise ValueError(
                "self-attention Img2 weight must be in [0, 1]")
        if (image_condition_self_attention_guidance_rms_cap is not None
                and image_condition_self_attention_guidance_rms_cap <= 0):
            raise ValueError(
                "self-attention guidance RMS cap must be positive")
        if image_condition_self_attention_guidance_stop_step is not None:
            if not image_condition_self_attention_guidance:
                raise ValueError(
                    "self-attention guidance stop step requires the "
                    "self-attention-guidance ablation")
            if not (0 <= image_condition_self_attention_guidance_stop_step <
                    sampling_steps):
                raise ValueError(
                    "self-attention guidance stop step must be in "
                    "[0, sampling_steps)")
        if image_condition_self_attention_guidance and any(
                value is not None for value in (
                    image_condition_paired_fade_start_step,
                    image_condition_paired_fade_end_step,
                    image_condition_paired_suffix_fade_start_step,
                    image_condition_paired_suffix_fade_end_step,
                    image_condition_paired_prefix_route_switch_step)):
            raise ValueError(
                "paired fading and prefix routing do not apply to the "
                "self-attention-guidance ablation")
        multi_stream_mode = (image_condition_paired_anchor_attention
                             or image_condition_self_attention_guidance)
        if image_condition_paired_anchor_attention:
            if not 0.0 <= image_condition_paired_stream_weight <= 1.0:
                raise ValueError(
                    "paired stream weight must be between zero and one")
            fade_pairs = {
                "shared": (image_condition_paired_fade_start_step,
                           image_condition_paired_fade_end_step),
                "suffix": (image_condition_paired_suffix_fade_start_step,
                           image_condition_paired_suffix_fade_end_step),
            }
            for fade_name, fade_steps in fade_pairs.items():
                if ((fade_steps[0] is None) != (fade_steps[1] is None)):
                    raise ValueError(
                        f"paired {fade_name} fade start and end steps must be "
                        "supplied together")
                if (fade_steps[0] is not None
                        and not (0 <= fade_steps[0] < fade_steps[1]
                                 < sampling_steps)):
                    raise ValueError(
                        f"paired {fade_name} fade steps must satisfy "
                        "0 <= start < end < sampling_steps")
            if (image_condition_paired_prefix_route_switch_step is not None
                    and not (0 <=
                             image_condition_paired_prefix_route_switch_step
                             < sampling_steps)):
                raise ValueError(
                    "paired prefix route switch step must be in "
                    "[0, sampling_steps)")
            suffix_fade_start_step = (
                image_condition_paired_suffix_fade_start_step
                if image_condition_paired_suffix_fade_start_step is not None
                else image_condition_paired_fade_start_step)
            suffix_fade_end_step = (
                image_condition_paired_suffix_fade_end_step
                if image_condition_paired_suffix_fade_end_step is not None
                else image_condition_paired_fade_end_step)
        if multi_stream_mode:
            if not isinstance(img, (list, tuple)) or len(img) != 2:
                raise ValueError(
                    "the selected multi-image ablation requires exactly two "
                    "images")
            if (image_latent_indices is None
                    or len(image_latent_indices) != 2
                    or image_latent_indices[0] != 0
                    or image_latent_indices[1] <= 0):
                raise ValueError(
                    "the selected multi-image ablation requires indices "
                    "[0, positive]")
            imgs = [
                TF.to_tensor(value).sub_(0.5).div_(0.5).to(self.device)
                for value in img
            ]
            anchor_index = int(image_latent_indices[1])
        else:
            if isinstance(img, (list, tuple)):
                if len(img) != 1:
                    raise ValueError(
                        "multiple images require paired anchor attention")
                img = img[0]
            imgs = [TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)]
            anchor_index = None

        F = frame_num
        h, w = imgs[0].shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        latent_frame_num = (F - 1) // self.vae_stride[0] + 1
        if (multi_stream_mode
                and anchor_index >= latent_frame_num - 1):
            raise ValueError(
                "paired Img2 anchor must have at least one latent frame on "
                "its right")
        # Paired mode inserts Img2 as a temporary diffusion latent instead of
        # replacing an output latent. It is removed before VAE decoding, so an
        # 81-frame request still decodes 21 retained latents.
        model_latent_frame_num = (latent_frame_num + 1
                                  if multi_stream_mode
                                  else latent_frame_num)
        max_seq_len = model_latent_frame_num * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            model_latent_frame_num,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        def first_frame_mask(pixel_frames):
            mask = torch.ones(
                1, pixel_frames, lat_h, lat_w, device=self.device)
            mask[:, 1:] = 0
            mask = torch.concat([
                torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1),
                mask[:, 1:]
            ], dim=1)
            mask = mask.view(1, mask.shape[1] // 4, 4, lat_h, lat_w)
            return mask.transpose(1, 2)[0]

        msk = first_frame_mask(F)

        def insert_latent_slot(value, index):
            """Insert a temporary slot while preserving every retained slot."""
            return torch.cat([
                value[:, :index], value[:, index:index + 1], value[:, index:]
            ], dim=1)

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

        def conditioning_video(image, pixel_frames):
            return torch.concat([
                torch.nn.functional.interpolate(
                    image[None].cpu(), size=(h, w),
                    mode='bicubic').transpose(0, 1),
                torch.zeros(3, pixel_frames - 1, h, w)
            ], dim=1).to(self.device)

        def joint_anchor_conditioning_video(first_image, anchor_image,
                                            pixel_frames, latent_index):
            """Build Img1 ... Img2x4 ... for one causal joint VAE encode."""
            first = torch.nn.functional.interpolate(
                first_image[None].cpu(), size=(h, w),
                mode='bicubic').to(self.device)[0]
            anchor = torch.nn.functional.interpolate(
                anchor_image[None].cpu(), size=(h, w),
                mode='bicubic').to(self.device)[0]
            video = torch.zeros(
                3, pixel_frames, h, w, device=self.device)
            video[:, 0].copy_(first)
            anchor_pixel_start = (1 + (latent_index - 1)
                                  * self.vae_stride[0])
            anchor_pixel_end = anchor_pixel_start + self.vae_stride[0]
            if anchor_pixel_end > pixel_frames:
                raise ValueError("joint Img2 anchor exceeds conditioning video")
            video[:, anchor_pixel_start:anchor_pixel_end].copy_(
                anchor[:, None].expand(-1, self.vae_stride[0], -1, -1))
            return video

        if not 0.0 <= image_condition_paired_prefix_future_eta <= 1.0:
            raise ValueError("paired prefix future eta must be in [0, 1]")
        if image_condition_paired_anchor_attention:
            suffix_latent_frame_num = model_latent_frame_num - anchor_index
            suffix_frame_num = (1 + self.vae_stride[0]
                                * (suffix_latent_frame_num - 1))
            expanded_frame_num = (1 + self.vae_stride[0]
                                  * (model_latent_frame_num - 1))
            encoded = self.vae.encode([
                conditioning_video(imgs[0], F),
                conditioning_video(imgs[1], suffix_frame_num),
                joint_anchor_conditioning_video(
                    imgs[0], imgs[1], expanded_frame_num, anchor_index),
            ])
            if (encoded[0].size(1) != latent_frame_num
                    or encoded[1].size(1) != suffix_latent_frame_num
                    or encoded[2].size(1) != model_latent_frame_num):
                raise RuntimeError("paired B/C conditioning lengths do not align")
            expanded_msk = insert_latent_slot(msk, anchor_index)
            expanded_main = insert_latent_slot(encoded[0], anchor_index)
            # C's inserted slot is transport only and is absent from retained
            # attention. Every retained C frame therefore stays Img1-only.
            expanded_msk[:, anchor_index:anchor_index + 1].zero_()
            expanded_main[:, anchor_index:anchor_index + 1].zero_()
            # H is a second representation of the first retained suffix frame
            # (expanded index anchor+1), not a second Img2-anchor token. It
            # carries that frame's joint causal-VAE history but shares the
            # exact same noisy latent as regular C at every denoising step.
            joint_mask = first_frame_mask(expanded_frame_num)
            joint_mask[:, anchor_index:anchor_index + 1].fill_(1)
            proxy_index = anchor_index + 1
            proxy_y = torch.concat([
                joint_mask[:, proxy_index:proxy_index + 1],
                encoded[2][:, proxy_index:proxy_index + 1],
            ])
            suffix_y = torch.concat([
                first_frame_mask(suffix_frame_num), encoded[1]
            ])
            hybrid_y = torch.concat([expanded_msk, expanded_main])
            del encoded, expanded_msk, expanded_main, joint_mask
            logging.info(
                "Paired proxy attention: Img1-only C, fresh-cache Img2-first "
                "B, and one-frame joint-conditioned H aligned to retained "
                "frame %d. C transport slot %d is hidden. Prefix queries use "
                "H instead of regular C%d, then read later ordinary C frames; "
                "framewise future eta=%.3f. Retained C suffix blends C/B "
                "attention at weight %.3f. H receives aligned B1 attention "
                "each layer, but H's final prediction is discarded; regular "
                "C%d supplies the retained frame velocity. B0 still supplies "
                "the hidden transport velocity, which is removed before "
                "decoding",
                proxy_index, anchor_index, proxy_index,
                image_condition_paired_prefix_future_eta,
                image_condition_paired_stream_weight, proxy_index)
            if suffix_fade_start_step is not None:
                logging.info("B suffix weight cosine fade: steps %d through %d",
                             suffix_fade_start_step, suffix_fade_end_step)
        elif image_condition_self_attention_guidance:
            guidance_local_latent_frame_num = (
                model_latent_frame_num - anchor_index)
            guidance_local_frame_num = (
                1 + self.vae_stride[0]
                * (guidance_local_latent_frame_num - 1))
            expanded_frame_num = (1 + self.vae_stride[0]
                                  * (model_latent_frame_num - 1))
            encoded = self.vae.encode([
                conditioning_video(imgs[0], F),
                joint_anchor_conditioning_video(
                    imgs[0], imgs[1], expanded_frame_num, anchor_index),
                conditioning_video(imgs[1], guidance_local_frame_num),
            ])
            if (encoded[0].size(1) != latent_frame_num
                    or encoded[1].size(1) != model_latent_frame_num
                    or encoded[2].size(1) !=
                    guidance_local_latent_frame_num):
                raise RuntimeError(
                    "self-attention-guidance conditioning lengths do not "
                    "match the expanded layout")
            # M expands ordinary Img1-only conditioning with a masked transport
            # slot. That slot is excluded from attention. J uses the joint
            # multi-image encode, B is standard Img2-first I2V, and J uses one
            # history-aware anchor for both prefix and suffix.
            primary_mask = insert_latent_slot(
                first_frame_mask(F), anchor_index)
            primary_encoded = insert_latent_slot(encoded[0], anchor_index)
            primary_mask[:, anchor_index:anchor_index + 1].zero_()
            joint_mask = first_frame_mask(expanded_frame_num)
            joint_mask[:, anchor_index:anchor_index + 1].fill_(1)
            primary_y = torch.concat([primary_mask, primary_encoded])
            joint_y = torch.concat([joint_mask, encoded[1]])
            guidance_local_y = torch.concat([
                first_frame_mask(guidance_local_frame_num), encoded[2]
            ])
            del encoded, primary_mask, primary_encoded, joint_mask
            logging.info(
                "Self-attention-guidance ablation: expanded M and joint J "
                "have %d latents with a temporary slot at %d. M's slot is "
                "masked and excluded from all attention; its retained "
                "neighbours are adjacent under compressed RoPE. B is a "
                "standard %d-latent Img2-first suffix. J-prefix and J-suffix "
                "queries both use J's single history-aware joint anchor; the "
                "anchor reads itself plus J's suffix at every layer. Removing "
                "J's anchor "
                "aligns J with M's retained positions. Before handoff, both "
                "prefix and suffix M use M/J weights %.3f/%.3f and B1: is "
                "excluded. After handoff, suffix M uses M/B weights "
                "%.3f/%.3f. B0's predicted "
                "velocity replaces M's temporary anchor velocity after each "
                "active model call; the slot is discarded before decoding. "
                "J-to-M handoff step (M/B continue): %s",
                model_latent_frame_num, anchor_index,
                guidance_local_latent_frame_num,
                (1.0 - image_condition_self_attention_guidance_scale),
                image_condition_self_attention_guidance_scale,
                (1.0 - image_condition_self_attention_guidance_img2_weight),
                image_condition_self_attention_guidance_img2_weight,
                ("disabled" if
                 image_condition_self_attention_guidance_stop_step is None
                 else str(image_condition_self_attention_guidance_stop_step)))
        else:
            y = self.vae.encode([conditioning_video(imgs[0], F)])[0]
            y = torch.concat([msk, y])
            suffix_y = None

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
                                    noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
                                     noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_low_noise(),
                no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

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

            if image_condition_paired_anchor_attention:
                arg_c = {
                    'context': [context[0], context[0], context[0]],
                    'seq_len': max_seq_len,
                    'y': [suffix_y, hybrid_y, proxy_y],
                    'paired_anchor_index': anchor_index,
                    'paired_stream_weight': image_condition_paired_stream_weight,
                    'paired_prefix_anchor_active': True,
                    'paired_prefix_future_eta':
                    image_condition_paired_prefix_future_eta,
                }
                arg_null = {
                    'context': [context_null[0], context_null[0],
                                context_null[0]],
                    'seq_len': max_seq_len,
                    'y': [suffix_y, hybrid_y, proxy_y],
                    'paired_anchor_index': anchor_index,
                    'paired_stream_weight': image_condition_paired_stream_weight,
                    'paired_prefix_anchor_active': True,
                    'paired_prefix_future_eta':
                    image_condition_paired_prefix_future_eta,
                }
            elif image_condition_self_attention_guidance:
                arg_c = {
                    'context': [context[0], context[0], context[0]],
                    'seq_len': max_seq_len,
                    'y': [primary_y, joint_y, guidance_local_y],
                    'self_attention_guidance_scale':
                    image_condition_self_attention_guidance_scale,
                    'self_attention_guidance_img2_weight':
                    image_condition_self_attention_guidance_img2_weight,
                    'self_attention_guidance_rms_cap':
                    image_condition_self_attention_guidance_rms_cap,
                    'attention_guidance_anchor_index': anchor_index,
                }
                m_b_arg_c = {
                    'context': [context[0], context[0]],
                    'seq_len': max_seq_len,
                    'y': [primary_y, guidance_local_y],
                    'self_attention_guidance_scale': 0.0,
                    'self_attention_guidance_img2_weight':
                    image_condition_self_attention_guidance_img2_weight,
                    'self_attention_guidance_rms_cap': None,
                    'attention_guidance_anchor_index': anchor_index,
                }
                arg_null = {
                    'context': [context_null[0]],
                    'seq_len': max_seq_len,
                    'y': [primary_y],
                    'self_attention_guidance_scale': 0.0,
                    'self_attention_guidance_img2_weight': 0.0,
                    'self_attention_guidance_rms_cap': None,
                    'attention_guidance_anchor_index': anchor_index,
                }
                logging.info(
                    "Self-attention guidance applies only to the conditional "
                    "CFG call. The negative-prompt call runs Img1-only M "
                    "throughout sampling, with no J/B guidance or B0 "
                    "anchor-velocity substitution")
            else:
                arg_c = {
                    'context': [context[0]],
                    'seq_len': max_seq_len,
                    'y': [y],
                }
                arg_null = {
                    'context': context_null,
                    'seq_len': max_seq_len,
                    'y': [y],
                }

            if offload_model:
                torch.cuda.empty_cache()

            def paired_fade_factor(sampling_step, fade_start_step,
                                   fade_end_step):
                if fade_start_step is None:
                    return 1.0
                if sampling_step <= fade_start_step:
                    return 1.0
                if sampling_step >= fade_end_step:
                    return 0.0
                fade_progress = (
                    (sampling_step - fade_start_step)
                    / (fade_end_step - fade_start_step))
                return 0.5 * (1.0 + math.cos(math.pi * fade_progress))

            def assemble_paired_prediction(predictions):
                if len(predictions) != 3:
                    raise RuntimeError(
                        "paired model must return B, C, and H predictions")
                suffix_prediction, hybrid_prediction, proxy_prediction = predictions
                if (suffix_prediction.size(1) != suffix_latent_frame_num
                        or hybrid_prediction.size(1) != model_latent_frame_num
                        or proxy_prediction.size(1) != 1):
                    raise RuntimeError(
                        "paired prediction lengths do not match B/C/H")
                # Reproduce the early local-anchor routing at prediction
                # level: after all DiT layers, substitute B0's Img2-first
                # velocity for the temporary C anchor. This happens
                # independently for conditional and unconditional predictions
                # before CFG, so the scheduler receives the CFG-combined B0
                # velocity at the anchor while evolving the latent normally.
                hybrid_prediction = hybrid_prediction.clone()
                hybrid_prediction[:, anchor_index:anchor_index + 1].copy_(
                    suffix_prediction[:, :1])
                # H is attention memory only. In particular, never copy its
                # prediction into regular C[anchor+1]; the scheduler uses C's
                # own prediction for that retained frame.
                return hybrid_prediction

            for sampling_step, t in enumerate(tqdm(timesteps)):
                current_arg_c = arg_c
                current_arg_null = arg_null
                self_attention_guidance_active = False
                if image_condition_paired_anchor_attention:
                    suffix_fade_factor = paired_fade_factor(
                        sampling_step, suffix_fade_start_step,
                        suffix_fade_end_step)
                    anchor_is_enabled = False
                    current_stream_weight = (
                        image_condition_paired_stream_weight
                        * suffix_fade_factor)
                    arg_c['paired_stream_weight'] = current_stream_weight
                    arg_c['paired_prefix_anchor_active'] = anchor_is_enabled
                    arg_null['paired_stream_weight'] = current_stream_weight
                    arg_null['paired_prefix_anchor_active'] = anchor_is_enabled
                    latent_model_input = [
                        latent[:, anchor_index:].to(self.device),
                        latent.to(self.device),
                        latent[:, anchor_index + 1:anchor_index + 2].to(
                            self.device),
                    ]
                    timestep = [t, t, t]
                elif image_condition_self_attention_guidance:
                    current_latent = latent.to(self.device)
                    self_attention_guidance_active = (
                        image_condition_self_attention_guidance_stop_step is
                        None or sampling_step <
                        image_condition_self_attention_guidance_stop_step)
                    if self_attention_guidance_active:
                        # M/J start from the exact same current noisy latent;
                        # B receives its aligned anchor/suffix slice.
                        latent_model_input = [
                            current_latent,
                            current_latent,
                            current_latent[:, anchor_index:],
                        ]
                        timestep = [t, t, t]
                    else:
                        # J handoff: keep M and the aligned Img2-first B.
                        # M prefix stays pure M while B continues to guide M's
                        # suffix and supplies B0's anchor velocity.
                        latent_model_input = [
                            current_latent,
                            current_latent[:, anchor_index:],
                        ]
                        timestep = [t, t]
                        current_arg_c = m_b_arg_c
                        if (sampling_step ==
                                image_condition_self_attention_guidance_stop_step):
                            logging.info(
                                "Self-attention guidance stopped at sampling "
                                "step %d: Stream J is disabled; Stream B "
                                "continues suffix guidance and B0 anchor "
                                "velocity replacement",
                                sampling_step)
                else:
                    latent_model_input = [latent.to(self.device)]
                    timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                model = self._prepare_model_for_timestep(
                    t,
                    boundary,
                    offload_model,
                    release_high_noise_model=
                    multi_stream_mode)
                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]

                collect_value_diagnostics = (
                    image_condition_self_attention_guidance
                    and self_attention_guidance_active
                    and sampling_step % 5 == 0)
                if collect_value_diagnostics:
                    begin_value_rms_diagnostics(sampling_step)
                predictions_cond = model(
                    latent_model_input, t=timestep, **current_arg_c)
                value_diagnostics = (
                    end_value_rms_diagnostics()
                    if collect_value_diagnostics else None)
                if value_diagnostics is not None and self.rank == 0:
                    retained_labels = list(range(anchor_index)) + list(
                        range(anchor_index + 1, model_latent_frame_num))
                    def format_metric(values):
                        return "[" + ", ".join(
                            f"{value:.4f}" for value in values) + "]"
                    logging.info(
                        "Self-attention value RMS diagnostics at sampling "
                        "step %d, averaged over %d DiT layers; retained "
                        "expanded-frame labels=%s; J anchor=%d",
                        value_diagnostics['sampling_step'],
                        value_diagnostics['layers'], retained_labels,
                        anchor_index)
                    logging.info(
                        "Value RMS M per retained frame: %s",
                        format_metric(value_diagnostics['m_rms']))
                    logging.info(
                        "Value RMS J per retained frame: %s",
                        format_metric(value_diagnostics['j_rms']))
                    logging.info(
                        "Value RMS J-M per retained frame: %s",
                        format_metric(value_diagnostics['delta_rms']))
                    logging.info(
                        "Value RMS J/M ratio per retained frame: %s; "
                        "J anchor RMS=%.4f",
                        format_metric(value_diagnostics['j_over_m']),
                        value_diagnostics['j_anchor_rms'])
                if image_condition_paired_anchor_attention:
                    noise_pred_cond = assemble_paired_prediction(
                        predictions_cond)
                    del predictions_cond
                elif image_condition_self_attention_guidance:
                    expected_predictions = (
                        3 if self_attention_guidance_active else 2)
                    if len(predictions_cond) != expected_predictions:
                        raise RuntimeError(
                            "self-attention guidance returned the wrong "
                            "number of conditional predictions")
                    suffix_prediction_cond = predictions_cond[
                        2 if self_attention_guidance_active else 1]
                    if (suffix_prediction_cond.size(1) !=
                            guidance_local_latent_frame_num):
                        raise RuntimeError(
                            "B conditional prediction has the "
                            "wrong temporal length")
                    noise_pred_cond = predictions_cond[0].clone()
                    noise_pred_cond[:, anchor_index:anchor_index + 1].copy_(
                        suffix_prediction_cond[:, :1])
                    del suffix_prediction_cond
                    del predictions_cond
                else:
                    noise_pred_cond = predictions_cond[0]
                    del predictions_cond
                if offload_model:
                    torch.cuda.empty_cache()
                # The ablation's negative CFG branch always runs M alone.
                # Keep the shared expanded layout and compressed RoPE, while
                # isolating its temporary slot from retained-frame attention.
                uncond_model_input = (
                    [current_latent] if image_condition_self_attention_guidance
                    else latent_model_input)
                uncond_timestep = (
                    timestep[:1] if image_condition_self_attention_guidance
                    else timestep)
                predictions_uncond = model(
                    uncond_model_input, t=uncond_timestep, **current_arg_null)
                del uncond_model_input, uncond_timestep
                if image_condition_paired_anchor_attention:
                    noise_pred_uncond = assemble_paired_prediction(
                        predictions_uncond)
                    del predictions_uncond
                elif image_condition_self_attention_guidance:
                    if len(predictions_uncond) != 1:
                        raise RuntimeError(
                            "Img1-only negative CFG call must return only M")
                    if predictions_uncond[0].size(1) != model_latent_frame_num:
                        raise RuntimeError(
                            "M unconditional prediction has the "
                            "wrong temporal length")
                    noise_pred_uncond = predictions_uncond[0]
                    del predictions_uncond
                else:
                    noise_pred_uncond = predictions_uncond[0]
                    del predictions_uncond
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + sample_guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                del noise_pred_cond, noise_pred_uncond, noise_pred
                latent = temp_x0.squeeze(0)
                del temp_x0

                x0 = [latent]
                del latent_model_input, timestep
                if image_condition_self_attention_guidance:
                    del current_latent

            if offload_model:
                self.low_noise_model.cpu()
                if self.high_noise_model is not None:
                    self.high_noise_model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                if multi_stream_mode:
                    retained_x0 = torch.cat([
                        x0[0][:, :anchor_index],
                        x0[0][:, anchor_index + 1:]
                    ], dim=1)
                    if retained_x0.size(1) != latent_frame_num:
                        raise RuntimeError(
                            "discarding the paired anchor produced the wrong "
                            "number of output latents")
                    videos = self.vae.decode([retained_x0])
                    del retained_x0
                else:
                    videos = self.vae.decode(x0)

        del noise, latent, x0
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
