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
from numbers import Integral

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import free_model, shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.frame_conditioning import (build_retained_frame_indices,
                                       center_crop_resize_geometry,
                                       pixel_frame_range_for_latent)


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
        self._high_noise_model_permanently_released = False

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

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
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
            if (required_model_name == 'low_noise_model'
                    and offload_model_name == 'high_noise_model'
                    and offload_model and self.dit_fsdp
                    and not self._high_noise_model_permanently_released):
                # Wan2.2's sampling schedule crosses this boundary only once:
                # after entering the low-noise phase, the high-noise DiT is
                # never used again. FSDP ``.to('cpu')`` can retain flat-param
                # or all-gather storage long enough to overlap with loading
                # the low-noise shard. Explicitly free the outgoing flat
                # parameters before moving the incoming model to CUDA.
                if dist.is_initialized():
                    dist.barrier()
                torch.cuda.synchronize(self.device)
                outgoing_model = self.high_noise_model
                free_model(outgoing_model)
                self.high_noise_model = None
                self._high_noise_model_permanently_released = True
                del outgoing_model
                gc.collect()
                torch.cuda.empty_cache()
                if dist.is_initialized():
                    dist.barrier()
                logging.info(
                    "Permanently released the high-noise FSDP model before "
                    "loading the low-noise model")

            offloaded_model = False
            offload_model_instance = getattr(self, offload_model_name)
            if (offload_model_instance is not None
                    and next(offload_model_instance.parameters()).device.type
                    == 'cuda'):
                offload_model_instance.to('cpu')
                offloaded_model = True
            if offloaded_model:
                # FSDP model moves can leave no room for the incoming noise
                # model unless dead Python objects and cached CUDA blocks are
                # reclaimed between the two transfers.
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
                 size=None,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 image_latent_indices=None,
                 image_condition_switch_step=None,
                 image_condition_rope_blend_weight=0.25,
                 image_condition_rope_blend_start_step=None,
                 image_condition_dual_guidance=False,
                 image_condition_prefix_multi_guidance_scale=0.6,
                 image_condition_suffix_multi_guidance_scale=0.35,
                 image_condition_multi_stop_step=None,
                 image_condition_late_suffix_img2_weight=0.5,
                 image_condition_anchor_block_radius=0,
                 image_condition_anchor_block_start_layer=0,
                 image_condition_anchor_block_start_step=None,
                 sacrificial_image_latent_index=None,
                 init_video=None,
                 init_video_noise_step=None,
                 attention_probe=None):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image or sequence[PIL.Image.Image]):
                One or more conditioning images.
            size (tuple[int, int], *optional*):
                Exact output ``(width, height)``. Conditioning images are
                isotropically resized to cover the target, then minimally
                center-cropped. When omitted, the legacy first-image
                aspect-ratio behavior is used.
            image_latent_indices (sequence[int], *optional*):
                Latent-frame index for each conditioning image. A single image
                defaults to latent frame 0.
            image_condition_switch_step (int, *optional*):
                Zero-based sampling step at which to disable the expanded
                multi-image view and continue with the final-length first-image
                global branch. The scheduler retains temporary placeholder latents
                until denoising completes, then removes them before decoding.
            image_condition_rope_blend_weight (float, *optional*):
                Weight of the expanded multi-image compressed-RoPE prediction
                before the conditioning switch. The remaining weight uses a
                compact, final-length Img1-only regular-I2V prediction.
            image_condition_rope_blend_start_step (int, *optional*):
                Zero-based sampling step at which to begin a cosine transition
                from fully compressed multi-image prediction to compact
                Img1-only regular I2V. The transition reaches fully regular at
                ``image_condition_switch_step``. When provided, this schedule
                replaces the fixed ``image_condition_rope_blend_weight``.
            image_condition_dual_guidance (bool, *optional*):
                Use an Img1-only global base for the prefix and an Img2-only
                local-I2V base for the suffix, guiding both toward the aligned
                expanded multi-image prediction. Supports exactly two images.
            image_condition_prefix_multi_guidance_scale (float, *optional*):
                Scale of ``multi - Img1`` applied to prefix predictions.
            image_condition_suffix_multi_guidance_scale (float, *optional*):
                Scale of ``multi - Img2-local`` applied to suffix predictions.
            image_condition_multi_stop_step (int, *optional*):
                Sampling step at which to stop evaluating the expanded
                multi-image guide while retaining the Img1-only and Img2-local
                branches. Requires dual-guidance mode.
            image_condition_late_suffix_img2_weight (float, *optional*):
                Img2-local interpolation weight for suffix predictions after
                ``image_condition_multi_stop_step``. The complementary weight
                uses the aligned Img1-only suffix prediction.
            image_condition_anchor_block_radius (int, *optional*):
                Symmetric exclusion radius around every non-first image
                anchor. The nearest N ordinary query frames on either side
                hide that anchor while ordinary-frame visibility is unchanged.
            image_condition_anchor_block_start_layer (int, *optional*):
                Zero-based transformer layer at which the anchor exclusion
                radius becomes active. Earlier layers use radius zero at every
                denoising step.
            image_condition_anchor_block_start_step (int, *optional*):
                Sampling step at which the configured exclusion radius becomes
                active in every transformer layer. Before this step, the
                configured layer threshold remains in effect.
            sacrificial_image_latent_index (int, *optional*):
                Deprecated compatibility argument. Sacrificial anchors are
                now inferred from every nonzero image latent index.
            init_video (torch.Tensor, *optional*):
                Existing normalized video with shape [3, frame_num, H, W].
                It is VAE-encoded and noised before refinement denoising.
            init_video_noise_step (int, *optional*):
                Zero-based sampling-schedule index at which to noise
                ``init_video``. Denoising starts at this step.
            attention_probe (dict, *optional*):
                Configuration for measuring temporal self-attention without
                changing the model output.
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

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        if self._high_noise_model_permanently_released:
            raise RuntimeError(
                "this WanI2V pipeline permanently released its high-noise "
                "FSDP model during a previous generation; create a new "
                "pipeline instance before generating again")

        # preprocess
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale
        if attention_probe is not None and self.sp_size != 1:
            raise ValueError(
                "attention probing currently supports single-GPU inference only"
            )
        imgs = list(img) if isinstance(img, (list, tuple)) else [img]
        if not imgs:
            raise ValueError("at least one conditioning image is required")
        if image_latent_indices is None:
            if len(imgs) != 1:
                raise ValueError(
                    "image_latent_indices is required for multiple images")
            image_latent_indices = [0]
        else:
            image_latent_indices = list(image_latent_indices)
        if any(not isinstance(index, Integral)
               for index in image_latent_indices):
            raise TypeError("image latent frame indices must be integers")
        image_latent_indices = [int(index) for index in image_latent_indices]
        if len(imgs) != len(image_latent_indices):
            raise ValueError(
                "img and image_latent_indices must have the same length")
        if len(set(image_latent_indices)) != len(image_latent_indices):
            raise ValueError("image latent frame indices must be unique")
        if image_condition_switch_step is not None:
            if not isinstance(image_condition_switch_step, Integral):
                raise TypeError("image condition switch step must be an integer")
            image_condition_switch_step = int(image_condition_switch_step)
            if not 0 <= image_condition_switch_step < sampling_steps:
                raise ValueError(
                    "image condition switch step must be in "
                    f"[0, {sampling_steps})")
            if len(imgs) < 2:
                raise ValueError(
                    "image condition switch step requires multiple images")
        if not isinstance(image_condition_rope_blend_weight, (int, float)):
            raise TypeError("image condition RoPE blend weight must be numeric")
        image_condition_rope_blend_weight = float(
            image_condition_rope_blend_weight)
        if not 0.0 <= image_condition_rope_blend_weight <= 1.0:
            raise ValueError(
                "image condition RoPE blend weight must be in [0, 1]")
        if image_condition_rope_blend_start_step is not None:
            if not isinstance(image_condition_rope_blend_start_step, Integral):
                raise TypeError(
                    "image condition RoPE blend start step must be an integer")
            image_condition_rope_blend_start_step = int(
                image_condition_rope_blend_start_step)
            if image_condition_switch_step is None:
                raise ValueError(
                    "image condition RoPE blend start step requires an image "
                    "condition switch step")
            if not (0 <= image_condition_rope_blend_start_step <
                    image_condition_switch_step):
                raise ValueError(
                    "image condition RoPE blend start step must be in "
                    "[0, image_condition_switch_step)")
        if not isinstance(image_condition_dual_guidance, bool):
            raise TypeError("image condition dual guidance must be boolean")
        for name, value in (
            ("prefix", image_condition_prefix_multi_guidance_scale),
            ("suffix", image_condition_suffix_multi_guidance_scale),
        ):
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"image condition {name} multi-guidance scale must be "
                    "numeric")
            if value < 0:
                raise ValueError(
                    f"image condition {name} multi-guidance scale must be "
                    "non-negative")
        image_condition_prefix_multi_guidance_scale = float(
            image_condition_prefix_multi_guidance_scale)
        image_condition_suffix_multi_guidance_scale = float(
            image_condition_suffix_multi_guidance_scale)
        if image_condition_multi_stop_step is not None:
            if not isinstance(image_condition_multi_stop_step, Integral):
                raise TypeError(
                    "image_condition_multi_stop_step must be an integer")
            image_condition_multi_stop_step = int(
                image_condition_multi_stop_step)
            if not image_condition_dual_guidance:
                raise ValueError(
                    "image_condition_multi_stop_step requires dual guidance")
            if not 0 <= image_condition_multi_stop_step < sampling_steps:
                raise ValueError(
                    "image_condition_multi_stop_step must be in "
                    f"[0, {sampling_steps})")
        if not isinstance(image_condition_late_suffix_img2_weight,
                          (int, float)):
            raise TypeError(
                "image_condition_late_suffix_img2_weight must be numeric")
        image_condition_late_suffix_img2_weight = float(
            image_condition_late_suffix_img2_weight)
        if not 0.0 <= image_condition_late_suffix_img2_weight <= 1.0:
            raise ValueError(
                "image_condition_late_suffix_img2_weight must be in [0, 1]")
        if image_condition_dual_guidance:
            if len(imgs) != 2:
                raise ValueError(
                    "image condition dual guidance currently requires exactly "
                    "two images")
            if image_latent_indices[0] != 0:
                raise ValueError(
                    "image condition dual guidance requires Img1 at latent 0")
        if not isinstance(image_condition_anchor_block_radius, Integral):
            raise TypeError("image condition anchor block radius must be an "
                            "integer")
        image_condition_anchor_block_radius = int(
            image_condition_anchor_block_radius)
        if image_condition_anchor_block_radius < 0:
            raise ValueError(
                "image condition anchor block radius must be non-negative")
        if not isinstance(image_condition_anchor_block_start_layer, Integral):
            raise TypeError("image condition anchor block start layer must be "
                            "an integer")
        image_condition_anchor_block_start_layer = int(
            image_condition_anchor_block_start_layer)
        if not (0 <= image_condition_anchor_block_start_layer <=
                self.config.num_layers):
            raise ValueError(
                "image condition anchor block start layer must be in "
                f"[0, {self.config.num_layers}]")
        if image_condition_anchor_block_start_step is not None:
            if not isinstance(image_condition_anchor_block_start_step,
                              Integral):
                raise TypeError(
                    "image condition anchor block start step must be an "
                    "integer")
            image_condition_anchor_block_start_step = int(
                image_condition_anchor_block_start_step)
            if not (0 <= image_condition_anchor_block_start_step <
                    sampling_steps):
                raise ValueError(
                    "image condition anchor block start step must be in "
                    f"[0, {sampling_steps})")
        if sacrificial_image_latent_index is not None:
            if not isinstance(sacrificial_image_latent_index, Integral):
                raise TypeError(
                    "sacrificial image latent index must be an integer")
            sacrificial_image_latent_index = int(
                sacrificial_image_latent_index)
        if (len(imgs) > 1
                and attention_probe is not None):
            raise ValueError(
                "attention probing is not supported while temporary image "
                "anchors are active")
        if init_video is not None:
            if init_video_noise_step is None:
                raise ValueError(
                    "init_video requires init_video_noise_step")
            if not isinstance(init_video_noise_step, Integral):
                raise TypeError("init video noise step must be an integer")
            init_video_noise_step = int(init_video_noise_step)
            if not 0 <= init_video_noise_step < sampling_steps:
                raise ValueError(
                    "init video noise step must be in "
                    f"[0, {sampling_steps})")
            if image_condition_switch_step is not None:
                raise ValueError(
                    "video refinement cannot use an image condition switch")
            if len(imgs) != 1 or image_latent_indices != [0]:
                raise ValueError(
                    "video refinement requires exactly one image condition "
                    "at latent index 0")
            if (not torch.is_tensor(init_video) or init_video.ndim != 4
                    or init_video.size(0) != 3):
                raise ValueError(
                    "init_video must be a [3, frame_num, H, W] tensor")
        elif init_video_noise_step is not None:
            raise ValueError("init_video_noise_step requires init_video")

        image_conditions = sorted(
            zip(image_latent_indices, imgs), key=lambda item: item[0])
        image_latent_indices = [item[0] for item in image_conditions]
        imgs = [
            TF.to_tensor(item[1]).sub_(0.5).div_(0.5)
            for item in image_conditions
        ]
        if len(imgs) > 1 and image_latent_indices[0] != 0:
            raise ValueError(
                "multi-image temporary-anchor conditioning requires an image "
                "at latent frame 0")
        sacrificial_image_latent_indices = (
            image_latent_indices[1:] if len(imgs) > 1 else [])
        if sacrificial_image_latent_index is not None:
            if (sacrificial_image_latent_index
                    not in sacrificial_image_latent_indices):
                raise ValueError(
                    "deprecated sacrificial image latent index must match "
                    "one of the nonzero image anchors")
            logging.warning(
                "--sacrificial_image_latent_index is deprecated; every "
                "nonzero image anchor is now treated as temporary and removed "
                "automatically after denoising")

        F = frame_num
        temporal_stride = self.vae_stride[0]
        if temporal_stride != 4:
            raise ValueError(
                "Wan I2V conditioning expects a temporal VAE stride of 4")
        if (F - 1) % temporal_stride != 0:
            raise ValueError(
                "I2V frame_num must be one plus a multiple of the temporal "
                "VAE stride")
        latent_frame_num = (F - 1) // temporal_stride + 1
        sacrificial_mode = bool(sacrificial_image_latent_indices)
        sacrificial_count = len(sacrificial_image_latent_indices)
        multi_latent_frame_num = latent_frame_num + sacrificial_count
        multi_frame_num = F + temporal_stride * sacrificial_count
        index_limit = (multi_latent_frame_num
                       if sacrificial_mode else latent_frame_num)
        if any(index < 0 or index >= index_limit
               for index in image_latent_indices):
            raise ValueError(
                "image latent frame indices must be in "
                f"[0, {index_limit})")
        if sacrificial_mode and any(
                anchor <= 0 or anchor >= multi_latent_frame_num - 1
                for anchor in sacrificial_image_latent_indices):
            raise ValueError(
                "every temporary image anchor must have retained frames on "
                "both sides of the expanded latent sequence")
        if sacrificial_mode:
            if image_condition_switch_step is None:
                logging.info(
                    "Sacrificial-anchor mode: sample %d latents (%d pixel "
                    "frames) with expanded multi-image conditioning for all "
                    "sampling steps; keep temporary latents %s through "
                    "denoising, then remove them before decoding %d latents "
                    "(%d output frames)", multi_latent_frame_num,
                    multi_frame_num, sacrificial_image_latent_indices,
                    latent_frame_num, F)
            else:
                logging.info(
                    "Sacrificial-anchor mode: sample %d latents (%d pixel "
                    "frames); disable expanded anchor conditioning at step "
                    "%d, keep temporary latents %s through denoising, then "
                    "remove them before decoding %d latents (%d output "
                    "frames)", multi_latent_frame_num, multi_frame_num,
                    image_condition_switch_step,
                    sacrificial_image_latent_indices, latent_frame_num, F)
            if image_condition_anchor_block_radius > 0:
                if image_condition_anchor_block_start_step is None:
                    logging.info(
                        "Image-anchor exclusion radius: %d ordinary query "
                        "frame(s) on each side of every non-first anchor, "
                        "active from transformer layer %d",
                        image_condition_anchor_block_radius,
                        image_condition_anchor_block_start_layer)
                else:
                    logging.info(
                        "Image-anchor exclusion radius: %d ordinary query "
                        "frame(s) on each side; active from layer %d before "
                        "sampling step %d, then active in all layers",
                        image_condition_anchor_block_radius,
                        image_condition_anchor_block_start_layer,
                        image_condition_anchor_block_start_step)
        if (attention_probe is not None
                and not 0 <= attention_probe["query_frame"] < latent_frame_num):
            raise ValueError(
                "attention probe query frame must be in "
                f"[0, {latent_frame_num})")
        if size is None:
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
        else:
            if (not isinstance(size, (tuple, list)) or len(size) != 2
                    or any(not isinstance(value, Integral) for value in size)):
                raise TypeError("size must be a (width, height) integer pair")
            w, h = (int(value) for value in size)
            if w <= 0 or h <= 0:
                raise ValueError("size dimensions must be positive")
            spatial_multiple_h = self.vae_stride[1] * self.patch_size[1]
            spatial_multiple_w = self.vae_stride[2] * self.patch_size[2]
            if h % spatial_multiple_h or w % spatial_multiple_w:
                raise ValueError(
                    "size dimensions must be divisible by the VAE-patch "
                    f"multiples ({spatial_multiple_w}, {spatial_multiple_h})")
            lat_h = h // self.vae_stride[1]
            lat_w = w // self.vae_stride[2]

        max_seq_len = multi_latent_frame_num * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            multi_latent_frame_num,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        # Every image anchor occupies every packed mask channel. Intermediate
        # anchors are repeated across their complete four-frame VAE group.
        multi_msk = torch.zeros(
            temporal_stride,
            multi_latent_frame_num,
            lat_h,
            lat_w,
            dtype=torch.float32,
            device=self.device)
        multi_msk[:, image_latent_indices] = 1

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

        resized_images = []
        crop_ranges = []
        for image in imgs:
            source_h, source_w = image.shape[-2:]
            resized_h, resized_w, top, left = center_crop_resize_geometry(
                source_h, source_w, h, w)
            if (resized_h, resized_w) != (source_h, source_w):
                image = torch.nn.functional.interpolate(
                    image[None],
                    size=(resized_h, resized_w),
                    mode='bicubic',
                    align_corners=False)[0]
            image = image[:, top:top + h, left:left + w]
            resized_images.append(image.unsqueeze(1).to(self.device))
            crop_ranges.append((source_w, source_h, resized_w, resized_h,
                                left, top, left + w, top + h))
        logging.info(
            "Conditioning image resize/crops (source_width, source_height, "
            "resized_width, resized_height, left, top, right, bottom): %s; "
            "output size: %dx%d",
            crop_ranges, w, h)
        refinement_video = None
        refinement_latents = None
        if init_video is not None:
            if init_video.size(1) != F:
                raise ValueError(
                    f"init_video must contain exactly {F} sampled frames")
            source_h, source_w = init_video.shape[-2:]
            target_aspect = w / h
            source_aspect = source_w / source_h
            if source_aspect > target_aspect:
                crop_w = max(1, round(source_h * target_aspect))
                left = (source_w - crop_w) // 2
                init_video = init_video[:, :, :, left:left + crop_w]
            elif source_aspect < target_aspect:
                crop_h = max(1, round(source_w / target_aspect))
                top = (source_h - crop_h) // 2
                init_video = init_video[:, :, top:top + crop_h, :]
            refinement_video = torch.nn.functional.interpolate(
                init_video.permute(1, 0, 2, 3),
                size=(h, w),
                mode='bicubic',
                align_corners=False,
            ).permute(1, 0, 2, 3).clamp_(-1, 1).to(self.device)
        if next(self.vae.model.parameters()).device.type == 'cpu':
            self.vae.model.to(self.device)
        if sacrificial_mode:
            # Before the switch, use the expanded continuous multi-image
            # encoding. Later images fill their complete four-pixel-frame VAE
            # groups and have [1, 1, 1, 1] packed masks.
            continuous_video = resized_images[0].new_zeros(
                3, multi_frame_num, h, w)
            for latent_index, image in zip(image_latent_indices,
                                           resized_images):
                start, end = pixel_frame_range_for_latent(
                    latent_index, temporal_stride)
                continuous_video[:, start:end] = image.expand(
                    -1, end - start, -1, -1)

            # The compressed global branch also predicts the disposable image
            # anchors. Its conditioning remains the complete joint causal VAE
            # encoding, including the ordinary cached Img2x4 anchor latent and
            # all later latents carrying that same continuous history.
            local_anchor_latent_count = None
            local_conditioning_videos = []
            if image_condition_dual_guidance:
                anchor_index = image_latent_indices[1]
                local_anchor_latent_count = multi_latent_frame_num - anchor_index
                local_anchor_frame_count = (
                    1 + temporal_stride * (local_anchor_latent_count - 1))
                local_anchor_video = resized_images[1].new_zeros(
                    3, local_anchor_frame_count, h, w)
                local_anchor_video[:, 0:1] = resized_images[1]
                # This independent suffix is used only by the Img2-local base;
                # it does not replace any joint multi-image latent.
                local_conditioning_videos = [local_anchor_video]

            needs_compact_global = (
                image_condition_switch_step is not None
                or image_condition_dual_guidance)
            if not needs_compact_global:
                conditioning_videos = [continuous_video,
                                       *local_conditioning_videos]
                continuous_encoding_index = 0
                local_encoding_index = 1
            else:
                # The compact Img1 view is used after the switch and as the
                # prefix base prediction in dual-guidance mode.
                global_video = resized_images[0].new_zeros(3, F, h, w)
                global_video[:, 0:1] = resized_images[0]
                global_mask = multi_msk.new_zeros(
                    temporal_stride, latent_frame_num, lat_h, lat_w)
                global_mask[:, 0] = 1
                conditioning_videos = [global_video, continuous_video,
                                       *local_conditioning_videos]
                continuous_encoding_index = 1
                local_encoding_index = 2

            encoded_conditioning = self.vae.encode(conditioning_videos)
            if len(encoded_conditioning) != len(conditioning_videos):
                raise ValueError(
                    "VAE did not return one encoding per conditioning video")
            continuous_latents = encoded_conditioning[
                continuous_encoding_index]
            if continuous_latents.size(1) != multi_latent_frame_num:
                raise ValueError(
                    "continuous multi-image VAE conditioning has the wrong "
                    "latent length")
            logging.info(
                "Retained joint cached Img2x4 conditioning at intermediate "
                "image anchors %s and the continuous causal VAE history in "
                "all later conditioning latents",
                image_latent_indices[1:])
            y_pre_switch = torch.concat([multi_msk, continuous_latents])
            if not needs_compact_global:
                y_global = y_pre_switch
            else:
                global_latents = encoded_conditioning[0]
                if global_latents.size(1) != latent_frame_num:
                    raise ValueError(
                        "global Img1 VAE conditioning has the wrong latent "
                        "length")
                y_global = torch.concat([global_mask, global_latents])

            y_local_anchor = None
            if image_condition_dual_guidance:
                local_latents = encoded_conditioning[local_encoding_index]
                if local_latents.size(1) != local_anchor_latent_count:
                    raise ValueError(
                        "Img2-local VAE conditioning has the wrong latent "
                        "length")
                local_mask = multi_msk.new_zeros(
                    temporal_stride, local_anchor_latent_count, lat_h, lat_w)
                local_mask[:, 0] = 1
                y_local_anchor = torch.concat([local_mask, local_latents])
                logging.info(
                    "Dual guidance: Img1-global prefix base, %d-latent "
                    "Img2-local suffix base, and expanded multi-image guide "
                    "with prefix/suffix scales %.3f/%.3f",
                    local_anchor_latent_count,
                    image_condition_prefix_multi_guidance_scale,
                    image_condition_suffix_multi_guidance_scale)
                if image_condition_multi_stop_step is not None:
                    logging.info(
                        "Dual guidance stops the expanded multi-image guide "
                        "at sampling step %d; later prefix is Img1-only, "
                        "anchor is Img2-local, and suffix uses %.1f%% "
                        "Img2-local / %.1f%% Img1-only",
                        image_condition_multi_stop_step,
                        100.0 * image_condition_late_suffix_img2_weight,
                        100.0 *
                        (1.0 - image_condition_late_suffix_img2_weight))

            if image_condition_multi_stop_step is not None:
                logging.info(
                    "Expanded multi-image guidance remains active through "
                    "sampling step %d; later dual-guidance steps use pure "
                    "Img1 prefix, pure Img2-local anchor, and the configured "
                    "Img1/Img2-local suffix blend",
                    image_condition_multi_stop_step - 1)
            elif image_condition_switch_step is None:
                logging.info(
                    "Expanded continuous multi-image conditioning and "
                    "compressed RoPE remain active for every sampling step")
            elif image_condition_rope_blend_start_step is None:
                logging.info(
                    "Hard-routed conditioning: expanded continuous multi-image "
                    "view, including globally predicted temporary anchors, "
                    "until sampling step %d; pre-switch retained-frame "
                    "prediction uses %.1f%% compact Img1-only regular I2V and "
                    "%.1f%% expanded multi-image compressed RoPE; then "
                    "%d-latent Img1-only global view",
                    image_condition_switch_step,
                    100.0 * (1.0 - image_condition_rope_blend_weight),
                    100.0 * image_condition_rope_blend_weight,
                    latent_frame_num)
            else:
                logging.info(
                    "Hard-routed conditioning: expanded continuous multi-image "
                    "view with globally predicted temporary anchors; "
                    "retained frames use fully compressed multi-image RoPE "
                    "through sampling step %d, then cosine-transition toward "
                    "compact Img1-only regular I2V and become fully regular at "
                    "sampling step %d; then %d-latent Img1-only global view",
                    image_condition_rope_blend_start_step,
                    image_condition_switch_step,
                    latent_frame_num)
        else:
            # Ordinary single-image I2V conditioning.
            conditioning_video = resized_images[0].new_zeros(
                3, multi_frame_num, h, w)
            image_pixel_ranges = []
            for latent_index, image in zip(image_latent_indices,
                                           resized_images):
                start, end = pixel_frame_range_for_latent(
                    latent_index, temporal_stride)
                conditioning_video[:, start:end] = image.expand(
                    -1, end - start, -1, -1)
                image_pixel_ranges.append((latent_index, start, end))
            logging.info("Conditioning image pixel-frame ranges: %s",
                         image_pixel_ranges)
            videos_to_encode = [conditioning_video]
            if refinement_video is not None:
                videos_to_encode.append(refinement_video)
            encoded_conditioning = self.vae.encode(videos_to_encode)
            continuous_latents = encoded_conditioning[0]
            if continuous_latents.size(1) != multi_latent_frame_num:
                raise ValueError(
                    "continuous VAE conditioning has the wrong latent length")
            y_global = torch.concat([multi_msk, continuous_latents])
            y_pre_switch = y_global
            if refinement_video is not None:
                refinement_latents = encoded_conditioning[1]
                if refinement_latents.shape != noise.shape:
                    raise ValueError(
                        "VAE-encoded init video shape does not match the "
                        "sampling latent shape: "
                        f"{tuple(refinement_latents.shape)} versus "
                        f"{tuple(noise.shape)}")

        # The DiT consumes only the packed masks and VAE latents above. Drop
        # the much larger pixel-space videos, plus redundant references to the
        # individual VAE outputs, before entering the denoising loop. Without
        # this cleanup these live Python references keep several GiB allocated
        # on every distributed rank for the duration of sampling.
        if sacrificial_mode:
            del conditioning_videos
            del continuous_video, continuous_latents
            if needs_compact_global:
                del global_video, global_latents, global_mask
            if image_condition_dual_guidance:
                del local_anchor_video, local_latents, local_mask
        else:
            del conditioning_video, videos_to_encode, continuous_latents
        del encoded_conditioning, resized_images, multi_msk, image
        if refinement_video is not None:
            del refinement_video
        if offload_model:
            # Encoding is complete and the VAE is not used again until final
            # decoding. Keep its small CUDA-resident scale tensors intact, but
            # move the model weights off GPU during the memory-intensive DiT
            # sampling loop.
            self.vae.model.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()

        retained_frame_indices = build_retained_frame_indices(
            multi_latent_frame_num, sacrificial_image_latent_indices)
        retained_index_tensor = torch.tensor(
            retained_frame_indices, dtype=torch.long, device=self.device)
        if len(retained_frame_indices) != latent_frame_num:
            raise ValueError("retained latent mapping has the wrong length")
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

            # Sample from pure noise, or forward-noise a VAE-encoded existing
            # video and continue only over the remaining schedule.
            sampling_step_offset = 0
            if refinement_latents is None:
                latent = noise
            else:
                refinement_timestep = timesteps[
                    init_video_noise_step:init_video_noise_step + 1]
                latent = sample_scheduler.add_noise(
                    refinement_latents.unsqueeze(0),
                    noise.unsqueeze(0),
                    refinement_timestep,
                ).squeeze(0)
                timesteps = timesteps[init_video_noise_step:]
                sampling_step_offset = init_video_noise_step
                logging.info(
                    "Video refinement: VAE-encoded %d frames into %d "
                    "latents, added noise at sampling step %d (timestep "
                    "%.4f), and will run %d denoising steps with first-frame "
                    "I2V conditioning",
                    F, refinement_latents.size(1),
                    init_video_noise_step,
                    float(refinement_timestep[0].item()), len(timesteps))

            arg_c = {
                'context': [context[0]],
                'seq_len': max_seq_len,
                'y': [y_global],
                'conditioned_frame_indices': (
                    None if sacrificial_mode else [image_latent_indices]),
                'sacrificial_frame_indices': None,
                'frame_attention_mode': None,
                'local_anchor_indices': None,
                'anchor_block_radius': 0,
                'anchor_block_start_layer': 0,
            }

            arg_null = {
                'context': context_null,
                'seq_len': max_seq_len,
                'y': [y_global],
                'conditioned_frame_indices': (
                    None if sacrificial_mode else [image_latent_indices]),
                'sacrificial_frame_indices': None,
                'frame_attention_mode': None,
                'local_anchor_indices': None,
                'anchor_block_radius': 0,
                'anchor_block_start_layer': 0,
            }

            if attention_probe is not None and attention_probe.get(
                    "output_path"):
                output_path = attention_probe["output_path"]
                output_directory = os.path.dirname(output_path)
                if output_directory:
                    os.makedirs(output_directory, exist_ok=True)
                with open(output_path, "w", encoding="utf-8"):
                    pass

            if offload_model:
                torch.cuda.empty_cache()

            for sampling_step, t in enumerate(
                    tqdm(timesteps), start=sampling_step_offset):
                active_anchor_block_start_layer = (
                    0 if image_condition_anchor_block_start_step is not None
                    and sampling_step >=
                    image_condition_anchor_block_start_step else
                    image_condition_anchor_block_start_layer)
                arg_c['anchor_block_radius'] = (
                    image_condition_anchor_block_radius)
                arg_null['anchor_block_radius'] = (
                    image_condition_anchor_block_radius)
                arg_c['anchor_block_start_layer'] = (
                    active_anchor_block_start_layer)
                arg_null['anchor_block_start_layer'] = (
                    active_anchor_block_start_layer)
                expanded_view_active = (
                    sacrificial_mode and (
                        image_condition_switch_step is None or
                        sampling_step < image_condition_switch_step))
                if (image_condition_switch_step is not None
                        and sampling_step == image_condition_switch_step):
                    logging.info(
                        "Disabled expanded temporary-anchor view at sampling step %d; "
                        "continuing with final-length Img1-only conditioning",
                        sampling_step)
                    if (sample_solver == 'unipc'
                            and image_condition_rope_blend_start_step is None):
                        # Conditioning changes discontinuously at this step.
                        # Drop only UniPC's multistep/corrector history so the
                        # first Img1-only update is first-order. Preserve the
                        # scheduler's step index, timesteps, and sigma schedule.
                        sample_scheduler.model_outputs = [
                            None
                        ] * len(sample_scheduler.model_outputs)
                        sample_scheduler.timestep_list = [
                            None
                        ] * len(sample_scheduler.timestep_list)
                        sample_scheduler.lower_order_nums = 0
                        sample_scheduler.last_sample = None
                        logging.info(
                            "Cleared UniPC multistep history at sampling step "
                            "%d without resetting timesteps or the noise "
                            "schedule", sampling_step)
                    elif sample_solver == 'unipc':
                        logging.info(
                            "Preserved UniPC multistep history at sampling "
                            "step %d because the conditioning prediction was "
                            "cosine-transitioned to regular I2V",
                            sampling_step)
                    else:
                        logging.warning(
                            "Conditioning switched at sampling step %d, but "
                            "scheduler-history reset is implemented only for "
                            "UniPC; %s history was left unchanged",
                            sampling_step, sample_solver)

                timestep = torch.stack([t]).to(self.device)

                model = self._prepare_model_for_timestep(
                    t, boundary, offload_model)
                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]

                step_probe = None
                if (attention_probe is not None
                        and sampling_step in attention_probe["sampling_steps"]):
                    step_probe = dict(
                        attention_probe,
                        sampling_step=sampling_step,
                        timestep=float(t.item()))
                arg_c['attention_probe'] = step_probe

                # Before the switch, blend a compact final-length Img1-only
                # view with the expanded compressed multi-image view for every
                # retained frame. The same global pass predicts temporary
                # anchors. After the switch, use only
                # ordinary final-length Img1-only I2V.
                expanded_prefix_view = expanded_view_active
                compact_global_latent = (
                    latent.index_select(1, retained_index_tensor)
                    if sacrificial_mode else latent)
                arg_c['frame_attention_mode'] = None
                arg_null['frame_attention_mode'] = None
                arg_c['local_anchor_indices'] = None
                arg_null['local_anchor_indices'] = None

                def run_global_prediction(model_latent, conditioning,
                                          conditioned_indices,
                                          sacrificial_indices):
                    arg_c['y'] = [conditioning]
                    arg_null['y'] = [conditioning]
                    arg_c['conditioned_frame_indices'] = conditioned_indices
                    arg_null['conditioned_frame_indices'] = (
                        conditioned_indices)
                    arg_c['sacrificial_frame_indices'] = sacrificial_indices
                    arg_null['sacrificial_frame_indices'] = (
                        sacrificial_indices)
                    noise_pred_cond = model(
                        [model_latent.to(self.device)],
                        t=timestep,
                        **arg_c)[0]
                    if offload_model:
                        torch.cuda.empty_cache()
                    noise_pred_uncond = model(
                        [model_latent.to(self.device)],
                        t=timestep,
                        **arg_null)[0]
                    if offload_model:
                        torch.cuda.empty_cache()
                    return noise_pred_uncond + sample_guide_scale * (
                        noise_pred_cond - noise_pred_uncond)

                if expanded_prefix_view:
                    if image_condition_switch_step is None:
                        blend_weight = 1.0
                    elif image_condition_rope_blend_start_step is None:
                        blend_weight = image_condition_rope_blend_weight
                    elif sampling_step <= image_condition_rope_blend_start_step:
                        blend_weight = 1.0
                    else:
                        blend_progress = (
                            (sampling_step -
                             image_condition_rope_blend_start_step) /
                            (image_condition_switch_step -
                             image_condition_rope_blend_start_step))
                        blend_weight = 0.5 * (
                            1.0 + math.cos(math.pi * blend_progress))
                    compressed_prediction = None
                    normal_prediction = None
                    late_single_image_blend = (
                        image_condition_dual_guidance
                        and image_condition_multi_stop_step is not None
                        and sampling_step >= image_condition_multi_stop_step)
                    if late_single_image_blend:
                        # The multi-image guide is intentionally absent in
                        # this late phase. Keep the prefix on standard Img1
                        # I2V, keep the disposable anchor exactly on its
                        # Img2-first local trajectory, and interpolate only
                        # the aligned retained suffix predictions.
                        normal_compact_prediction = run_global_prediction(
                            compact_global_latent,
                            y_global,
                            None,
                            None)
                        anchor_index = image_latent_indices[1]
                        local_model_latent = latent[:, anchor_index:]
                        local_prediction = run_global_prediction(
                            local_model_latent,
                            y_local_anchor,
                            None,
                            None)
                        if local_prediction.size(1) != (
                                latent.size(1) - anchor_index):
                            raise RuntimeError(
                                "Img2-local prediction length does not align "
                                "with the expanded suffix")
                        img1_suffix = normal_compact_prediction[:,
                                                                 anchor_index:]
                        img2_suffix = local_prediction[:, 1:]
                        if img1_suffix.size(1) != img2_suffix.size(1):
                            raise RuntimeError(
                                "Img1 and Img2-local suffix predictions do "
                                "not align after removing the temporary anchor")

                        global_prediction = torch.zeros_like(latent)
                        global_prediction[:, :anchor_index] = (
                            normal_compact_prediction[:, :anchor_index])
                        global_prediction[:, anchor_index:anchor_index + 1] = (
                            local_prediction[:, :1])
                        global_prediction[:, anchor_index + 1:] = (
                            (1.0 - image_condition_late_suffix_img2_weight) *
                            img1_suffix +
                            image_condition_late_suffix_img2_weight *
                            img2_suffix)
                        del (normal_compact_prediction, local_model_latent,
                             local_prediction, img1_suffix, img2_suffix)
                    elif image_condition_dual_guidance and blend_weight > 0.0:
                        # Evaluate three CFG predictions at the same x_t and
                        # timestep. Align them before taking deltas, then route
                        # the Img1-based result to the prefix and the
                        # Img2-local result to the anchor/suffix. UniPC still
                        # receives exactly one composite prediction.
                        compressed_prediction = run_global_prediction(
                            latent,
                            y_pre_switch,
                            [image_latent_indices],
                            [sacrificial_image_latent_indices])
                        normal_compact_prediction = run_global_prediction(
                            compact_global_latent,
                            y_global,
                            None,
                            None)
                        anchor_index = image_latent_indices[1]
                        local_model_latent = latent[:, anchor_index:]
                        local_prediction = run_global_prediction(
                            local_model_latent,
                            y_local_anchor,
                            None,
                            None)
                        if local_prediction.size(1) != (
                                latent.size(1) - anchor_index):
                            raise RuntimeError(
                                "Img2-local prediction length does not align "
                                "with the expanded suffix")

                        routed_prediction = torch.zeros_like(latent)
                        prefix_base = normal_compact_prediction[:, :anchor_index]
                        prefix_guide = compressed_prediction[:, :anchor_index]
                        routed_prediction[:, :anchor_index] = (
                            prefix_base +
                            image_condition_prefix_multi_guidance_scale *
                            (prefix_guide - prefix_base))
                        # Keep the disposable image anchor itself exactly on
                        # the Img2-first local-I2V trajectory. Apply the
                        # multi-image delta only to ordinary suffix frames.
                        routed_prediction[:, anchor_index:anchor_index + 1] = (
                            local_prediction[:, 0:1])
                        suffix_base = local_prediction[:, 1:]
                        suffix_guide = compressed_prediction[
                            :, anchor_index + 1:]
                        routed_prediction[:, anchor_index + 1:] = (
                            suffix_base +
                            image_condition_suffix_multi_guidance_scale *
                            (suffix_guide - suffix_base))

                        normal_prediction = torch.zeros_like(latent)
                        normal_prediction.index_copy_(
                            1, retained_index_tensor,
                            normal_compact_prediction)
                        if blend_weight == 1.0:
                            global_prediction = routed_prediction
                        else:
                            global_prediction = (
                                (1.0 - blend_weight) * normal_prediction +
                                blend_weight * routed_prediction)
                        # The compact Img1 branch has no temporary-anchor
                        # counterpart. Keep the active anchor exactly local
                        # even while retained frames cosine-transition toward
                        # compact Img1-only I2V.
                        global_prediction[
                            :, anchor_index:anchor_index + 1] = (
                                routed_prediction[
                                    :, anchor_index:anchor_index + 1])
                        del (local_model_latent, local_prediction,
                             normal_compact_prediction, normal_prediction,
                             prefix_base, prefix_guide, suffix_base,
                             suffix_guide, routed_prediction,
                             compressed_prediction)
                    else:
                        if blend_weight > 0.0:
                            compressed_prediction = run_global_prediction(
                                latent,
                                y_pre_switch,
                                [image_latent_indices],
                                [sacrificial_image_latent_indices])
                    if (not image_condition_dual_guidance
                            and blend_weight < 1.0):
                        # This branch is an actual regular-I2V view: remove
                        # temporary anchors from x, use final-length Img1-only
                        # conditioning, and apply ordinary global attention.
                        normal_compact_prediction = run_global_prediction(
                            compact_global_latent,
                            y_global,
                            None,
                            None)
                        normal_prediction = torch.zeros_like(latent)
                        normal_prediction.index_copy_(
                            1, retained_index_tensor,
                            normal_compact_prediction)
                        del normal_compact_prediction
                    if (not image_condition_dual_guidance):
                        if blend_weight == 0.0:
                            global_prediction = normal_prediction
                        elif blend_weight == 1.0:
                            global_prediction = compressed_prediction
                        else:
                            global_prediction = (
                                (1.0 - blend_weight) * normal_prediction +
                                blend_weight * compressed_prediction)
                        del normal_prediction, compressed_prediction
                    elif (blend_weight == 0.0
                          and not late_single_image_blend):
                        # The outer cosine transition has reached regular I2V;
                        # skip both the local and expanded guide branches.
                        normal_compact_prediction = run_global_prediction(
                            compact_global_latent,
                            y_global,
                            None,
                            None)
                        global_prediction = torch.zeros_like(latent)
                        global_prediction.index_copy_(
                            1, retained_index_tensor,
                            normal_compact_prediction)
                        del normal_compact_prediction
                else:
                    global_prediction = run_global_prediction(
                        compact_global_latent,
                        y_global,
                        None if sacrificial_mode else [image_latent_indices],
                        None)

                if sacrificial_mode:
                    noise_pred = torch.zeros_like(latent)

                    if expanded_view_active:
                        # The single compressed multi-image pass predicts both
                        # retained frames and disposable anchors. Temporary
                        # anchors are denoised normally in this same pass.
                        noise_pred.copy_(global_prediction)
                    else:
                        noise_pred.index_copy_(
                            1, retained_index_tensor, global_prediction)
                else:
                    noise_pred = global_prediction

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                del noise_pred
                latent = temp_x0.squeeze(0)
                del temp_x0

                x0 = [latent]
                del timestep, global_prediction

            if sacrificial_mode:
                latent = latent.index_select(1, retained_index_tensor)
                x0 = [latent]
                logging.info(
                    "Removed temporary image latents %s after denoising; "
                    "decoding %d final latents",
                    sacrificial_image_latent_indices, latent.size(1))
            if offload_model:
                if self.low_noise_model is not None:
                    self.low_noise_model.cpu()
                if self.high_noise_model is not None:
                    self.high_noise_model.cpu()
                gc.collect()
                torch.cuda.empty_cache()

            if self.rank == 0:
                if offload_model:
                    self.vae.model.to(self.device)
                videos = self.vae.decode(x0)

        del noise, latent, x0
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
