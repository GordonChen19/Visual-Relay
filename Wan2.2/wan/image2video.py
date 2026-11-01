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
                 image_prompts=None,
                 image_condition_one_way_prompt_guidance=False,
                 image_condition_one_way_prompt_guidance_weight=0.25,
                 image_condition_paired_anchor_attention=False,
                 image_condition_paired_stream_weight=0.25,
                 image_condition_paired_memory_alpha=None,
                 image_condition_paired_prefix_future_eta=0.0,
                 image_condition_paired_fade_start_step=None,
                 image_condition_paired_fade_end_step=None,
                 image_condition_paired_suffix_fade_start_step=None,
                 image_condition_paired_suffix_fade_end_step=None,
                 image_condition_paired_prefix_route_switch_step=None,
                 image_condition_self_attention_guidance=False,
                 image_condition_self_attention_guidance_scale=0.5,
                 image_condition_self_attention_guidance_scale_end=None,
                 image_condition_self_attention_guidance_scale_end_step=None,
                 image_condition_self_attention_guidance_img2_weight=0.25,
                 image_condition_self_attention_guidance_post_stop_img2_suffix=False,
                 image_condition_self_attention_guidance_joint_query_single_memory_alpha=None,
                 image_condition_self_attention_guidance_selective_suffix_memory_alpha=None,
                 image_condition_self_attention_guidance_selective_suffix_memory_alpha_end=None,
                 image_condition_self_attention_guidance_selective_suffix_memory_alpha_end_step=None,
                 image_condition_self_attention_guidance_rms_cap=None,
                 image_condition_self_attention_guidance_anchor_prior=False,
                 image_condition_self_attention_guidance_anchor_prior_start=0.8,
                 image_condition_self_attention_guidance_anchor_prior_end=0.2,
                 image_condition_self_attention_guidance_anchor_prior_end_step=None,
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
                Enable Img1-only C/M and fresh-cache Img2-first B. Retained C
                queries use C prefix plus aligned C/B1: suffix K/V in one
                source-prior-normalized attention. B0 is excluded from C
                memory and maintains only the hidden transport state.
            image_condition_paired_stream_weight (`float`, *optional*):
                Legacy fallback for paired-memory alpha.
            image_condition_paired_memory_alpha (`float`, *optional*):
                B-source prior used when retained C queries attend to aligned
                C/B suffix K/V inside one normalization. The complementary
                source priors prevent duplicated suffix attention mass.
            image_condition_paired_prefix_future_eta (`float`, *optional*):
                Legacy compatibility option; ignored by paired M/B memory
                attention.
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
            image_condition_self_attention_guidance_scale_end (`float`, *optional*):
                Optional final J attention-output weight for cosine scheduling.
            image_condition_self_attention_guidance_scale_end_step (`int`, *optional*):
                Zero-based step at which the J weight reaches its final value.
            image_condition_self_attention_guidance_img2_weight (`float`, *optional*):
                B attention-output residual weight used by optional post-stop
                Img2 suffix guidance.
            image_condition_self_attention_guidance_post_stop_img2_suffix (`bool`, *optional*):
                After J stops, retain M plus an aligned Img2-first B stream.
                B1: self-attention outputs guide only M's retained suffix;
                neither the prefix nor temporary anchor receives B updates.
            image_condition_self_attention_guidance_joint_query_single_memory_alpha (`float`, *optional*):
                Optional Img1-only M-memory prior for joint J prefix and suffix
                queries. J and M memories are split into exact LSE-merged
                attention groups, preventing automatic mass doubling. The
                results directly update M's aligned prefix and suffix without
                completed-output mixing. The joint anchor query remains J-only.
            image_condition_self_attention_guidance_selective_suffix_memory_alpha (`float`, *optional*):
                Optional M-source prior applied only to aligned J/M suffix
                memory. J prefix queries read unique J prefix, the J anchor and
                paired J/M suffix. At the anchor/first-suffix shared RoPE
                position, the anchor prior divides anchor versus neighbour
                mass and alpha divides the neighbour mass between J/M. J
                suffix queries read J prefix, the J anchor and paired J/M
                suffix. The result directly updates aligned M/J positions.
            image_condition_self_attention_guidance_selective_suffix_memory_alpha_end (`float`, *optional*):
                Optional final M-suffix source prior for a cosine alpha
                schedule. Omit it to keep selective alpha constant.
            image_condition_self_attention_guidance_selective_suffix_memory_alpha_end_step (`int`, *optional*):
                Zero-based sampling step at which the selective alpha schedule
                reaches its final value. It remains fixed afterward.
            image_condition_self_attention_guidance_rms_cap (`float`, *optional*):
                Optional per-frame, per-head J-M attention residual RMS cap,
                expressed relative to M's attention-output RMS.
            image_condition_self_attention_guidance_anchor_prior (`bool`, *optional*):
                Enable exact prior-weighted shared-position attention for J.
            image_condition_self_attention_guidance_anchor_prior_start (`float`, *optional*):
                Anchor share at the first denoising step.
            image_condition_self_attention_guidance_anchor_prior_end (`float`, *optional*):
                Final anchor share.
            image_condition_self_attention_guidance_anchor_prior_end_step (`int`, *optional*):
                Zero-based sampling step at which the anchor prior reaches its
                end value. It remains fixed afterward. Defaults to the final
                sampling step.
            image_condition_self_attention_guidance_stop_step (`int`, *optional*):
                Zero-based sampling step at which J and B are removed. Both
                CFG calls then use Img1-only M with no guidance or anchor
                velocity substitution. The temporary slot stays isolated.

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
        enabled_multi_image_modes = sum(bool(value) for value in (
            image_condition_one_way_prompt_guidance,
            image_condition_paired_anchor_attention,
            image_condition_self_attention_guidance,
        ))
        if enabled_multi_image_modes > 1:
            raise ValueError(
                "multi-image attention ablations are mutually exclusive")
        if not 0 <= image_condition_one_way_prompt_guidance_weight <= 1:
            raise ValueError("one-way Img2 guidance weight must be in [0, 1]")
        if not 0 <= image_condition_self_attention_guidance_scale <= 1:
            raise ValueError(
                "self-attention guidance scale must be in [0, 1]")
        if image_condition_self_attention_guidance_scale_end is not None:
            if not image_condition_self_attention_guidance:
                raise ValueError(
                    "guidance-scale scheduling requires self-attention guidance")
            if not (0 <= image_condition_self_attention_guidance_scale_end <=
                    image_condition_self_attention_guidance_scale):
                raise ValueError(
                    "guidance-scale end must be in [0, guidance-scale]")
        if image_condition_self_attention_guidance_scale_end_step is not None:
            if image_condition_self_attention_guidance_scale_end is None:
                raise ValueError(
                    "guidance-scale end step requires guidance-scale end")
            if not (1 <=
                    image_condition_self_attention_guidance_scale_end_step <
                    sampling_steps):
                raise ValueError(
                    "guidance-scale end step must be in [1, sampling_steps)")
        if not 0 <= image_condition_self_attention_guidance_img2_weight <= 1:
            raise ValueError(
                "self-attention Img2 weight must be in [0, 1]")
        if image_condition_self_attention_guidance_post_stop_img2_suffix:
            if not image_condition_self_attention_guidance:
                raise ValueError(
                    "post-stop Img2 suffix guidance requires self-attention "
                    "guidance")
            if image_condition_self_attention_guidance_stop_step is None:
                raise ValueError(
                    "post-stop Img2 suffix guidance requires a stop step")
        if (image_condition_self_attention_guidance_joint_query_single_memory_alpha
                is not None):
            if not image_condition_self_attention_guidance:
                raise ValueError(
                    "joint-query split memory requires self-attention guidance")
            if not (0 <=
                    image_condition_self_attention_guidance_joint_query_single_memory_alpha
                    <= 1):
                raise ValueError(
                    "joint-query single-memory alpha must be in [0, 1]")
        if (image_condition_self_attention_guidance_selective_suffix_memory_alpha
                is not None):
            if not image_condition_self_attention_guidance:
                raise ValueError(
                    "selective suffix memory requires self-attention guidance")
            if not (0 <=
                    image_condition_self_attention_guidance_selective_suffix_memory_alpha
                    <= 1):
                raise ValueError(
                    "selective suffix-memory alpha must be in [0, 1]")
        if (image_condition_self_attention_guidance_selective_suffix_memory_alpha_end
                is not None):
            if (image_condition_self_attention_guidance_selective_suffix_memory_alpha
                    is None):
                raise ValueError(
                    "selective suffix-memory alpha end requires selective "
                    "suffix-memory mode")
            if not (0 <=
                    image_condition_self_attention_guidance_selective_suffix_memory_alpha_end
                    <= 1):
                raise ValueError(
                    "selective suffix-memory alpha end must be in [0, 1]")
        if (image_condition_self_attention_guidance_selective_suffix_memory_alpha_end_step
                is not None):
            if (image_condition_self_attention_guidance_selective_suffix_memory_alpha_end
                    is None):
                raise ValueError(
                    "selective suffix-memory alpha end step requires alpha end")
            if not (1 <=
                    image_condition_self_attention_guidance_selective_suffix_memory_alpha_end_step
                    < sampling_steps):
                raise ValueError(
                    "selective suffix-memory alpha end step must be in "
                    "[1, sampling_steps)")
        if (image_condition_self_attention_guidance_joint_query_single_memory_alpha
                is not None
                and image_condition_self_attention_guidance_selective_suffix_memory_alpha
                is not None):
            raise ValueError(
                "full-memory and selective-suffix memory modes are exclusive")
        if (image_condition_paired_memory_alpha is not None
                and not 0 <= image_condition_paired_memory_alpha <= 1):
            raise ValueError("paired memory alpha must be in [0, 1]")
        if (image_condition_self_attention_guidance_rms_cap is not None
                and image_condition_self_attention_guidance_rms_cap <= 0):
            raise ValueError(
                "self-attention guidance RMS cap must be positive")
        if image_condition_self_attention_guidance_anchor_prior:
            if not image_condition_self_attention_guidance:
                raise ValueError(
                    "anchor-prior scheduling requires self-attention guidance")
            if not (0.0 <
                    image_condition_self_attention_guidance_anchor_prior_end
                    <= image_condition_self_attention_guidance_anchor_prior_start
                    < 1.0):
                raise ValueError(
                    "anchor-prior values must satisfy 0 < end <= start < 1")
        if image_condition_self_attention_guidance_anchor_prior_end_step is not None:
            if not image_condition_self_attention_guidance_anchor_prior:
                raise ValueError(
                    "anchor-prior end step requires anchor-prior scheduling")
            if not (1 <=
                    image_condition_self_attention_guidance_anchor_prior_end_step
                    < sampling_steps):
                raise ValueError(
                    "anchor-prior end step must be in [1, sampling_steps)")
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
        if (image_condition_self_attention_guidance
                or image_condition_one_way_prompt_guidance) and any(
                value is not None for value in (
                    image_condition_paired_fade_start_step,
                    image_condition_paired_fade_end_step,
                    image_condition_paired_suffix_fade_start_step,
                    image_condition_paired_suffix_fade_end_step,
                    image_condition_paired_prefix_route_switch_step)):
            raise ValueError(
                "paired fading and prefix routing do not apply to the "
                "selected guidance ablation")
        temporary_anchor_mode = (image_condition_paired_anchor_attention
                                 or image_condition_self_attention_guidance)
        multi_stream_mode = (temporary_anchor_mode
                             or image_condition_one_way_prompt_guidance)
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
            if image_condition_one_way_prompt_guidance:
                if (not isinstance(image_prompts, (list, tuple))
                        or len(image_prompts) != 2
                        or not all(isinstance(value, str) and value
                                   for value in image_prompts)):
                    raise ValueError(
                        "one-way prompt guidance requires two non-empty "
                        "image prompts")
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
        # Legacy paired modes insert Img2 as a temporary diffusion latent instead of
        # replacing an output latent. It is removed before VAE decoding, so an
        # 81-frame request still decodes 21 retained latents.
        model_latent_frame_num = (latent_frame_num + 1
                                  if (temporary_anchor_mode
                                      or image_condition_one_way_prompt_guidance)
                                  else latent_frame_num)
        max_seq_len = model_latent_frame_num * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            (latent_frame_num if image_condition_one_way_prompt_guidance else
             model_latent_frame_num),
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
        text_prompts = (list(image_prompts)
                        if image_condition_one_way_prompt_guidance else
                        [input_prompt])
        null_prompts = [n_prompt] * len(text_prompts)
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder(text_prompts, self.device)
            context_null = self.text_encoder(null_prompts, self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder(text_prompts, torch.device('cpu'))
            context_null = self.text_encoder(null_prompts, torch.device('cpu'))
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
        if image_condition_one_way_prompt_guidance:
            suffix_latent_frame_num = latent_frame_num - anchor_index
            suffix_frame_num = (1 + self.vae_stride[0]
                                * (suffix_latent_frame_num - 1))
            encoded = self.vae.encode([
                conditioning_video(imgs[0], F),
                joint_anchor_conditioning_video(
                    imgs[0], imgs[1], F, anchor_index),
                conditioning_video(imgs[1], suffix_frame_num),
            ])
            if (encoded[0].size(1) != latent_frame_num
                    or encoded[1].size(1) != latent_frame_num
                    or encoded[2].size(1) != suffix_latent_frame_num):
                raise RuntimeError(
                    "one-way prompt-guidance conditioning lengths do not "
                    "align")
            # Keep ordinary Img1-only conditioning everywhere except the real
            # output anchor. Only that slot is taken from the joint causal
            # Img1 -> Img2 encode; joint suffix history is intentionally
            # discarded so frames after the anchor remain standard Img1 I2V.
            primary_mask = insert_latent_slot(msk, anchor_index)
            primary_mask[:, anchor_index:anchor_index + 1].fill_(1)
            primary_encoded = insert_latent_slot(encoded[0], anchor_index)
            primary_encoded[:, anchor_index:anchor_index + 1].copy_(
                encoded[1][:, anchor_index:anchor_index + 1])
            primary_y = torch.concat([primary_mask, primary_encoded])
            suffix_y = torch.concat([
                first_frame_mask(suffix_frame_num), encoded[2]
            ])
            del encoded, primary_mask, primary_encoded
            logging.info(
                "One-way per-image-prompt guidance: expanded %d-latent "
                "main stream M uses prompt 1 and full bidirectional attention. "
                "It retains the ordinary Img1-only M%d anchor and adds one "
                "auxiliary joint Img1-to-Img2 anchor at the same integer RoPE "
                "position. Standard %d-latent Img2-first B uses prompt 2 "
                "and evolves independently. Prefix queries read the auxiliary "
                "anchor key paired with B0's value but not ordinary M%d; "
                "ordinary anchor/suffix queries "
                "read ordinary M%d but not the auxiliary anchor. At every "
                "DiT layer the auxiliary anchor uses its own query to attend "
                "to the prefix plus itself; B0 completely replaces ordinary "
                "M%d's attention output, and B1: updates only M%d: at weight "
                "%.3f. "
                "The auxiliary anchor is "
                "discarded before the scheduler, and the retained M velocity "
                "is used without B0 replacement. At the next denoising step "
                "the auxiliary anchor is recreated from M%d's current noisy "
                "latent. Information flows only B -> M and retained M "
                "positions are not compressed",
                model_latent_frame_num, anchor_index, suffix_latent_frame_num,
                anchor_index, anchor_index, anchor_index, anchor_index + 1,
                image_condition_one_way_prompt_guidance_weight, anchor_index)
        elif image_condition_paired_anchor_attention:
            suffix_latent_frame_num = model_latent_frame_num - anchor_index
            suffix_frame_num = (1 + self.vae_stride[0]
                                * (suffix_latent_frame_num - 1))
            encoded = self.vae.encode([
                conditioning_video(imgs[0], F),
                conditioning_video(imgs[1], suffix_frame_num),
            ])
            if (encoded[0].size(1) != latent_frame_num
                    or encoded[1].size(1) != suffix_latent_frame_num):
                raise RuntimeError("paired B/C conditioning lengths do not align")
            expanded_msk = insert_latent_slot(msk, anchor_index)
            expanded_main = insert_latent_slot(encoded[0], anchor_index)
            # C's inserted slot is transport only and is absent from retained
            # attention. Every retained C frame therefore stays Img1-only.
            expanded_msk[:, anchor_index:anchor_index + 1].zero_()
            expanded_main[:, anchor_index:anchor_index + 1].zero_()
            suffix_y = torch.concat([
                first_frame_mask(suffix_frame_num), encoded[1]
            ])
            hybrid_y = torch.concat([expanded_msk, expanded_main])
            paired_memory_alpha = (
                image_condition_paired_stream_weight
                if image_condition_paired_memory_alpha is None else
                image_condition_paired_memory_alpha)
            del encoded, expanded_msk, expanded_main
            logging.info(
                "Paired M/B memory attention: Img1-only C/M and fresh-cache "
                "Img2-first B. C transport slot %d is hidden. Every retained "
                "C query reads C prefix plus aligned C/B1: suffix K/V inside "
                "one exact normalization with B-source alpha %.3f. B0 is not "
                "C memory; it maintains the hidden transport state and is "
                "removed before decoding. No joint multi-image stream or H "
                "proxy is used",
                anchor_index, paired_memory_alpha)
            if suffix_fade_start_step is not None:
                logging.info("B memory-alpha cosine fade: steps %d through %d",
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
                "standard %d-latent Img2-first suffix. Under default routing, "
                "J-prefix and J-suffix queries both use J's single history-aware "
                "joint anchor in one mathematical softmax across all visible "
                "keys, including anchor-adjacent representations sharing a "
                "RoPE coordinate. "
                "The "
                "anchor reads itself plus J's suffix at every layer. Removing "
                "J's anchor "
                "aligns J with M's retained positions. Before handoff, both "
                "prefix and suffix M initially use M/J weights %.3f/%.3f "
                "and B1: is "
                "excluded. After handoff, both CFG calls use Img1-only M "
                "without J/B guidance. B0's predicted velocity replaces M's "
                "temporary anchor velocity only in conditional calls before "
                "handoff; the slot is discarded before decoding. "
                "Guidance stop step: %s",
                model_latent_frame_num, anchor_index,
                guidance_local_latent_frame_num,
                (1.0 - image_condition_self_attention_guidance_scale),
                image_condition_self_attention_guidance_scale,
                ("disabled" if
                 image_condition_self_attention_guidance_stop_step is None
                 else str(image_condition_self_attention_guidance_stop_step)))
            if image_condition_self_attention_guidance_scale_end is not None:
                logging.info(
                    "Self-attention J guidance-scale cosine schedule: %.3f "
                    "-> %.3f by sampling step %d, then held at %.3f",
                    image_condition_self_attention_guidance_scale,
                    image_condition_self_attention_guidance_scale_end,
                    (sampling_steps - 1 if
                     image_condition_self_attention_guidance_scale_end_step
                     is None else
                     image_condition_self_attention_guidance_scale_end_step),
                    image_condition_self_attention_guidance_scale_end)
            if (image_condition_self_attention_guidance_joint_query_single_memory_alpha
                    is not None):
                logging.info(
                    "Joint-query split prefix/suffix memory enabled: J prefix "
                    "and suffix Q read full J K/V and retained Img1-only M K/V with "
                    "M-source prior alpha %.3f; FlashAttention LSE merging "
                    "prevents automatic memory-mass doubling, and the split "
                    "outputs directly update M's aligned prefix and suffix",
                    image_condition_self_attention_guidance_joint_query_single_memory_alpha)
            if (image_condition_self_attention_guidance_selective_suffix_memory_alpha
                    is not None):
                logging.info(
                    "Selective suffix-memory attention enabled: J prefix Q "
                    "reads unique J prefix, J anchor and aligned J/M suffix "
                    "with hierarchical shared-position mass control; J "
                    "suffix Q reads J prefix, J anchor and aligned J/M "
                    "suffix. M-source suffix prior alpha %.3f; "
                    "the exact LSE-merged outputs directly update aligned M/J",
                    image_condition_self_attention_guidance_selective_suffix_memory_alpha)
                if (image_condition_self_attention_guidance_selective_suffix_memory_alpha_end
                        is not None):
                    logging.info(
                        "Selective suffix-memory alpha cosine schedule: %.3f "
                        "-> %.3f by sampling step %d, then held at %.3f",
                        image_condition_self_attention_guidance_selective_suffix_memory_alpha,
                        image_condition_self_attention_guidance_selective_suffix_memory_alpha_end,
                        (sampling_steps - 1 if
                         image_condition_self_attention_guidance_selective_suffix_memory_alpha_end_step
                         is None else
                         image_condition_self_attention_guidance_selective_suffix_memory_alpha_end_step),
                        image_condition_self_attention_guidance_selective_suffix_memory_alpha_end)
            if image_condition_self_attention_guidance_post_stop_img2_suffix:
                logging.info(
                    "After J stops, aligned Img2-first B1: self-attention "
                    "outputs guide only M's retained suffix at weight %.3f; "
                    "M prefix and temporary anchor receive no B update, and "
                    "B0 does not replace anchor velocity",
                    image_condition_self_attention_guidance_img2_weight)
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

            anchor_prior_schedule_end_step = (
                sampling_steps - 1
                if image_condition_self_attention_guidance_anchor_prior_end_step
                is None else
                image_condition_self_attention_guidance_anchor_prior_end_step)
            guidance_scale_schedule_end_step = (
                sampling_steps - 1
                if image_condition_self_attention_guidance_scale_end_step
                is None else
                image_condition_self_attention_guidance_scale_end_step)
            selective_alpha_schedule_end_step = (
                sampling_steps - 1
                if image_condition_self_attention_guidance_selective_suffix_memory_alpha_end_step
                is None else
                image_condition_self_attention_guidance_selective_suffix_memory_alpha_end_step)

            # sample videos
            latent = noise

            if image_condition_one_way_prompt_guidance:
                arg_c = {
                    'context': [context[0], context[1]],
                    'seq_len': max_seq_len,
                    'y': [primary_y, suffix_y],
                    'one_way_img2_anchor_index': anchor_index,
                    'one_way_img2_weight':
                    image_condition_one_way_prompt_guidance_weight,
                }
                arg_null = {
                    'context': [context_null[0], context_null[1]],
                    'seq_len': max_seq_len,
                    'y': [primary_y, suffix_y],
                    'one_way_img2_anchor_index': anchor_index,
                    'one_way_img2_weight':
                    image_condition_one_way_prompt_guidance_weight,
                }
            elif image_condition_paired_anchor_attention:
                arg_c = {
                    'context': [context[0], context[0]],
                    'seq_len': max_seq_len,
                    'y': [suffix_y, hybrid_y],
                    'paired_anchor_index': anchor_index,
                    'paired_stream_weight': paired_memory_alpha,
                    'paired_prefix_anchor_active': True,
                    'paired_prefix_future_eta':
                    image_condition_paired_prefix_future_eta,
                }
                arg_null = {
                    'context': [context_null[0], context_null[0]],
                    'seq_len': max_seq_len,
                    'y': [suffix_y, hybrid_y],
                    'paired_anchor_index': anchor_index,
                    'paired_stream_weight': paired_memory_alpha,
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
                    'self_attention_guidance_joint_query_single_memory_alpha':
                    image_condition_self_attention_guidance_joint_query_single_memory_alpha,
                    'self_attention_guidance_selective_suffix_memory_alpha':
                    image_condition_self_attention_guidance_selective_suffix_memory_alpha,
                    'self_attention_guidance_rms_cap':
                    image_condition_self_attention_guidance_rms_cap,
                    'self_attention_guidance_anchor_prior': None,
                    'attention_guidance_anchor_index': anchor_index,
                }
                m_only_arg_c = {
                    'context': [context[0]],
                    'seq_len': max_seq_len,
                    'y': [primary_y],
                    'self_attention_guidance_scale': 0.0,
                    'self_attention_guidance_img2_weight': 0.0,
                    'self_attention_guidance_joint_query_single_memory_alpha':
                    None,
                    'self_attention_guidance_selective_suffix_memory_alpha':
                    None,
                    'self_attention_guidance_rms_cap': None,
                    'self_attention_guidance_anchor_prior': None,
                    'attention_guidance_anchor_index': anchor_index,
                }
                arg_null = {
                    'context': [context_null[0], context_null[0],
                                context_null[0]],
                    'seq_len': max_seq_len,
                    'y': [primary_y, joint_y, guidance_local_y],
                    'self_attention_guidance_scale':
                    image_condition_self_attention_guidance_scale,
                    'self_attention_guidance_img2_weight':
                    image_condition_self_attention_guidance_img2_weight,
                    'self_attention_guidance_joint_query_single_memory_alpha':
                    image_condition_self_attention_guidance_joint_query_single_memory_alpha,
                    'self_attention_guidance_selective_suffix_memory_alpha':
                    image_condition_self_attention_guidance_selective_suffix_memory_alpha,
                    'self_attention_guidance_rms_cap':
                    image_condition_self_attention_guidance_rms_cap,
                    'self_attention_guidance_anchor_prior': None,
                    'attention_guidance_anchor_index': anchor_index,
                }
                m_only_arg_null = {
                    'context': [context_null[0]],
                    'seq_len': max_seq_len,
                    'y': [primary_y],
                    'self_attention_guidance_scale': 0.0,
                    'self_attention_guidance_img2_weight': 0.0,
                    'self_attention_guidance_joint_query_single_memory_alpha':
                    None,
                    'self_attention_guidance_selective_suffix_memory_alpha':
                    None,
                    'self_attention_guidance_rms_cap': None,
                    'self_attention_guidance_anchor_prior': None,
                    'attention_guidance_anchor_index': anchor_index,
                }
                post_stop_m_b_arg_c = {
                    'context': [context[0], context[0]],
                    'seq_len': max_seq_len,
                    'y': [primary_y, guidance_local_y],
                    'self_attention_guidance_scale': 0.0,
                    'self_attention_guidance_img2_weight':
                    image_condition_self_attention_guidance_img2_weight,
                    'self_attention_guidance_joint_query_single_memory_alpha':
                    None,
                    'self_attention_guidance_selective_suffix_memory_alpha':
                    None,
                    'self_attention_guidance_rms_cap': None,
                    'self_attention_guidance_anchor_prior': None,
                    'attention_guidance_anchor_index': anchor_index,
                }
                post_stop_m_b_arg_null = {
                    'context': [context_null[0], context_null[0]],
                    'seq_len': max_seq_len,
                    'y': [primary_y, guidance_local_y],
                    'self_attention_guidance_scale': 0.0,
                    'self_attention_guidance_img2_weight':
                    image_condition_self_attention_guidance_img2_weight,
                    'self_attention_guidance_joint_query_single_memory_alpha':
                    None,
                    'self_attention_guidance_selective_suffix_memory_alpha':
                    None,
                    'self_attention_guidance_rms_cap': None,
                    'self_attention_guidance_anchor_prior': None,
                    'attention_guidance_anchor_index': anchor_index,
                }
                logging.info(
                    "Self-attention guidance uses the same M/J/B image "
                    "routing in both CFG calls; only the text context differs. "
                    "Before handoff, B0 replaces the temporary anchor velocity "
                    "independently in both predictions. After handoff, both "
                    "calls run %s. Shared-position anchor prior: %s",
                    ("Img1-only M plus layerwise B1: suffix guidance"
                     if image_condition_self_attention_guidance_post_stop_img2_suffix
                     else "Img1-only M"),
                    ("disabled" if not
                     image_condition_self_attention_guidance_anchor_prior else
                     "cosine %.3f -> %.3f by sampling step %d, then held at "
                     "%.3f" % (
                         image_condition_self_attention_guidance_anchor_prior_start,
                         image_condition_self_attention_guidance_anchor_prior_end,
                         anchor_prior_schedule_end_step,
                         image_condition_self_attention_guidance_anchor_prior_end)))
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
                if len(predictions) != 2:
                    raise RuntimeError(
                        "paired model must return B and C predictions")
                suffix_prediction, hybrid_prediction = predictions
                if (suffix_prediction.size(1) != suffix_latent_frame_num
                        or hybrid_prediction.size(1) != model_latent_frame_num):
                    raise RuntimeError(
                        "paired prediction lengths do not match B/C")
                # Reproduce the early local-anchor routing at prediction
                # level: after all DiT layers, substitute B0's Img2-first
                # velocity for the temporary C anchor. This happens
                # independently for conditional and unconditional predictions
                # before CFG, so the scheduler receives the CFG-combined B0
                # velocity at the anchor while evolving the latent normally.
                hybrid_prediction = hybrid_prediction.clone()
                hybrid_prediction[:, anchor_index:anchor_index + 1].copy_(
                    suffix_prediction[:, :1])
                return hybrid_prediction

            for sampling_step, t in enumerate(tqdm(timesteps)):
                current_arg_c = arg_c
                current_arg_null = arg_null
                self_attention_guidance_active = False
                post_stop_img2_suffix_active = False
                if image_condition_one_way_prompt_guidance:
                    current_latent = latent.to(self.device)
                    latent_model_input = [
                        insert_latent_slot(current_latent, anchor_index),
                        current_latent[:, anchor_index:],
                    ]
                    timestep = [t, t]
                elif image_condition_paired_anchor_attention:
                    suffix_fade_factor = paired_fade_factor(
                        sampling_step, suffix_fade_start_step,
                        suffix_fade_end_step)
                    anchor_is_enabled = False
                    current_stream_weight = (
                        paired_memory_alpha
                        * suffix_fade_factor)
                    arg_c['paired_stream_weight'] = current_stream_weight
                    arg_c['paired_prefix_anchor_active'] = anchor_is_enabled
                    arg_null['paired_stream_weight'] = current_stream_weight
                    arg_null['paired_prefix_anchor_active'] = anchor_is_enabled
                    latent_model_input = [
                        latent[:, anchor_index:].to(self.device),
                        latent.to(self.device),
                    ]
                    timestep = [t, t]
                elif image_condition_self_attention_guidance:
                    current_latent = latent.to(self.device)
                    self_attention_guidance_active = (
                        image_condition_self_attention_guidance_stop_step is
                        None or sampling_step <
                        image_condition_self_attention_guidance_stop_step)
                    if self_attention_guidance_active:
                        current_guidance_scale = (
                            image_condition_self_attention_guidance_scale)
                        if (image_condition_self_attention_guidance_scale_end
                                is not None):
                            scale_progress = min(
                                1.0,
                                sampling_step /
                                guidance_scale_schedule_end_step)
                            scale_cosine_weight = 0.5 * (
                                1.0 + math.cos(math.pi * scale_progress))
                            current_guidance_scale = (
                                image_condition_self_attention_guidance_scale_end
                                + (
                                    image_condition_self_attention_guidance_scale
                                    - image_condition_self_attention_guidance_scale_end
                                ) * scale_cosine_weight)
                        current_arg_c['self_attention_guidance_scale'] = (
                            current_guidance_scale)
                        current_arg_null['self_attention_guidance_scale'] = (
                            current_guidance_scale)
                        current_selective_alpha = (
                            image_condition_self_attention_guidance_selective_suffix_memory_alpha)
                        if (current_selective_alpha is not None
                                and image_condition_self_attention_guidance_selective_suffix_memory_alpha_end
                                is not None):
                            alpha_progress = min(
                                1.0,
                                sampling_step /
                                selective_alpha_schedule_end_step)
                            alpha_cosine_weight = 0.5 * (
                                1.0 + math.cos(math.pi * alpha_progress))
                            current_selective_alpha = (
                                image_condition_self_attention_guidance_selective_suffix_memory_alpha_end
                                + (
                                    image_condition_self_attention_guidance_selective_suffix_memory_alpha
                                    - image_condition_self_attention_guidance_selective_suffix_memory_alpha_end
                                ) * alpha_cosine_weight)
                        current_arg_c[
                            'self_attention_guidance_selective_suffix_memory_alpha'] = (
                                current_selective_alpha)
                        current_arg_null[
                            'self_attention_guidance_selective_suffix_memory_alpha'] = (
                                current_selective_alpha)
                        current_anchor_prior = None
                        if image_condition_self_attention_guidance_anchor_prior:
                            schedule_progress = (
                                1.0 if anchor_prior_schedule_end_step == 0
                                else min(
                                    1.0,
                                    sampling_step /
                                    anchor_prior_schedule_end_step))
                            cosine_weight = 0.5 * (
                                1.0 + math.cos(math.pi * schedule_progress))
                            current_anchor_prior = (
                                image_condition_self_attention_guidance_anchor_prior_end
                                + (
                                    image_condition_self_attention_guidance_anchor_prior_start
                                    - image_condition_self_attention_guidance_anchor_prior_end
                                ) * cosine_weight)
                        current_arg_c[
                            'self_attention_guidance_anchor_prior'] = (
                                current_anchor_prior)
                        current_arg_null[
                            'self_attention_guidance_anchor_prior'] = (
                                current_anchor_prior)
                        # M/J start from the exact same current noisy latent;
                        # B receives its aligned anchor/suffix slice.
                        latent_model_input = [
                            current_latent,
                            current_latent,
                            current_latent[:, anchor_index:],
                        ]
                        timestep = [t, t, t]
                    else:
                        post_stop_img2_suffix_active = (
                            image_condition_self_attention_guidance_post_stop_img2_suffix)
                        if post_stop_img2_suffix_active:
                            # M remains the primary state. B0 is present only
                            # so B1: can evolve as a standard Img2-first I2V
                            # suffix; attention.py aligns B1: with M's retained
                            # post-anchor frames and never injects B0.
                            latent_model_input = [
                                current_latent,
                                current_latent[:, anchor_index:],
                            ]
                            timestep = [t, t]
                            current_arg_c = post_stop_m_b_arg_c
                            current_arg_null = post_stop_m_b_arg_null
                        else:
                            # Pure Img1 I2V on retained positions. The isolated
                            # transport slot remains only for scheduler layout.
                            latent_model_input = [current_latent]
                            timestep = [t]
                            current_arg_c = m_only_arg_c
                            current_arg_null = m_only_arg_null
                        if (sampling_step ==
                                image_condition_self_attention_guidance_stop_step):
                            if post_stop_img2_suffix_active:
                                logging.info(
                                    "Joint J guidance stopped at sampling "
                                    "step %d: M/B suffix-residual guidance "
                                    "begins; B1: updates only retained frames "
                                    "after Img2 and B0 anchor velocity is not "
                                    "used",
                                    sampling_step)
                            else:
                                logging.info(
                                    "Self-attention guidance stopped at "
                                    "sampling step %d: J and B are disabled; "
                                    "both CFG calls run Img1-only M with no "
                                    "attention guidance or anchor velocity "
                                    "replacement",
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

                predictions_cond = model(
                    latent_model_input, t=timestep, **current_arg_c)
                if image_condition_one_way_prompt_guidance:
                    if len(predictions_cond) != 2:
                        raise RuntimeError(
                            "one-way prompt guidance returned the wrong "
                            "number of conditional predictions")
                    expanded_prediction_cond = predictions_cond[0]
                    if (expanded_prediction_cond.size(1) !=
                            model_latent_frame_num):
                        raise RuntimeError(
                            "one-way M conditional prediction has the wrong "
                            "expanded temporal length")
                    if (predictions_cond[1].size(1) !=
                            suffix_latent_frame_num):
                        raise RuntimeError(
                            "one-way Img2 conditional prediction has the "
                            "wrong temporal length")
                    # Remove only the auxiliary joint anchor. The retained
                    # Img1-only M anchor/suffix already received B's layerwise
                    # attention guidance; do not replace any final velocity.
                    noise_pred_cond = torch.cat([
                        expanded_prediction_cond[:, :anchor_index],
                        expanded_prediction_cond[:, anchor_index + 1:]
                    ], dim=1)
                    del predictions_cond
                elif image_condition_paired_anchor_attention:
                    noise_pred_cond = assemble_paired_prediction(
                        predictions_cond)
                    del predictions_cond
                elif image_condition_self_attention_guidance:
                    expected_predictions = (
                        3 if self_attention_guidance_active else
                        (2 if post_stop_img2_suffix_active else 1))
                    if len(predictions_cond) != expected_predictions:
                        raise RuntimeError(
                            "self-attention guidance returned the wrong "
                            "number of conditional predictions")
                    noise_pred_cond = predictions_cond[0]
                    if self_attention_guidance_active:
                        suffix_prediction_cond = predictions_cond[2]
                        if (suffix_prediction_cond.size(1) !=
                                guidance_local_latent_frame_num):
                            raise RuntimeError(
                                "B conditional prediction has the "
                                "wrong temporal length")
                        noise_pred_cond = noise_pred_cond.clone()
                        noise_pred_cond[:, anchor_index:anchor_index + 1].copy_(
                            suffix_prediction_cond[:, :1])
                        del suffix_prediction_cond
                    elif post_stop_img2_suffix_active:
                        # B has already influenced M suffix attention inside
                        # every DiT layer. Its final prediction is auxiliary;
                        # do not replace the anchor or any scheduler velocity.
                        if (predictions_cond[1].size(1) !=
                                guidance_local_latent_frame_num):
                            raise RuntimeError(
                                "post-stop B conditional prediction has the "
                                "wrong temporal length")
                    del predictions_cond
                else:
                    noise_pred_cond = predictions_cond[0]
                    del predictions_cond
                if offload_model:
                    torch.cuda.empty_cache()
                predictions_uncond = model(
                    latent_model_input, t=timestep, **current_arg_null)
                if image_condition_one_way_prompt_guidance:
                    if len(predictions_uncond) != 2:
                        raise RuntimeError(
                            "one-way prompt guidance returned the wrong "
                            "number of unconditional predictions")
                    expanded_prediction_uncond = predictions_uncond[0]
                    if (expanded_prediction_uncond.size(1) !=
                            model_latent_frame_num):
                        raise RuntimeError(
                            "one-way M unconditional prediction has the "
                            "wrong expanded temporal length")
                    if (predictions_uncond[1].size(1) !=
                            suffix_latent_frame_num):
                        raise RuntimeError(
                            "one-way Img2 unconditional prediction has the "
                            "wrong temporal length")
                    noise_pred_uncond = torch.cat([
                        expanded_prediction_uncond[:, :anchor_index],
                        expanded_prediction_uncond[:, anchor_index + 1:]
                    ], dim=1)
                    del predictions_uncond
                elif image_condition_paired_anchor_attention:
                    noise_pred_uncond = assemble_paired_prediction(
                        predictions_uncond)
                    del predictions_uncond
                elif image_condition_self_attention_guidance:
                    expected_predictions = (
                        3 if self_attention_guidance_active else
                        (2 if post_stop_img2_suffix_active else 1))
                    if len(predictions_uncond) != expected_predictions:
                        raise RuntimeError(
                            "self-attention guidance returned the wrong "
                            "number of unconditional predictions")
                    noise_pred_uncond = predictions_uncond[0]
                    if self_attention_guidance_active:
                        suffix_prediction_uncond = predictions_uncond[2]
                        if (suffix_prediction_uncond.size(1) !=
                                guidance_local_latent_frame_num):
                            raise RuntimeError(
                                "B unconditional prediction has the wrong "
                                "temporal length")
                        noise_pred_uncond = noise_pred_uncond.clone()
                        noise_pred_uncond[:,
                                          anchor_index:anchor_index + 1].copy_(
                                              suffix_prediction_uncond[:, :1])
                    elif post_stop_img2_suffix_active:
                        # As in the conditional pass, B is used only for
                        # layerwise suffix attention guidance.
                        if (predictions_uncond[1].size(1) !=
                                guidance_local_latent_frame_num):
                            raise RuntimeError(
                                "post-stop B unconditional prediction has "
                                "the wrong temporal length")
                    if noise_pred_uncond.size(1) != model_latent_frame_num:
                        raise RuntimeError(
                            "unconditional prediction has the wrong temporal "
                            "length")
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
                if (image_condition_self_attention_guidance
                        or image_condition_one_way_prompt_guidance):
                    del current_latent

            if offload_model:
                self.low_noise_model.cpu()
                if self.high_noise_model is not None:
                    self.high_noise_model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                if temporary_anchor_mode:
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
