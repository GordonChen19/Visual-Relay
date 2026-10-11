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
                 image_condition_paired_anchor_attention=False,
                 image_condition_paired_stream_weight=0.25):
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
                Enable the per-layer Img1-main/Img2-suffix attention fusion.
            image_condition_paired_stream_weight (`float`, *optional*):
                Fixed weight assigned to the full Img2 suffix-attention output
                when paired mode fuses suffix tokens at every DiT layer.

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
        if image_condition_paired_anchor_attention:
            if not 0.0 <= image_condition_paired_stream_weight <= 1.0:
                raise ValueError(
                    "paired stream weight must be between zero and one")
            if not isinstance(img, (list, tuple)) or len(img) != 2:
                raise ValueError(
                    "paired anchor attention requires exactly two images")
            if (image_latent_indices is None
                    or len(image_latent_indices) != 2
                    or image_latent_indices[0] != 0
                    or image_latent_indices[1] <= 0):
                raise ValueError(
                    "paired anchor attention requires indices [0, positive]")
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
        if (image_condition_paired_anchor_attention
                and anchor_index >= latent_frame_num - 1):
            raise ValueError(
                "paired Img2 anchor must have at least one latent frame on "
                "its right")
        # Paired mode inserts Img2 as a temporary diffusion latent instead of
        # replacing an output latent. It is removed before VAE decoding, so an
        # 81-frame request still decodes 21 retained latents.
        model_latent_frame_num = (latent_frame_num + 1
                                  if image_condition_paired_anchor_attention
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

        if image_condition_paired_anchor_attention:
            suffix_latent_frame_num = model_latent_frame_num - anchor_index
            suffix_frame_num = (1 + self.vae_stride[0]
                                * (suffix_latent_frame_num - 1))
            encoded = self.vae.encode([
                conditioning_video(imgs[0], F),
                conditioning_video(imgs[1], suffix_frame_num),
            ])
            if (encoded[0].size(1) != latent_frame_num
                    or encoded[1].size(1) != suffix_latent_frame_num):
                raise RuntimeError(
                    "paired conditioning VAE lengths do not align")
            # Duplicate the aligned Img1-only conditioning slot solely as a
            # shape placeholder. The temporary anchor is excluded from Stream
            # A K/V, and its final prediction is copied entirely from Stream B.
            expanded_msk = insert_latent_slot(msk, anchor_index)
            expanded_main = insert_latent_slot(encoded[0], anchor_index)
            y = torch.concat([expanded_msk, expanded_main])
            suffix_y = torch.concat([
                first_frame_mask(suffix_frame_num), encoded[1]
            ])
            logging.info(
                "Paired anchor attention: %d retained latents plus temporary "
                "Img2 anchor at %d (%d working latents), with a %d-latent "
                "fresh-cache Img2 suffix; fixed full-suffix fusion weight "
                "%.3f. The temporary anchor is removed before decoding",
                latent_frame_num, anchor_index, model_latent_frame_num,
                suffix_latent_frame_num,
                image_condition_paired_stream_weight)
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
                    'context': [context[0], context[0]],
                    'seq_len': max_seq_len,
                    'y': [y, suffix_y],
                    'paired_anchor_index': anchor_index,
                    'paired_stream_weight': image_condition_paired_stream_weight,
                }
                arg_null = {
                    'context': [context_null[0], context_null[0]],
                    'seq_len': max_seq_len,
                    'y': [y, suffix_y],
                    'paired_anchor_index': anchor_index,
                    'paired_stream_weight': image_condition_paired_stream_weight,
                }
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

            for _, t in enumerate(tqdm(timesteps)):
                if image_condition_paired_anchor_attention:
                    latent_model_input = [
                        latent.to(self.device),
                        latent[:, anchor_index:].to(self.device),
                    ]
                    timestep = [t, t]
                else:
                    latent_model_input = [latent.to(self.device)]
                    timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                model = self._prepare_model_for_timestep(
                    t,
                    boundary,
                    offload_model,
                    release_high_noise_model=
                    image_condition_paired_anchor_attention)
                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]

                predictions_cond = model(
                    latent_model_input, t=timestep, **arg_c)
                if image_condition_paired_anchor_attention:
                    predictions_cond[0][:, anchor_index:anchor_index + 1].copy_(
                        predictions_cond[1][:, :1])
                noise_pred_cond = predictions_cond[0]
                del predictions_cond
                if offload_model:
                    torch.cuda.empty_cache()
                predictions_uncond = model(
                    latent_model_input, t=timestep, **arg_null)
                if image_condition_paired_anchor_attention:
                    predictions_uncond[0][:,
                                          anchor_index:anchor_index + 1].copy_(
                                              predictions_uncond[1][:, :1])
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

            if offload_model:
                self.low_noise_model.cpu()
                if self.high_noise_model is not None:
                    self.high_noise_model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                if image_condition_paired_anchor_attention:
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
