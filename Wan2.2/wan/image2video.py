# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import copy
import gc
import json
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
from .modules.attention import override_frame_weights, reverse_stream_weight
from .modules.attn_probe import ATTN_PROBE
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def fuse_reverse_prediction(noise_pred, rev_pred, frame_weights):
    """Visual Relay round 2, Stream R prediction fusion.

    noise_pred: [C, F, H, W] guided prediction of the working latents.
    rev_pred:   [C, a, H, W] guided Stream-R prediction, Stream-R order
                (frame j is working latent a - j; frame 0 is the anchor slot).
    Prefix latent t = 1..a-1 becomes (1 - w_t) noise_pred[t] + w_t rev_pred[a - t];
    every other latent is unchanged.
    """
    anchor_index = rev_pred.size(1)
    fused = noise_pred.clone()
    for frame in range(1, anchor_index):
        weight = float(frame_weights[frame])
        if weight == 0.0:
            continue
        fused[:, frame] = ((1.0 - weight) * noise_pred[:, frame]
                           + weight * rev_pred[:, anchor_index - frame])
    return fused



def build_step_plan(num_steps, resample=None):
    """Visual Relay round 4, resampling. Returns the visiting order as
    (step, first_visit, restart) triples. resample = (J, K, U): after step K,
    go back to step J's noise level and run steps J..K again, U times."""
    if not resample or resample[2] <= 0:
        return [(step, True, False) for step in range(num_steps)]
    first, last, cycles = (int(v) for v in resample)
    if not 0 <= first <= last < num_steps:
        raise ValueError("resample steps must satisfy 0 <= from <= to < steps")
    plan = [(step, True, False) for step in range(last + 1)]
    for _ in range(cycles):
        plan += [(step, False, step == first) for step in range(first, last + 1)]
    plan += [(step, True, False) for step in range(last + 1, num_steps)]
    return plan


def relay_stat_targets(stats1, stats2, frames, anchor_index):
    """Visual Relay round 4, colour anchor. stats = (mean, std), each [C], of
    a static 4-frame latent of Img1 / Img2. Latent f gets the straight
    interpolation at lam = min(f / a, 1): Img1 at 0, Img2 from the Img2
    latent on. Returns mean and std, each [C, F, 1, 1]."""
    lam = torch.tensor([min(f / anchor_index, 1.0) for f in range(frames)],
                       dtype=torch.float32, device=stats1[0].device)
    mean = ((1 - lam)[None] * stats1[0][:, None]
            + lam[None] * stats2[0][:, None])
    std = ((1 - lam)[None] * stats1[1][:, None]
           + lam[None] * stats2[1][:, None])
    return mean[..., None, None], std[..., None, None]


def anchor_frame_stats(velocity, sample, sigma, target_mean, target_std=None,
                       strength=1.0, first_frame=1):
    """Visual Relay round 4, colour anchor. Moves each latent frame's
    per-channel spatial mean (and, with target_std, its spatial std) of the
    predicted clean latent x0 = sample - sigma * v toward the targets, then
    returns the velocity that predicts the corrected x0 from the same sample.
    Frames before first_frame (Img1) are left alone.

    velocity, sample: [C, F, H, W]; target_mean / target_std: [C, F, 1, 1]."""
    if sigma <= 0:
        return velocity
    x0 = sample.float() - sigma * velocity.float()
    part = x0[:, first_frame:]
    mean = part.mean(dim=(2, 3), keepdim=True)
    new = part
    if target_std is not None:
        std = part.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        ratio = (1.0 - strength) + strength * (
            target_std[:, first_frame:] / std)
        new = (part - mean) * ratio + mean
    new = new + strength * (target_mean[:, first_frame:] - mean)
    x0 = torch.cat([x0[:, :first_frame], new], dim=1)
    return ((sample.float() - x0) / sigma).to(velocity.dtype)


def pin_latent_prediction(velocity, sample, sigma, index, target):
    """Visual Relay round 6, Direction B (render pass). Flow matching: the
    predicted clean latent is x0 = sample - sigma * v. Replace latent frame
    ``index`` of x0 by ``target`` ([C, H, W]) and return the velocity that
    predicts the pinned x0 from the same sample; every other frame keeps its
    prediction exactly (up to float rounding)."""
    if sigma <= 0:
        raise ValueError("pinning needs sigma > 0")
    x0 = sample.float() - sigma * velocity.float()
    x0[:, index] = target.to(x0.dtype)
    return ((sample.float() - x0) / sigma).to(velocity.dtype)


def smooth_latent_stats(z, lo, hi, mode):
    """Visual Relay round 10 (round 9's decode fix, same as smooth_stats in
    vr_latent_fix.py). z [C, T, h, w]: latents lo..hi get each channel's
    spatial mean ('mean') or mean and std ('meanstd') on the straight line
    between latents lo-1 and hi+1; latent j sits at (j - lo + 1) /
    (hi - lo + 2) of the way. Each latent keeps its spatial pattern."""
    if not 1 <= lo <= hi < z.shape[1] - 1:
        raise ValueError(
            f"decode fix window {lo}-{hi} needs a latent on each side "
            f"(latents 0..{z.shape[1] - 1})")
    if mode not in ('mean', 'meanstd'):
        raise ValueError(f"unknown decode fix mode {mode}")
    out = z.clone()
    m0, m1 = z[:, lo - 1].mean(dim=(1, 2)), z[:, hi + 1].mean(dim=(1, 2))
    s0, s1 = z[:, lo - 1].std(dim=(1, 2)), z[:, hi + 1].std(dim=(1, 2))
    n = hi - lo + 2
    for j in range(lo, hi + 1):
        t = (j - lo + 1) / n
        mean_t, std_t = m0 + t * (m1 - m0), s0 + t * (s1 - s0)
        cur = z[:, j]
        cur_mean = cur.mean(dim=(1, 2))[:, None, None]
        if mode == 'mean':
            out[:, j] = cur - cur_mean + mean_t[:, None, None]
        else:
            cur_std = cur.std(dim=(1, 2)).clamp(min=1e-6)[:, None, None]
            out[:, j] = ((cur - cur_mean) / cur_std * std_t[:, None, None]
                         + mean_t[:, None, None])
    return out


def gaussian_blur_latent(x, sigma):
    """Visual Relay round 7. Separable Gaussian blur over the last two
    (spatial) dimensions, replicate padding; sigma in latent pixels, <= 0
    returns x unchanged. Keeps layout-level structure only."""
    if sigma <= 0:
        return x
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32,
                          device=x.device)
    kernel = torch.exp(-0.5 * (coords / sigma)**2)
    kernel = kernel / kernel.sum()
    shape = x.shape
    height, width = shape[-2], shape[-1]
    weights = [float(v) for v in kernel.tolist()]
    # Weighted sums of shifted copies, not conv2d: exact float32 on every
    # device, whatever the sampler's bfloat16 autocast or cuDNN's TF32 do.
    with torch.amp.autocast(x.device.type, enabled=False):
        flat = x.reshape(-1, 1, height, width).float()
        padded = torch.nn.functional.pad(flat, (radius, radius, 0, 0),
                                         mode='replicate')
        flat = sum(wk * padded[..., k:k + width]
                   for k, wk in enumerate(weights))
        padded = torch.nn.functional.pad(flat, (0, 0, radius, radius),
                                         mode='replicate')
        flat = sum(wk * padded[..., k:k + height, :]
                   for k, wk in enumerate(weights))
    return flat.reshape(shape).to(x.dtype)


def reach_pattern(channels, height, width, blur, seed):
    """Visual Relay round 7, reach test. A layout-level nudge: Gaussian noise
    [channels, height, width] from a CPU generator (identical on every rank),
    blurred by ``blur`` latent pixels, scaled to RMS 1."""
    gen = torch.Generator(device='cpu')
    gen.manual_seed(int(seed))
    pattern = torch.randn(channels, height, width, generator=gen)
    pattern = gaussian_blur_latent(pattern, blur)
    return pattern / pattern.pow(2).mean().sqrt().clamp_min(1e-12)


def reach_response(sample, sigma, norm, delta, v_plus, v_minus, v_base,
                   blur):
    """Visual Relay round 7, reach test. Response of the predicted clean
    latent x0 = x - sigma v to the nudges x +- delta, from the velocities at
    x + delta, x - delta and x. Per latent frame k, RMS over channels and
    space, divided by ``norm`` (the nudge RMS):

      raw / low:   first-order change (x0(x+d) - x0(x-d)) / 2, unblurred and
                   after the same blur as the nudge (layout level);
      model_low:   the model's part of it, -sigma (v+ - v-) / 2, blurred
                   (equal to low except on the nudged latent itself);
      second_low:  second-order part (x0(x+d) + x0(x-d)) / 2 - x0(x), blurred.
    Returns a dict of [F] tensors."""
    sample = sample.float()
    x0_plus = (sample + delta) - sigma * v_plus.float()
    x0_minus = (sample - delta) - sigma * v_minus.float()
    first = 0.5 * (x0_plus - x0_minus)
    second = 0.5 * (x0_plus + x0_minus) - (sample - sigma * v_base.float())
    model_part = -sigma * 0.5 * (v_plus.float() - v_minus.float())

    def per_frame(value):
        return value.pow(2).mean(dim=(0, 2, 3)).sqrt() / norm

    return dict(raw=per_frame(first),
                low=per_frame(gaussian_blur_latent(first, blur)),
                model_low=per_frame(gaussian_blur_latent(model_part, blur)),
                second_low=per_frame(gaussian_blur_latent(second, blur)))


def reach_summary(record):
    """Visual Relay round 7: human-readable table of a reach-test record."""
    meta = record['meta']
    a = meta['target']
    lines = [
        "Visual Relay round 7: forward-only reach test",
        f"plain I2V of Img1, relay prompt from latent {meta['relay_from']}; "
        f"target = latent {a} (Img2's position); conditional pass only",
        f"nudge: one latent at a time, Gaussian noise blurred by "
        f"{meta['blur']} latent px, RMS = {meta['eps']} x RMS(x_t), both "
        f"signs; {meta['patterns']} pattern(s)",
        "response = RMS change of the predicted clean latent "
        "x0 = x_t - sigma v at the target, per unit of nudge RMS, after the "
        "same blur (layout level). 'vs self' divides by the response to a "
        "nudge on the target itself. 'floor' = the nudge on latent a-1 x "
        "1/100, in full-nudge units (mostly bfloat16 rounding noise). 'x1/4' = the nudge on "
        "latent a-1 at a quarter of the size (equal to the a-1 row if the "
        "response is linear).", ""]
    for step in sorted(record['steps'], key=int):
        rec = record['steps'][step]
        lines.append(f"step {step} (t = {rec['timestep']:.0f}, sigma = "
                     f"{rec['sigma']:.3f}, nudge RMS {rec['nudge_rms']:.4f})")
        pats = rec['patterns']
        frames = len(pats[0]['rows'])
        resp = [[p['rows'][j]['low'][a] for p in pats] for j in range(frames)]
        mean = [sum(r) / len(r) for r in resp]
        self_scale = max(mean[a], 1e-12)
        floor = sum(p['floor']['low'][a] for p in pats) / len(pats)
        lines.append("  nudged latent | response at target per pattern | "
                     "mean | vs self | 2nd order / 1st")
        for j in range(frames):
            second = [p['rows'][j]['second_low'][a] for p in pats]
            ratio2 = (sum(second) / len(second)) / max(mean[j], 1e-12)
            mark = '  <- target' if j == a else ''
            lines.append(
                f"  {j:3d} (d={a - j:+3d}) | "
                + ' '.join(f"{v:.4f}" for v in resp[j])
                + f" | {mean[j]:.4f} | {mean[j] / self_scale:6.3f} | "
                f"{ratio2:.2f}{mark}")
        quarter = [p['quarter']['low'][a] for p in pats]
        lines.append(f"  floor at target: {floor:.4f} ({floor / self_scale:.3f}"
                     f" of self); latent {a - 1} nudged x1/4: "
                     f"{sum(quarter) / len(quarter):.4f} (full: "
                     f"{mean[a - 1]:.4f})")
        lines.append("")
    return "\n".join(lines) + "\n"


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
            if convert_model_dtype:
                # from_pretrained loads float32 weights and FSDP keeps them in
                # float32 (14B params = 28.6 GB per GPU at 2-way sharding), too
                # much for 40 GB A100s. FSDP already casts every parameter to
                # bfloat16 for compute (MixedPrecision param_dtype), so storing
                # bfloat16 gives the same forward pass at half the memory.
                model.to(self.param_dtype)
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
                 image_condition_paired_fade_start_step=None,
                 image_condition_paired_fade_end_step=None,
                 image_condition_paired_suffix_fade_start_step=None,
                 image_condition_paired_suffix_fade_end_step=None,
                 image_condition_paired_prefix_route_switch_step=None,
                 image_condition_paired_variant=None,
                 image_condition_paired_variant_until_step=None,
                 probe_steps=None,
                 probe_samples=48,
                 probe_path=None,
                 suffix_prompt=None,
                 keyframe_guidance_scale=0.0,
                 keyframe_guidance_until_step=10,
                 reverse_prompt=None,
                 reverse_stream_mode='off',
                 reverse_stream_strength=1.0,
                 reverse_stream_shape='linear',
                 reverse_stream_until_step=10,
                 reverse_stream_couple=False,
                 override_stream=None,
                 override_min_weight=1.0,
                 override_radius=3,
                 override_blend='linear',
                 override_latent='shared',
                 override_img2_span='latent',
                 override_save_stream2=False,
                 relay_prompt=None,
                 relay_from_latent=None,
                 stat_anchor='off',
                 stat_strength=1.0,
                 stat_streams='one',
                 stat_std_from_step=10,
                 resample_from_step=None,
                 resample_to_step=None,
                 resample_cycles=0,
                 override_profile='damp',
                 override_handover_step=None,
                 stream_b_weight=None,
                 stream_b_fade_start_step=None,
                 stream_b_fade_end_step=None,
                 override_mode='blend',
                 rerender_from_step=None,
                 img2_motion_frames=None,
                 reach_test=None,
                 img2_context_video=None,
                 img2_fill='img2',
                 img2_code_stats=None,
                 save_latents=False,
                 img2_release_step=None,
                 decode_fix=None,
                 decode_fix_window=None,
                 img2_span_schedule=None,
                 img2_release_input='joint',
                 override_img2_mask=None,
                 override_img2_value=None,
                 override_single=False):
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
                Hard anchor-removal step. Prefix queries always attend ordinary
                Stream-A frames on both sides of the anchor. Before this step,
                they additionally attend a hybrid anchor using the
                history-aware Stream-A key and fresh-cache Stream-B Img2
                value; at and after it, only that hybrid route is removed.
            image_condition_paired_variant (`dict`, *optional*):
                Visual Relay round-1 change of the prefix -> anchor route
                (see ``wan.modules.attention.VARIANT_DEFAULTS``). ``None``
                keeps the baseline route exactly.
            image_condition_paired_variant_until_step (`int`, *optional*):
                Use the variant only for sampling steps below this index;
                later steps use the baseline route.
            probe_steps (sequence[int], *optional*):
                Sampling steps at which the read-only attention probe records
                the conditional pass. ``None`` disables the probe.
            probe_samples (`int`, *optional*):
                Spatial positions sampled per query latent frame.
            probe_path (`str`, *optional*):
                Output file of the probe (rank 0 also writes a summary).
            suffix_prompt (`str`, *optional*):
                Visual Relay round 2: own prompt for the Img2-first Stream B
                (default: the main prompt).
            keyframe_guidance_scale (`float`, *optional*):
                Visual Relay round 2 keyframe guidance k. For prefix latents
                1..anchor-1 the prediction becomes
                uncond + g (cond - cond_uncond) + k (cond - cond_without_Img2),
                where cond_without_Img2 is an extra conditional pass with the
                anchor route off and Stream-B fusion weight 0. 0 disables it.
            keyframe_guidance_until_step (`int`, *optional*):
                Apply keyframe guidance only for sampling steps below this.
            reverse_prompt (`str`, *optional*):
                Visual Relay round 2: prompt of Stream R, describing the
                prefix played backward from Img2 (default: the main prompt).
            reverse_stream_mode (`str`, *optional*):
                Stream R, an ordinary Img2-first I2V stream that runs backward
                in time over prefix latents a-1..1. 'off' (default); 'attn'
                fuses its attention output into prefix frame t at every layer
                with weight w_t (the mirror of Stream B in the suffix); 'pred'
                runs it as a separate plain I2V pass and fuses its guided
                prediction into prefix frame t with weight w_t.
            reverse_stream_strength (`float`, *optional*):
                s in w_t = s * f(t / anchor), in [0, 1].
            reverse_stream_shape (`str`, *optional*):
                f: 'linear' (t / anchor) or 'cosine'.
            reverse_stream_until_step (`int`, *optional*):
                Use Stream R only for sampling steps below this index.
            override_stream (`str`, *optional*):
                Visual Relay round 3, Gordon's Method 1. Two streams share the
                prompt and the noisy latent: stream 1 is plain I2V of Img1,
                stream 2 also has Img2 at latent ``image_latent_indices[1]``.
                In every layer stream 1 swaps its self-attention output for
                (a blend with) stream 2's; stream 1 is the video. 'joint':
                stream 2's conditioning is one causal VAE pass of
                Img1, zeros, Img2 x4, zeros. 'splice': Img1-only encoding with
                only the Img2 latent taken from that joint pass. None = off.
            override_min_weight (`float`, *optional*):
                Blend weight w at the Img2 latent; it rises linearly to 1 at
                ``override_radius`` latents away. 1 = full override.
            override_radius (`int`, *optional*):
                Distance in latents over which the weight returns to 1.
            override_blend (`str`, *optional*):
                'linear' (O1 + w (O2 - O1)) or 'norm' (same direction, length
                (1 - w)|O1| + w|O2| per token and head).
            override_latent (`str`, *optional*):
                'shared': stream 2 reads stream 1's noisy latent every step.
                'own': stream 2 starts from the same noise but keeps its own
                latent, updated with its own prediction.
            reverse_stream_couple (`bool`, *optional*):
                Visual Relay round 3, 'attn' mode only: Stream R frame a - t
                also takes the mixed output of prefix latent t, so Stream R
                sees Img1's side and both streams share one trajectory.
            override_img2_span (`str`, *optional*):
                Visual Relay round 4. 'latent': stream 2 holds Img2 for all
                4 frames of its latent (round 3). 'frame': Img2 is only the
                first frame of that latent, with the mask on that frame only,
                so the next 3 frames are free.
            override_save_stream2 (`bool`, *optional*):
                With override_latent='own', also decode stream 2's latent; the
                video is left in ``self.last_stream2_video``.
            relay_prompt (`str`, *optional*):
                Prompt relay: latent frames >= relay_from_latent (default: the
                one after Img2) read this prompt in cross-attention, in both
                streams; earlier frames keep ``input_prompt``.
            stat_anchor (`str`, *optional*):
                Colour anchor: 'off', 'mean' or 'meanstd'. Every step, each
                latent frame's per-channel mean (and std from
                ``stat_std_from_step``) of stream 1's predicted clean latent is
                moved toward a straight Img1 -> Img2 interpolation, by
                ``stat_strength``. ``stat_streams``='both' also anchors
                stream 2 (own latent only).
            override_profile (`str`, *optional*):
                Visual Relay round 5. 'damp' (default, round 3): w is
                ``override_min_weight`` at Img2's latent and 1 from
                ``override_radius`` latents away. 'local': w is 1 at Img2's
                latent and falls to 0 at ``override_radius`` latents away.
            override_handover_step (`int`, *optional*):
                From this sampling step on, stream 1 takes stream 2's
                attention only at Img2's latent (w = 1 there, 0 elsewhere):
                stream 2 plans the layout and motion, stream 1 finishes as
                plain I2V around Img2.
            stream_b_weight (`float`, *optional*):
                Stream B (paired design): an ordinary Img2-first I2V over
                latents a..F-1 reading stream 1's latent there, with the
                relay prompt if set. Stream 1's latents after Img2 become
                base + lambda (O_B - base). None = off.
            stream_b_fade_start_step, stream_b_fade_end_step:
                Cosine fade of lambda to 0 between these steps (as round 2's
                suffix fade); Stream B leaves the batch once it reaches 0.
            override_stream 'fresh' (round 5): stream 2's Img2 latent and the
                latents after it are encoded like an Img2-first I2V (a
                one-frame Img2 latent, then zeros), mask on all of latent a,
                exactly as Img1 is presented in latent 0.
            resample_from_step, resample_to_step, resample_cycles:
                Resampling: after step K = resample_to_step, re-noise each
                stream's predicted clean latent to step J =
                resample_from_step's noise level with fresh noise and run
                steps J..K again, ``resample_cycles`` times. J..K must use
                one expert. Off when resample_cycles is 0.
            override_stream 'clean' (round 6, Direction A): stream 2 is the
                joint encoding, except Img2's latent a, which is latent 1 of
                Img2 held for 5 frames and encoded on its own (a 4-frame
                static Img2 latent whose history is Img2, not grey padding).
            override_mode (`str`, *optional*):
                Round 6, Direction C. 'blend' (default): stream 1 takes stream
                2's attention output (Method 1). 'route': stream 1 attends with
                stream 2's queries and keys but its own values,
                attn(q2, k2, v1), on every latent except Img2's, which takes
                O2. Needs w = 1 everywhere, no handover, no Stream B.
            rerender_from_step (`int`, *optional*):
                Round 6, Direction B (plan, then render). After the whole
                override run (the plan), stream 1's final latent, with Img2's
                latent replaced by the clean static Img2 latent, is re-noised
                to this step's noise level with fresh noise, and steps S..end
                run again as plain single-stream I2V of Img1 (relay prompt
                if set, fresh solver), with Img2's latent pinned to the static
                Img2 latent in every step's predicted clean latent. S must be
                a low-noise step. The plan's own video is left in
                ``self.last_plan_video``.
            override_img2_span 'last' / 'motion' (round 7): 'last' puts Img2
                on the last frame of its latent only (frame 4a), mask on that
                frame, the other 3 frames free (grey). 'motion' fills the
                latent with Img2 then ``img2_motion_frames`` (3 frames, e.g.
                frames 1-3 of a plain I2V run from Img2), mask on all 4.
                With override_stream 'clean' the latent is encoded on its
                own, after its first frame as history.
            img2_motion_frames (`torch.Tensor`, *optional*):
                [3, 3, H, W] in [-1, 1] at the working size; span 'motion'
                only.
            override_stream 'long' / 'context' (round 8): stream 2 is the
                joint encoding, except Img2's latent a, which is encoded deep
                inside a clip, so it has neither the start-of-clip offset of
                the clean encoding (which decodes darker mid-video) nor grey
                frames (which decode as flat grey). The clip is frames
                0..4a-4, then the latent's 4 frames, and the code is latent a
                of its encoding. 'long': frames 0..4a-4 are Img2 (single
                pass). 'context': they are frames of ``img2_context_video``
                (a first-pass video; two-pass). Span 'latent' (mask on all 4
                frames) or 'last' (mask on frame 4a only).
            img2_context_video (`torch.Tensor`, *optional*):
                [3, N, H, W] in [-1, 1], N >= 4a - 3 (4a with fill 'video');
                resized to the working size if needed. Stream 'context' only.
            img2_fill (`str`, *optional*):
                Round 8, the first 3 frames of Img2's latent in the 'context'
                encoding with span 'last': 'img2' (default; Img2 x4, also
                what 'long' and span 'latent' use), 'hold' (frame 4a-4 held,
                then Img2), 'video' (the context video's frames 4a-3..4a-1
                with each channel's mean moved onto the straight line from
                frame 4a-4 to Img2, then Img2). Round 9: 'grey' (grey x3,
                then Img2: the layout of Wan's last-frame conditioning, but
                deep inside a clip; streams 'long' and 'context', span
                'last').
            img2_code_stats (`str`, *optional*):
                Round 9: None (default), 'mean' or 'meanstd'. After stream 2's
                input is built, Img2's latent a keeps its spatial pattern
                but each channel's spatial mean (and std) is set to the long
                code's (Img2 held from frame 0, Img2 x4), which decodes at
                Img2's own brightness. Span 'latent'; not with 'long'.
            save_latents (`bool`, *optional*):
                Round 9: keep the final latents in ``self.last_latents``
                (rank 0; dict of stream1, stream2, plan, anchor; float32 on
                the CPU) so they can be corrected and decoded afterwards.
            img2_release_step (`int`, *optional*):
                Round 10: from this sampling step on, stream 2 reads PR-LF's
                input instead of M1-PR's: the joint encoding with Img2 on the
                last frame of its latent only (mask on that frame only).
                Before it, all 4 frames are marked (the orbit); after it, the
                first 3 frames of Img2's latent are free (the hold
                released). Needs the joint encoding with span 'latent'.
            decode_fix (`str`, *optional*):
                Round 10: None (default), 'mean' or 'meanstd'. The main
                video's final latents lo..hi (``decode_fix_window``) get each
                channel's spatial mean (and std) on the line between latents
                lo-1 and hi+1, and are decoded again into
                ``self.last_fixed_video`` (rank 0); the main video itself is
                unchanged. Round 9's lab correction, inside the run.
            decode_fix_window (`tuple`, *optional*):
                Round 10: (lo, hi), the latents the decode fix corrects.
            img2_span_schedule (`sequence of int`, *optional*):
                Round 12: how many frames of Img2's latent stream 2 is given
                at each sampling step, Img2 on the last k frames (frames
                4a-k+1..4a, grey before them, the mask on those k frames),
                joint encoding; the last entry holds for the remaining steps.
                (4, 3, 2, 1): M1-PR's input at step 0, then 3 and 2 frames,
                then PR-LF's input from step 3. Needs M1-PR's input.
            img2_release_input (`str`, *optional*):
                Round 12: 'joint' (default, PR-LF's input after the release)
                or 'own' (CL-LF's input after the release: the same joint
                encoding, with Img2's latent encoded on its own, grey x4 then
                Img2, latent 1). Needs the release step.
            override_img2_mask (`str`, *optional*):
                Round 12: None (default, the mask follows the span) or 'last':
                Img2's latent keeps M1-PR's code (Img2 on all 4 frames, joint
                encoding) but only its last frame (RGB frame 4a) is marked, as
                in PR-LF.
            override_img2_value (`str`, *optional*):
                Round 13, Gordon's K_A12 / V_B0: None (default), 'before' or
                'all'. Stream B (Img2-first I2V on stream 2's latents a..,
                weight 0) runs every step only to supply values: in stream 2's
                self-attention, latents 0..a-1 ('before') or every latent but a
                ('all') read Img2's latent a with stream 2's keys and Stream
                B's frame-0 values (Img2 encoded with no frames before it).
                Joint or splice encoding, blend mode.
            override_single (`bool`, *optional*):
                Round 14: run stream 2 alone (its own input: Img1 and Img2,
                the span, release or schedule, splice, the relay prompt) as
                the video; no stream 1. With ``override_img2_value`` Stream B
                runs beside it only to supply the clean Img2 values. Stream 2
                never read stream 1, so this is round 4-13's _s2 video.
            reach_test (`dict`, *optional*):
                Round 7, forward-only reach test on plain I2V (no Img2 in
                the input): keys steps, target, eps, blur, patterns, seed,
                path. At each listed step (first visit), every latent j in
                turn is nudged by +-delta (a blurred noise pattern, RMS = eps
                x RMS(x_t)) in one batch-2 conditional pass, and the change
                of the predicted clean latent is recorded per latent; plus a
                quarter-size nudge and a 1/100 nudge (rounding floor) on
                latent target-1. The sampling itself is unchanged. Rank 0
                writes ``path`` (JSON) and its .txt summary; the record is
                left in ``self.last_reach``.

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        # preprocess
        self.last_stream2_video = None
        self.last_plan_video = None
        self.last_latents = None
        self.last_fixed_video = None
        self.last_reach = None
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale
        override_on = override_stream is not None
        y2_late = None  # round 10: stream 2's input after the Img2 release
        img2_spans = None  # round 12: frames of Img2 given at each step
        y2_by_span = None  # round 12: stream 2's input for each frame count
        if override_on:
            if image_condition_paired_anchor_attention:
                raise ValueError(
                    "the self-attention override (Method 1) and paired anchor "
                    "attention are separate methods; use one")
            if override_stream not in ('joint', 'splice', 'fresh', 'clean',
                                       'long', 'context'):
                raise ValueError(
                    "override stream must be 'joint', 'splice', 'fresh', "
                    "'clean', 'long' or 'context'")
            if (override_stream in ('long', 'context')
                    and override_img2_span not in ('latent', 'last')):
                raise ValueError(
                    "the long and context Img2 encodings take span 'latent' "
                    "or 'last'")
            if (override_stream == 'context') != (
                    img2_context_video is not None):
                raise ValueError(
                    "the context Img2 encoding and its context video go "
                    "together")
            if img2_fill not in ('img2', 'hold', 'video', 'grey'):
                raise ValueError(
                    "Img2 fill must be 'img2', 'hold', 'video' or 'grey'")
            if img2_fill in ('hold', 'video') and (
                    override_stream != 'context'
                    or override_img2_span != 'last'):
                raise ValueError(
                    "Img2 fill 'hold' or 'video' needs the context encoding "
                    "with span 'last'")
            if img2_fill == 'grey' and (
                    override_stream not in ('long', 'context')
                    or override_img2_span != 'last'):
                raise ValueError(
                    "Img2 fill 'grey' needs the long or context encoding "
                    "with span 'last'")
            if img2_code_stats not in (None, 'mean', 'meanstd'):
                raise ValueError(
                    "Img2 code stats must be None, 'mean' or 'meanstd'")
            if img2_code_stats is not None and (
                    override_img2_span != 'latent'
                    or override_stream == 'long'):
                raise ValueError(
                    "Img2 code stats need span 'latent' and a stream other "
                    "than 'long'")
            if img2_release_step is not None:
                if (override_stream not in ('joint', 'splice')
                        or override_img2_span != 'latent'
                        or img2_code_stats is not None):
                    raise ValueError(
                        "the Img2 release needs M1-PR's input: the joint or "
                        "splice encoding, span 'latent', no code stats")
                if not 0 < img2_release_step < sampling_steps:
                    raise ValueError(
                        "the Img2 release step must be in (0, sampling_steps)")
                if (stream_b_weight is not None
                        or rerender_from_step is not None or resample_cycles):
                    raise ValueError(
                        "the Img2 release does not combine with Stream B, "
                        "re-rendering or resampling")
            if img2_release_input not in ('joint', 'own'):
                raise ValueError(
                    "the Img2 release input must be 'joint' or 'own'")
            if img2_release_input != 'joint' and img2_release_step is None:
                raise ValueError(
                    "the Img2 release input 'own' needs a release step")
            if img2_span_schedule is not None:
                img2_spans = [int(k) for k in img2_span_schedule]
                if (not 2 <= len(img2_spans) <= sampling_steps
                        or any(k not in (1, 2, 3, 4) for k in img2_spans)):
                    raise ValueError(
                        "the Img2 span schedule lists 2 to sampling_steps "
                        "frame counts, each 1 to 4")
                if (override_stream not in ('joint', 'splice')
                        or override_img2_span != 'latent'
                        or img2_code_stats is not None):
                    raise ValueError(
                        "the Img2 span schedule needs M1-PR's input: the joint "
                        "or splice encoding, span 'latent', no code stats")
                if (img2_release_step is not None
                        or stream_b_weight is not None
                        or rerender_from_step is not None or resample_cycles):
                    raise ValueError(
                        "the Img2 span schedule does not combine with the "
                        "release, Stream B, re-rendering or resampling")
            if override_img2_value not in (None, 'before', 'all'):
                raise ValueError(
                    "the clean Img2 values must be None, 'before' or 'all'")
            if override_img2_value is not None and (
                    override_stream not in ('joint', 'splice')
                    or override_mode != 'blend'
                    or stream_b_weight is not None
                    or rerender_from_step is not None or resample_cycles):
                raise ValueError(
                    "the clean Img2 values need the joint or splice encoding "
                    "in blend mode, and no Stream B weight, re-rendering or "
                    "resampling")
            if override_single and (
                    override_mode != 'blend' or stream_b_weight is not None
                    or rerender_from_step is not None or resample_cycles
                    or override_handover_step is not None
                    or override_save_stream2 or stat_anchor != 'off'):
                raise ValueError(
                    "the single stream does not combine with route mode, a "
                    "Stream B weight, re-rendering, resampling, the "
                    "handover, saving stream 2 or the colour anchor")
            if override_img2_mask not in (None, 'last'):
                raise ValueError("the Img2 mask must be None or 'last'")
            if override_img2_mask is not None and (
                    override_stream != 'joint'
                    or override_img2_span != 'latent'
                    or img2_release_step is not None
                    or img2_span_schedule is not None
                    or rerender_from_step is not None):
                raise ValueError(
                    "the Img2 mask 'last' needs the joint encoding with span "
                    "'latent', and no release, span schedule or re-rendering")
            if override_stream == 'fresh' and override_img2_span != 'latent':
                raise ValueError("the fresh Img2 encoding fills its latent")
            if override_stream == 'clean' and override_img2_span == 'frame':
                raise ValueError(
                    "the clean Img2 encoding takes span 'latent', 'last' or "
                    "'motion'")
            if (override_img2_span == 'motion') != (
                    img2_motion_frames is not None):
                raise ValueError(
                    "Img2 span 'motion' and its 3 motion frames go together")
            if (rerender_from_step is not None
                    and override_img2_span != 'latent'):
                raise ValueError(
                    "re-rendering pins Img2 held 4 frames; it needs span "
                    "'latent'")
            if override_mode not in ('blend', 'route'):
                raise ValueError("override mode must be 'blend' or 'route'")
            if override_mode == 'route' and (
                    override_min_weight != 1.0 or override_profile != 'damp'
                    or override_handover_step is not None
                    or stream_b_weight is not None):
                raise ValueError(
                    "route mode needs w = 1 everywhere (min weight 1, damp "
                    "profile), no handover and no Stream B")
            if rerender_from_step is not None:
                if not 0 < rerender_from_step < sampling_steps:
                    raise ValueError(
                        "re-render step must be in (0, sampling_steps)")
                if resample_cycles:
                    raise ValueError(
                        "re-rendering and resampling are separate; use one")
            if override_profile not in ('damp', 'local'):
                raise ValueError("override profile must be 'damp' or 'local'")
            if override_handover_step is not None and not (
                    0 < override_handover_step < sampling_steps):
                raise ValueError(
                    "override handover step must be in (0, sampling_steps)")
            if stream_b_weight is not None:
                if not 0.0 <= stream_b_weight <= 1.0:
                    raise ValueError("Stream B weight must be in [0, 1]")
                if ((stream_b_fade_start_step is None)
                        != (stream_b_fade_end_step is None)):
                    raise ValueError(
                        "Stream B fade start and end go together")
                if stream_b_fade_start_step is not None and not (
                        0 <= stream_b_fade_start_step < stream_b_fade_end_step
                        <= sampling_steps):
                    raise ValueError(
                        "Stream B fade needs 0 <= start < end <= steps")
            elif stream_b_fade_start_step is not None:
                raise ValueError("a Stream B fade needs a Stream B weight")
            if override_blend not in ('linear', 'norm'):
                raise ValueError("override blend must be 'linear' or 'norm'")
            if override_latent not in ('shared', 'own'):
                raise ValueError("override latent must be 'shared' or 'own'")
            if not 0.0 <= override_min_weight <= 1.0:
                raise ValueError("override min weight must be in [0, 1]")
            if override_radius < 0:
                raise ValueError("override radius must be >= 0")
            if override_img2_span not in ('latent', 'frame', 'last', 'motion'):
                raise ValueError(
                    "override Img2 span must be 'latent', 'frame', 'last' or "
                    "'motion'")
            if override_save_stream2 and override_latent != 'own':
                raise ValueError("saving stream 2 needs override latent 'own'")
            if stat_anchor not in ('off', 'mean', 'meanstd'):
                raise ValueError("stat anchor must be 'off', 'mean' or 'meanstd'")
            if stat_streams not in ('one', 'both'):
                raise ValueError("stat streams must be 'one' or 'both'")
            if stat_streams == 'both' and override_latent != 'own':
                raise ValueError("anchoring stream 2 needs override latent 'own'")
            if not 0.0 <= stat_strength <= 1.0:
                raise ValueError("stat strength must be in [0, 1]")
        elif (override_img2_span != 'latent' or override_save_stream2
              or (relay_prompt and image_condition_paired_anchor_attention)
              or stat_anchor != 'off'
              or override_profile != 'damp'
              or override_handover_step is not None
              or stream_b_weight is not None
              or override_mode != 'blend'
              or rerender_from_step is not None
              or img2_motion_frames is not None
              or img2_context_video is not None
              or img2_fill != 'img2'
              or img2_code_stats is not None
              or img2_release_step is not None
              or img2_release_input != 'joint'
              or img2_span_schedule is not None
              or override_img2_mask is not None
              or override_img2_value is not None
              or override_single):
            raise ValueError(
                "Img2 span, stream-2 saving, prompt relay in paired mode, the "
                "colour anchor, the local profile, the handover, Stream B, "
                "route mode, re-rendering, motion frames, the context video, "
                "the Img2 fill, the Img2 code stats, the Img2 release and its "
                "input, the Img2 span schedule, the Img2 mask, the clean "
                "Img2 values and the single stream require the self-attention "
                "override")
        elif relay_prompt and relay_from_latent is None:
            # Plain I2V (round 7): no Img2, so the relay start is explicit.
            raise ValueError(
                "prompt relay in plain I2V needs an explicit relay start "
                "latent")
        if decode_fix not in (None, 'mean', 'meanstd'):
            raise ValueError("decode fix must be None, 'mean' or 'meanstd'")
        if (decode_fix is None) != (decode_fix_window is None):
            raise ValueError("the decode fix and its latent window go together")
        if decode_fix is not None:
            fix_lo, fix_hi = (int(v) for v in decode_fix_window)
            out_latents = (frame_num - 1) // self.vae_stride[0] + 1
            if not 1 <= fix_lo <= fix_hi < out_latents - 1:
                raise ValueError(
                    f"decode fix window {fix_lo}-{fix_hi} needs a latent on "
                    f"each side (latents 0..{out_latents - 1})")
        reach_on = reach_test is not None
        if reach_on:
            if override_on or image_condition_paired_anchor_attention:
                raise ValueError("the reach test runs on plain I2V only")
            if resample_cycles:
                raise ValueError("the reach test does not take resampling")
            reach_steps = sorted({int(s) for s in reach_test['steps']})
            if not reach_steps or not all(
                    0 <= s < sampling_steps for s in reach_steps):
                raise ValueError("reach steps must be in [0, sampling_steps)")
            reach_target = int(reach_test['target'])
            reach_eps = float(reach_test.get('eps', 0.1))
            reach_blur = float(reach_test.get('blur', 4.0))
            reach_patterns = int(reach_test.get('patterns', 1))
            reach_seed = int(reach_test.get('seed', 1234))
            if reach_eps <= 0 or reach_blur < 0 or reach_patterns < 1:
                raise ValueError(
                    "reach test needs eps > 0, blur >= 0 and patterns >= 1")
        resample = None
        if resample_cycles:
            if image_condition_paired_anchor_attention:
                raise ValueError("resampling is not supported in paired mode")
            if resample_cycles < 0 or resample_from_step is None or (
                    resample_to_step is None):
                raise ValueError(
                    "resampling needs from/to steps and cycles >= 1")
            resample = (int(resample_from_step), int(resample_to_step),
                        int(resample_cycles))
            build_step_plan(sampling_steps, resample)  # validates the steps
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
            if reverse_stream_mode not in ('off', 'attn', 'pred'):
                raise ValueError(
                    "reverse stream mode must be 'off', 'attn' or 'pred'")
            if reverse_stream_mode != 'off':
                if not 0.0 <= reverse_stream_strength <= 1.0:
                    raise ValueError("reverse stream strength must be in [0, 1]")
                if reverse_stream_shape not in ('linear', 'cosine'):
                    raise ValueError(
                        "reverse stream shape must be 'linear' or 'cosine'")
                if not 0 < reverse_stream_until_step <= sampling_steps:
                    raise ValueError(
                        "reverse stream until-step must be in "
                        "(0, sampling_steps]")
                if anchor_index < 2:
                    raise ValueError(
                        "Stream R needs at least one prefix latent before Img2")
            if reverse_stream_couple and reverse_stream_mode != 'attn':
                raise ValueError(
                    "coupled Stream R requires reverse stream mode 'attn'")
        else:
            if (image_condition_paired_variant is not None
                    or suffix_prompt or keyframe_guidance_scale
                    or reverse_prompt or reverse_stream_mode != 'off'):
                raise ValueError(
                    "paired variants, suffix/reverse prompts, keyframe "
                    "guidance and Stream R require paired anchor attention")
        if override_on:
            if not isinstance(img, (list, tuple)) or len(img) != 2:
                raise ValueError("the override needs exactly two images")
            if (image_latent_indices is None
                    or len(image_latent_indices) != 2
                    or image_latent_indices[0] != 0
                    or image_latent_indices[1] <= 0):
                raise ValueError("the override needs indices [0, positive]")
            imgs = [
                TF.to_tensor(value).sub_(0.5).div_(0.5).to(self.device)
                for value in img
            ]
            anchor_index = int(image_latent_indices[1])
        elif not image_condition_paired_anchor_attention:
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
        if override_on and anchor_index >= latent_frame_num:
            raise ValueError("override Img2 latent must be inside the video")
        if reach_on and not 1 <= reach_target < latent_frame_num:
            raise ValueError("reach target must be a latent in [1, frames)")
        if img2_context_video is not None:
            if (not torch.is_tensor(img2_context_video)
                    or img2_context_video.dim() != 4
                    or img2_context_video.size(0) != 3):
                raise ValueError("context video must be [3, N, H, W]")
            context_frames = 1 + self.vae_stride[0] * (anchor_index - 1) + (
                self.vae_stride[0] - 1 if img2_fill == 'video' else 0)
            if img2_context_video.size(1) < context_frames:
                raise ValueError(
                    f"context video needs at least {context_frames} frames "
                    f"for Img2's latent {anchor_index}, got "
                    f"{img2_context_video.size(1)}")
            img2_context_video = img2_context_video[:, :context_frames].float()
            if tuple(img2_context_video.shape[2:]) != (h, w):
                logging.info("Context video resized from %s to %s",
                             tuple(img2_context_video.shape[2:]), (h, w))
                img2_context_video = torch.nn.functional.interpolate(
                    img2_context_video.transpose(0, 1).contiguous(),
                    size=(h, w), mode='bicubic').transpose(0, 1).clamp(-1, 1)
        if img2_motion_frames is not None:
            img2_motion_frames = img2_motion_frames.float()
            if tuple(img2_motion_frames.shape) != (3, self.vae_stride[0] - 1,
                                                   h, w):
                raise ValueError(
                    f"motion frames must be [3, {self.vae_stride[0] - 1}, "
                    f"{h}, {w}] (the working size), got "
                    f"{tuple(img2_motion_frames.shape)}")
        if relay_prompt:
            if relay_from_latent is None:
                relay_from_latent = anchor_index + 1
            if not 1 <= relay_from_latent < latent_frame_num:
                raise ValueError("relay start latent must be in [1, frames)")
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

        # 40 GB GPUs: two bfloat16 experts (2 x 14.3 GB per GPU at 2-way FSDP)
        # plus T5-XXL (11.4 GB) do not fit together. With offloading on, park
        # the low-noise expert on the CPU now; the step-10 expert switch in
        # _prepare_model_for_timestep moves it back to the GPU.
        if (offload_model and self.dit_fsdp
                and self.low_noise_model is not None
                and next(self.low_noise_model.parameters()).device.type
                == 'cuda'):
            self.low_noise_model.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()

        rev_on = (image_condition_paired_anchor_attention
                  and reverse_stream_mode != 'off')
        rev_prompt_used = reverse_prompt if rev_on and reverse_prompt else None

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            context_suffix = (self.text_encoder([suffix_prompt], self.device)
                              if suffix_prompt else context)
            context_rev = (self.text_encoder([rev_prompt_used], self.device)
                           if rev_prompt_used else context)
            context_relay = (self.text_encoder([relay_prompt], self.device)
                             if relay_prompt else None)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]
            if suffix_prompt:
                context_suffix = [
                    t.to(self.device) for t in self.text_encoder(
                        [suffix_prompt], torch.device('cpu'))
                ]
            else:
                context_suffix = context
            if rev_prompt_used:
                context_rev = [
                    t.to(self.device) for t in self.text_encoder(
                        [rev_prompt_used], torch.device('cpu'))
                ]
            else:
                context_rev = context
            context_relay = ([
                t.to(self.device) for t in self.text_encoder(
                    [relay_prompt], torch.device('cpu'))
            ] if relay_prompt else None)
        if suffix_prompt:
            logging.info("Stream B prompt: %s", suffix_prompt)
        if rev_on:
            logging.info("Stream R prompt: %s",
                         rev_prompt_used or "(main prompt)")
        if relay_prompt:
            logging.info("Prompt relay from latent %d (frame %d on): %s",
                         relay_from_latent,
                         1 + self.vae_stride[0] * (relay_from_latent - 1),
                         relay_prompt)

        def conditioning_video(image, pixel_frames):
            return torch.concat([
                torch.nn.functional.interpolate(
                    image[None].cpu(), size=(h, w),
                    mode='bicubic').transpose(0, 1),
                torch.zeros(3, pixel_frames - 1, h, w)
            ], dim=1).to(self.device)

        def joint_anchor_conditioning_video(first_image, anchor_image,
                                            pixel_frames, latent_index,
                                            span='latent'):
            """Build Img1 ... Img2x4 ... for one causal joint VAE encode
            (span 'frame': Img2 only on the first frame of its latent)."""
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
                img2_latent_frames(anchor, span))
            return video

        def img2_latent_frames(anchor, span):
            """The 4 RGB frames given for Img2's latent, [3, 4, h, w], grey
            (0) where free: 'latent' Img2 x4; 'frame' Img2 first; 'last'
            (round 7) Img2 last; 'motion' (round 7) Img2, then the 3 motion
            frames."""
            frames = torch.zeros(3, self.vae_stride[0], h, w,
                                 device=anchor.device)
            if span == 'latent':
                frames.copy_(anchor[:, None].expand(-1, self.vae_stride[0],
                                                    -1, -1))
            elif span == 'frame':
                frames[:, 0].copy_(anchor)
            elif span == 'last':
                frames[:, -1].copy_(anchor)
            elif span in ('last2', 'last3'):  # round 12: the last k frames
                k = int(span[-1])
                frames[:, -k:].copy_(anchor[:, None].expand(-1, k, -1, -1))
            elif span == 'motion':
                frames[:, 0].copy_(anchor)
                frames[:, 1:].copy_(img2_motion_frames.to(anchor.device))
            else:
                raise ValueError(f"unknown Img2 span {span}")
            return frames

        def own_img2_latent(image, span):
            """Round 7, clean encoding of a non-static Img2 latent: its 4
            frames encoded on their own after their first frame as history
            (Img2 for 'motion', grey for 'last'); latent 1 of that clip."""
            anchor = torch.nn.functional.interpolate(
                image[None].cpu(), size=(h, w),
                mode='bicubic').to(self.device)[0]
            frames = img2_latent_frames(anchor, span)
            clip = torch.cat([frames[:, :1], frames], dim=1)
            return self.vae.encode([clip])[0][:, 1].float()

        def lift_frame_means(frames, start, end):
            """Round 8, fill 'video': frames [3, n, h, w] in [-1, 1] with
            each frame's per-channel mean (0-1 scale) scaled onto the straight
            line from ``start``'s means to ``end``'s ([3, h, w] each); frame k
            sits at (k + 1) / (n + 1) of the way."""
            n = frames.size(1)
            scaled = (frames.float() + 1) / 2
            first = ((start.float() + 1) / 2).mean(dim=(1, 2))
            last = ((end.float() + 1) / 2).mean(dim=(1, 2))
            pos = torch.arange(1, n + 1, dtype=torch.float32,
                               device=scaled.device) / (n + 1)
            target = first[:, None] + pos[None] * (last - first)[:, None]
            gain = target / scaled.mean(dim=(2, 3)).clamp(min=1e-4)
            return (scaled * gain[:, :, None, None]).clamp(0, 1) * 2 - 1

        def steady_img2_latent(image, stream, span, fill):
            """Round 8: Img2's latent encoded deep inside a clip (no
            start-of-clip offset, no grey frames): frames 0..4a-4 (Img2 for
            'long', the context video's for 'context'), then the latent's 4
            frames (Img2 x4; with span 'last', fill 'hold' or 'video' changes
            the first 3); latent a of that clip."""
            anchor = torch.nn.functional.interpolate(
                image[None].cpu(), size=(h, w),
                mode='bicubic').to(self.device)[0]
            stride = self.vae_stride[0]
            before_frames = 1 + stride * (anchor_index - 1)
            if stream == 'long':
                before = anchor[:, None].expand(-1, before_frames, -1, -1)
            else:
                before = img2_context_video[:, :before_frames].to(
                    anchor.device)
            frames = anchor[:, None].expand(-1, stride, -1, -1).clone()
            if span == 'last' and fill == 'hold':
                frames[:, :-1] = before[:, -1:]
            elif span == 'last' and fill == 'grey':
                frames[:, :-1] = 0
            elif span == 'last' and fill == 'video':
                frames[:, :-1] = lift_frame_means(
                    img2_context_video[:, before_frames:before_frames
                                       + stride - 1].to(anchor.device),
                    before[:, -1], anchor)
            clip = torch.cat([before, frames], dim=1).contiguous()
            latent = self.vae.encode([clip])[0]
            if latent.size(1) != anchor_index + 1:
                raise RuntimeError("steady Img2 encoding length is wrong")
            return latent[:, anchor_index].float()

        def static_latent(image):
            """A static 4-frame latent [C, H, W]: latent 1 of the image held
            for 5 frames, encoded on its own (its history is the image)."""
            still = torch.nn.functional.interpolate(
                image[None].cpu(), size=(h, w), mode='bicubic')[0]
            still = still[:, None].expand(-1, 1 + self.vae_stride[0], -1,
                                          -1).contiguous().to(self.device)
            return self.vae.encode([still])[0][:, 1].float()

        def static_latent_stats(image):
            """Per-channel spatial mean and std of a static 4-frame latent
            (latent 1 of the image held for 5 frames)."""
            lat = static_latent(image)
            return lat.mean(dim=(1, 2)), lat.std(dim=(1, 2))

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
                raise RuntimeError(
                    "paired conditioning VAE lengths do not align")
            # Ordinary Stream-A slots retain the compact Img1-only encoding.
            # Only its temporary anchor slot uses the joint causal encoding;
            # Stream B independently uses a fresh-cache Img2-first suffix.
            expanded_msk = insert_latent_slot(msk, anchor_index)
            expanded_main = insert_latent_slot(encoded[0], anchor_index)
            expanded_msk[:, anchor_index:anchor_index + 1].fill_(1)
            expanded_main[:, anchor_index:anchor_index + 1].copy_(
                encoded[2][:, anchor_index:anchor_index + 1])
            y = torch.concat([expanded_msk, expanded_main])
            suffix_y = torch.concat([
                first_frame_mask(suffix_frame_num), encoded[1]
            ])
            del encoded, expanded_msk, expanded_main
            if rev_on:
                # Stream R: ordinary fresh-cache Img2-first I2V conditioning
                # over anchor latents (Img2 + prefix latents a-1..1).
                rev_frame_num = 1 + self.vae_stride[0] * (anchor_index - 1)
                rev_encoded = self.vae.encode(
                    [conditioning_video(imgs[1], rev_frame_num)])[0]
                if rev_encoded.size(1) != anchor_index:
                    raise RuntimeError("Stream R VAE length does not align")
                y_rev = torch.concat(
                    [first_frame_mask(rev_frame_num), rev_encoded])
                del rev_encoded
                rev_seq_len = anchor_index * lat_h * lat_w // (
                    self.patch_size[1] * self.patch_size[2])
                rev_seq_len = int(math.ceil(
                    rev_seq_len / self.sp_size)) * self.sp_size
                rev_frame_weights = [
                    reverse_stream_weight(frame, anchor_index,
                                          reverse_stream_strength,
                                          reverse_stream_shape)
                    for frame in range(anchor_index)
                ]
                logging.info(
                    "Stream R (%s fusion%s): Img2-first stream backward over "
                    "prefix latents %d..1 for sampling steps < %d; prefix "
                    "weights w_t = %s", reverse_stream_mode,
                    ', coupled' if reverse_stream_couple else '',
                    anchor_index - 1, reverse_stream_until_step,
                    ', '.join(f"{w_t:.2f}" for w_t in rev_frame_weights[1:]))
            logging.info(
                "Paired anchor attention: %d retained latents plus temporary "
                "Img2 anchor at %d (%d working latents), with a %d-latent "
                "fresh-cache Img2 suffix; fixed full-suffix fusion weight "
                "%.3f. While its anchor route is active, every prefix query "
                "uses one softmax over all ordinary Stream-A frames plus a "
                "hybrid anchor with the history-aware Stream-A key and "
                "fresh-cache Stream-B Img2 value. "
                "Stream B independently "
                "evolves a fresh-cache Img2 "
                "anchor, while the joint-conditioned Stream-A anchor keeps "
                "its own prediction and is removed prior to decoding",
                latent_frame_num, anchor_index, model_latent_frame_num,
                suffix_latent_frame_num,
                image_condition_paired_stream_weight)
            if suffix_fade_start_step is not None:
                logging.info(
                    "Paired Stream-B suffix cosine fade: sampling steps %d "
                    "through %d; weight %.3f -> 0; scheduler state is not reset",
                    suffix_fade_start_step,
                    suffix_fade_end_step,
                    image_condition_paired_stream_weight)
            if image_condition_paired_prefix_route_switch_step is not None:
                logging.info(
                    "Paired prefix routing handoff at sampling step %d: before "
                    "the step, prefix queries see all ordinary Stream-A frames "
                    "plus a history-aware Stream-A anchor key paired with the "
                    "fresh-cache Stream-B Img2 value; at and after the step, "
                    "only that hybrid anchor route is removed",
                    image_condition_paired_prefix_route_switch_step)
            if image_condition_paired_variant is not None:
                logging.info(
                    "Visual Relay prefix-route variant %s (until step %s)",
                    image_condition_paired_variant,
                    image_condition_paired_variant_until_step)
            else:
                logging.info("Visual Relay prefix route: baseline")
        elif override_on:
            # Method 1: stream 1 = plain I2V of Img1; stream 2 = Img1 and Img2.
            img2_first_latents = latent_frame_num - anchor_index
            img2_first_frames = 1 + self.vae_stride[0] * (
                img2_first_latents - 1)
            needs_img2_first = (override_stream == 'fresh'
                                or stream_b_weight is not None
                                or override_img2_value is not None)
            encoded = self.vae.encode([
                conditioning_video(imgs[0], F),
                joint_anchor_conditioning_video(imgs[0], imgs[1], F,
                                                anchor_index,
                                                override_img2_span),
            ] + ([conditioning_video(imgs[1], img2_first_frames)]
                 if needs_img2_first else []))
            if (encoded[0].size(1) != latent_frame_num
                    or encoded[1].size(1) != latent_frame_num):
                raise RuntimeError("override conditioning lengths differ")
            if needs_img2_first and encoded[2].size(1) != img2_first_latents:
                raise RuntimeError("Img2-first encoding length is wrong")
            # Img2-first I2V conditioning (one-frame Img2 latent, then zeros
            # encoded after it): Stream B's input, and stream 2's latents
            # a.. with override_stream 'fresh'.
            img2_first = encoded[2] if needs_img2_first else None
            y_b = (torch.concat([first_frame_mask(img2_first_frames),
                                 img2_first])
                   if (stream_b_weight is not None
                       or override_img2_value is not None) else None)
            y = torch.concat([msk, encoded[0]])
            msk2 = msk.clone()
            if override_img2_span == 'frame':
                msk2[0, anchor_index] = 1  # RGB frame 4a - 3 only
            elif override_img2_span == 'last':
                msk2[-1, anchor_index] = 1  # RGB frame 4a only (round 7)
            else:
                msk2[:, anchor_index] = 1
            if override_img2_mask == 'last':
                # Round 12 (PR-4C): M1-PR's code, PR-LF's mask.
                msk2 = msk.clone()
                msk2[-1, anchor_index] = 1
                logging.info(
                    "Img2's latent %d: joint code of Img2 on all 4 frames, "
                    "mask on its last frame only (RGB frame %d)", anchor_index,
                    self.vae_stride[0] * anchor_index)
            # Round 6: the static 4-frame Img2 latent (Direction A's stream 2
            # Img2 latent; Direction B's pinned latent).
            img2_static = (static_latent(imgs[1])
                           if (override_stream == 'clean'
                               and override_img2_span == 'latent')
                           or rerender_from_step is not None else None)
            if override_stream == 'joint':
                cond2 = encoded[1]
            elif override_stream == 'fresh':
                cond2 = encoded[0].clone()
                cond2[:, anchor_index:] = img2_first
            elif override_stream == 'clean':
                cond2 = encoded[1].clone()
                if override_img2_span == 'latent':
                    cond2[:, anchor_index] = img2_static.to(cond2.dtype)
                else:  # round 7: last frame only, or Img2 + motion frames
                    cond2[:, anchor_index] = own_img2_latent(
                        imgs[1], override_img2_span).to(cond2.dtype)
            elif override_stream in ('long', 'context'):
                # Round 8: the joint encoding with Img2's latent re-encoded
                # deep inside a clip.
                cond2 = encoded[1].clone()
                cond2[:, anchor_index] = steady_img2_latent(
                    imgs[1], override_stream, override_img2_span,
                    img2_fill).to(cond2.dtype)
                logging.info(
                    "Img2's latent %d: %s encoding (frames 0..%d before it "
                    "are %s), span %s, fill %s", anchor_index,
                    override_stream,
                    self.vae_stride[0] * (anchor_index - 1),
                    'Img2' if override_stream == 'long'
                    else 'the context video', override_img2_span, img2_fill)
            else:
                cond2 = encoded[0].clone()
                cond2[:, anchor_index] = encoded[1][:, anchor_index]
            if img2_code_stats is not None:
                # Round 9: keep the latent's pattern, take the long code's
                # per-channel colour statistics.
                cond2 = cond2.clone()
                ref = steady_img2_latent(imgs[1], 'long', 'latent', 'img2')
                cur = cond2[:, anchor_index].float()
                cur_mean = cur.mean(dim=(1, 2))[:, None, None]
                ref_mean = ref.mean(dim=(1, 2))[:, None, None]
                if img2_code_stats == 'mean':
                    fixed = cur - cur_mean + ref_mean
                else:
                    cur_std = cur.std(dim=(1, 2)).clamp(min=1e-6)[:, None,
                                                                   None]
                    ref_std = ref.std(dim=(1, 2))[:, None, None]
                    fixed = (cur - cur_mean) / cur_std * ref_std + ref_mean
                cond2[:, anchor_index] = fixed.to(cond2.dtype)
                logging.info(
                    "Img2's latent %d: %s encoding with the long code's "
                    "per-channel %s", anchor_index, override_stream,
                    'mean' if img2_code_stats == 'mean' else 'mean and std')
            y2 = torch.concat([msk2, cond2])

            def stream2_code(code):
                """Round 13: with the splice encoding (Gordon's surgical
                swap) only Img2's latent a comes from the pass with Img2 in
                it; every other latent from the Img1-only pass."""
                if override_stream != 'splice':
                    return code
                spliced = encoded[0].clone()
                spliced[:, anchor_index] = code[:, anchor_index].to(
                    spliced.dtype)
                return spliced

            if img2_release_step is not None:
                # Round 10: PR-LF's input (the joint encoding with Img2 on its
                # latent's last frame, grey on the 3 before; mask on that
                # frame only), read by stream 2 from the release step on.
                late = stream2_code(self.vae.encode([
                    joint_anchor_conditioning_video(imgs[0], imgs[1], F,
                                                    anchor_index, 'last')
                ])[0])
                if img2_release_input == 'own':
                    # Round 12 (RLc): CL-LF's input after the release.
                    late = late.clone()
                    late[:, anchor_index] = own_img2_latent(
                        imgs[1], 'last').to(late.dtype)
                msk_late = msk.clone()
                msk_late[-1, anchor_index] = 1
                y2_late = torch.concat([msk_late, late])
                del late, msk_late
                logging.info(
                    "Img2 release at step %d: stream 2 switches from Img2 on "
                    "all 4 frames of latent %d to Img2 on its last frame only "
                    "(RGB frame %d), %s", img2_release_step,
                    anchor_index, self.vae_stride[0] * anchor_index,
                    'joint encoding' if img2_release_input == 'joint' else
                    "joint encoding with Img2's latent encoded on its own")
            if img2_spans is not None:
                # Round 12: Img2 on the last k frames of its latent, k per
                # step (k = 4 is y2 itself, M1-PR's input).
                y2_by_span = {}
                for k in sorted(set(img2_spans) - {4}):
                    code = stream2_code(self.vae.encode([
                        joint_anchor_conditioning_video(
                            imgs[0], imgs[1], F, anchor_index,
                            'last' if k == 1 else f'last{k}')
                    ])[0])
                    msk_k = msk.clone()
                    msk_k[self.vae_stride[0] - k:, anchor_index] = 1
                    y2_by_span[k] = torch.concat([msk_k, code])
                    del code, msk_k
                logging.info(
                    "Img2 span schedule: frames of Img2's latent %d given at "
                    "steps 0..%d: %s (the last holds after), %s encoding",
                    anchor_index, len(img2_spans) - 1, img2_spans,
                    override_stream)
            if override_stream == 'splice':
                logging.info(
                    "Splice encoding (Gordon's surgical swap): stream 2's hint "
                    "latents come from the Img1-only VAE pass except Img2's "
                    "latent %d", anchor_index)
            if override_img2_value is not None:
                logging.info(
                    "Clean Img2 values (Gordon's K_A12 / V_B0): Stream B "
                    "(Img2-first I2V on stream 2's latents %d..) supplies the "
                    "values of Img2's latent to stream 2's queries on %s",
                    anchor_index, 'latents 0..%d' % (anchor_index - 1)
                    if override_img2_value == 'before' else 'every other latent')
            suffix_y = None
            del encoded, cond2
            override_weights = override_frame_weights(
                latent_frame_num, anchor_index, override_min_weight,
                override_radius, override_profile)
            handover_weights = [1.0 if f == anchor_index else 0.0
                                for f in range(latent_frame_num)]
            if override_handover_step is not None:
                logging.info(
                    "Override handover at step %d: from then on stream 1 "
                    "takes stream 2's attention only at Img2's latent %d",
                    override_handover_step, anchor_index)
            if stream_b_weight is not None:
                logging.info(
                    "Stream B: Img2-first I2V over latents %d..%d (reads "
                    "stream 1's latent), lambda %.2f on latents %d.., fade "
                    "%s", anchor_index, latent_frame_num - 1, stream_b_weight,
                    anchor_index + 1,
                    (stream_b_fade_start_step, stream_b_fade_end_step))
            override_cfg = dict(weights=override_weights,
                                blend=override_blend, anchor=anchor_index)
            if override_mode == 'route':
                override_cfg['mode'] = 'route'
                logging.info(
                    "Route mode: stream 1 attends with stream 2's queries and "
                    "keys and its own values on every latent but Img2's "
                    "latent %d, which takes stream 2's output", anchor_index)
            if rerender_from_step is not None:
                logging.info(
                    "Plan, then render: after the override run, re-noise "
                    "stream 1's final latent to step %d and run steps %d.. "
                    "as plain I2V with Img2's latent %d pinned to the static "
                    "Img2 latent", rerender_from_step, rerender_from_step,
                    anchor_index)
            stat_targets = None
            if stat_anchor != 'off':
                stat_targets = relay_stat_targets(
                    static_latent_stats(imgs[0]),
                    static_latent_stats(imgs[1]), latent_frame_num,
                    anchor_index)
                logging.info(
                    "Colour anchor (%s, strength %.2f, streams %s, std from "
                    "step %d): per-latent channel stats move toward Img1 -> "
                    "Img2 interpolated targets", stat_anchor, stat_strength,
                    stat_streams, stat_std_from_step)
            logging.info(
                "Self-attention override (Method 1): stream 2 = %s encoding "
                "with Img2 at latent %d, %s latent, %s blend; per-latent "
                "weights %s; Img2 span: %s", override_stream, anchor_index,
                override_latent,
                override_blend,
                ', '.join(f"{w_f:.2f}" for w_f in override_weights),
                override_img2_span)
        else:
            y = self.vae.encode([conditioning_video(imgs[0], F)])[0]
            y = torch.concat([msk, y])
            suffix_y = None

        @contextmanager
        def noop_no_sync():
            yield

        def no_sync_of(model):
            # no_sync() only matters for gradients. Entering it runs FSDP's
            # lazy init, which rejects a model parked on the CPU, so skip it
            # for a CPU-resident expert (FSDP initialises it at its first
            # forward, after it is moved back to the GPU).
            if (model is None or next(model.parameters()).device.type
                    != 'cuda'):
                return noop_no_sync
            return getattr(model, 'no_sync', noop_no_sync)

        no_sync_low_noise = no_sync_of(self.low_noise_model)
        no_sync_high_noise = no_sync_of(self.high_noise_model)

        if probe_steps:
            ATTN_PROBE.configure(probe_steps, samples=probe_samples)
            logging.info("Attention probe ON at sampling steps %s -> %s",
                         sorted(set(int(s) for s in probe_steps)), probe_path)

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

            step_sigmas = [float(value) for value in sample_scheduler.sigmas]
            step_plan = build_step_plan(len(timesteps), resample)
            if resample is not None:
                first_rs, last_rs = resample[0], resample[1]
                if ((timesteps[first_rs].item() >= boundary)
                        != (timesteps[last_rs].item() >= boundary)):
                    raise ValueError(
                        "resample steps J..K must all use one expert")
                pristine_scheduler = copy.deepcopy(sample_scheduler)
                logging.info(
                    "Resampling: after step %d, re-noise to step %d (sigma "
                    "%.4f) and repeat steps %d-%d, %d time(s)", last_rs,
                    first_rs, step_sigmas[first_rs], first_rs, last_rs,
                    resample[2])
            if rerender_from_step is not None:
                if timesteps[rerender_from_step].item() >= boundary:
                    raise ValueError(
                        "the re-render pass must start at a low-noise step "
                        "(the high-noise expert is released at the switch)")
                # A fresh solver for the render pass (no history).
                render_scheduler = copy.deepcopy(sample_scheduler)
            x0_hat = x0_hat2 = None

            # sample videos
            latent = noise
            override_own = (override_on and override_latent == 'own'
                            and not override_single)
            if override_single:
                logging.info(
                    "Single stream (round 14): stream 2's input alone is the "
                    "video; no stream 1%s", '; Stream B supplies the clean '
                    'Img2 values' if override_img2_value else '')
            if override_own:
                # Stream 2 starts from the same noise and keeps its own latent.
                latent2 = noise.clone()
                scheduler2 = copy.deepcopy(sample_scheduler)

            if image_condition_paired_anchor_attention:
                arg_c = {
                    'context': [context[0], context_suffix[0]],
                    'seq_len': max_seq_len,
                    'y': [y, suffix_y],
                    'paired_anchor_index': anchor_index,
                    'paired_stream_weight': image_condition_paired_stream_weight,
                    'paired_prefix_anchor_active': True,
                    'paired_variant': image_condition_paired_variant,
                }
                arg_null = {
                    'context': [context_null[0], context_null[0]],
                    'seq_len': max_seq_len,
                    'y': [y, suffix_y],
                    'paired_anchor_index': anchor_index,
                    'paired_stream_weight': image_condition_paired_stream_weight,
                    'paired_prefix_anchor_active': True,
                    'paired_variant': image_condition_paired_variant,
                }
            elif override_on:
                arg_c = {
                    'context': [context[0], context[0]],
                    'seq_len': max_seq_len,
                    'y': [y, y2],
                    'override': override_cfg,
                }
                if relay_prompt:
                    # The negative prompt stays the same for every frame.
                    arg_c['relay'] = dict(
                        context=[context_relay[0], context_relay[0]],
                        from_frame=relay_from_latent)
                arg_null = {
                    'context': [context_null[0], context_null[0]],
                    'seq_len': max_seq_len,
                    'y': [y, y2],
                    'override': override_cfg,
                }
                override_args = (arg_c, arg_null)
                if stream_b_weight is not None or override_img2_value:
                    # Stream B reads the relay prompt (else the main prompt)
                    # on every one of its tokens.
                    ctx_b = context_relay[0] if relay_prompt else context[0]
                    arg_c_b = dict(arg_c, context=arg_c['context'] + [ctx_b],
                                   y=[y, y2, y_b])
                    if relay_prompt:
                        arg_c_b['relay'] = dict(
                            context=[context_relay[0]] * 3,
                            from_frame=relay_from_latent)
                    arg_null_b = dict(
                        arg_null, context=arg_null['context'] + [
                            context_null[0]], y=[y, y2, y_b])
                    override_args_b = (arg_c_b, arg_null_b)
                if override_single:
                    # Round 14: stream 2 alone (plus Stream B for the values).
                    single_c = {'context': [context[0]], 'seq_len': max_seq_len}
                    single_null = {'context': [context_null[0]],
                                   'seq_len': max_seq_len}
                    if relay_prompt:
                        single_c['relay'] = dict(context=[context_relay[0]],
                                                 from_frame=relay_from_latent)
                    if override_img2_value:
                        ctx_b = context_relay[0] if relay_prompt else context[0]
                        single_c_b = dict(single_c, context=[context[0], ctx_b])
                        if relay_prompt:
                            single_c_b['relay'] = dict(
                                context=[context_relay[0]] * 2,
                                from_frame=relay_from_latent)
                        single_null_b = dict(
                            single_null, context=[context_null[0]] * 2)
            else:
                arg_c = {
                    'context': [context[0]],
                    'seq_len': max_seq_len,
                    'y': [y],
                }
                if relay_prompt:
                    # Round 7: plain I2V with the prompt relay (conditional
                    # pass only, as in every relay run).
                    arg_c['relay'] = dict(context=[context_relay[0]],
                                          from_frame=relay_from_latent)
                arg_null = {
                    'context': context_null,
                    'seq_len': max_seq_len,
                    'y': [y],
                }
            if reach_on:
                # Both signs of a nudge in one batch-2 conditional pass.
                reach_c = dict(arg_c, context=[context[0]] * 2, y=[y, y])
                if relay_prompt:
                    reach_c['relay'] = dict(context=[context_relay[0]] * 2,
                                            from_frame=relay_from_latent)
                reach_record = dict(
                    meta=dict(target=reach_target, eps=reach_eps,
                              blur=reach_blur, patterns=reach_patterns,
                              seed=reach_seed, steps=reach_steps,
                              relay_from=relay_from_latent,
                              relay_prompt=relay_prompt,
                              latent_frames=latent_frame_num,
                              latent_hw=[lat_h, lat_w], video_seed=seed),
                    steps={})
                logging.info(
                    "Reach test at steps %s: nudge each of %d latents by "
                    "+-(blurred noise, sigma %.1f latent px, RMS %.3f x "
                    "RMS(x_t)), %d pattern(s); target latent %d",
                    reach_steps, latent_frame_num, reach_blur, reach_eps,
                    reach_patterns, reach_target)

            def measure_reach(model, sample, t, sigma, v_base):
                """Round 7 reach test at one step (see reach_test)."""
                sample = sample.to(self.device).float()
                v_base = v_base.float()
                nudge_rms = reach_eps * float(sample.pow(2).mean().sqrt())
                t2 = torch.stack([t, t]).to(self.device)

                def respond(delta, norm):
                    ATTN_PROBE.begin_forward(-1, 'reach')  # never probed
                    out = model([sample + delta, sample - delta], t=t2,
                                **reach_c)
                    if offload_model:
                        torch.cuda.empty_cache()
                    resp = reach_response(sample, sigma, norm, delta, out[0],
                                          out[1], v_base, reach_blur)
                    return {k: [float(x) for x in v.tolist()]
                            for k, v in resp.items()}

                rec = dict(sigma=float(sigma), timestep=float(t),
                           nudge_rms=nudge_rms, patterns=[])
                for p in range(reach_patterns):
                    pattern = reach_pattern(
                        sample.size(0), lat_h, lat_w, reach_blur,
                        reach_seed + p).to(self.device)
                    rows = []
                    for j in range(latent_frame_num):
                        delta = torch.zeros_like(sample)
                        delta[:, j] = nudge_rms * pattern
                        rows.append(respond(delta, nudge_rms))
                    near = reach_target - 1
                    delta = torch.zeros_like(sample)
                    delta[:, near] = 0.25 * nudge_rms * pattern
                    quarter = respond(delta, 0.25 * nudge_rms)
                    delta = torch.zeros_like(sample)
                    delta[:, near] = 1e-2 * nudge_rms * pattern
                    floor = respond(delta, nudge_rms)  # in full-nudge units
                    rec['patterns'].append(dict(seed=reach_seed + p,
                                                rows=rows, quarter=quarter,
                                                floor=floor))
                    del pattern, delta
                return rec

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

            kg_on = (image_condition_paired_anchor_attention
                     and keyframe_guidance_scale > 0)
            if kg_on:
                # Prefix latents 1..anchor-1 only (frame 0 is Img1).
                kg_mask = torch.zeros(1, model_latent_frame_num, 1, 1,
                                      device=self.device)
                kg_mask[:, 1:anchor_index] = 1.0
                logging.info(
                    "Keyframe guidance k=%.2f on prefix latents 1..%d for "
                    "sampling steps < %d", keyframe_guidance_scale,
                    anchor_index - 1, keyframe_guidance_until_step)

            for sampling_step, first_visit, restart in tqdm(step_plan):
                t = timesteps[sampling_step]
                if restart:
                    # Resampling: back to step J's noise level from each
                    # stream's predicted clean latent at step K, fresh noise,
                    # fresh solver history.
                    sigma_j = step_sigmas[sampling_step]
                    renoise = torch.randn(
                        *latent.shape, dtype=torch.float32, generator=seed_g,
                        device=self.device)
                    latent = (1.0 - sigma_j) * x0_hat + sigma_j * renoise
                    sample_scheduler = copy.deepcopy(pristine_scheduler)
                    if override_own:
                        latent2 = ((1.0 - sigma_j) * x0_hat2
                                   + sigma_j * renoise)
                        scheduler2 = copy.deepcopy(pristine_scheduler)
                    del renoise
                # Repeated (resampled) steps are not recorded by the probe.
                probe_step = sampling_step if first_visit else -1
                if image_condition_paired_anchor_attention:
                    suffix_fade_factor = paired_fade_factor(
                        sampling_step, suffix_fade_start_step,
                        suffix_fade_end_step)
                    anchor_is_enabled = (
                        image_condition_paired_prefix_route_switch_step is None
                        or sampling_step <
                        image_condition_paired_prefix_route_switch_step)
                    current_stream_weight = (
                        image_condition_paired_stream_weight
                        * suffix_fade_factor)
                    arg_c['paired_stream_weight'] = current_stream_weight
                    arg_c['paired_prefix_anchor_active'] = anchor_is_enabled
                    arg_null['paired_stream_weight'] = current_stream_weight
                    arg_null['paired_prefix_anchor_active'] = anchor_is_enabled
                    current_variant = image_condition_paired_variant
                    if (image_condition_paired_variant_until_step is not None
                            and sampling_step >=
                            image_condition_paired_variant_until_step):
                        # Prefix variant off; keep the Stream-B timing fix.
                        current_variant = (
                            {'suffix_time_fix': True}
                            if current_variant
                            and current_variant.get('suffix_time_fix') else None)
                    arg_c['paired_variant'] = current_variant
                    arg_null['paired_variant'] = current_variant
                    latent_model_input = [
                        latent.to(self.device),
                        latent[:, anchor_index:].to(self.device),
                    ]
                    timestep = [t, t]
                elif override_on and override_single:
                    y2_now = y2
                    if (y2_late is not None
                            and sampling_step >= img2_release_step):
                        y2_now = y2_late
                    if y2_by_span is not None:
                        k_now = img2_spans[min(sampling_step,
                                               len(img2_spans) - 1)]
                        if k_now != 4:
                            y2_now = y2_by_span[k_now]
                    latent_model_input = [latent.to(self.device)]
                    timestep = [t]
                    if override_img2_value is not None:
                        latent_model_input.append(
                            latent[:, anchor_index:].to(self.device))
                        timestep.append(t)
                        single_ov = dict(single=True, anchor=anchor_index,
                                         img2_value=override_img2_value)
                        arg_c = dict(single_c_b, y=[y2_now, y_b],
                                     override=single_ov)
                        arg_null = dict(single_null_b, y=[y2_now, y_b],
                                        override=single_ov)
                    else:
                        arg_c = dict(single_c, y=[y2_now])
                        arg_null = dict(single_null, y=[y2_now])
                elif override_on:
                    latent_model_input = [
                        latent.to(self.device),
                        (latent2 if override_own else latent).to(self.device),
                    ]
                    timestep = [t, t]
                    current_override = dict(override_cfg)
                    if (override_handover_step is not None
                            and sampling_step >= override_handover_step):
                        current_override['weights'] = handover_weights
                    lam_now = 0.0
                    if stream_b_weight is not None:
                        lam_now = stream_b_weight * paired_fade_factor(
                            sampling_step, stream_b_fade_start_step,
                            stream_b_fade_end_step)
                    if override_img2_value is not None:
                        # Round 13 (Gordon's V_B0): Stream B on stream 2's
                        # latents a.., weight 0, only for Img2's clean values.
                        current_override['stream_b'] = dict(weight=0.0)
                        current_override['img2_value'] = override_img2_value
                        latent_model_input.append(
                            (latent2 if override_own else latent)[
                                :, anchor_index:].to(self.device))
                        timestep.append(t)
                        arg_c = dict(override_args_b[0],
                                     override=current_override)
                        arg_null = dict(override_args_b[1],
                                        override=current_override)
                    elif lam_now > 0.0:
                        # Stream B: Img2-first I2V on stream 1's latents a..
                        current_override['stream_b'] = dict(weight=lam_now)
                        latent_model_input.append(
                            latent[:, anchor_index:].to(self.device))
                        timestep.append(t)
                        arg_c = dict(override_args_b[0],
                                     override=current_override)
                        arg_null = dict(override_args_b[1],
                                        override=current_override)
                    else:
                        arg_c = dict(override_args[0],
                                     override=current_override)
                        arg_null = dict(override_args[1],
                                        override=current_override)
                    if (y2_late is not None
                            and sampling_step >= img2_release_step):
                        # Round 10: the hold on Img2 is released.
                        arg_c = dict(arg_c, y=[y, y2_late] + arg_c['y'][2:])
                        arg_null = dict(arg_null,
                                        y=[y, y2_late] + arg_null['y'][2:])
                    if y2_by_span is not None:
                        # Round 12: Img2 on the scheduled number of frames.
                        k_now = img2_spans[min(sampling_step,
                                               len(img2_spans) - 1)]
                        if k_now != 4:
                            arg_c = dict(arg_c, y=[y, y2_by_span[k_now]]
                                         + arg_c['y'][2:])
                            arg_null = dict(arg_null, y=[y, y2_by_span[k_now]]
                                            + arg_null['y'][2:])
                else:
                    latent_model_input = [latent.to(self.device)]
                    timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                model = self._prepare_model_for_timestep(
                    t,
                    boundary,
                    offload_model,
                    # Free the finished high-noise expert in every run: on
                    # 40 GB GPUs plain I2V ran out of memory at the step-10
                    # expert switch without it.
                    release_high_noise_model=True)
                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]

                rev_now = rev_on and sampling_step < reverse_stream_until_step
                step_input, step_timestep = latent_model_input, timestep
                step_c, step_null = arg_c, arg_null
                if rev_now:
                    # Stream R frame j = working latent a - j (j = 0..a-1):
                    # the anchor slot first, then the prefix backward.
                    rev_latent = latent[:, 1:anchor_index + 1].flip(1).to(
                        self.device)
                if rev_now and reverse_stream_mode == 'attn':
                    rev_variant = dict(current_variant or {},
                                       rev_strength=reverse_stream_strength,
                                       rev_shape=reverse_stream_shape)
                    if reverse_stream_couple:
                        rev_variant['rev_couple'] = True
                    step_input = latent_model_input + [rev_latent]
                    step_timestep = torch.stack([t, t, t]).to(self.device)
                    step_c = dict(arg_c,
                                  context=arg_c['context'] + [context_rev[0]],
                                  y=arg_c['y'] + [y_rev],
                                  paired_variant=rev_variant)
                    step_null = dict(
                        arg_null,
                        context=arg_null['context'] + [context_null[0]],
                        y=arg_null['y'] + [y_rev],
                        paired_variant=rev_variant)

                ATTN_PROBE.begin_forward(probe_step, 'cond')
                predictions_cond = model(
                    step_input, t=step_timestep, **step_c)
                noise_pred_cond = predictions_cond[0]
                stream2_cond = predictions_cond[1] if override_own else None
                del predictions_cond
                if offload_model:
                    torch.cuda.empty_cache()
                ATTN_PROBE.begin_forward(probe_step, 'uncond')
                predictions_uncond = model(
                    step_input, t=step_timestep, **step_null)
                noise_pred_uncond = predictions_uncond[0]
                stream2_uncond = (predictions_uncond[1] if override_own
                                  else None)
                del predictions_uncond
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + sample_guide_scale * (
                    noise_pred_cond - noise_pred_uncond)
                del step_input, step_c, step_null
                if rev_now and reverse_stream_mode == 'pred':
                    # Stream R as its own guided plain I2V prediction, fused
                    # into prefix latents 1..a-1 with weights w_t.
                    rev_t = torch.stack([t]).to(self.device)
                    ATTN_PROBE.begin_forward(probe_step, 'rev')
                    rev_cond = model([rev_latent], t=rev_t,
                                     context=[context_rev[0]],
                                     seq_len=rev_seq_len, y=[y_rev])[0]
                    if offload_model:
                        torch.cuda.empty_cache()
                    ATTN_PROBE.begin_forward(probe_step, 'rev')
                    rev_uncond = model([rev_latent], t=rev_t,
                                       context=[context_null[0]],
                                       seq_len=rev_seq_len, y=[y_rev])[0]
                    if offload_model:
                        torch.cuda.empty_cache()
                    noise_pred = fuse_reverse_prediction(
                        noise_pred,
                        rev_uncond + sample_guide_scale * (
                            rev_cond - rev_uncond), rev_frame_weights)
                    del rev_cond, rev_uncond, rev_t
                if rev_now:
                    del rev_latent
                if kg_on and sampling_step < keyframe_guidance_until_step:
                    # Same text condition, Img2 removed from every
                    # non-anchor token: anchor route off, Stream B weight 0.
                    arg_kg = dict(arg_c)
                    arg_kg['paired_prefix_anchor_active'] = False
                    arg_kg['paired_stream_weight'] = 0.0
                    arg_kg['paired_variant'] = None
                    ATTN_PROBE.begin_forward(probe_step, 'kg')
                    predictions_kg = model(
                        latent_model_input, t=timestep, **arg_kg)
                    noise_pred_no_img2 = predictions_kg[0]
                    del predictions_kg
                    if offload_model:
                        torch.cuda.empty_cache()
                    noise_pred = noise_pred + keyframe_guidance_scale * (
                        kg_mask * (noise_pred_cond - noise_pred_no_img2))
                    del noise_pred_no_img2

                if (reach_on and first_visit
                        and sampling_step in reach_steps):
                    # Round 7: read-only; the step below is unchanged.
                    reach_record['steps'][str(sampling_step)] = measure_reach(
                        model, latent, t, step_sigmas[sampling_step],
                        noise_pred_cond)
                sigma_now = step_sigmas[sampling_step]
                capture_x0 = (resample is not None
                              and sampling_step == resample[1])
                if stat_anchor != 'off':
                    stat_std = (stat_targets[1] if stat_anchor == 'meanstd'
                                and sampling_step >= stat_std_from_step
                                else None)
                    noise_pred = anchor_frame_stats(
                        noise_pred, latent.to(noise_pred.device), sigma_now,
                        stat_targets[0], stat_std, stat_strength)
                if override_own:
                    pred2 = stream2_uncond + sample_guide_scale * (
                        stream2_cond - stream2_uncond)
                    if stat_anchor != 'off' and stat_streams == 'both':
                        pred2 = anchor_frame_stats(
                            pred2, latent2.to(pred2.device), sigma_now,
                            stat_targets[0], stat_std, stat_strength)
                    if capture_x0:
                        x0_hat2 = latent2 - sigma_now * pred2
                    latent2 = scheduler2.step(
                        pred2.unsqueeze(0),
                        t,
                        latent2.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0].squeeze(0)
                    del stream2_cond, stream2_uncond, pred2
                if capture_x0:
                    x0_hat = latent - sigma_now * noise_pred
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

            if reach_on:
                self.last_reach = reach_record
                if self.rank == 0 and reach_test.get('path'):
                    with open(reach_test['path'], 'w') as handle:
                        json.dump(reach_record, handle)
                    summary_path = (os.path.splitext(reach_test['path'])[0]
                                    + '.txt')
                    with open(summary_path, 'w') as handle:
                        handle.write(reach_summary(reach_record))
                    logging.info("Reach test written to %s and %s",
                                 reach_test['path'], summary_path)
            ATTN_PROBE.record = False
            if ATTN_PROBE.enabled:
                ATTN_PROBE.finalize(
                    probe_path,
                    extra_meta=dict(
                        timesteps=[int(value) for value in timesteps.tolist()],
                        boundary_timestep=float(boundary),
                        stream_weight=(image_condition_paired_stream_weight
                                       if image_condition_paired_anchor_attention
                                       else None),
                        route_switch_step=(
                            image_condition_paired_prefix_route_switch_step),
                        suffix_fade=(suffix_fade_start_step,
                                     suffix_fade_end_step)
                        if image_condition_paired_anchor_attention else None,
                        variant=image_condition_paired_variant,
                        variant_until_step=(
                            image_condition_paired_variant_until_step),
                        guide_scale=list(guide_scale),
                        shift=shift,
                        seed=seed,
                        suffix_prompt=suffix_prompt,
                        keyframe_guidance=(keyframe_guidance_scale,
                                           keyframe_guidance_until_step),
                        reverse_stream=dict(
                            mode=reverse_stream_mode,
                            strength=reverse_stream_strength,
                            shape=reverse_stream_shape,
                            until_step=reverse_stream_until_step,
                            couple=bool(reverse_stream_couple),
                            prompt=reverse_prompt) if rev_on else None,
                        override=dict(
                            stream=override_stream,
                            mode=override_mode,
                            rerender_from_step=rerender_from_step,
                            min_weight=override_min_weight,
                            radius=override_radius,
                            blend=override_blend,
                            latent=override_latent,
                            anchor_index=anchor_index,
                            weights=override_weights,
                            img2_span=override_img2_span,
                            img2_mask=override_img2_mask,
                            img2_span_schedule=img2_spans,
                            img2_release=dict(step=img2_release_step,
                                              input=img2_release_input),
                            img2_value=override_img2_value,
                            profile=override_profile,
                            handover_step=override_handover_step,
                            stream_b=dict(
                                weight=stream_b_weight,
                                fade=(stream_b_fade_start_step,
                                      stream_b_fade_end_step))
                            if stream_b_weight is not None else None)
                        if override_on else None,
                        round4=dict(
                            relay_prompt=relay_prompt,
                            relay_from_latent=relay_from_latent,
                            stat_anchor=stat_anchor,
                            stat_strength=stat_strength,
                            stat_streams=stat_streams,
                            stat_std_from_step=stat_std_from_step,
                            resample=resample),
                    ))

            plan_latent = None
            if rerender_from_step is not None:
                # Direction B, pass 2: render the plan as plain I2V of Img1.
                plan_latent = latent
                sigma_s = step_sigmas[rerender_from_step]
                x0_plan = latent.clone()
                x0_plan[:, anchor_index] = img2_static.to(x0_plan.dtype)
                renoise = torch.randn(
                    *latent.shape, dtype=torch.float32, generator=seed_g,
                    device=self.device)
                latent = (1.0 - sigma_s) * x0_plan + sigma_s * renoise
                del x0_plan, renoise
                render_c = {
                    'context': [context[0]],
                    'seq_len': max_seq_len,
                    'y': [y],
                }
                if relay_prompt:
                    render_c['relay'] = dict(context=[context_relay[0]],
                                             from_frame=relay_from_latent)
                render_null = {
                    'context': [context_null[0]],
                    'seq_len': max_seq_len,
                    'y': [y],
                }
                for sampling_step in tqdm(
                        range(rerender_from_step, len(timesteps))):
                    t = timesteps[sampling_step]
                    model = self._prepare_model_for_timestep(
                        t, boundary, offload_model,
                        release_high_noise_model=True)
                    sample_guide_scale = guide_scale[1] if t.item(
                    ) >= boundary else guide_scale[0]
                    timestep = torch.stack([t]).to(self.device)
                    ATTN_PROBE.begin_forward(-1, 'cond')  # never probed
                    render_cond = model([latent.to(self.device)], t=timestep,
                                        **render_c)[0]
                    if offload_model:
                        torch.cuda.empty_cache()
                    ATTN_PROBE.begin_forward(-1, 'uncond')
                    render_uncond = model([latent.to(self.device)],
                                          t=timestep, **render_null)[0]
                    if offload_model:
                        torch.cuda.empty_cache()
                    noise_pred = pin_latent_prediction(
                        render_uncond + sample_guide_scale * (
                            render_cond - render_uncond),
                        latent.to(render_cond.device),
                        step_sigmas[sampling_step], anchor_index, img2_static)
                    del render_cond, render_uncond
                    latent = render_scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latent.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0].squeeze(0)
                    del noise_pred, timestep
                x0 = [latent]

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
                    if save_latents:
                        self.last_latents = dict(
                            stream1=retained_x0.float().cpu(), stream2=None,
                            plan=None, anchor=anchor_index)
                    videos = self.vae.decode([retained_x0])
                    fix_source = retained_x0
                    del retained_x0
                else:
                    if save_latents:
                        # Round 9: the final latents, before decoding.
                        self.last_latents = dict(
                            stream1=x0[0].float().cpu(),
                            stream2=(latent2.float().cpu()
                                     if override_own else None),
                            plan=(plan_latent.float().cpu()
                                  if plan_latent is not None else None),
                            anchor=anchor_index)
                    videos = self.vae.decode(x0)
                    fix_source = x0[0]
                if decode_fix is not None:
                    # Round 10: round 9's decode fix inside the run; the main
                    # video above stays the uncorrected one.
                    try:
                        torch.cuda.empty_cache()
                        fixed = smooth_latent_stats(fix_source.float(), fix_lo,
                                                    fix_hi, decode_fix)
                        self.last_fixed_video = self.vae.decode([fixed])[0]
                        del fixed
                        logging.info(
                            "Decode fix (%s) on latents %d-%d decoded",
                            decode_fix, fix_lo, fix_hi)
                    except RuntimeError as error:  # keep the main video
                        logging.warning("fixed video not decoded: %s", error)
                        self.last_fixed_video = None
                        torch.cuda.empty_cache()
                del fix_source
                if plan_latent is not None:
                    try:
                        torch.cuda.empty_cache()
                        self.last_plan_video = self.vae.decode(
                            [plan_latent])[0]
                    except RuntimeError as error:  # keep the main video
                        logging.warning("plan video not decoded: %s", error)
                        self.last_plan_video = None
                        torch.cuda.empty_cache()
                if override_save_stream2:
                    try:
                        torch.cuda.empty_cache()
                        self.last_stream2_video = self.vae.decode([latent2])[0]
                    except RuntimeError as error:  # keep the main video
                        logging.warning("stream 2 video not decoded: %s",
                                        error)
                        self.last_stream2_video = None
                        torch.cuda.empty_cache()

        del noise, latent, x0
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
