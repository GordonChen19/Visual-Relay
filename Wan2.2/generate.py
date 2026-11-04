# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import merge_video_audio, save_video, str2bool


EXAMPLE_PROMPT = {
    "t2v-A14B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "i2v-A14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
    "ti2v-5B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "animate-14B": {
        "prompt": "视频中的人在做动作",
        "video": "",
        "pose": "",
        "mask": "",
    },
    "s2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
        "audio":
            "examples/talk.wav",
        "tts_prompt_audio":
            "examples/zero_shot_prompt.wav",
        "tts_prompt_text":
            "希望你以后能够做的比我还好呦。",
        "tts_text":
            "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = [EXAMPLE_PROMPT[args.task]["image"]]
    if args.audio is None and args.enable_tts is False and "audio" in EXAMPLE_PROMPT[args.task]:
        args.audio = EXAMPLE_PROMPT[args.task]["audio"]
    if (args.tts_prompt_audio is None or args.tts_text is None) and args.enable_tts is True and "audio" in EXAMPLE_PROMPT[args.task]:
        args.tts_prompt_audio = EXAMPLE_PROMPT[args.task]["tts_prompt_audio"]
        args.tts_prompt_text = EXAMPLE_PROMPT[args.task]["tts_prompt_text"]
        args.tts_text = EXAMPLE_PROMPT[args.task]["tts_text"]

    if args.task == "i2v-A14B":
        assert args.image is not None, "Please specify the image path for i2v."
        assert 0.0 <= args.image_condition_paired_stream_weight <= 1.0, (
            "--image_condition_paired_stream_weight must be in [0, 1].")
        paired_fade_pairs = {
            "shared": (
                args.image_condition_paired_fade_start_step,
                args.image_condition_paired_fade_end_step),
            "suffix": (
                args.image_condition_paired_suffix_fade_start_step,
                args.image_condition_paired_suffix_fade_end_step),
        }
        for fade_name, fade_steps in paired_fade_pairs.items():
            assert ((fade_steps[0] is None) == (fade_steps[1] is None)), (
                f"Paired {fade_name} fade start and end steps must be "
                "supplied together.")
        if args.image_condition_paired_anchor_attention:
            assert not args.vr_override, (
                "--vr_override (Method 1) and paired anchor attention are "
                "separate methods; use one.")
            assert len(args.image) == 2, (
                "Paired anchor attention requires exactly two images.")
            assert 0.0 < args.vr_mirror_mass_gain, (
                "--vr_mirror_mass_gain must be positive.")
            assert 0.0 < args.vr_mass_cap < 1.0, (
                "--vr_mass_cap must be in (0, 1).")
            assert (args.image_latent_index is not None
                    and len(args.image_latent_index) == 2
                    and args.image_latent_index[0] == 0
                    and args.image_latent_index[1] > 0), (
                        "Paired anchor attention requires image latent "
                        "indices [0, positive].")
        else:
            assert all(steps == (None, None)
                       for steps in paired_fade_pairs.values()), (
                "Paired-attention fading requires "
                "--image_condition_paired_anchor_attention.")
            assert args.image_condition_paired_prefix_route_switch_step is None, (
                "Prefix routing requires "
                "--image_condition_paired_anchor_attention.")
            assert _build_paired_variant(args) is None, (
                "--vr_anchor_key/--vr_anchor_mirror/--vr_prefix_mass require "
                "--image_condition_paired_anchor_attention.")
            assert args.vr_variant_until_step is None, (
                "--vr_variant_until_step requires "
                "--image_condition_paired_anchor_attention.")
            assert not args.vr_suffix_prompt and not args.vr_kg_scale, (
                "--vr_suffix_prompt and --vr_kg_scale require "
                "--image_condition_paired_anchor_attention.")
            assert args.vr_rev_mode == "off" and not args.vr_rev_prompt, (
                "--vr_rev_mode and --vr_rev_prompt require "
                "--image_condition_paired_anchor_attention.")
            if args.vr_override:
                # Method 1 (round 3): two images, Img2 at an output latent.
                assert len(args.image) == 2, (
                    "--vr_override needs exactly two --image arguments.")
                assert (args.image_latent_index is not None
                        and len(args.image_latent_index) == 2
                        and args.image_latent_index[0] == 0
                        and args.image_latent_index[1] > 0), (
                            "--vr_override needs image latent indices "
                            "[0, positive].")
                assert 0.0 <= args.vr_override_min_weight <= 1.0, (
                    "--vr_override_min_weight must be in [0, 1].")
                assert args.vr_override_radius >= 0, (
                    "--vr_override_radius must be >= 0.")
                assert not args.vr_override_save_stream2 or (
                    args.vr_override_latent == "own"), (
                    "--vr_override_save_stream2 needs --vr_override_latent "
                    "own.")
                assert args.vr_stat_streams == "one" or (
                    args.vr_override_latent == "own"), (
                    "--vr_stat_streams both needs --vr_override_latent own.")
                assert 0.0 <= args.vr_stat_strength <= 1.0, (
                    "--vr_stat_strength must be in [0, 1].")
            else:
                assert len(args.image) == 1, (
                    "Multiple images require "
                    "--image_condition_paired_anchor_attention or "
                    "--vr_override.")
                assert (args.image_latent_index is None
                        or args.image_latent_index == [0]), (
                            "Standard I2V only supports image latent index 0.")
    else:
        assert not args.image_condition_paired_anchor_attention, (
            "Paired anchor attention is supported only by i2v-A14B.")
        assert args.image_latent_index is None, (
            "--image_latent_index is supported only by paired I2V.")
        assert args.image is None or len(args.image) == 1, (
            "Only paired I2V accepts multiple --image arguments.")

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.vr_probe_steps is not None:
        args.vr_probe_steps = sorted({
            int(value) for value in args.vr_probe_steps.split(',')
            if value.strip() != ''
        })
        assert args.vr_probe_steps and all(
            0 <= value < args.sample_steps for value in args.vr_probe_steps), (
                "--vr_probe_steps must list sampling-step indices in "
                "[0, sample_steps).")
        assert args.task == "i2v-A14B", (
            "--vr_probe_steps is supported only by i2v-A14B.")
    if args.vr_variant_until_step is not None:
        assert 0 < args.vr_variant_until_step <= args.sample_steps, (
            "--vr_variant_until_step must be in (0, sample_steps].")
    assert not args.vr_rev_couple or args.vr_rev_mode == "attn", (
        "--vr_rev_couple requires --vr_rev_mode attn.")
    if args.vr_rev_mode != "off":
        assert 0.0 <= args.vr_rev_strength <= 1.0, (
            "--vr_rev_strength must be in [0, 1].")
        assert 0 < args.vr_rev_until_step <= args.sample_steps, (
            "--vr_rev_until_step must be in (0, sample_steps].")

    if args.task == "i2v-A14B":
        for fade_name, fade_steps in paired_fade_pairs.items():
            if fade_steps[0] is not None:
                assert (0 <= fade_steps[0] < fade_steps[1]
                        < args.sample_steps), (
                            f"Paired {fade_name} fade steps must satisfy "
                            "0 <= start < end < sample_steps.")
        if args.image_condition_paired_prefix_route_switch_step is not None:
            assert (0 <= args.image_condition_paired_prefix_route_switch_step
                    < args.sample_steps), (
                        "--image_condition_paired_prefix_route_switch_step "
                        "must be in [0, sample_steps).")

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale
    if args.vr_high_noise_guide_scale is not None:
        low = (args.sample_guide_scale[0] if isinstance(
            args.sample_guide_scale, (tuple, list)) else
               args.sample_guide_scale)
        # (low-noise, high-noise), as WanI2V.generate reads it.
        args.sample_guide_scale = (float(low),
                                   float(args.vr_high_noise_guide_scale))
    # Round 12: "4,3,2,1" -> (4, 3, 2, 1).
    if args.vr_img2_span_schedule is not None:
        parts = [p.strip() for p in args.vr_img2_span_schedule.split(",")]
        assert (2 <= len(parts) <= args.sample_steps
                and all(p in ("1", "2", "3", "4") for p in parts)), (
            "--vr_img2_span_schedule lists 2 to --sample_steps frame counts, "
            "each 1-4, e.g. 4,3,2,1.")
        args.vr_img2_span_schedule = tuple(int(p) for p in parts)
    if not args.vr_override:
        assert (args.vr_override_img2_span == "latent"
                and not args.vr_override_save_stream2
                and args.vr_stat_anchor == "off"
                and args.vr_override_profile == "damp"
                and args.vr_override_handover_step is None
                and args.vr_stream_b_weight is None
                and args.vr_override_mode == "blend"
                and args.vr_rerender_from_step is None
                and args.vr_img2_motion_frames is None
                and args.vr_img2_context_video is None
                and args.vr_img2_fill == "img2"
                and args.vr_img2_code_stats is None
                and args.vr_img2_release_step is None
                and args.vr_img2_release_input == "joint"
                and args.vr_img2_span_schedule is None
                and args.vr_override_img2_mask is None
                and args.vr_override_img2_value is None
                and not args.vr_override_single), (
            "--vr_override_img2_span/--vr_override_save_stream2/"
            "--vr_stat_anchor/--vr_override_profile/"
            "--vr_override_handover_step/--vr_stream_b_weight/"
            "--vr_override_mode/--vr_rerender_from_step/"
            "--vr_img2_motion_frames/--vr_img2_context_video/--vr_img2_fill/"
            "--vr_img2_code_stats/--vr_img2_release_step/"
            "--vr_img2_release_input/--vr_img2_span_schedule/"
            "--vr_override_img2_mask/--vr_override_img2_value/"
            "--vr_override_single require --vr_override.")
        # Round 7: plain I2V may read a relay prompt from an explicit latent.
        assert not args.vr_relay_prompt or (
            not args.image_condition_paired_anchor_attention
            and args.vr_relay_from_latent is not None), (
            "--vr_relay_prompt without --vr_override is for plain I2V and "
            "needs --vr_relay_from_latent.")
    else:
        assert args.vr_stream_b_weight is None or (
            0.0 <= args.vr_stream_b_weight <= 1.0), (
            "--vr_stream_b_weight must be in [0, 1].")
        assert ((args.vr_stream_b_fade_start_step is None)
                == (args.vr_stream_b_fade_end_step is None)), (
            "Stream B fade start and end steps go together.")
        assert not (args.vr_override == "fresh"
                    and args.vr_override_img2_span != "latent"), (
            "--vr_override fresh fills Img2's whole latent.")
        assert not (args.vr_override == "clean"
                    and args.vr_override_img2_span == "frame"), (
            "--vr_override clean takes --vr_override_img2_span latent, last "
            "or motion.")
        assert not (args.vr_override in ("long", "context")
                    and args.vr_override_img2_span not in ("latent", "last")), (
            "--vr_override long/context take --vr_override_img2_span latent "
            "or last.")
        assert (args.vr_override == "context") == (
            args.vr_img2_context_video is not None), (
            "--vr_override context and --vr_img2_context_video go together.")
        assert args.vr_img2_fill not in ("hold", "video") or (
            args.vr_override == "context"
            and args.vr_override_img2_span == "last"), (
            "--vr_img2_fill hold/video needs --vr_override context with "
            "--vr_override_img2_span last.")
        assert args.vr_img2_fill != "grey" or (
            args.vr_override in ("long", "context")
            and args.vr_override_img2_span == "last"), (
            "--vr_img2_fill grey needs --vr_override long or context with "
            "--vr_override_img2_span last.")
        assert args.vr_img2_code_stats is None or (
            args.vr_override_img2_span == "latent"
            and args.vr_override != "long"), (
            "--vr_img2_code_stats needs --vr_override_img2_span latent and "
            "a stream other than long.")
        assert args.vr_img2_release_step is None or (
            args.vr_override in ("joint", "splice")
            and args.vr_override_img2_span == "latent"
            and args.vr_img2_code_stats is None), (
            "--vr_img2_release_step needs M1-PR's input: --vr_override joint "
            "or splice with --vr_override_img2_span latent and no "
            "--vr_img2_code_stats.")
        assert args.vr_img2_release_step is None or (
            0 < args.vr_img2_release_step < args.sample_steps
            and args.vr_stream_b_weight is None
            and args.vr_rerender_from_step is None
            and not args.vr_resample_cycles), (
            "--vr_img2_release_step must be in (0, --sample_steps) and does "
            "not combine with Stream B, re-rendering or resampling.")
        assert (args.vr_img2_release_input == "joint"
                or args.vr_img2_release_step is not None), (
            "--vr_img2_release_input own needs --vr_img2_release_step.")
        assert args.vr_img2_span_schedule is None or (
            args.vr_override in ("joint", "splice")
            and args.vr_override_img2_span == "latent"
            and args.vr_img2_code_stats is None
            and args.vr_img2_release_step is None
            and args.vr_stream_b_weight is None
            and args.vr_rerender_from_step is None
            and not args.vr_resample_cycles), (
            "--vr_img2_span_schedule needs M1-PR's input (--vr_override joint "
            "or splice, --vr_override_img2_span latent, no "
            "--vr_img2_code_stats) and does "
            "not combine with the release, Stream B, re-rendering or "
            "resampling.")
        assert args.vr_override_img2_value is None or (
            args.vr_override in ("joint", "splice")
            and args.vr_override_mode == "blend"
            and args.vr_stream_b_weight is None
            and args.vr_rerender_from_step is None
            and not args.vr_resample_cycles), (
            "--vr_override_img2_value needs --vr_override joint or splice in "
            "blend mode, and no --vr_stream_b_weight, re-rendering or "
            "resampling.")
        assert not args.vr_override_single or (
            args.vr_override_mode == "blend"
            and args.vr_stream_b_weight is None
            and args.vr_rerender_from_step is None
            and not args.vr_resample_cycles
            and args.vr_override_handover_step is None
            and not args.vr_override_save_stream2
            and args.vr_stat_anchor == "off"), (
            "--vr_override_single does not combine with route mode, "
            "--vr_stream_b_weight, re-rendering, resampling, the handover, "
            "--vr_override_save_stream2 or the colour anchor.")
        assert args.vr_override_img2_mask is None or (
            args.vr_override == "joint"
            and args.vr_override_img2_span == "latent"
            and args.vr_img2_release_step is None
            and args.vr_img2_span_schedule is None
            and args.vr_rerender_from_step is None), (
            "--vr_override_img2_mask last needs --vr_override joint with "
            "--vr_override_img2_span latent, and no release, span schedule or "
            "re-rendering.")
        assert (args.vr_override_img2_span == "motion") == (
            args.vr_img2_motion_frames is not None), (
            "--vr_override_img2_span motion and --vr_img2_motion_frames go "
            "together.")
        assert args.vr_rerender_from_step is None or (
            args.vr_override_img2_span == "latent"), (
            "--vr_rerender_from_step needs --vr_override_img2_span latent.")
        assert args.vr_override_mode == "blend" or (
            args.vr_override_min_weight == 1.0
            and args.vr_override_profile == "damp"
            and args.vr_override_handover_step is None
            and args.vr_stream_b_weight is None), (
            "--vr_override_mode route needs a full override (min weight 1, "
            "damp profile), no handover and no Stream B.")
        assert args.vr_rerender_from_step is None or (
            0 < args.vr_rerender_from_step < args.sample_steps
            and not args.vr_resample_cycles), (
            "--vr_rerender_from_step must be in (0, --sample_steps) and "
            "cannot be combined with resampling.")
    if args.vr_reach_steps is not None:
        args.vr_reach_steps = sorted({
            int(value) for value in args.vr_reach_steps.split(',')
            if value.strip()
        })
        assert args.vr_reach_steps and all(
            0 <= value < args.sample_steps for value in args.vr_reach_steps), (
                "--vr_reach_steps must list sampling steps in "
                "[0, sample_steps).")
        assert not args.vr_override and not (
            args.image_condition_paired_anchor_attention), (
            "--vr_reach_steps runs on plain I2V (no --vr_override, not "
            "paired).")
        assert args.vr_reach_target is not None and args.vr_reach_target > 0, (
            "--vr_reach_steps needs --vr_reach_target (Img2's latent).")
        assert (args.vr_reach_eps > 0 and args.vr_reach_blur >= 0
                and args.vr_reach_patterns >= 1), (
            "--vr_reach_eps > 0, --vr_reach_blur >= 0, --vr_reach_patterns "
            ">= 1.")
        assert not args.vr_resample_cycles, (
            "--vr_reach_steps does not take resampling.")
    assert args.vr_save_head_frames >= 0, "--vr_save_head_frames must be >= 0."
    if args.vr_resample_cycles:
        assert not args.image_condition_paired_anchor_attention, (
            "Resampling is not supported with paired anchor attention.")
        assert (args.vr_resample_from_step is not None
                and args.vr_resample_to_step is not None
                and args.vr_resample_cycles > 0), (
            "Resampling needs --vr_resample_from_step, --vr_resample_to_step "
            "and --vr_resample_cycles >= 1.")

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    # Round 10: round 9's decode fix inside the run (<stem>_fix.mp4).
    assert (args.vr_decode_fix is None) == (args.vr_decode_fix_window is None), (
        "--vr_decode_fix and --vr_decode_fix_window go together.")
    if args.vr_decode_fix_window is not None:
        assert args.task == "i2v-A14B", (
            "--vr_decode_fix is supported only by i2v-A14B.")
        parts = args.vr_decode_fix_window.split("-")
        assert len(parts) == 2 and all(p.strip().isdigit() for p in parts), (
            "--vr_decode_fix_window takes two latent indices, e.g. 10-11.")
        args.vr_decode_fix_window = tuple(int(p) for p in parts)
        out_latents = (args.frame_num - 1) // cfg.vae_stride[0] + 1
        lo, hi = args.vr_decode_fix_window
        assert 1 <= lo <= hi < out_latents - 1, (
            f"--vr_decode_fix_window {lo}-{hi} needs a latent on each side "
            f"(latents 0..{out_latents - 1}).")

    if args.vr_override:
        latent_frame_num = ((args.frame_num - 1) // cfg.vae_stride[0] + 1)
        assert args.image_latent_index[1] < latent_frame_num, (
            f"Img2 must be an output latent below {latent_frame_num}.")
    if args.image_condition_paired_anchor_attention:
        latent_frame_num = ((args.frame_num - 1) // cfg.vae_stride[0] + 1)
        assert args.image_latent_index[1] < latent_frame_num - 1, (
            "Img2 must leave at least one latent frame on its right; use an "
            f"index below {latent_frame_num - 1}.")

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    if not 's2v' in args.task:
        assert args.size in SUPPORTED_SIZES[
            args.
            task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _build_paired_variant(args):
    """Visual Relay round-1 prefix-route variant, or None for the baseline."""
    if (args.vr_anchor_key == "A" and not args.vr_anchor_mirror
            and args.vr_prefix_mass == "natural"
            and not args.vr_suffix_time_fix):
        return None
    return dict(
        anchor_key=args.vr_anchor_key,
        mirror=bool(args.vr_anchor_mirror),
        mass=args.vr_prefix_mass,
        split_shape=args.vr_split_shape,
        mirror_gain=float(args.vr_mirror_mass_gain),
        mass_cap=float(args.vr_mass_cap),
        suffix_time_fix=bool(args.vr_suffix_time_fix),
    )


def _probe_path(args):
    if args.vr_probe_steps is None:
        return None
    if args.vr_probe_out:
        return args.vr_probe_out
    if args.save_file:
        return os.path.splitext(args.save_file)[0] + "_probe.pt"
    return f"probe_{args.task}_seed{args.base_seed}.pt"


def _reach_test(args):
    """Round 7: the reach-test settings for WanI2V.generate, or None."""
    if args.vr_reach_steps is None:
        return None
    stem = (os.path.splitext(args.save_file)[0] if args.save_file else
            f"reach_{args.task}_seed{args.base_seed}")
    return dict(steps=args.vr_reach_steps, target=args.vr_reach_target,
                eps=args.vr_reach_eps, blur=args.vr_reach_blur,
                patterns=args.vr_reach_patterns, seed=1234,
                path=stem + "_reach.json")


def _load_motion_frames(path):
    """Round 7: frames 1-3 of a <stem>_head.pt file, [3, 3, H, W]."""
    if path is None:
        return None
    head = torch.load(path, map_location="cpu")
    if (not torch.is_tensor(head) or head.dim() != 4 or head.size(0) != 3
            or head.size(1) < 4):
        raise ValueError(f"{path} is not a [3, N >= 4, H, W] frame tensor")
    return head[:, 1:4].float().clone()


def _load_context_video(path):
    """Round 8: a first-pass video as [3, N, H, W] float in [-1, 1], from an
    .mp4 (read with OpenCV) or a .pt tensor of that shape."""
    if path is None:
        return None
    if path.endswith(".pt"):
        video = torch.load(path, map_location="cpu")
        if not torch.is_tensor(video) or video.dim() != 4 or video.size(0) != 3:
            raise ValueError(f"{path} is not a [3, N, H, W] frame tensor")
        return video.float()
    import cv2
    import numpy as np
    capture = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"could not read any frame from {path}")
    video = torch.from_numpy(np.stack(frames)).float()  # [N, H, W, 3], 0-255
    return video.permute(3, 0, 1, 2).div(127.5).sub(1.0).contiguous()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        action="append",
        default=None,
        help=("An input image. Repeat exactly twice when using paired anchor "
              "attention."))
    parser.add_argument(
        "--image_latent_index",
        type=int,
        action="append",
        default=None,
        help=("Latent index corresponding to each --image in paired anchor "
              "attention; use 0 for Img1 and a positive index for Img2."))
    parser.add_argument(
        "--image_condition_paired_anchor_attention",
        action="store_true",
        help=("Use a full Img1 I2V stream and an aligned Img2-first suffix. "
              "Suffix attention outputs from the two streams are fused with "
              "a fixed user-controlled weight."))
    parser.add_argument(
        "--image_condition_paired_stream_weight",
        type=float,
        default=0.25,
        help=("Fixed Img2-stream attention-output weight used by paired "
              "anchor attention; must be in [0, 1] (default: 0.25)."))
    parser.add_argument(
        "--image_condition_paired_fade_start_step",
        type=int,
        default=None,
        help=("Optional sampling-step index at which a cosine fade of the "
              "suffix fusion weight begins. Must be "
              "supplied with "
              "--image_condition_paired_fade_end_step."))
    parser.add_argument(
        "--image_condition_paired_fade_end_step",
        type=int,
        default=None,
        help=("Sampling-step index at which the suffix fusion weight reaches "
              "zero."))
    parser.add_argument(
        "--image_condition_paired_suffix_fade_start_step",
        type=int,
        default=None,
        help=("Optional Stream-B-suffix-only cosine-fade start step. "
              "Overrides the shared paired fade for the suffix when supplied "
              "with its matching end step."))
    parser.add_argument(
        "--image_condition_paired_suffix_fade_end_step",
        type=int,
        default=None,
        help=("Stream-B-suffix-only cosine-fade end step; must be supplied "
              "with --image_condition_paired_suffix_fade_start_step."))
    parser.add_argument(
        "--image_condition_paired_prefix_route_switch_step",
        type=int,
        default=None,
        help=("Optional hard Img2-anchor removal step. Prefix queries always "
              "attend ordinary Stream-A frames on both sides of the anchor. "
              "Before this step they additionally attend the shifted "
              "fresh-cache Stream-B Img2 "
              "anchor; at and after it only that anchor is removed."))
    # Visual Relay round 1: prefix -> Img2 anchor route variants and probe.
    parser.add_argument(
        "--vr_anchor_key",
        type=str,
        default="A",
        choices=["A", "B"],
        help=("Key of the Img2 anchor seen by prefix queries: A = "
              "history-aware Stream-A key (baseline), B = fresh-cache "
              "Stream-B frame-0 key moved to the anchor's RoPE position."))
    parser.add_argument(
        "--vr_anchor_mirror",
        action="store_true",
        default=False,
        help=("Mirror the anchor key's temporal RoPE per prefix frame t "
              "(position 2t - anchor), so Img2 appears to the query's left "
              "at its true distance, like a first frame."))
    parser.add_argument(
        "--vr_prefix_mass",
        type=str,
        default="natural",
        choices=["natural", "split", "mirror_mass", "img1_fade"],
        help=("Prefix attention-mass policy: natural softmax (baseline), "
              "split = keep the natural Img1+anchor mass but give a fraction "
              "t/anchor of it to the anchor, mirror_mass = anchor mass of "
              "frame t equals the Img1 mass of frame anchor-t, img1_fade = "
              "Img1 logit bias log(1 - t/anchor), no extra Img2 pull."))
    parser.add_argument(
        "--vr_split_shape",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
        help=("Shape of the fraction t/anchor used by split and img1_fade "
              "(cosine = 0.5 - 0.5 cos(pi t/anchor))."))
    parser.add_argument(
        "--vr_mirror_mass_gain",
        type=float,
        default=1.0,
        help="Scale of the mirrored anchor mass (mirror_mass policy).")
    parser.add_argument(
        "--vr_mass_cap",
        type=float,
        default=0.95,
        help="Upper bound of the mirrored anchor mass (mirror_mass policy).")
    parser.add_argument(
        "--vr_variant_until_step",
        type=int,
        default=None,
        help=("Apply the variant only for sampling steps below this index "
              "(default: while the prefix anchor route is active)."))
    parser.add_argument(
        "--vr_suffix_time_fix",
        action="store_true",
        default=False,
        help=("ABLATION of Gordon's offset (not a bug fix): shift Stream-B "
              "latents 1.. one RoPE unit left so the first suffix latent "
              "shares Img2's position in Stream B."))
    parser.add_argument(
        "--vr_suffix_prompt",
        type=str,
        default=None,
        help="Own prompt for the Img2-first Stream B (default: --prompt).")
    parser.add_argument(
        "--vr_kg_scale",
        type=float,
        default=0.0,
        help=("Keyframe guidance k on prefix latents: adds k * (cond - cond "
              "without Img2); costs one extra forward per guided step."))
    parser.add_argument(
        "--vr_kg_until_step",
        type=int,
        default=10,
        help="Keyframe guidance only for sampling steps below this index.")
    # Visual Relay round 2: Stream R, a reverse Img2-first stream for the prefix.
    parser.add_argument(
        "--vr_rev_mode",
        type=str,
        default="off",
        choices=["off", "attn", "pred"],
        help=("Stream R: an ordinary Img2-first I2V stream running backward "
              "over the latents before Img2. attn = fuse its attention output "
              "into prefix latent t at every layer with weight w_t (mirror of "
              "Stream B); pred = separate guided pass, fuse its prediction "
              "into prefix latent t with weight w_t."))
    parser.add_argument(
        "--vr_rev_strength",
        type=float,
        default=1.0,
        help="Stream R: s in w_t = s * f(t / anchor), in [0, 1].")
    parser.add_argument(
        "--vr_rev_shape",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
        help="Stream R: f = t / anchor (linear) or 0.5 - 0.5 cos(pi t / anchor).")
    parser.add_argument(
        "--vr_rev_until_step",
        type=int,
        default=10,
        help="Stream R only for sampling steps below this index.")
    # Visual Relay round 3: Gordon's Method 1, the self-attention override.
    parser.add_argument(
        "--vr_override",
        type=str,
        default=None,
        choices=["joint", "splice", "fresh", "clean", "long", "context"],
        help=("Method 1: stream 1 (plain I2V of Img1) takes stream 2's "
              "self-attention output in every layer; stream 2 also has Img2 "
              "at the second --image_latent_index. joint = stream 2 encoded "
              "in one causal VAE pass (Img1, zeros, Img2 x4, zeros); splice = "
              "Img1-only encoding with only the Img2 latent from that pass; "
              "fresh (round 5) = Img2's latent and the ones after it encoded "
              "like an Img2-first I2V, as Img1 is in latent 0; clean (round "
              "6) = joint, but Img2's latent is Img2 held 4 frames and encoded "
              "on its own (no grey history); long (round 8) = joint, but "
              "Img2's latent is latent a of Img2 held from frame 0 (encoded "
              "deep inside a clip, no grey); context (round 8, two-pass) = "
              "joint, but Img2's latent is latent a of "
              "--vr_img2_context_video's frames 0..4a-4 followed by Img2's "
              "latent frames."))
    parser.add_argument(
        "--vr_override_min_weight",
        type=float,
        default=1.0,
        help=("Override weight at the Img2 latent, rising linearly to 1 at "
              "--vr_override_radius latents away (1 = full override)."))
    parser.add_argument(
        "--vr_override_radius",
        type=int,
        default=3,
        help="Latents over which the override weight returns to 1.")
    parser.add_argument(
        "--vr_override_blend",
        type=str,
        default="linear",
        choices=["linear", "norm"],
        help=("linear: O1 + w (O2 - O1); norm: same direction with length "
              "(1 - w)|O1| + w|O2| per token and head."))
    parser.add_argument(
        "--vr_override_latent",
        type=str,
        default="shared",
        choices=["shared", "own"],
        help=("shared: stream 2 reads stream 1's noisy latent every step; "
              "own: stream 2 starts from the same noise but keeps its own "
              "latent."))
    # Visual Relay round 4: fixes on top of Method 1.
    parser.add_argument(
        "--vr_override_img2_span",
        type=str,
        default="latent",
        choices=["latent", "frame", "last", "motion"],
        help=("Stream 2's Img2: latent = all 4 frames of its latent (round 3, "
              "a 4-frame hold); frame = only the first frame of that latent, "
              "mask on that frame only; last (round 7) = only the last frame "
              "of that latent (frame 4a), mask on that frame only; motion "
              "(round 7) = Img2, then frames 1-3 of --vr_img2_motion_frames, "
              "mask on all 4."))
    parser.add_argument(
        "--vr_override_save_stream2",
        action="store_true",
        default=False,
        help=("With --vr_override_latent own, also save stream 2's video "
              "(the plain two-keyframe run) as <save_file stem>_s2.mp4."))
    parser.add_argument(
        "--vr_relay_prompt",
        type=str,
        default=None,
        help=("Prompt relay: latents from --vr_relay_from_latent on (default: "
              "the one after Img2) read this prompt in cross-attention, in "
              "both streams."))
    parser.add_argument(
        "--vr_relay_from_latent",
        type=int,
        default=None,
        help="First latent that reads --vr_relay_prompt (default: Img2's + 1).")
    parser.add_argument(
        "--vr_stat_anchor",
        type=str,
        default="off",
        choices=["off", "mean", "meanstd"],
        help=("Colour anchor: every step, move each latent's per-channel mean "
              "(and std) of stream 1's predicted clean latent toward an Img1 "
              "-> Img2 interpolation."))
    parser.add_argument(
        "--vr_stat_strength",
        type=float,
        default=1.0,
        help="Colour anchor strength in [0, 1].")
    parser.add_argument(
        "--vr_stat_streams",
        type=str,
        default="one",
        choices=["one", "both"],
        help="Anchor stream 1 only, or both streams (own latent only).")
    parser.add_argument(
        "--vr_stat_std_from_step",
        type=int,
        default=10,
        help="meanstd: match the std only from this sampling step on.")
    parser.add_argument(
        "--vr_resample_from_step",
        type=int,
        default=None,
        help="Resampling: go back to this step's noise level (J).")
    parser.add_argument(
        "--vr_resample_to_step",
        type=int,
        default=None,
        help="Resampling: after this step (K).")
    parser.add_argument(
        "--vr_resample_cycles",
        type=int,
        default=0,
        help="Resampling: how many times steps J..K are repeated (0 = off).")
    # Visual Relay round 5.
    parser.add_argument(
        "--vr_override_profile",
        type=str,
        default="damp",
        choices=["damp", "local"],
        help=("damp: w = min weight at Img2, 1 from --vr_override_radius "
              "away (round 3); local: w = 1 at Img2, 0 from "
              "--vr_override_radius away."))
    parser.add_argument(
        "--vr_override_handover_step",
        type=int,
        default=None,
        help=("From this step on, stream 1 takes stream 2's attention only at "
              "Img2's latent and finishes as plain I2V elsewhere."))
    parser.add_argument(
        "--vr_stream_b_weight",
        type=float,
        default=None,
        help=("Stream B lambda: an Img2-first I2V over Img2's latent and the "
              "rest, mixed into stream 1's latents after Img2 (paired design's "
              "suffix). Uses --vr_relay_prompt if set."))
    parser.add_argument(
        "--vr_stream_b_fade_start_step",
        type=int,
        default=None,
        help="Stream B lambda fades to 0 from this step ...")
    parser.add_argument(
        "--vr_stream_b_fade_end_step",
        type=int,
        default=None,
        help="... to this step (then Stream B leaves the batch).")
    # Visual Relay round 6.
    parser.add_argument(
        "--vr_override_mode",
        type=str,
        default="blend",
        choices=["blend", "route"],
        help=("blend: stream 1 takes stream 2's attention output (Method 1); "
              "route: stream 1 attends with stream 2's queries and keys but "
              "its own values, attn(q2, k2, v1), except at Img2's latent, "
              "which takes stream 2's output. Needs a full override."))
    parser.add_argument(
        "--vr_rerender_from_step",
        type=int,
        default=None,
        help=("Plan, then render: after the override run, re-noise stream "
              "1's final latent to this (low-noise) step and run the rest as "
              "plain I2V of Img1, with Img2's latent pinned to Img2 held 4 "
              "frames. The plan is also saved as <save_file stem>_plan.mp4."))
    # Visual Relay round 7.
    parser.add_argument(
        "--vr_img2_motion_frames",
        type=str,
        default=None,
        help=("A <stem>_head.pt file saved by --vr_save_head_frames from a "
              "plain I2V run of Img2: its frames 1-3 follow Img2 in Img2's "
              "latent (--vr_override_img2_span motion)."))
    # Visual Relay round 8.
    parser.add_argument(
        "--vr_img2_context_video",
        type=str,
        default=None,
        help=("--vr_override context: a first-pass video (.mp4, or a .pt "
              "[3, N, H, W] tensor in -1..1) whose frames 0..4a-4 are encoded "
              "before Img2's latent frames."))
    parser.add_argument(
        "--vr_img2_fill",
        type=str,
        default="img2",
        choices=["img2", "hold", "video", "grey"],
        help=("--vr_override context with --vr_override_img2_span last: the "
              "first 3 frames of Img2's latent. img2 = Img2; hold = frame "
              "4a-4 of the context video held; video = the context video's "
              "frames 4a-3..4a-1 with their colour means moved onto the line "
              "from frame 4a-4 to Img2; grey (round 9, also with --vr_override "
              "long) = grey frames. Frame 4a is Img2 in every case."))
    parser.add_argument(
        "--vr_img2_code_stats",
        type=str,
        default=None,
        choices=["mean", "meanstd"],
        help=("Round 9: Img2's latent in stream 2's input keeps its spatial "
              "pattern but takes each channel's mean (and std) from the long "
              "code (Img2 held from frame 0), which decodes at Img2's own "
              "brightness. With --vr_override_img2_span latent."))
    parser.add_argument(
        "--vr_save_latents",
        action="store_true",
        default=False,
        help=("Round 9: also save the final latents (stream 1, stream 2, "
              "plan; float32) as <save_file stem>_latents.pt, for "
              "vr_latent_fix.py."))
    parser.add_argument(
        "--vr_img2_release_step",
        type=int,
        default=None,
        help=("Round 10: from this sampling step on, stream 2 reads PR-LF's "
              "input (the joint encoding with Img2 on its latent's last "
              "frame only) instead of M1-PR's (Img2 on all 4 frames). Needs "
              "--vr_override joint with --vr_override_img2_span latent."))
    parser.add_argument(
        "--vr_img2_release_input",
        type=str,
        default="joint",
        choices=["joint", "own"],
        help=("Round 12: stream 2's input after --vr_img2_release_step: "
              "joint (PR-LF's, default) or own (CL-LF's: Img2's latent "
              "encoded on its own, grey x4 then Img2)."))
    parser.add_argument(
        "--vr_img2_span_schedule",
        type=str,
        default=None,
        help=("Round 12: frames of Img2's latent given at each sampling step, "
              "Img2 on the last k (e.g. 4,3,2,1: all 4 at step 0, frames "
              "4a-2..4a at step 1, 4a-1..4a at step 2, frame 4a from step 3 "
              "on; the last entry holds). Joint encoding; needs --vr_override "
              "joint with --vr_override_img2_span latent."))
    parser.add_argument(
        "--vr_override_img2_mask",
        type=str,
        default=None,
        choices=["last"],
        help=("Round 12: last = Img2's latent keeps M1-PR's code (Img2 on all "
              "4 frames, joint encoding) but only its last frame is marked, "
              "as in PR-LF."))
    parser.add_argument(
        "--vr_override_img2_value",
        type=str,
        default=None,
        choices=["before", "all"],
        help=("Round 13, Gordon's K_A12 / V_B0: Stream B (Img2-first I2V on "
              "stream 2's latents a.., weight 0) runs every step and gives "
              "the values of Img2's latent to stream 2's queries on latents "
              "before Img2 (before) or on every other latent (all); keys stay "
              "stream 2's."))
    parser.add_argument(
        "--vr_override_single",
        action="store_true",
        default=False,
        help=("Round 14: run stream 2 alone as the video (no stream 1): its "
              "input as set by --vr_override and the Img2 options. With "
              "--vr_override_img2_value, Stream B runs beside it only for the "
              "clean values. Gives what earlier rounds saved as _s2.mp4."))
    parser.add_argument(
        "--vr_decode_fix",
        type=str,
        default=None,
        choices=["mean", "meanstd"],
        help=("Round 10: round 9's decode fix inside the run. The main "
              "video's final latents in --vr_decode_fix_window get each "
              "channel's spatial mean (meanstd: and std) on the line between "
              "the latents on either side, and are decoded again as <save_file "
              "stem>_fix.mp4; the main video is saved uncorrected."))
    parser.add_argument(
        "--vr_decode_fix_window",
        type=str,
        default=None,
        help=("Round 10: the latents the decode fix corrects, LO-HI (e.g. "
              "10-11 for FNR, frames 37-44)."))
    parser.add_argument(
        "--vr_save_head_frames",
        type=int,
        default=0,
        help=("Also save the first N decoded frames ([3, N, H, W], float, "
              "-1..1) as <save_file stem>_head.pt (0 = off)."))
    parser.add_argument(
        "--vr_reach_steps",
        type=str,
        default=None,
        help=("Forward-only reach test on plain I2V at these sampling steps, "
              "e.g. '0,5,9': nudge each latent by +- a blurred pattern and "
              "record how the predicted clean latent changes. Written to "
              "<save_file stem>_reach.json and _reach.txt."))
    parser.add_argument(
        "--vr_reach_target",
        type=int,
        default=None,
        help="Reach test: the latent whose response is read (Img2's latent).")
    parser.add_argument(
        "--vr_reach_eps",
        type=float,
        default=0.1,
        help="Reach test: nudge RMS as a fraction of RMS(x_t).")
    parser.add_argument(
        "--vr_reach_blur",
        type=float,
        default=4.0,
        help="Reach test: blur of the nudge and the read-out, latent pixels.")
    parser.add_argument(
        "--vr_reach_patterns",
        type=int,
        default=2,
        help="Reach test: number of random blurred nudge patterns.")
    parser.add_argument(
        "--vr_high_noise_guide_scale",
        type=float,
        default=None,
        help=("CFG scale for the high-noise expert only (steps 0-9); the "
              "low-noise expert keeps --sample_guide_scale / the default."))
    parser.add_argument(
        "--vr_rev_couple",
        action="store_true",
        default=False,
        help=("Round 3, --vr_rev_mode attn only: Stream R frame a - t also "
              "takes prefix latent t's mixed output, so both streams follow "
              "one trajectory."))
    parser.add_argument(
        "--vr_rev_prompt",
        type=str,
        default=None,
        help=("Stream R prompt: the prefix described played backward from "
              "Img2 (default: --prompt)."))
    parser.add_argument(
        "--vr_probe_steps",
        type=str,
        default=None,
        help=("Comma-separated sampling steps at which the read-only "
              "attention probe records the conditional pass, e.g. "
              "'0,2,5,9,14,20,24,30,39'."))
    parser.add_argument(
        "--vr_probe_samples",
        type=int,
        default=48,
        help="Spatial positions sampled per query latent frame by the probe.")
    parser.add_argument(
        "--vr_probe_out",
        type=str,
        default=None,
        help=("Probe output file (default: <save_file stem>_probe.pt). A "
              "text summary is written next to it."))
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")

    # animate
    parser.add_argument(
        "--src_root_path",
        type=str,
        default=None,
        help="The file of the process output path. Default None.")
    parser.add_argument(
        "--refert_num",
        type=int,
        default=77,
        help="How many frames used for temporal guidance. Recommended to be 1 or 5."
    )
    parser.add_argument(
        "--replace_flag",
        action="store_true",
        default=False,
        help="Whether to use replace.")
    parser.add_argument(
        "--use_relighting_lora",
        action="store_true",
        default=False,
        help="Whether to use relighting lora.")
    
    # following args only works for s2v
    parser.add_argument(
        "--num_clip",
        type=int,
        default=None,
        help="Number of video clips to generate, the whole video will not exceed the length of audio."
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to the audio file, e.g. wav, mp3")
    parser.add_argument(
        "--enable_tts",
        action="store_true",
        default=False,
        help="Use CosyVoice to synthesis audio")
    parser.add_argument(
        "--tts_prompt_audio",
        type=str,
        default=None,
        help="Path to the tts prompt audio file, e.g. wav, mp3. Must be greater than 16khz, and between 5s to 15s.")
    parser.add_argument(
        "--tts_prompt_text",
        type=str,
        default=None,
        help="Content to the tts prompt audio. If provided, must exactly match tts_prompt_audio")
    parser.add_argument(
        "--tts_text",
        type=str,
        default=None,
        help="Text wish to synthesize")
    parser.add_argument(
        "--pose_video",
        type=str,
        default=None,
        help="Provide Dw-pose sequence to do Pose Driven")
    parser.add_argument(
        "--start_from_ref",
        action="store_true",
        default=False,
        help="whether set the reference image as the starting point for generation"
    )
    parser.add_argument(
        "--infer_frames",
        type=int,
        default=80,
        help="Number of frames per clip, 48 or 80 or others (must be multiple of 4) for 14B s2v"
    )
    args = parser.parse_args()
    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)
    video_stream2 = None  # round 4: stream 2's video when it is saved
    video_plan = None  # round 6: the plan of a re-rendered run
    latents_final = None  # round 9: final latents, with --vr_save_latents
    video_fixed = None  # round 10: the decode-fixed video, with --vr_decode_fix

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    image_paths = args.image or []
    if image_paths:
        opened_images = [
            Image.open(path).convert("RGB") for path in image_paths
        ]
        img = (opened_images if (args.image_condition_paired_anchor_attention
                                 or args.vr_override) else opened_images[0])
        logging.info("Input image(s): %s", image_paths)

    # prompt extend
    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt_output = prompt_expander(
                args.prompt,
                image=img[0] if isinstance(img, list) else img,
                tar_lang=args.prompt_extend_target_lang,
                seed=args.base_seed)
            if prompt_output.status == False:
                logging.info(
                    f"Extending prompt failed: {prompt_output.message}")
                logging.info("Falling back to original prompt.")
                input_prompt = args.prompt
            else:
                input_prompt = prompt_output.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")

    if "t2v" in args.task:
        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        logging.info(f"Generating video ...")
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
    elif "ti2v" in args.task:
        logging.info("Creating WanTI2V pipeline.")
        wan_ti2v = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        logging.info(f"Generating video ...")
        video = wan_ti2v.generate(
            args.prompt,
            img=img,
            size=SIZE_CONFIGS[args.size],
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
    elif "animate" in args.task:
        logging.info("Creating Wan-Animate pipeline.")
        wan_animate = wan.WanAnimate(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
            use_relighting_lora=args.use_relighting_lora
        )

        logging.info(f"Generating video ...")
        video = wan_animate.generate(
            src_root_path=args.src_root_path,
            replace_flag=args.replace_flag,
            refert_num = args.refert_num,
            clip_len=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
    elif "s2v" in args.task:
        logging.info("Creating WanS2V pipeline.")
        wan_s2v = wan.WanS2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )
        logging.info(f"Generating video ...")
        video = wan_s2v.generate(
            input_prompt=args.prompt,
            ref_image_path=image_paths[0] if image_paths else None,
            audio_path=args.audio,
            enable_tts=args.enable_tts,
            tts_prompt_audio=args.tts_prompt_audio,
            tts_prompt_text=args.tts_prompt_text,
            tts_text=args.tts_text,
            num_repeat=args.num_clip,
            pose_video=args.pose_video,
            max_area=MAX_AREA_CONFIGS[args.size],
            infer_frames=args.infer_frames,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            init_first_frame=args.start_from_ref,
        )
    else:
        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )
        logging.info("Generating video ...")
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            image_latent_indices=args.image_latent_index,
            image_condition_paired_anchor_attention=(
                args.image_condition_paired_anchor_attention),
            image_condition_paired_stream_weight=(
                args.image_condition_paired_stream_weight),
            image_condition_paired_fade_start_step=(
                args.image_condition_paired_fade_start_step),
            image_condition_paired_fade_end_step=(
                args.image_condition_paired_fade_end_step),
            image_condition_paired_suffix_fade_start_step=(
                args.image_condition_paired_suffix_fade_start_step),
            image_condition_paired_suffix_fade_end_step=(
                args.image_condition_paired_suffix_fade_end_step),
            image_condition_paired_prefix_route_switch_step=(
                args.image_condition_paired_prefix_route_switch_step),
            image_condition_paired_variant=_build_paired_variant(args),
            image_condition_paired_variant_until_step=(
                args.vr_variant_until_step),
            probe_steps=args.vr_probe_steps,
            probe_samples=args.vr_probe_samples,
            probe_path=_probe_path(args),
            suffix_prompt=args.vr_suffix_prompt,
            keyframe_guidance_scale=args.vr_kg_scale,
            keyframe_guidance_until_step=args.vr_kg_until_step,
            reverse_prompt=args.vr_rev_prompt,
            reverse_stream_mode=args.vr_rev_mode,
            reverse_stream_strength=args.vr_rev_strength,
            reverse_stream_shape=args.vr_rev_shape,
            reverse_stream_until_step=args.vr_rev_until_step,
            reverse_stream_couple=args.vr_rev_couple,
            override_stream=args.vr_override,
            override_min_weight=args.vr_override_min_weight,
            override_radius=args.vr_override_radius,
            override_blend=args.vr_override_blend,
            override_latent=args.vr_override_latent,
            override_img2_span=args.vr_override_img2_span,
            override_save_stream2=args.vr_override_save_stream2,
            relay_prompt=args.vr_relay_prompt,
            relay_from_latent=args.vr_relay_from_latent,
            stat_anchor=args.vr_stat_anchor,
            stat_strength=args.vr_stat_strength,
            stat_streams=args.vr_stat_streams,
            stat_std_from_step=args.vr_stat_std_from_step,
            resample_from_step=args.vr_resample_from_step,
            resample_to_step=args.vr_resample_to_step,
            resample_cycles=args.vr_resample_cycles,
            override_profile=args.vr_override_profile,
            override_handover_step=args.vr_override_handover_step,
            stream_b_weight=args.vr_stream_b_weight,
            stream_b_fade_start_step=args.vr_stream_b_fade_start_step,
            stream_b_fade_end_step=args.vr_stream_b_fade_end_step,
            override_mode=args.vr_override_mode,
            rerender_from_step=args.vr_rerender_from_step,
            img2_motion_frames=_load_motion_frames(args.vr_img2_motion_frames),
            reach_test=_reach_test(args),
            img2_context_video=_load_context_video(args.vr_img2_context_video),
            img2_fill=args.vr_img2_fill,
            img2_code_stats=args.vr_img2_code_stats,
            save_latents=args.vr_save_latents,
            img2_release_step=args.vr_img2_release_step,
            decode_fix=args.vr_decode_fix,
            decode_fix_window=args.vr_decode_fix_window,
            img2_span_schedule=args.vr_img2_span_schedule,
            img2_release_input=args.vr_img2_release_input,
            override_img2_mask=args.vr_override_img2_mask,
            override_img2_value=args.vr_override_img2_value,
            override_single=args.vr_override_single)
        latents_final = getattr(wan_i2v, 'last_latents', None)
        video_fixed = getattr(wan_i2v, 'last_fixed_video', None)
        video_stream2 = getattr(wan_i2v, 'last_stream2_video', None)
        video_plan = getattr(wan_i2v, 'last_plan_video', None)

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.mp4'
            args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}" + suffix

        logging.info(f"Saving generated video to {args.save_file}")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        if args.vr_save_head_frames:
            head_file = os.path.splitext(args.save_file)[0] + "_head.pt"
            logging.info(f"Saving the first {args.vr_save_head_frames} frames "
                         f"to {head_file}")
            torch.save(video[:, :args.vr_save_head_frames].float().cpu()
                       .clone(), head_file)
        if args.vr_save_latents and latents_final is not None:
            latents_file = os.path.splitext(args.save_file)[0] + "_latents.pt"
            logging.info(f"Saving the final latents to {latents_file}")
            torch.save(latents_final, latents_file)
        if video_fixed is not None:
            fixed_file = os.path.splitext(args.save_file)[0] + "_fix.mp4"
            logging.info(f"Saving the decode-fixed video ({args.vr_decode_fix}"
                         f", latents {args.vr_decode_fix_window[0]}-"
                         f"{args.vr_decode_fix_window[1]}) to {fixed_file}")
            save_video(
                tensor=video_fixed[None],
                save_file=fixed_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
        if video_stream2 is not None:
            stream2_file = os.path.splitext(args.save_file)[0] + "_s2.mp4"
            logging.info(f"Saving stream 2 (two-keyframe) video to "
                         f"{stream2_file}")
            save_video(
                tensor=video_stream2[None],
                save_file=stream2_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
        if video_plan is not None:
            plan_file = os.path.splitext(args.save_file)[0] + "_plan.mp4"
            logging.info(f"Saving the plan (override run before re-rendering) "
                         f"to {plan_file}")
            save_video(
                tensor=video_plan[None],
                save_file=plan_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
        if "s2v" in args.task:
            if args.enable_tts is False:
                merge_video_audio(video_path=args.save_file, audio_path=args.audio)
            else:
                merge_video_audio(video_path=args.save_file, audio_path="tts.wav")
    del video, video_stream2, video_plan, video_fixed

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
