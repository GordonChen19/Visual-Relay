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
        enabled_multi_image_modes = sum(bool(value) for value in (
            args.image_condition_one_way_prompt_guidance,
            args.image_condition_joint_img2_guidance,
            args.image_condition_cross_tail_bridge,
            args.image_condition_paired_anchor_attention,
            args.image_condition_self_attention_guidance,
        ))
        assert enabled_multi_image_modes <= 1, (
            "Multi-image attention ablations are mutually exclusive.")
        assert 0.0 <= args.image_condition_one_way_prompt_guidance_weight <= 1.0, (
            "--image_condition_one_way_prompt_guidance_weight must be in "
            "[0, 1].")
        assert (0.0 <
                args.image_condition_one_way_prompt_guidance_aux_anchor_prior
                < 1.0), (
                    "--image_condition_one_way_prompt_guidance_aux_anchor_prior "
                    "must be strictly between 0 and 1.")
        if args.image_condition_one_way_prompt_guidance_prefix_m10_only:
            assert args.image_condition_one_way_prompt_guidance, (
                "--image_condition_one_way_prompt_guidance_prefix_m10_only "
                "requires --image_condition_one_way_prompt_guidance.")
        if args.image_condition_one_way_prompt_guidance_full_img2_stream:
            assert args.image_condition_one_way_prompt_guidance, (
                "--image_condition_one_way_prompt_guidance_full_img2_stream "
                "requires --image_condition_one_way_prompt_guidance.")
        if args.image_condition_one_way_prompt_guidance_shared_anchor_update:
            assert args.image_condition_one_way_prompt_guidance, (
                "--image_condition_one_way_prompt_guidance_shared_anchor_update "
                "requires --image_condition_one_way_prompt_guidance.")
        if args.image_condition_one_way_prompt_guidance_guided_frames is not None:
            assert args.image_condition_one_way_prompt_guidance, (
                "--image_condition_one_way_prompt_guidance_guided_frames "
                "requires --image_condition_one_way_prompt_guidance.")
            assert args.image_condition_one_way_prompt_guidance_guided_frames >= 1, (
                "--image_condition_one_way_prompt_guidance_guided_frames "
                "must be at least 1.")
        assert 0.0 <= args.image_condition_joint_img2_guidance_weight <= 1.0, (
            "--image_condition_joint_img2_guidance_weight must be in "
            "[0, 1].")
        if args.image_condition_joint_img2_guidance_shifted_anchor:
            assert args.image_condition_joint_img2_guidance, (
                "--image_condition_joint_img2_guidance_shifted_anchor "
                "requires --image_condition_joint_img2_guidance.")
        if args.image_condition_joint_img2_guidance_rope_compression:
            assert args.image_condition_joint_img2_guidance_shifted_anchor, (
                "--image_condition_joint_img2_guidance_rope_compression "
                "requires --image_condition_joint_img2_guidance_shifted_anchor.")
        if args.image_condition_joint_img2_guidance_static_b_prompt:
            assert args.image_condition_joint_img2_guidance, (
                "--image_condition_joint_img2_guidance_static_b_prompt "
                "requires --image_condition_joint_img2_guidance.")
        if args.image_condition_joint_img2_guidance_persistent_b:
            assert args.image_condition_joint_img2_guidance_shifted_anchor, (
                "--image_condition_joint_img2_guidance_persistent_b "
                "requires --image_condition_joint_img2_guidance_shifted_anchor.")
        if args.image_condition_joint_img2_guidance_save_b_video:
            assert args.image_condition_joint_img2_guidance_persistent_b, (
                "--image_condition_joint_img2_guidance_save_b_video "
                "requires --image_condition_joint_img2_guidance_persistent_b.")
        if args.image_condition_cross_tail_bridge:
            assert args.image_condition_cross_tail_bridge_tail_frames >= 1, (
                "--image_condition_cross_tail_bridge_tail_frames must be "
                "at least 1.")
            assert (0.0 <=
                    args.image_condition_cross_tail_bridge_memory_prior <=
                    1.0), (
                        "--image_condition_cross_tail_bridge_memory_prior "
                        "must be in [0, 1].")
        if args.image_condition_cross_tail_bridge_uncontrolled_mass:
            assert args.image_condition_cross_tail_bridge, (
                "--image_condition_cross_tail_bridge_uncontrolled_mass "
                "requires --image_condition_cross_tail_bridge.")
        if (args.image_condition_one_way_prompt_guidance
                or args.image_condition_joint_img2_guidance
                or args.image_condition_cross_tail_bridge):
            assert args.image_prompt is not None and len(args.image_prompt) == 2, (
                "Per-stream prompt guidance requires exactly two "
                "--image_prompt arguments.")
            assert all(value for value in args.image_prompt), (
                "Each --image_prompt must be non-empty.")
            assert not args.use_prompt_extend, (
                "Prompt extension is not supported with per-image prompts.")
        else:
            assert args.image_prompt is None, (
                "--image_prompt is only used by a per-stream prompt-guidance "
                "ablation.")
        assert 0.0 <= args.image_condition_self_attention_guidance_scale <= 1.0, (
            "--image_condition_self_attention_guidance_scale must be "
            "in [0, 1].")
        if args.image_condition_self_attention_guidance_scale_end is not None:
            assert args.image_condition_self_attention_guidance, (
                "--image_condition_self_attention_guidance_scale_end requires "
                "--image_condition_self_attention_guidance.")
            assert (0.0 <=
                    args.image_condition_self_attention_guidance_scale_end <=
                    args.image_condition_self_attention_guidance_scale), (
                        "guidance-scale end must be in [0, guidance-scale].")
        if (args.image_condition_self_attention_guidance_scale_end_step
                is not None):
            assert (args.image_condition_self_attention_guidance_scale_end
                    is not None), (
                        "--image_condition_self_attention_guidance_scale_end_step "
                        "requires --image_condition_self_attention_guidance_scale_end.")
            assert (args.image_condition_self_attention_guidance_scale_end_step
                    >= 1), (
                        "--image_condition_self_attention_guidance_scale_end_step "
                        "must be at least 1.")
        assert 0.0 <= args.image_condition_self_attention_guidance_img2_weight <= 1.0, (
            "--image_condition_self_attention_guidance_img2_weight must be "
            "in [0, 1].")
        if (args.image_condition_self_attention_guidance_joint_query_single_memory_alpha
                is not None):
            assert args.image_condition_self_attention_guidance, (
                "--image_condition_self_attention_guidance_joint_query_single_memory_alpha "
                "requires --image_condition_self_attention_guidance.")
            assert (0.0 <=
                    args.image_condition_self_attention_guidance_joint_query_single_memory_alpha
                    <= 1.0), (
                        "joint-query single-memory alpha must be in [0, 1].")
        if (args.image_condition_self_attention_guidance_selective_suffix_memory_alpha
                is not None):
            assert args.image_condition_self_attention_guidance, (
                "--image_condition_self_attention_guidance_selective_suffix_memory_alpha "
                "requires --image_condition_self_attention_guidance.")
            assert (0.0 <=
                    args.image_condition_self_attention_guidance_selective_suffix_memory_alpha
                    <= 1.0), (
                        "selective suffix-memory alpha must be in [0, 1].")
        if (args.image_condition_self_attention_guidance_selective_suffix_memory_alpha_end
                is not None):
            assert (args.image_condition_self_attention_guidance_selective_suffix_memory_alpha
                    is not None), (
                        "selective suffix-memory alpha end requires the "
                        "selective suffix-memory mode.")
            assert (0.0 <=
                    args.image_condition_self_attention_guidance_selective_suffix_memory_alpha_end
                    <= 1.0), (
                        "selective suffix-memory alpha end must be in [0, 1].")
        if (args.image_condition_self_attention_guidance_selective_suffix_memory_alpha_end_step
                is not None):
            assert (args.image_condition_self_attention_guidance_selective_suffix_memory_alpha_end
                    is not None), (
                        "selective suffix-memory alpha end step requires "
                        "selective suffix-memory alpha end.")
            assert (args.image_condition_self_attention_guidance_selective_suffix_memory_alpha_end_step
                    >= 1), (
                        "selective suffix-memory alpha end step must be at least 1.")
        assert not (
            args.image_condition_self_attention_guidance_joint_query_single_memory_alpha
            is not None
            and args.image_condition_self_attention_guidance_selective_suffix_memory_alpha
            is not None), (
                "full-memory and selective-suffix memory modes are exclusive.")
        assert (args.image_condition_self_attention_guidance_rms_cap is None
                or args.image_condition_self_attention_guidance_rms_cap > 0), (
            "--image_condition_self_attention_guidance_rms_cap must be "
            "positive.")
        if args.image_condition_self_attention_guidance_anchor_prior:
            assert args.image_condition_self_attention_guidance, (
                "--image_condition_self_attention_guidance_anchor_prior "
                "requires --image_condition_self_attention_guidance.")
            assert (0.0 <
                    args.image_condition_self_attention_guidance_anchor_prior_end
                    <= args.image_condition_self_attention_guidance_anchor_prior_start
                    < 1.0), (
                        "anchor-prior values must satisfy "
                        "0 < end <= start < 1.")
        if (args.image_condition_self_attention_guidance_anchor_prior_end_step
                is not None):
            assert args.image_condition_self_attention_guidance_anchor_prior, (
                "--image_condition_self_attention_guidance_anchor_prior_end_step "
                "requires --image_condition_self_attention_guidance_anchor_prior.")
            assert (args.image_condition_self_attention_guidance_anchor_prior_end_step
                    >= 1), (
                        "--image_condition_self_attention_guidance_anchor_prior_end_step "
                        "must be at least 1.")
        if args.image_condition_self_attention_guidance_stop_step is not None:
            assert args.image_condition_self_attention_guidance, (
                "--image_condition_self_attention_guidance_stop_step requires "
                "--image_condition_self_attention_guidance.")
        if args.image_condition_self_attention_guidance_post_stop_img2_suffix:
            assert args.image_condition_self_attention_guidance, (
                "post-stop Img2 suffix guidance requires "
                "--image_condition_self_attention_guidance.")
            assert (args.image_condition_self_attention_guidance_stop_step
                    is not None), (
                        "post-stop Img2 suffix guidance requires "
                        "--image_condition_self_attention_guidance_stop_step.")
        assert args.image is not None, "Please specify the image path for i2v."
        assert 0.0 <= args.image_condition_paired_stream_weight <= 1.0, (
            "--image_condition_paired_stream_weight must be in [0, 1].")
        assert (args.image_condition_paired_memory_alpha is None
                or 0.0 <= args.image_condition_paired_memory_alpha <= 1.0), (
            "--image_condition_paired_memory_alpha must be in [0, 1].")
        assert 0.0 <= args.image_condition_paired_prefix_future_eta <= 1.0, (
            "--image_condition_paired_prefix_future_eta must be in [0, 1].")
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
        if args.image_condition_one_way_prompt_guidance:
            assert all(steps == (None, None)
                       for steps in paired_fade_pairs.values()), (
                "Paired-attention fading does not apply to one-way prompt "
                "guidance.")
            assert args.image_condition_paired_prefix_route_switch_step is None, (
                "Paired prefix routing does not apply to one-way prompt "
                "guidance.")
        if (args.image_condition_one_way_prompt_guidance
                or args.image_condition_joint_img2_guidance
                or args.image_condition_cross_tail_bridge
                or args.image_condition_paired_anchor_attention
                or args.image_condition_self_attention_guidance):
            assert len(args.image) == 2, (
                "The selected multi-image ablation requires exactly two "
                "images.")
            assert (args.image_latent_index is not None
                    and len(args.image_latent_index) == 2
                    and args.image_latent_index[0] == 0
                    and args.image_latent_index[1] > 0), (
                        "The selected multi-image ablation requires image "
                        "latent "
                        "indices [0, positive].")
            if args.image_condition_self_attention_guidance:
                assert all(steps == (None, None)
                           for steps in paired_fade_pairs.values()), (
                    "Paired-attention fading cannot be used with the "
                    "self-attention-guidance ablation.")
                assert (args.image_condition_paired_prefix_route_switch_step
                        is None), (
                    "Paired prefix routing cannot be used with the "
                    "self-attention-guidance ablation.")
        else:
            assert all(steps == (None, None)
                       for steps in paired_fade_pairs.values()), (
                "Paired-attention fading requires "
                "--image_condition_paired_anchor_attention.")
            assert args.image_condition_paired_prefix_route_switch_step is None, (
                "Prefix routing requires "
                "--image_condition_paired_anchor_attention.")
            assert len(args.image) == 1, (
                "Multiple images require "
                "--image_condition_paired_anchor_attention.")
            assert (args.image_latent_index is None
                    or args.image_latent_index == [0]), (
                        "Standard I2V only supports image latent index 0.")
    else:
        assert not args.image_condition_one_way_prompt_guidance, (
            "One-way prompt guidance is supported only by i2v-A14B.")
        assert not args.image_condition_joint_img2_guidance, (
            "Joint Img2 guidance is supported only by i2v-A14B.")
        assert not args.image_condition_cross_tail_bridge, (
            "Cross-tail bridge is supported only by i2v-A14B.")
        assert not args.image_condition_paired_anchor_attention, (
            "Paired anchor attention is supported only by i2v-A14B.")
        assert not args.image_condition_self_attention_guidance, (
            "Self-attention guidance is supported only by i2v-A14B.")
        assert args.image_latent_index is None, (
            "--image_latent_index is supported only by paired I2V.")
        assert args.image_prompt is None, (
            "--image_prompt is supported only by one-way paired I2V.")
        assert args.image is None or len(args.image) == 1, (
            "Only paired I2V accepts multiple --image arguments.")

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

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
        if args.image_condition_self_attention_guidance_stop_step is not None:
            assert (0 <=
                    args.image_condition_self_attention_guidance_stop_step <
                    args.sample_steps), (
                        "--image_condition_self_attention_guidance_stop_step "
                        "must be in [0, sample_steps).")
        if (args.image_condition_self_attention_guidance_anchor_prior_end_step
                is not None):
            assert (
                args.image_condition_self_attention_guidance_anchor_prior_end_step
                < args.sample_steps), (
                    "--image_condition_self_attention_guidance_anchor_prior_end_step "
                    "must be in [1, sample_steps).")
        if (args.image_condition_self_attention_guidance_scale_end_step
                is not None):
            assert (
                args.image_condition_self_attention_guidance_scale_end_step <
                args.sample_steps), (
                    "--image_condition_self_attention_guidance_scale_end_step "
                    "must be in [1, sample_steps).")
        if (args.image_condition_self_attention_guidance_selective_suffix_memory_alpha_end_step
                is not None):
            assert (
                args.image_condition_self_attention_guidance_selective_suffix_memory_alpha_end_step
                < args.sample_steps), (
                    "selective suffix-memory alpha end step must be in "
                    "[1, sample_steps).")

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    if (args.image_condition_one_way_prompt_guidance
            or args.image_condition_joint_img2_guidance
            or args.image_condition_cross_tail_bridge
            or args.image_condition_paired_anchor_attention
            or args.image_condition_self_attention_guidance):
        latent_frame_num = ((args.frame_num - 1) // cfg.vae_stride[0] + 1)
        assert args.image_latent_index[1] < latent_frame_num - 1, (
            "Img2 must leave at least one latent frame on its right; use an "
            f"index below {latent_frame_num - 1}.")
        if args.image_condition_cross_tail_bridge:
            img2_to_end_frames = (
                latent_frame_num - args.image_latent_index[1])
            assert (args.image_condition_cross_tail_bridge_tail_frames <
                    img2_to_end_frames), (
                        "--image_condition_cross_tail_bridge_tail_frames "
                        "is the overlap count and must be smaller than the "
                        "Img2-to-end window "
                        f"({img2_to_end_frames} latent frames).")
        if (args.image_condition_one_way_prompt_guidance_guided_frames
                is not None):
            retained_overlap_frames = (
                latent_frame_num - args.image_latent_index[1])
            assert (
                args.image_condition_one_way_prompt_guidance_guided_frames
                <= retained_overlap_frames), (
                    "--image_condition_one_way_prompt_guidance_guided_frames "
                    "cannot exceed the retained anchor-to-end overlap "
                    f"({retained_overlap_frames} latent frames).")

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    if not 's2v' in args.task:
        assert args.size in SUPPORTED_SIZES[
            args.
            task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


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
        "--image_prompt",
        type=str,
        action="append",
        default=None,
        help=("Prompt paired positionally with each repeated --image. In "
              "one-way prompt guidance and joint Img2 guidance these are "
              "local prompt 1 and local prompt 2, while --prompt is the "
              "global prompt."))
    parser.add_argument(
        "--image_condition_one_way_prompt_guidance",
        action="store_true",
        help=("Run native-length Img1 M and aligned Img2-first B. M uses "
              "--prompt as a global prompt plus both --image_prompt values "
              "as temporally relayed local prompts. B uses the global prompt "
              "plus local prompt 2 for its entire sequence. Both streams use ordinary "
              "bidirectional attention. M keeps its full Img1-only timeline "
              "and adds one auxiliary joint Img1-to-Img2 anchor sharing the "
              "ordinary anchor's integer RoPE position. Prefix queries use "
              "both the auxiliary anchor and ordinary anchor with a neutral "
              "shared-position prior, so together they receive one frame's "
              "evidence rather than an automatic factor-of-two advantage. Ordinary "
              "anchor/suffix queries see the ordinary anchor but not the "
              "auxiliary one. The auxiliary anchor uses its own query to "
              "attend to the prefix plus itself. The ordinary anchor keeps "
              "its own M-stream self-attention output, while B1: guides only "
              "the aligned post-anchor suffix; B0 does not update M. There is no "
              "M-to-B influence. Before CFG, conditional B0 replaces the "
              "conditional ordinary-anchor velocity; the unconditional "
              "pass is one ordinary retained Img1-only M stream with no "
              "auxiliary anchor, B stream, or prompt relay."))
    parser.add_argument(
        "--image_condition_one_way_prompt_guidance_weight",
        type=float,
        default=0.25,
        help=("Layerwise B1: output weight applied only to M's aligned "
              "post-anchor suffix; M's ordinary anchor keeps its own "
              "self-attention output (default: 0.25)."))
    parser.add_argument(
        "--image_condition_one_way_prompt_guidance_prefix_m10_only",
        action="store_true",
        help=("One-way prompt-guidance ablation: prefix queries exclude the "
              "auxiliary joint A10 and attend to retained ordinary M10. "
              "Omit this flag to retain the default shared A10/M10 route."))
    parser.add_argument(
        "--image_condition_one_way_prompt_guidance_aux_anchor_prior",
        type=float,
        default=0.5,
        help=("A10's prior within the shared-position A10/M10 group read by "
              "prefix queries; M10 receives the complementary 1-prior. "
              "This controls group evidence before the final softmax, not "
              "an absolute fraction of total attention (default: 0.5)."))
    parser.add_argument(
        "--image_condition_one_way_prompt_guidance_full_img2_stream",
        action="store_true",
        help=("Denoise B as a full native-length Img2-first video regardless "
              "of anchor position. B uses its complete timeline internally, "
              "but only B0 and the aligned overlapping B suffix outputs can "
              "affect M."))
    parser.add_argument(
        "--image_condition_one_way_prompt_guidance_guided_frames",
        type=int,
        default=None,
        help=("Optional number of aligned latent frames, counted from the "
              "retained anchor, that receive layerwise B guidance. For "
              "example, 3 applies B0:B2 to the retained anchor and its next "
              "two frames. Omit this option to retain the legacy route where "
              "B1: guides every post-anchor frame and B0 does not guide the "
              "anchor layerwise."))
    parser.add_argument(
        "--image_condition_one_way_prompt_guidance_shared_anchor_update",
        action="store_true",
        help=("Make auxiliary A10 and retained M10 receive the exact same "
              "completed self-attention output at every DiT layer. When "
              "--image_condition_one_way_prompt_guidance_guided_frames is "
              "set, this is the M10/B0 interpolated anchor output. Their "
              "residual states, cross-attention, and FFNs remain separate."))
    parser.add_argument(
        "--image_condition_joint_img2_guidance",
        action="store_true",
        help=("Run a native 21-latent jointly Img1/Img2 VAE-conditioned J "
              "stream and an aligned standard Img2-first B suffix. J uses "
              "--prompt globally plus both --image_prompt values with "
              "temporal Prompt Relay; B uses the global prompt plus local "
              "prompt 2. J uses ordinary bidirectional attention except "
              "that its anchor query reads only itself and its suffix. B "
              "uses ordinary bidirectional attention; both retain ordinary "
              "RoPE. At every "
              "DiT layer B0 and B1: attention outputs guide J's anchor and "
              "suffix respectively; J's prefix is unchanged. After the DiT, "
              "B0 replaces J's anchor velocity in each CFG branch before "
              "the scheduler step."))
    parser.add_argument(
        "--image_condition_joint_img2_guidance_weight",
        type=float,
        default=0.25,
        help=("Layerwise B attention-output weight applied to J's aligned "
              "anchor and suffix (default: 0.25)."))
    parser.add_argument(
        "--image_condition_joint_img2_guidance_shifted_anchor",
        action="store_true",
        help=("Shift the auxiliary standard Img2-first B view one latent "
              "earlier, so B1 aligns with J's Img2 anchor. J and B retain "
              "independent ordinary bidirectional self-attention. After the "
              "DiT, B1 "
              "replaces J's aligned anchor velocity. B is rebuilt from J "
              "noise each step and is never scheduled independently."))
    parser.add_argument(
        "--image_condition_joint_img2_guidance_rope_compression",
        action="store_true",
        help=("Opt-in query-dependent RoPE compression for shifted-anchor J. "
              "Prefix queries see the first suffix frame compressed onto the "
              "Img2 anchor coordinate; suffix queries see the anchor "
              "compressed onto the preceding frame coordinate while suffix "
              "queries/keys shift consistently. The anchor reads only itself "
              "and future frames. "
              "The anchor is discarded before decoding. B keeps ordinary "
              "RoPE."))
    parser.add_argument(
        "--image_condition_joint_img2_guidance_static_b_prompt",
        action="store_true",
        help=("Condition B on the global --prompt followed by 'No motion in "
              "the video occurs.' instead of global prompt plus local image "
              "prompt 2. J and its Prompt Relay are unchanged."))
    parser.add_argument(
        "--image_condition_joint_img2_guidance_persistent_b",
        action="store_true",
        help=("Initialize shifted B once from J[anchor-1:] and evolve it with "
              "a separate identically configured scheduler using B's own "
              "CFG-combined prediction. B1 and J's anchor start from the "
              "same noise, and B1 continues to supply J's anchor velocity."))
    parser.add_argument(
        "--image_condition_joint_img2_guidance_save_b_video",
        action="store_true",
        help=("Decode and save the final persistent B stream beside the main "
              "video using an _B filename suffix."))
    parser.add_argument(
        "--image_condition_cross_tail_bridge",
        action="store_true",
        help=("Run overlapping Img1 and Img2 temporal windows with "
              "simultaneous crossed self-attention memory. M spans frame 0 "
              "through T-1 frames after the Img2 anchor; B spans Img2 "
              "through the output end. M's final T and B's first T frames "
              "are aligned and start from identical noise. Every query also "
              "reads its complete native stream. The decoded video keeps M "
              "and appends B only after the overlap; B's CFG-combined "
              "velocity replaces M's aligned overlap velocity before every "
              "scheduler step."))
    parser.add_argument(
        "--image_condition_cross_tail_bridge_tail_frames",
        type=int,
        default=4,
        help=("Number T of aligned overlap latents, including the Img2 "
              "anchor (default: 4)."))
    parser.add_argument(
        "--image_condition_cross_tail_bridge_memory_prior",
        type=float,
        default=0.2,
        help=("Cross-stream source prior in each length-normalized native + "
              "cross-memory softmax (default: 0.2)."))
    parser.add_argument(
        "--image_condition_cross_tail_bridge_uncontrolled_mass",
        action="store_true",
        help=("Disable cross-tail source priors and length normalization. "
              "Each route directly concatenates native and cross K/V in one "
              "ordinary softmax, so attention mass is determined only by "
              "token count and QK logits."))
    parser.add_argument(
        "--image_condition_paired_anchor_attention",
        action="store_true",
        help=("Use Img1-only C/M and aligned Img2-first B. Every retained C "
              "query reads C prefix plus aligned C/B1: suffix K/V in one "
              "mass-normalized attention. B0 is excluded from C memory and "
              "maintains only the hidden transport state."))
    parser.add_argument(
        "--image_condition_self_attention_guidance",
        action="store_true",
        help=("Run expanded Img1-only M, joint Img1/Img2 J, and standard "
              "Img2-first suffix B in both CFG branches. M's temporary slot "
              "is excluded from attention. Removing J's anchor aligns its "
              "retained attention output with M for M/J guidance. J retained "
              "queries use one mathematical softmax across every visible key, "
              "including anchor-adjacent representations sharing a RoPE "
              "coordinate. "
              "B0 replaces the temporary anchor velocity before CFG."))
    parser.add_argument(
        "--image_condition_self_attention_guidance_scale",
        type=float,
        default=0.5,
        help=("J output weight used in both prefix and suffix fusion "
              "(default: 0.5)."))
    parser.add_argument(
        "--image_condition_self_attention_guidance_scale_end",
        type=float,
        default=None,
        help=("Optional final J output weight for a cosine guidance-scale "
              "schedule. It remains fixed after the end step."))
    parser.add_argument(
        "--image_condition_self_attention_guidance_scale_end_step",
        type=int,
        default=None,
        help=("Zero-based sampling step at which the cosine guidance-scale "
              "schedule reaches its final value. Defaults to the last step."))
    parser.add_argument(
        "--image_condition_self_attention_guidance_img2_weight",
        type=float,
        default=0.25,
        help=("B attention-output residual weight for optional post-stop "
              "Img2 suffix guidance (default: 0.25)."))
    parser.add_argument(
        "--image_condition_self_attention_guidance_post_stop_img2_suffix",
        action="store_true",
        help=("After the J stop step, run aligned M and Img2-first B streams. "
              "B1: layerwise self-attention outputs guide only M frames after "
              "Img2; the prefix and temporary anchor remain M-only and B0 "
              "does not replace the anchor velocity."))
    parser.add_argument(
        "--image_condition_self_attention_guidance_joint_query_single_memory_alpha",
        type=float,
        default=None,
        help=("Optional M-memory source prior for J prefix and suffix queries. "
              "Each J query group attends separately to full J memory and "
              "retained Img1-only M memory, then combines the groups exactly "
              "from their FlashAttention log-normalizers. The split results "
              "directly update M's aligned prefix and suffix without another "
              "M/J output interpolation. The anchor query remains J-only. "
              "Zero is J-memory-only and one is M-memory-only."))
    parser.add_argument(
        "--image_condition_self_attention_guidance_selective_suffix_memory_alpha",
        type=float,
        default=None,
        help=("Optional M-source prior for aligned suffix memory only. J "
              "prefix queries read unique J prefix, the J anchor and paired "
              "J/M suffix. At the anchor/first-suffix shared RoPE position, "
              "the anchor prior divides anchor versus neighbour mass and "
              "alpha divides the neighbour mass between J/M. J suffix "
              "queries read J prefix, the J anchor and paired J/M suffix. "
              "The exact LSE-merged "
              "outputs directly update aligned M/J positions. Zero uses "
              "J suffix memory only and one uses M suffix memory only."))
    parser.add_argument(
        "--image_condition_self_attention_guidance_selective_suffix_memory_alpha_end",
        type=float,
        default=None,
        help=("Optional final M-suffix prior for a cosine alpha schedule. "
              "It remains fixed after the end step. Omit this option to keep "
              "the selective suffix-memory alpha constant."))
    parser.add_argument(
        "--image_condition_self_attention_guidance_selective_suffix_memory_alpha_end_step",
        type=int,
        default=None,
        help=("Zero-based sampling step at which selective suffix-memory "
              "alpha reaches its final value. Defaults to the final sampling "
              "step when alpha-end is supplied."))
    parser.add_argument(
        "--image_condition_self_attention_guidance_rms_cap",
        type=float,
        default=None,
        help=("Optional per-frame, per-head cap on the injected J-M attention "
              "residual RMS, expressed as a fraction of M RMS."))
    parser.add_argument(
        "--image_condition_self_attention_guidance_anchor_prior",
        action="store_true",
        help=("Use exact three-group LSE merging for J prefix/suffix shared "
              "positions. The anchor prior follows a cosine schedule from "
              "anchor-prior-start to anchor-prior-end, then remains at the "
              "end value."))
    parser.add_argument(
        "--image_condition_self_attention_guidance_anchor_prior_start",
        type=float,
        default=0.8,
        help=("Anchor share of the shared-position prior at the first "
              "denoising step (default: 0.8)."))
    parser.add_argument(
        "--image_condition_self_attention_guidance_anchor_prior_end",
        type=float,
        default=0.2,
        help=("Final anchor share reached at anchor-prior-end-step "
              "(default: 0.2)."))
    parser.add_argument(
        "--image_condition_self_attention_guidance_anchor_prior_end_step",
        type=int,
        default=None,
        help=("Zero-based sampling step at which the cosine anchor-prior "
              "schedule reaches its end value. It remains there afterward. "
              "By default, it reaches the end value on the final step."))
    parser.add_argument(
        "--image_condition_self_attention_guidance_stop_step",
        type=int,
        default=None,
        help=("Optional zero-based sampling step at which J and B stop. "
              "Afterward both CFG calls use Img1-only M, with no guidance "
              "or anchor velocity replacement."))
    parser.add_argument(
        "--image_condition_paired_prefix_future_eta",
        type=float, default=0.0,
        help=("Legacy compatibility option; ignored by paired M/B memory "
              "attention."))
    parser.add_argument(
        "--image_condition_paired_stream_weight",
        type=float,
        default=0.25,
        help=("Legacy fallback for paired-memory alpha when the explicit "
              "paired-memory option is omitted (default: 0.25)."))
    parser.add_argument(
        "--image_condition_paired_memory_alpha",
        type=float,
        default=None,
        help=("B-source prior alpha for paired M-query attention over aligned "
              "M/B suffix K/V. Complementary log priors prevent doubled "
              "suffix mass. Defaults to --image_condition_paired_stream_weight."))
    parser.add_argument(
        "--image_condition_paired_fade_start_step",
        type=int,
        default=None,
        help=("Optional sampling-step index at which a cosine fade of the "
              "paired B-memory alpha begins. Must be "
              "supplied with "
              "--image_condition_paired_fade_end_step."))
    parser.add_argument(
        "--image_condition_paired_fade_end_step",
        type=int,
        default=None,
        help=("Sampling-step index at which the paired B-memory alpha reaches "
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
        help="Legacy compatibility option; currently ignored in paired mode.")
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
        img = (opened_images if (
            args.image_condition_one_way_prompt_guidance
            or args.image_condition_joint_img2_guidance
            or args.image_condition_cross_tail_bridge
            or args.image_condition_paired_anchor_attention
            or args.image_condition_self_attention_guidance)
               else opened_images[0])
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
            image_prompts=args.image_prompt,
            image_condition_one_way_prompt_guidance=(
                args.image_condition_one_way_prompt_guidance),
            image_condition_one_way_prompt_guidance_weight=(
                args.image_condition_one_way_prompt_guidance_weight),
            image_condition_one_way_prompt_guidance_prefix_m10_only=(
                args.image_condition_one_way_prompt_guidance_prefix_m10_only),
            image_condition_one_way_prompt_guidance_aux_anchor_prior=(
                args.image_condition_one_way_prompt_guidance_aux_anchor_prior),
            image_condition_one_way_prompt_guidance_full_img2_stream=(
                args.image_condition_one_way_prompt_guidance_full_img2_stream),
            image_condition_one_way_prompt_guidance_guided_frames=(
                args.image_condition_one_way_prompt_guidance_guided_frames),
            image_condition_one_way_prompt_guidance_shared_anchor_update=(
                args.image_condition_one_way_prompt_guidance_shared_anchor_update),
            image_condition_joint_img2_guidance=(
                args.image_condition_joint_img2_guidance),
            image_condition_joint_img2_guidance_weight=(
                args.image_condition_joint_img2_guidance_weight),
            image_condition_joint_img2_guidance_shifted_anchor=(
                args.image_condition_joint_img2_guidance_shifted_anchor),
            image_condition_joint_img2_guidance_rope_compression=(
                args.image_condition_joint_img2_guidance_rope_compression),
            image_condition_joint_img2_guidance_static_b_prompt=(
                args.image_condition_joint_img2_guidance_static_b_prompt),
            image_condition_joint_img2_guidance_persistent_b=(
                args.image_condition_joint_img2_guidance_persistent_b),
            return_persistent_b_video=(
                args.image_condition_joint_img2_guidance_save_b_video),
            image_condition_cross_tail_bridge=(
                args.image_condition_cross_tail_bridge),
            image_condition_cross_tail_bridge_tail_frames=(
                args.image_condition_cross_tail_bridge_tail_frames),
            image_condition_cross_tail_bridge_memory_prior=(
                args.image_condition_cross_tail_bridge_memory_prior),
            image_condition_cross_tail_bridge_uncontrolled_mass=(
                args.image_condition_cross_tail_bridge_uncontrolled_mass),
            image_condition_paired_anchor_attention=(
                args.image_condition_paired_anchor_attention),
            image_condition_paired_prefix_future_eta=(
                args.image_condition_paired_prefix_future_eta),
            image_condition_paired_stream_weight=(
                args.image_condition_paired_stream_weight),
            image_condition_paired_memory_alpha=(
                args.image_condition_paired_memory_alpha),
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
            image_condition_self_attention_guidance=(
                args.image_condition_self_attention_guidance),
            image_condition_self_attention_guidance_scale=(
                args.image_condition_self_attention_guidance_scale),
            image_condition_self_attention_guidance_scale_end=(
                args.image_condition_self_attention_guidance_scale_end),
            image_condition_self_attention_guidance_scale_end_step=(
                args.image_condition_self_attention_guidance_scale_end_step),
            image_condition_self_attention_guidance_img2_weight=(
                args.image_condition_self_attention_guidance_img2_weight),
            image_condition_self_attention_guidance_post_stop_img2_suffix=(
                args.image_condition_self_attention_guidance_post_stop_img2_suffix),
            image_condition_self_attention_guidance_joint_query_single_memory_alpha=(
                args.image_condition_self_attention_guidance_joint_query_single_memory_alpha),
            image_condition_self_attention_guidance_selective_suffix_memory_alpha=(
                args.image_condition_self_attention_guidance_selective_suffix_memory_alpha),
            image_condition_self_attention_guidance_selective_suffix_memory_alpha_end=(
                args.image_condition_self_attention_guidance_selective_suffix_memory_alpha_end),
            image_condition_self_attention_guidance_selective_suffix_memory_alpha_end_step=(
                args.image_condition_self_attention_guidance_selective_suffix_memory_alpha_end_step),
            image_condition_self_attention_guidance_rms_cap=(
                args.image_condition_self_attention_guidance_rms_cap),
            image_condition_self_attention_guidance_anchor_prior=(
                args.image_condition_self_attention_guidance_anchor_prior),
            image_condition_self_attention_guidance_anchor_prior_start=(
                args.image_condition_self_attention_guidance_anchor_prior_start),
            image_condition_self_attention_guidance_anchor_prior_end=(
                args.image_condition_self_attention_guidance_anchor_prior_end),
            image_condition_self_attention_guidance_anchor_prior_end_step=(
                args.image_condition_self_attention_guidance_anchor_prior_end_step),
            image_condition_self_attention_guidance_stop_step=(
                args.image_condition_self_attention_guidance_stop_step))

    if rank == 0:
        b_video = None
        if args.image_condition_joint_img2_guidance_save_b_video:
            video, b_video = video
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
        if b_video is not None:
            save_root, save_ext = os.path.splitext(args.save_file)
            b_save_file = save_root + "_B" + (save_ext or ".mp4")
            logging.info("Saving persistent B video to %s", b_save_file)
            save_video(
                tensor=b_video[None],
                save_file=b_save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
        if "s2v" in args.task:
            if args.enable_tts is False:
                merge_video_audio(video_path=args.save_file, audio_path=args.audio)
            else:
                merge_video_audio(video_path=args.save_file, audio_path="tts.wav")
    del video
    if rank == 0 and b_video is not None:
        del b_video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
