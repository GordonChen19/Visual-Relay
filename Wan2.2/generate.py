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
        assert 0.0 <= args.image_condition_joint_img2_guidance_weight <= 1.0, (
            "--image_condition_joint_img2_guidance_weight must be in "
            "[0, 1].")
        if args.image_condition_joint_img2_guidance_shifted_anchor:
            assert args.image_condition_joint_img2_guidance, (
                "--image_condition_joint_img2_guidance_shifted_anchor "
                "requires --image_condition_joint_img2_guidance.")
        if args.image_condition_joint_img2_guidance_align_b1:
            assert args.image_condition_joint_img2_guidance_shifted_anchor, (
                "--image_condition_joint_img2_guidance_align_b1 requires "
                "--image_condition_joint_img2_guidance_shifted_anchor.")
        if args.image_condition_joint_img2_guidance_single_frame_anchor:
            assert args.image_condition_joint_img2_guidance, (
                "--image_condition_joint_img2_guidance_single_frame_anchor "
                "requires --image_condition_joint_img2_guidance.")
        if args.image_condition_joint_img2_guidance_single_frame_anchor_four_history:
            assert args.image_condition_joint_img2_guidance_single_frame_anchor, (
                "--image_condition_joint_img2_guidance_single_frame_anchor_four_history "
                "requires --image_condition_joint_img2_guidance_single_frame_anchor.")
        if args.image_condition_joint_img2_guidance_rope_compression:
            assert args.image_condition_joint_img2_guidance_shifted_anchor, (
                "--image_condition_joint_img2_guidance_rope_compression "
                "requires --image_condition_joint_img2_guidance_shifted_anchor.")
        if args.image_condition_joint_img2_guidance_no_rope_compression:
            assert args.image_condition_joint_img2_guidance_shifted_anchor, (
                "--image_condition_joint_img2_guidance_no_rope_compression "
                "requires --image_condition_joint_img2_guidance_shifted_anchor.")
        if args.image_condition_joint_img2_guidance_anchor_rope_offset is not None:
            assert args.image_condition_joint_img2_guidance_shifted_anchor, (
                "--image_condition_joint_img2_guidance_anchor_rope_offset "
                "requires --image_condition_joint_img2_guidance_shifted_anchor.")
            assert (0.0 <= args.image_condition_joint_img2_guidance_anchor_rope_offset
                    <= 1.0), "Anchor RoPE offset must be in [0, 1]."
        if args.image_condition_joint_img2_guidance_backward_pair_rope_compression:
            assert args.image_condition_joint_img2_guidance_shifted_anchor, (
                "--image_condition_joint_img2_guidance_backward_pair_rope_compression "
                "requires --image_condition_joint_img2_guidance_shifted_anchor.")
            assert not args.image_condition_joint_img2_guidance_rope_compression, (
                "Backward-pair compression is a separate RoPE ablation.")
            assert not args.image_condition_joint_img2_guidance_no_rope_compression, (
                "Backward-pair compression is a separate RoPE ablation.")
            assert args.image_condition_joint_img2_guidance_anchor_rope_offset is None, (
                "Backward-pair compression replaces the fractional anchor offset.")
        if args.image_condition_joint_img2_guidance_static_b_prompt:
            assert args.image_condition_joint_img2_guidance, (
                "--image_condition_joint_img2_guidance_static_b_prompt "
                "requires --image_condition_joint_img2_guidance.")
        if args.image_condition_joint_img2_guidance_suffix_anchor_transfer:
            assert args.image_condition_joint_img2_guidance_shifted_anchor, (
                "Suffix anchor transfer requires shifted-anchor guidance.")
            assert (args.image_condition_joint_img2_guidance_rope_compression
                    or args.image_condition_joint_img2_guidance_no_rope_compression
                    or args.image_condition_joint_img2_guidance_anchor_rope_offset is not None
                    or args.image_condition_joint_img2_guidance_backward_pair_rope_compression), (
                "Suffix anchor transfer requires a joint RoPE routing option.")
            ratio = args.image_condition_joint_img2_guidance_suffix_anchor_transfer_ratio
            cap = args.image_condition_joint_img2_guidance_suffix_anchor_transfer_max_delta
            assert 0 <= ratio < float('inf'), "Transfer ratio must be finite and nonnegative."
            assert 0 <= cap <= 1, "Transfer cap must be in [0, 1]."
        if args.image_condition_joint_img2_guidance_persistent_b:
            assert args.image_condition_joint_img2_guidance_shifted_anchor, (
                "--image_condition_joint_img2_guidance_persistent_b "
                "requires --image_condition_joint_img2_guidance_shifted_anchor.")
        if args.image_condition_joint_img2_guidance_velocity_blend_weights is not None:
            assert args.image_condition_joint_img2_guidance, (
                "Velocity blend weights require joint Img2 guidance.")
            assert all(0.0 <= weight <= 1.0 for weight in
                       args.image_condition_joint_img2_guidance_velocity_blend_weights), (
                "Velocity blend weights must be in [0, 1].")
        if args.image_condition_joint_img2_guidance_save_b_video:
            assert args.image_condition_joint_img2_guidance_persistent_b, (
                "--image_condition_joint_img2_guidance_save_b_video "
                "requires --image_condition_joint_img2_guidance_persistent_b.")
        if args.image_condition_joint_img2_guidance_attention_diagnostics:
            assert args.image_condition_joint_img2_guidance_shifted_anchor, (
                "Joint attention diagnostics require shifted-anchor guidance.")
        if args.image_condition_joint_img2_guidance_adaptive_pair_attention:
            assert (args.image_condition_joint_img2_guidance_rope_compression
                    or args.image_condition_joint_img2_guidance_no_rope_compression
                    or args.image_condition_joint_img2_guidance_anchor_rope_offset
                    is not None
                    or args.image_condition_joint_img2_guidance_backward_pair_rope_compression), (
                "Adaptive pair attention requires a joint Img2-guidance RoPE "
                "mode: rope_compression, no_rope_compression, or anchor_rope_offset.")
            assert (args.image_condition_joint_img2_guidance_adaptive_pair_end_step
                    >= 1), "Adaptive pair end step must be at least 1."
            assert (args.image_condition_joint_img2_guidance_adaptive_pair_margin
                    >= 1.0), "Adaptive pair margin must be at least 1."
            assert (0.5 <=
                    args.image_condition_joint_img2_guidance_adaptive_pair_alpha_max
                    < 1.0), "Adaptive pair alpha max must be in [0.5, 1)."
        if args.image_condition_joint_img2_guidance_handoff_img1_only:
            assert args.image_condition_joint_img2_guidance_adaptive_pair_attention, (
                "Img1-only handoff requires "
                "--image_condition_joint_img2_guidance_adaptive_pair_attention.")
        if args.image_condition_joint_img2_guidance:
            assert args.image_prompt is not None and len(args.image_prompt) == 2, (
                "Per-stream prompt guidance requires exactly two "
                "--image_prompt arguments.")
            assert all(value for value in args.image_prompt), (
                "Each --image_prompt must be non-empty.")
            assert not args.use_prompt_extend, (
                "Prompt extension is not supported with per-image prompts.")
        else:
            assert args.image_prompt is None, (
                "--image_prompt is only used by joint Img2 guidance.")
        assert args.image is not None, "Please specify the image path for i2v."
        if args.image_condition_joint_img2_guidance:
            assert len(args.image) == 2, (
                "Joint Img2 guidance requires exactly two images.")
            assert (args.image_latent_index is not None
                    and len(args.image_latent_index) == 2
                    and args.image_latent_index[0] == 0
                    and args.image_latent_index[1] > 0), (
                        "Joint Img2 guidance requires image latent indices "
                        "[0, positive].")
        else:
            assert len(args.image) == 1, (
                "Multiple images require --image_condition_joint_img2_guidance.")
            assert (args.image_latent_index is None
                    or args.image_latent_index == [0]), (
                        "Standard I2V only supports image latent index 0.")
    else:
        assert not args.image_condition_joint_img2_guidance, (
            "Joint Img2 guidance is supported only by i2v-A14B.")
        assert args.image_latent_index is None, (
            "--image_latent_index is supported only by joint I2V.")
        assert args.image_prompt is None, (
            "--image_prompt is supported only by joint I2V.")
        assert args.image is None or len(args.image) == 1, (
            "Only joint I2V accepts multiple --image arguments.")

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.task == "i2v-A14B":
        if args.image_condition_joint_img2_guidance_adaptive_pair_attention:
            assert (
                args.image_condition_joint_img2_guidance_adaptive_pair_end_step
                < args.sample_steps), (
                    "Adaptive pair end step must be in [1, sample_steps).")

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    if args.image_condition_joint_img2_guidance:
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
        help=("An input image. Repeat exactly twice when using joint Img2 "
              "guidance."))
    parser.add_argument(
        "--image_latent_index",
        type=int,
        action="append",
        default=None,
        help=("Latent index corresponding to each --image in joint Img2 "
              "guidance; use 0 for Img1 and a positive index for Img2."))
    parser.add_argument(
        "--image_prompt",
        type=str,
        action="append",
        default=None,
        help=("Prompt paired positionally with each repeated --image. In "
              "joint Img2 guidance these are local prompt 1 and local prompt "
              "2, while --prompt is the global prompt."))
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
        help=("Legacy-named velocity-only guidance mode: B is rebuilt from "
              "J[anchor:] every step, so B0 shares J's anchor noisy state "
              "and supplies its final velocity in both CFG branches. No "
              "layerwise B-output injection. With align_b1, B1 supplies the "
              "velocity instead and only B0 has an independent scheduler."))
    parser.add_argument(
        "--image_condition_joint_img2_guidance_align_b1",
        action="store_true",
        help=("Align B1 with J's anchor instead of B0. B0 has independent "
              "initial noise and its own scheduler; B1 onward are rebuilt "
              "from J[anchor:] each step. B1 replaces only J's anchor velocity "
              "in both CFG branches. Requires shifted-anchor mode."))
    parser.add_argument(
        "--image_condition_joint_img2_guidance_single_frame_anchor",
        action="store_true",
        help=("Experimental contextual single-frame Img2 VAE encoding: "
              "replace Img2x4 with one pixel frame, retain causal convolution "
              "history, and bypass the encoder's temporal downsampling at "
              "the anchor as for a first frame. Keeps the latent count, "
              "attention, B encoding, and velocity replacement unchanged."))
    parser.add_argument(
        "--image_condition_joint_img2_guidance_single_frame_anchor_four_history",
        action="store_true",
        help=("With the single-frame anchor option, run the first temporal "
              "downsampler on [four cached features, Img2] to make two "
              "outputs, then run the second on [one cached feature, two "
              "current features] to make one latent. Experimental."))
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
        "--image_condition_joint_img2_guidance_no_rope_compression",
        action="store_true",
        help=("Use ordinary integer J RoPE before the handoff: J9/J11 have "
              "temporal offset 2. Overrides --image_condition_joint_img2_guidance_rope_compression "
              "if both are set. Keeps anchor self/future-only attention, "
              "B0 velocity replacement, and the configured anchor-removal "
              "handoff unchanged. Requires shifted-anchor guidance."))
    parser.add_argument(
        "--image_condition_joint_img2_guidance_anchor_rope_offset",
        type=float,
        default=None,
        help=("Place the J anchor key at anchor_index-1+OFFSET for prefix "
              "and suffix queries, with OFFSET in [0, 1]. For example, "
              "OFFSET=0.8 puts J10 at 9.8 while J9=9 and J11=10. "
              "J10's own query reads itself/future at ordinary uncompressed "
              "offsets. Overrides either other joint RoPE option. Requires "
              "shifted-anchor guidance; handoff and velocity stay unchanged."))
    parser.add_argument(
        "--image_condition_joint_img2_guidance_backward_pair_rope_compression",
        action="store_true",
        help=("Query-dependent backward-pair RoPE ablation. Prefix queries "
              "see the frame before the anchor and the anchor at one shared "
              "coordinate. The anchor reads all J and sees the two preceding "
              "frames at one shared coordinate. Suffix queries use ordinary "
              "RoPE. Values and softmax mass are unchanged."))
    parser.add_argument(
        "--image_condition_joint_img2_guidance_suffix_anchor_transfer",
        action="store_true",
        help=("Only for queries in the first frame after Img2's anchor, "
              "transfer bounded attention probability from the entire prefix "
              "to the anchor. Keeps all suffix probabilities unchanged. "
              "Applied per query/head/layer in both CFG branches."))
    parser.add_argument(
        "--image_condition_joint_img2_guidance_suffix_anchor_transfer_ratio",
        type=float, default=1.0,
        help="Target anchor-to-total-prefix attention ratio (default: 1).")
    parser.add_argument(
        "--image_condition_joint_img2_guidance_suffix_anchor_transfer_max_delta",
        type=float, default=0.05,
        help="Maximum transferred probability per query/head (0.05 = 5 percentage points).")
    parser.add_argument(
        "--image_condition_joint_img2_guidance_static_b_prompt",
        action="store_true",
        help=("Deprecated compatibility flag. B always uses the global "
              "--prompt plus local image prompt 2; no static-motion sentence "
              "is appended."))
    parser.add_argument(
        "--image_condition_joint_img2_guidance_velocity_blend_weights",
        type=float,
        nargs="+",
        default=None,
        help=("Aligned B-to-J velocity weights beginning at the J anchor. "
              "For example, '1.0 0.6 0.25 0.1' replaces the anchor with B "
              "and blends the following three frames. Each value must be in "
              "[0, 1]. Omission preserves anchor-only replacement."))
    parser.add_argument(
        "--image_condition_joint_img2_guidance_persistent_b",
        action="store_true",
        help=("Deprecated compatibility flag. B0 and all later B frames now "
              "read J[anchor:] at every step; there is no independently "
              "scheduled B state."))
    parser.add_argument(
        "--image_condition_joint_img2_guidance_save_b_video",
        action="store_true",
        help=("Decode the last guided B view (J's canonical anchor and suffix) "
              "beside the main video using an _B filename suffix. If guidance "
              "ends early, this view is an intermediate noisy state."))
    parser.add_argument(
        "--image_condition_joint_img2_guidance_attention_diagnostics",
        action="store_true",
        help=("Print conditional attention percentages at every sampling step "
              "from anchor-1 to anchor/anchor+1 and from anchor+1 to "
              "anchor/anchor-1, plus adaptive prefix/suffix mean alpha, cap "
              "rate, and corrected anchor-win rate. Adds diagnostic "
              "computation without changing attention outputs."))
    parser.add_argument(
        "--image_condition_joint_img2_guidance_adaptive_pair_attention",
        action="store_true",
        help=("During the initial configured sampling steps, compress RoPE "
              "in both directions: prefix queries read all J frames with "
              "the anchor and first suffix frame colocated as temporal "
              "neighbours; suffix queries read all J frames with the anchor "
              "colocated with the last prefix frame. No "
              "attention-mass correction or gradual temporal shift is "
              "applied. At the end step, hand off to ordinary single-stream "
              "attention with the temporary anchor and B stream removed."))
    parser.add_argument(
        "--image_condition_joint_img2_guidance_adaptive_pair_end_step",
        type=int,
        default=10,
        help=("First zero-based switch step. It removes the temporary anchor "
              "and B stream and uses ordinary single-stream attention "
              "(default: 10)."))
    parser.add_argument(
        "--image_condition_joint_img2_guidance_handoff_img1_only",
        action="store_true",
        help=("At the adaptive-pair end step, remove the temporary anchor and "
              "continue as ordinary single-stream I2V conditioned only on "
              "Img1."))
    parser.add_argument(
        "--image_condition_joint_img2_guidance_adaptive_pair_margin",
        type=float,
        default=1.05,
        help=("Deprecated compatibility argument; ignored by the current "
              "prefix-only compression ablation."))
    parser.add_argument(
        "--image_condition_joint_img2_guidance_adaptive_pair_alpha_max",
        type=float,
        default=0.7,
        help=("Deprecated compatibility argument; ignored by the current "
              "prefix-only compression ablation."))
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
        img = (opened_images if args.image_condition_joint_img2_guidance
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
            image_condition_joint_img2_guidance=(
                args.image_condition_joint_img2_guidance),
            image_condition_joint_img2_guidance_weight=(
                args.image_condition_joint_img2_guidance_weight),
            image_condition_joint_img2_guidance_shifted_anchor=(
                args.image_condition_joint_img2_guidance_shifted_anchor),
            image_condition_joint_img2_guidance_align_b1=(
                args.image_condition_joint_img2_guidance_align_b1),
            image_condition_joint_img2_guidance_velocity_blend_weights=(
                args.image_condition_joint_img2_guidance_velocity_blend_weights),
            image_condition_joint_img2_guidance_single_frame_anchor=(
                args.image_condition_joint_img2_guidance_single_frame_anchor),
            image_condition_joint_img2_guidance_single_frame_anchor_four_history=(
                args.image_condition_joint_img2_guidance_single_frame_anchor_four_history),
            image_condition_joint_img2_guidance_rope_compression=(
                args.image_condition_joint_img2_guidance_rope_compression),
            image_condition_joint_img2_guidance_no_rope_compression=(
                args.image_condition_joint_img2_guidance_no_rope_compression),
            image_condition_joint_img2_guidance_anchor_rope_offset=(
                args.image_condition_joint_img2_guidance_anchor_rope_offset),
            image_condition_joint_img2_guidance_backward_pair_rope_compression=(
                args.image_condition_joint_img2_guidance_backward_pair_rope_compression),
            image_condition_joint_img2_guidance_static_b_prompt=(
                args.image_condition_joint_img2_guidance_static_b_prompt),
            image_condition_joint_img2_guidance_suffix_anchor_transfer=(
                args.image_condition_joint_img2_guidance_suffix_anchor_transfer),
            image_condition_joint_img2_guidance_suffix_anchor_transfer_ratio=(
                args.image_condition_joint_img2_guidance_suffix_anchor_transfer_ratio),
            image_condition_joint_img2_guidance_suffix_anchor_transfer_max_delta=(
                args.image_condition_joint_img2_guidance_suffix_anchor_transfer_max_delta),
            image_condition_joint_img2_guidance_persistent_b=(
                args.image_condition_joint_img2_guidance_persistent_b),
            return_persistent_b_video=(
                args.image_condition_joint_img2_guidance_save_b_video),
            image_condition_joint_img2_guidance_attention_diagnostics=(
                args.image_condition_joint_img2_guidance_attention_diagnostics),
            image_condition_joint_img2_guidance_adaptive_pair_attention=(
                args.image_condition_joint_img2_guidance_adaptive_pair_attention),
            image_condition_joint_img2_guidance_adaptive_pair_end_step=(
                args.image_condition_joint_img2_guidance_adaptive_pair_end_step),
            image_condition_joint_img2_guidance_handoff_img1_only=(
                args.image_condition_joint_img2_guidance_handoff_img1_only),
            image_condition_joint_img2_guidance_adaptive_pair_margin=(
                args.image_condition_joint_img2_guidance_adaptive_pair_margin),
            image_condition_joint_img2_guidance_adaptive_pair_alpha_max=(
                args.image_condition_joint_img2_guidance_adaptive_pair_alpha_max),
        )

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
