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


def _load_refinement_video(path, frame_num):
    """Load ``frame_num`` RGB frames as a normalized [C, T, H, W] tensor."""
    try:
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise ImportError(
            "--init_video requires decord, which is included in the Wan "
            "runtime dependencies") from exc

    reader = VideoReader(path, ctx=cpu(0))
    total_frames = len(reader)
    if total_frames < frame_num:
        raise ValueError(
            f"--init_video has {total_frames} frames, but --frame_num "
            f"requires at least {frame_num}")
    if frame_num == 1:
        indices = [0]
    else:
        indices = [
            round(index * (total_frames - 1) / (frame_num - 1))
            for index in range(frame_num)
        ]
    frames = reader.get_batch(indices).asnumpy()
    video = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()
    video = video.float().div_(127.5).sub_(1.0)
    return video, Image.fromarray(frames[0]).convert("RGB"), indices


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if (args.image is None and args.init_video is None
            and "image" in EXAMPLE_PROMPT[args.task]):
        args.image = [EXAMPLE_PROMPT[args.task]["image"]]
    elif isinstance(args.image, str):
        args.image = [args.image]
    if args.audio is None and args.enable_tts is False and "audio" in EXAMPLE_PROMPT[args.task]:
        args.audio = EXAMPLE_PROMPT[args.task]["audio"]
    if (args.tts_prompt_audio is None or args.tts_text is None) and args.enable_tts is True and "audio" in EXAMPLE_PROMPT[args.task]:
        args.tts_prompt_audio = EXAMPLE_PROMPT[args.task]["tts_prompt_audio"]
        args.tts_prompt_text = EXAMPLE_PROMPT[args.task]["tts_prompt_text"]
        args.tts_text = EXAMPLE_PROMPT[args.task]["tts_text"]

    if args.task == "i2v-A14B":
        assert args.image is not None or args.init_video is not None, (
            "Please specify --image or --init_video for i2v.")
        image_count = len(args.image) if args.image is not None else 1
        if args.image_latent_index is None:
            assert image_count == 1, (
                "Please provide one --image_latent_index for each --image.")
            args.image_latent_index = [0]
        assert image_count == len(args.image_latent_index), (
            "The number of --image and --image_latent_index arguments must "
            "match.")
        if args.init_video is not None:
            assert args.init_video_noise_step is not None, (
                "--init_video requires --init_video_noise_step.")
            assert image_count == 1 and args.image_latent_index == [0], (
                "Video refinement supports exactly one first-frame image "
                "condition at latent index 0.")
            assert args.image_condition_switch_step is None, (
                "Video refinement cannot be combined with "
                "--image_condition_switch_step.")
            assert args.sacrificial_image_latent_index is None, (
                "Video refinement cannot use a sacrificial image anchor.")
        else:
            assert args.init_video_noise_step is None, (
                "--init_video_noise_step requires --init_video.")
        if args.image_condition_switch_step is not None:
            assert len(args.image) >= 2, (
                "--image_condition_switch_step requires multiple images.")
            assert 0 in args.image_latent_index, (
                "--image_condition_switch_step requires an image at latent "
                "frame 0 for the default-I2V stage.")
        if args.sacrificial_image_latent_index is not None:
            assert args.image_condition_switch_step is not None, (
                "--sacrificial_image_latent_index requires "
                "--image_condition_switch_step.")
            assert args.sacrificial_image_latent_index > 0, (
                "--sacrificial_image_latent_index must be positive.")
            assert (args.sacrificial_image_latent_index
                    in args.image_latent_index), (
                        "--sacrificial_image_latent_index must match a "
                        "nonzero --image_latent_index.")
    else:
        assert args.image is None or len(args.image) == 1, (
            "Multiple images are currently supported only by i2v-A14B.")
        assert args.image_latent_index is None, (
            "--image_latent_index is supported only by i2v-A14B.")
        assert args.image_condition_switch_step is None, (
            "--image_condition_switch_step is supported only by i2v-A14B.")
        assert args.sacrificial_image_latent_index is None, (
            "--sacrificial_image_latent_index is supported only by i2v-A14B.")
        assert args.init_video is None, (
            "--init_video is supported only by i2v-A14B.")
        assert args.init_video_noise_step is None, (
            "--init_video_noise_step is supported only by i2v-A14B.")

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    if args.image_condition_switch_step is not None:
        assert 0 <= args.image_condition_switch_step < args.sample_steps, (
            "--image_condition_switch_step must be in "
            f"[0, {args.sample_steps}).")
    if args.image_condition_rope_blend_start_step is not None:
        assert args.image_condition_switch_step is not None, (
            "--image_condition_rope_blend_start_step requires "
            "--image_condition_switch_step.")
        assert (0 <= args.image_condition_rope_blend_start_step <
                args.image_condition_switch_step), (
                    "--image_condition_rope_blend_start_step must be in "
                    "[0, image_condition_switch_step).")
    assert 0.0 <= args.image_condition_rope_blend_weight <= 1.0, (
        "--image_condition_rope_blend_weight must be in [0, 1].")
    if args.image_condition_dual_guidance:
        assert args.task == "i2v-A14B", (
            "--image_condition_dual_guidance is supported only by i2v-A14B.")
        assert args.image is not None and len(args.image) == 2, (
            "--image_condition_dual_guidance currently requires exactly two "
            "images.")
        assert args.image_latent_index is not None, (
            "--image_condition_dual_guidance requires explicit image latent "
            "indices.")
        assert args.image_condition_prefix_multi_guidance_scale >= 0, (
            "--image_condition_prefix_multi_guidance_scale must be "
            "non-negative.")
        assert args.image_condition_suffix_multi_guidance_scale >= 0, (
            "--image_condition_suffix_multi_guidance_scale must be "
            "non-negative.")
    if args.image_condition_multi_stop_step is not None:
        assert args.image_condition_dual_guidance, (
            "--image_condition_multi_stop_step requires "
            "--image_condition_dual_guidance.")
        assert 0 <= args.image_condition_multi_stop_step < args.sample_steps, (
            "--image_condition_multi_stop_step must be in "
            f"[0, {args.sample_steps}).")
    assert 0.0 <= args.image_condition_late_suffix_img2_weight <= 1.0, (
        "--image_condition_late_suffix_img2_weight must be in [0, 1].")
    assert args.image_condition_anchor_block_radius >= 0, (
        "--image_condition_anchor_block_radius must be non-negative.")
    assert (0 <= args.image_condition_anchor_block_start_layer <=
            cfg.num_layers), (
                "--image_condition_anchor_block_start_layer must be in "
                f"[0, {cfg.num_layers}].")
    if args.image_condition_anchor_block_start_step is not None:
        assert (0 <= args.image_condition_anchor_block_start_step <
                args.sample_steps), (
                    "--image_condition_anchor_block_start_step must be in "
                    f"[0, {args.sample_steps}).")
    if args.init_video_noise_step is not None:
        assert 0 <= args.init_video_noise_step < args.sample_steps, (
            "--init_video_noise_step must be in "
            f"[0, {args.sample_steps}).")
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    probe_option_used = any([
        args.attention_probe_query_frame is not None,
        args.attention_probe_layer is not None,
        args.attention_probe_step is not None,
        args.attention_probe_query_row is not None,
        args.attention_probe_query_col is not None,
        args.attention_probe_output is not None,
    ])
    if probe_option_used:
        assert args.task == "i2v-A14B", (
            "The attention probe currently supports only i2v-A14B.")
        assert args.ulysses_size == 1, (
            "The attention probe currently supports single-GPU inference only."
        )
        assert args.image_condition_switch_step is None, (
            "The attention probe is not supported while temporary image "
            "anchors are active.")
        assert args.attention_probe_query_frame is not None, (
            "--attention_probe_query_frame is required when using probe options."
        )
        latent_frame_num = ((args.frame_num - 1) // cfg.vae_stride[0] + 1)
        assert 0 <= args.attention_probe_query_frame < latent_frame_num, (
            "--attention_probe_query_frame must refer to a generated latent "
            f"frame in [0, {latent_frame_num}).")
        assert ((args.attention_probe_query_row is None)
                == (args.attention_probe_query_col is None)), (
                    "Specify both --attention_probe_query_row and "
                    "--attention_probe_query_col, or neither for the center "
                    "patch.")
        if args.attention_probe_query_row is not None:
            assert args.attention_probe_query_row >= 0
            assert args.attention_probe_query_col >= 0

        if args.attention_probe_layer is None:
            args.attention_probe_layer = sorted(
                {0, cfg.num_layers // 2, cfg.num_layers - 1})
        else:
            args.attention_probe_layer = sorted(
                set(args.attention_probe_layer))
        assert all(0 <= layer < cfg.num_layers
                   for layer in args.attention_probe_layer), (
                       "--attention_probe_layer must be in "
                       f"[0, {cfg.num_layers}).")

        if args.attention_probe_step is None:
            args.attention_probe_step = sorted(
                {0, args.sample_steps // 2, args.sample_steps - 1})
        else:
            args.attention_probe_step = sorted(set(args.attention_probe_step))
        assert all(0 <= step < args.sample_steps
                   for step in args.attention_probe_step), (
                       "--attention_probe_step must be in "
                       f"[0, {args.sample_steps}).")

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
        help=("An image to condition on. Repeat for multiple i2v images; "
              "single-image usage remains unchanged."))
    parser.add_argument(
        "--image_latent_index",
        type=int,
        action="append",
        default=None,
        help=("Latent-frame index for the corresponding --image. Repeat in "
              "the same order as --image."))
    parser.add_argument(
        "--image_condition_switch_step",
        type=int,
        default=None,
        help=("Zero-based denoising step at which isolated local image "
              "branches are disabled. The final-length first-image global "
              "branch continues while the scheduler retains disposable "
              "placeholder anchors until VAE decoding. If omitted, expanded "
              "multi-image conditioning remains active for every step."))
    parser.add_argument(
        "--image_condition_rope_blend_weight",
        type=float,
        default=0.25,
        help=("Expanded multi-image compressed-RoPE prediction weight before "
              "the conditioning switch. 0 uses only compact Img1 regular "
              "I2V, 1 uses only the expanded compressed view, and values "
              "between them blend the two predictions. Defaults to 0.25."))
    parser.add_argument(
        "--image_condition_rope_blend_start_step",
        type=int,
        default=None,
        help=("Optional zero-based step at which a cosine transition begins "
              "from fully compressed multi-image prediction toward compact "
              "Img1-only regular I2V. The transition reaches fully regular "
              "at --image_condition_switch_step. When set, it overrides the "
              "fixed --image_condition_rope_blend_weight before the switch."))
    parser.add_argument(
        "--image_condition_dual_guidance",
        action="store_true",
        help=("Before the conditioning switch, route prefix predictions from "
              "an Img1-only regular-I2V base and suffix predictions from an "
              "Img2-only local-I2V base, guiding both toward the expanded "
              "multi-image prediction. Currently supports exactly two "
              "images."))
    parser.add_argument(
        "--image_condition_prefix_multi_guidance_scale",
        type=float,
        default=0.6,
        help=("Multi-image delta scale for retained frames before Img2 in "
              "dual-guidance mode. 0 uses Img1-only, 1 uses multi-image, and "
              "values above 1 extrapolate. Defaults to 0.6."))
    parser.add_argument(
        "--image_condition_suffix_multi_guidance_scale",
        type=float,
        default=0.35,
        help=("Multi-image delta scale for Img2 and its suffix in dual-"
              "guidance mode. 0 uses Img2-local I2V, 1 uses multi-image, and "
              "values above 1 extrapolate. Defaults to 0.35."))
    parser.add_argument(
        "--image_condition_multi_stop_step",
        type=int,
        default=None,
        help=("Optional zero-based denoising step at which dual guidance "
              "stops evaluating the expanded multi-image guide. The prefix "
              "then uses Img1-only I2V, the anchor uses Img2-local I2V, and "
              "the suffix blends the aligned Img1 and Img2-local predictions."))
    parser.add_argument(
        "--image_condition_late_suffix_img2_weight",
        type=float,
        default=0.5,
        help=("Img2-local weight for the suffix after "
              "--image_condition_multi_stop_step; the complementary weight "
              "uses the aligned Img1-only prediction. Defaults to 0.5."))
    parser.add_argument(
        "--image_condition_anchor_block_radius",
        "--image_condition_pre_anchor_block_frames",
        dest="image_condition_anchor_block_radius",
        type=int,
        default=0,
        help=("Symmetric exclusion radius around every non-first image "
              "anchor. The nearest N ordinary query frames on each side "
              "cannot attend to that anchor; ordinary-frame visibility is "
              "unchanged. The old --image_condition_pre_anchor_block_frames "
              "name remains accepted as an alias. Defaults to 0."))
    parser.add_argument(
        "--image_condition_anchor_block_start_layer",
        type=int,
        default=0,
        help=("Zero-based transformer layer at which the symmetric "
              "anchor-block radius becomes active. Earlier layers use "
              "radius 0 at every denoising step. Set this to the number of "
              "layers (40 for A14B) to disable layer-based blocking before "
              "an optional step override. Defaults to 0."))
    parser.add_argument(
        "--image_condition_anchor_block_start_step",
        type=int,
        default=None,
        help=("Optional zero-based denoising step at which the configured "
              "anchor-block radius becomes active in every transformer "
              "layer. Before this step, --image_condition_anchor_block_"
              "start_layer still controls the layer threshold."))
    parser.add_argument(
        "--init_video",
        type=str,
        default=None,
        help=("Existing video to VAE-encode, noise at "
              "--init_video_noise_step, and refine with standard first-frame "
              "I2V conditioning."))
    parser.add_argument(
        "--init_video_noise_step",
        type=int,
        default=None,
        help=("Zero-based index in the configured sampling schedule at which "
              "to noise --init_video. Denoising runs from this step through "
              "the end; lower values apply more noise."))
    parser.add_argument(
        "--sacrificial_image_latent_index",
        type=int,
        default=None,
        help=("Deprecated compatibility option. Every nonzero "
              "--image_latent_index is now treated as temporary and removed "
              "automatically after denoising."))
    parser.add_argument(
        "--attention_probe_query_frame",
        type=int,
        default=None,
        help=("Latent frame containing the query patch to diagnose. The "
              "probe does not change model attention output."))
    parser.add_argument(
        "--attention_probe_layer",
        type=int,
        action="append",
        default=None,
        help=("Transformer layer to probe; repeat for multiple layers. "
              "Defaults to the first, middle, and last layers."))
    parser.add_argument(
        "--attention_probe_step",
        type=int,
        action="append",
        default=None,
        help=("Zero-based denoising step to probe; repeat for multiple steps. "
              "Defaults to the first, middle, and last steps."))
    parser.add_argument(
        "--attention_probe_query_row",
        type=int,
        default=None,
        help="Spatial patch row to probe; defaults to the center row.")
    parser.add_argument(
        "--attention_probe_query_col",
        type=int,
        default=None,
        help="Spatial patch column to probe; defaults to the center column.")
    parser.add_argument(
        "--attention_probe_output",
        type=str,
        default=None,
        help=("Optional JSONL output path. Probe results are always logged to "
              "stdout."))
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

    if args.attention_probe_query_frame is not None:
        assert world_size == 1, (
            "The attention probe currently supports single-GPU inference only."
        )

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
    imgs = None
    img = None
    init_video = None
    init_video_first_frame = None
    if args.init_video is not None:
        init_video, init_video_first_frame, sampled_indices = (
            _load_refinement_video(args.init_video, args.frame_num))
        logging.info(
            "Input refinement video: %s (%d sampled frames; source indices "
            "%d..%d)", args.init_video, init_video.size(1),
            sampled_indices[0], sampled_indices[-1])
    if args.image is not None:
        imgs = [Image.open(path).convert("RGB") for path in args.image]
        img = imgs[0]
        logging.info(f"Input images: {args.image}")
    elif init_video_first_frame is not None:
        imgs = [init_video_first_frame]
        img = imgs[0]
        logging.info(
            "No --image supplied; using the first sampled video frame as "
            "the I2V condition")

    # prompt extend
    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt_output = prompt_expander(
                args.prompt,
                image=img,
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
            ref_image_path=args.image[0],
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
        attention_probe = None
        if args.attention_probe_query_frame is not None:
            attention_probe = {
                'query_frame': args.attention_probe_query_frame,
                'query_row': args.attention_probe_query_row,
                'query_col': args.attention_probe_query_col,
                'layers': args.attention_probe_layer,
                'sampling_steps': args.attention_probe_step,
                'output_path': args.attention_probe_output,
            }
        video = wan_i2v.generate(
            args.prompt,
            imgs,
            size=SIZE_CONFIGS[args.size],
            image_latent_indices=args.image_latent_index,
            image_condition_switch_step=args.image_condition_switch_step,
            image_condition_rope_blend_weight=(
                args.image_condition_rope_blend_weight),
            image_condition_rope_blend_start_step=(
                args.image_condition_rope_blend_start_step),
            image_condition_dual_guidance=(
                args.image_condition_dual_guidance),
            image_condition_prefix_multi_guidance_scale=(
                args.image_condition_prefix_multi_guidance_scale),
            image_condition_suffix_multi_guidance_scale=(
                args.image_condition_suffix_multi_guidance_scale),
            image_condition_multi_stop_step=(
                args.image_condition_multi_stop_step),
            image_condition_late_suffix_img2_weight=(
                args.image_condition_late_suffix_img2_weight),
            image_condition_anchor_block_radius=(
                args.image_condition_anchor_block_radius),
            image_condition_anchor_block_start_layer=(
                args.image_condition_anchor_block_start_layer),
            image_condition_anchor_block_start_step=(
                args.image_condition_anchor_block_start_step),
            sacrificial_image_latent_index=(
                args.sacrificial_image_latent_index),
            init_video=init_video,
            init_video_noise_step=args.init_video_noise_step,
            attention_probe=attention_probe,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

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
        if "s2v" in args.task:
            if args.enable_tts is False:
                merge_video_audio(video_path=args.save_file, audio_path=args.audio)
            else:
                merge_video_audio(video_path=args.save_file, audio_path="tts.wav")
    del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
