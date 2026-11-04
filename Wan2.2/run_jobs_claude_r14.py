"""
Visual Relay, round 14: the final compilation for the supervisor. Every configuration on both scenes, one stream,
raw and with Gordon's fix: 10 configurations x 2 scenes x 2 = 40 runs, 2 GPUs per run, on whichever GPUs are free.
New code: wan/image2video.py, wan/modules/attention.py, generate.py (--vr_override_single), test_r1_attention.py.

    python run_jobs_claude_r14.py --dry_run    # print the plan and every command; launch nothing
    python run_jobs_claude_r14.py              # preflight tests, then background scheduler
    python run_jobs_claude_r14.py --gpus 4,5,6,7   # use only these GPUs
    python run_jobs_claude_r14.py --only S-RL2,S-RL2-G --scenes sam   # a subset

One stream. With w = 1 the main video was stream 2's own video retouched on Img2's 4 frames (F75), and stream 2
never read stream 1, so every run here is stream 2 alone (--vr_override_single): Img1 on latent 0, Img2 on its
latent as each configuration sets it, prompt relay, seed 4827316. Earlier rounds' _s2.mp4 videos are the same runs
(r11 PR-LF and RL2, r12 SR4321, r13 SR421, SR141 and RL2-G): compare them as a check. No probe (it compares
stream 1 with stream 2) and no decode fix: every video is raw. The latents are still saved.

Gordon's fix (-G), as he ran it: (1) the surgical swap, every hint latent but Img2's from the Img1-only VAE pass
(--vr_override splice); (2) clean Img2 values, the latents before Img2 read Img2's latent with its own keys but the
values of Img2 encoded as a first frame (--vr_override_img2_value before). (2) needs Stream B, a short Img2-first
stream over Img2's latent to the end (9 latents SAM, 11 FNR) that only supplies those values; it is not a video.

Configurations, frames of Img2's latent given at each of the 40 steps (SAM 45-48, FNR 37-40; the last entry holds):
  S-M1-PR     all 4 at every step (the stall / hold at Img2)
  S-PR-LF     the last frame only at every step
  S-RL1..4    all 4 for the first K steps, then the last frame only
  S-SR421     4, 2, 1          S-SR4321   4, 3, 2, 1
  S-SR141     1, 4, 1          S-SR12421  1, 2, 4, 2, 1
First wave: RL3 and RL4 with Gordon's fix on both scenes (does a longer boost turn SAM with clean values?).
Outputs: r14_outputs/r14_<code>_<scene>.mp4 and _latents.pt.
"""
import json
import os
import shlex
import socket
import subprocess
import sys
import time

# --- SETTINGS (edit if your paths differ) -------------------------------------------------------
RUN_TAG = "r14"
CKPT_DIR = "/home/ycwang/Visual-Relay/Wan2.2-I2V-A14B"
IMAGE_DIR = "/home/ycwang/Visual-Relay/"
GPUS = None  # None = every GPU nvidia-smi lists (0-7 if it cannot be read); --gpus 0,1,2,3 restricts it
DEFAULT_GPUS = list(range(8))
OUT_DIR = "r14_outputs"
BUSY_MB = 500  # a GPU using more memory than this is treated as busy
POLL_S = float(os.environ.get("RELAY_POLL_S", "30"))
PREFLIGHT_TIMEOUT_S = 1200
# A run that fails this soon after launch failed to start (bad GPU, busy GPU, memory at load), not in the
# method: it goes back in the queue, preferably for another pair; a GPU in 2 such failures is retired.
FAST_FAIL_S = 300
GPU_FAST_FAIL_LIMIT = 2
TASK_FAST_FAIL_LIMIT = 2  # a run that fails fast this many times is given up
OMP_THREADS_PER_RANK = 16  # CPU threads per process; 2 jobs x 2 ranks x 16 = 64

SCENES = {
    "sam": dict(
        images=["Img1.png", "Img2.png"], anchor=12, weight=0.5, fix_window="11-13",
        prompt=("A continuous 180-degree orbit around two stationary warriors in a moonlit Japanese temple "
                "courtyard. The camera starts behind the red-armored samurai facing the black-clad ninja, "
                "then orbits smoothly around them and ends behind the black-clad ninja facing the "
                "red-armored samurai.")),
    "fnr": dict(
        images=["fnr1.png", "fnr2.png"], anchor=10, weight=1.0, fix_window="10-11",
        prompt=("A rock and a white feather fall downward from the same height above the ground. The rock "
                "drops quickly and reaches the ground first, while the feather descends more slowly, gently "
                "drifting and fluttering in the air. The camera remains static and the outdoor background "
                "stays unchanged.")),
}

# Supervisor's command (T5 on the GPU for 40 GB A100s); the seed is added per task.
SEED = 4827316
COMMON = ["--task", "i2v-A14B", "--size", "832*480", "--frame_num", "81", "--ckpt_dir", CKPT_DIR,
          "--sample_shift", "3", "--ulysses_size", "2", "--dit_fsdp",
          "--offload_model", "True", "--convert_model_dtype"]
SUFFIX_PROMPTS = {
    "sam": ("The camera stays behind the black-clad ninja, who faces the red-armored samurai in a moonlit "
            "Japanese temple courtyard. Both warriors remain stationary."),
    "fnr": ("The rock lies still on the ground while the white feather keeps drifting slowly downward, gently "
            "fluttering in the air. The camera remains static and the outdoor background stays unchanged."),
}
# Round 10, CL-DP: a section-2 prompt that describes Img2's two people and where they stand (no "faces the
# red-armored samurai" alone, which fits any red samurai).
DETAILED_PROMPTS = {
    "sam": ("The camera stays behind the black-clad ninja, seen from behind in the right foreground: black hood, "
            "black mask and dark clothing, holding a katana. Further back on the left stands the same samurai, "
            "with long dark hair tied in a topknot and no helmet, in red and black armor, his sword drawn, facing "
            "the ninja. A red torii gate, glowing lanterns, cherry blossoms and a full moon over a wet stone temple "
            "courtyard. Both warriors remain stationary."),
}
# Stream R runs backward in time from Img2 to (almost) Img1, so its prompt describes that stretch reversed.
REVERSE_PROMPTS = {
    "sam": ("A continuous 180-degree orbit around two stationary warriors in a moonlit Japanese temple "
            "courtyard. The camera starts behind the black-clad ninja facing the red-armored samurai, then "
            "orbits smoothly around them and ends behind the red-armored samurai facing the black-clad ninja."),
    "fnr": ("The white feather floats slowly upward through the air, gently fluttering, while the rock rests "
            "on the ground and then quickly rises back up to the feather's starting height. The camera "
            "remains static and the outdoor background stays unchanged."),
}
# PR-LAND: the feather settles on the ground after Img2 (FNR only).
LANDING_PROMPTS = {
    "fnr": ("The rock lies still on the ground while the white feather slowly drifts down, gently fluttering, and "
            "settles on the ground beside the rock. The camera remains static and the outdoor background stays "
            "unchanged."),
}
PROBE = ["--vr_probe_steps", "0,2,5,9,14,20,24,30,39"]


def rev(mode, until, strength=1.0, couple=False):
    return (["--vr_rev_mode", mode, "--vr_rev_until_step", str(until), "--vr_rev_strength", str(strength)]
            + (["--vr_rev_couple"] if couple else []))


def m1(stream="joint", min_weight=1.0, radius=3, blend="linear", latent="shared"):
    return dict(mode="override", args=["--vr_override", stream, "--vr_override_min_weight", str(min_weight),
                                       "--vr_override_radius", str(radius), "--vr_override_blend", blend,
                                       "--vr_override_latent", latent])


SEED2 = 20261024


def r4(*extra, **keys):
    """Round 4: M1-OWN plus one change; every run also saves stream 2's video."""
    ex = m1(stream=keys.pop("override_stream", "joint"), latent="own")
    ex["args"] = ex["args"] + ["--vr_override_save_stream2"] + list(extra)
    ex.update(keys)
    return ex


def r5(*extra, **keys):
    """Round 5: M1-PR (round 4 reference) plus one change."""
    keys.setdefault("relay_prompt", True)
    return r4(*extra, **keys)


r6 = r5  # Round 6: M1-PR plus one change, same as round 5
r7 = r5  # Round 7: M1-PR (SAM) or PR-CL (FNR, override_stream="clean") plus one change
r8 = r5  # Round 8b: CL-LF (FNR) or M1-PR (SAM) with a new Img2 latent
r9 = r5  # Round 9b: the same bases; every override run also saves its final latents
r10 = r5  # Round 10: M1-PR (SAM) or CL-LF plus one change; every override run also writes its decode-fixed video
r12 = r5  # Round 12: M1-PR or PR-LF plus one change
r13 = r5  # Round 13: RL2 (M1-PR released at step 2), PR-LF or a span schedule, plus one change


def r14(*extra, gordon=False, **keys):
    """Round 14, the final compilation: stream 2's input alone (--vr_override_single), prompt relay, no probe, no
    decode fix; gordon=True adds Gordon's fix (splice encoding and clean Img2 values)."""
    ex = m1(stream="splice" if gordon else "joint", latent="own")
    ex["args"] = ex["args"] + ["--vr_override_single"] + list(extra) + (
        ["--vr_override_img2_value", "before"] if gordon else [])
    ex.update(dict(relay_prompt=True, probe=False, decode_fix=False), **keys)
    return ex


FINAL = {  # configuration -> its flags (frames of Img2 given per step)
    "M1-PR": [],
    "PR-LF": ["--vr_override_img2_span", "last"],
    "RL1": ["--vr_img2_release_step", "1"],
    "RL2": ["--vr_img2_release_step", "2"],
    "RL3": ["--vr_img2_release_step", "3"],
    "RL4": ["--vr_img2_release_step", "4"],
    "SR421": ["--vr_img2_span_schedule", "4,2,1"],
    "SR4321": ["--vr_img2_span_schedule", "4,3,2,1"],
    "SR141": ["--vr_img2_span_schedule", "1,4,1"],
    "SR12421": ["--vr_img2_span_schedule", "1,2,4,2,1"],
}

# Round 8b, two-pass runs: the first pass whose frames 0..4a-4 are encoded before Img2 (first file found).
CONTEXT_VIDEOS = {
    "fnr": ["r7_outputs/r7_CL-LF_fnr.mp4"],
    "sam": ["r4_outputs/r4_M1-PR_sam.mp4", "r5_outputs/r5_M1-PR_sam.mp4"],
}


def context_video(scene):
    """The first existing context video for this scene (the first listed if none exists, for --dry_run)."""
    paths = CONTEXT_VIDEOS[scene]
    return next((p for p in paths if os.path.isfile(p)), paths[0])


def head_file(scene):
    """The IMG2 run's first frames (Img2, then its own motion), saved by --vr_save_head_frames."""
    return os.path.join(OUT_DIR, f"{RUN_TAG}_IMG2_{scene}_head.pt")


R = dict(mode="paired", suffix_prompt=True, reverse_prompt=True)  # every Stream R run: PB + reverse prompt

EXPERIMENTS = {
    # --- round 14: the final compilation (S- = single stream, -G = Gordon's fix) ---
    **{f"S-{name}": r14(*flags) for name, flags in FINAL.items()},
    **{f"S-{name}-G": r14(*flags, gordon=True) for name, flags in FINAL.items()},
    # --- round 13: Gordon's brightness fixes on RL2, and release schedules ---
    "RL2-VB": r13("--vr_img2_release_step", "2", "--vr_override_img2_value", "before"),
    "RL2-SP": r13("--vr_img2_release_step", "2", override_stream="splice"),
    "RL2-G": r13("--vr_img2_release_step", "2", "--vr_override_img2_value", "before", override_stream="splice"),
    "SR421": r13("--vr_img2_span_schedule", "4,2,1", only_scenes=("sam",)),
    "SR141": r13("--vr_img2_span_schedule", "1,4,1", only_scenes=("sam",)),
    "PR-LF-VB": r13("--vr_override_img2_span", "last", "--vr_override_img2_value", "before"),
    "RL2-VBA": r13("--vr_img2_release_step", "2", "--vr_override_img2_value", "all"),
    "SR431": r13("--vr_img2_span_schedule", "4,3,1", only_scenes=("sam",)),
    "SR1441": r13("--vr_img2_span_schedule", "1,4,4,1"),
    "RL2-DP": r13("--vr_img2_release_step", "2", relay_prompt="detailed", only_scenes=("sam",)),
    # --- round 12: changes to PR-LF's input ---
    "SR4321": r12("--vr_img2_span_schedule", "4,3,2,1"),
    "RLc2": r12("--vr_img2_release_step", "2", "--vr_img2_release_input", "own"),
    "PR-4C": r12("--vr_override_img2_mask", "last"),
    "PR-LF-DP": r12("--vr_override_img2_span", "last", relay_prompt="detailed", only_scenes=("sam",)),
    # --- rounds 10-11: the release (RL<K>) and the section-2 prompt tests ---
    "RL1": r10("--vr_img2_release_step", "1"),
    "RL2": r10("--vr_img2_release_step", "2"),
    "RL3": r10("--vr_img2_release_step", "3"),
    "RL4": r10("--vr_img2_release_step", "4"),
    "RL4-DP": r10("--vr_img2_release_step", "4", relay_prompt="detailed", only_scenes=("sam",)),
    "RL7": r10("--vr_img2_release_step", "7"),
    "RL10": r10("--vr_img2_release_step", "10"),
    "RL14": r10("--vr_img2_release_step", "14"),
    "CL-NR": r10("--vr_override_img2_span", "last", override_stream="clean", relay_prompt=False,
                 only_scenes=("sam",)),
    "CL-DP": r10("--vr_override_img2_span", "last", override_stream="clean", relay_prompt="detailed",
                 only_scenes=("sam",)),
    # --- catalog: round 9b ---
    "LGG-LF": r9("--vr_override_img2_span", "last", "--vr_img2_fill", "grey", override_stream="long"),
    "JM-4": r9("--vr_img2_code_stats", "meanstd"),
    # --- catalog: round 8b ---
    "LG-LF": r8("--vr_override_img2_span", "last", override_stream="long"),
    "LG-4": r8(override_stream="long"),
    "CX-4": r8(override_stream="context", context=True),
    "CX-LFi": r8("--vr_override_img2_span", "last", override_stream="context", context=True),
    "CX-LFh": r8("--vr_override_img2_span", "last", "--vr_img2_fill", "hold", override_stream="context",
                 context=True),
    "CX-LFv": r8("--vr_override_img2_span", "last", "--vr_img2_fill", "video", override_stream="context",
                 context=True),
    # --- catalog: round 7 ---
    "IMG2": dict(mode="img2", args=["--vr_save_head_frames", "8"], probe=False,
                 outputs=lambda scene: [head_file(scene)]),
    "CL-LF": r7("--vr_override_img2_span", "last", override_stream="clean"),
    "PR-LF": r7("--vr_override_img2_span", "last"),
    "CL-MV": r7("--vr_override_img2_span", "motion", override_stream="clean", needs="IMG2"),
    "PR-MV": r7("--vr_override_img2_span", "motion", needs="IMG2"),
    "REACH": dict(mode="reach", relay_prompt=True, probe=False,
                  args=["--vr_reach_steps", "0,5,9", "--vr_reach_eps", "0.1", "--vr_reach_blur", "4",
                        "--vr_reach_patterns", "2"],
                  outputs=lambda scene: [os.path.join(OUT_DIR, f"{RUN_TAG}_REACH_{scene}_reach.json")]),
    # --- catalog: round 6, directions A, B, C on top of M1-PR ---
    "PR-CL": r6(override_stream="clean"),
    "PR-RR14": r6("--vr_rerender_from_step", "14"),
    "PR-RR20": r6("--vr_rerender_from_step", "20"),
    "PR-QK": r6("--vr_override_mode", "route"),
    # --- catalog: round 5, fixes on top of M1-PR ---
    "PR-SB": r5(stream_b=True),
    "PR-HO10": r5("--vr_override_handover_step", "10"),
    "PR-HO20": r5("--vr_override_handover_step", "20"),
    "PR-HO30": r5("--vr_override_handover_step", "30"),
    "PR-FE": r5(override_stream="fresh"),
    "PR-RS": r5("--vr_resample_from_step", "3", "--vr_resample_to_step", "9", "--vr_resample_cycles", "1"),
    "PR-LOC": r5("--vr_override_profile", "local"),
    "PR-LAND": r5(relay_prompt="land", only_scenes=("fnr",)),
    # --- catalog: round 4, fixes on top of M1-OWN ---
    "M1-PR": r4(relay_prompt=True),
    "M1-SF": r4("--vr_override_img2_span", "frame"),
    "M1-SM": r4("--vr_stat_anchor", "mean"),
    "M1-RS": r4("--vr_resample_from_step", "3", "--vr_resample_to_step", "9", "--vr_resample_cycles", "1"),
    "M1-SEED2": r4(seed=SEED2),
    "M1-G5": r4("--vr_high_noise_guide_scale", "5.0"),
    "M1-OWN-S2": r4(),  # the round-3 reference again, with stream 2's video saved
    # --- catalog: round 3, Method 1 ---
    "M1": m1(),
    "M1-D50": m1(min_weight=0.5),
    "M1-D50N": m1(min_weight=0.5, blend="norm"),
    "M1-D25": m1(min_weight=0.25),
    "M1-SPL": m1(stream="splice"),
    "M1-OWN": m1(latent="own"),
    # --- catalog: earlier rounds ---
    "PB": dict(mode="paired", args=[], suffix_prompt=True),
    "KG3": dict(mode="paired", args=["--vr_kg_scale", "3", "--vr_kg_until_step", "10"], suffix_prompt=True),
    "RA10": dict(R, args=rev("attn", 10)),
    "RO10": dict(R, args=rev("pred", 10)),
    "RA25": dict(R, args=rev("attn", 25)),
    "RA10H": dict(R, args=rev("attn", 10, 0.5)),
    "RC10s50": dict(R, args=rev("attn", 10, 0.5, couple=True)),
    "RC10s100": dict(R, args=rev("attn", 10, 1.0, couple=True)),
    "BASE": dict(mode="paired", args=[]),
    "I2V": dict(mode="plain", args=[]),
    "A-FreshKey": dict(mode="paired", args=["--vr_anchor_key", "B"]),
    "B-MirrorSink": dict(mode="paired", args=["--vr_anchor_key", "B", "--vr_anchor_mirror"]),
    "C-SinkSplit": dict(mode="paired", args=["--vr_prefix_mass", "split", "--vr_split_shape", "linear"]),
    "D-MirrorMass": dict(mode="paired", args=["--vr_prefix_mass", "mirror_mass"]),
    "E-Img1Fade": dict(mode="paired", args=["--vr_prefix_mass", "img1_fade", "--vr_split_shape", "linear"]),
    "TFIX": dict(mode="paired", args=["--vr_suffix_time_fix"]),  # ablation of Gordon's offset, not a fix
}

# Priority order: one run per free GPU pair, the rest as pairs free up.
# Priority order: one run per free GPU pair, the rest as pairs free up. First the open question (a longer boost
# with Gordon's fix), then every configuration raw and fixed on both scenes.
FIRST = [("S-RL3-G", "sam"), ("S-RL4-G", "sam"), ("S-RL3-G", "fnr"), ("S-RL4-G", "fnr")]
TASKS = FIRST + [(f"S-{name}{fix}", scene) for name in FINAL for fix in ("", "-G") for scene in ("sam", "fnr")
                 if (f"S-{name}{fix}", scene) not in FIRST]

# Markers proving the round-14 files are installed.
INSTALLED_MARKERS = {
    "generate.py": "vr_override_single",
    os.path.join("wan", "image2video.py"): "elif override_on and override_single:",
    os.path.join("wan", "modules", "attention.py"): "def _single_stream_clean_values",
    os.path.join("wan", "modules", "attn_probe.py"): "_summary_override",
    os.path.join("wan", "modules", "model.py"): "def relay_local_split",
    os.path.join("wan", "distributed", "sequence_parallel.py"): "relay_local_split",
    os.path.join("wan", "distributed", "ulysses.py"): "override_attention",
    "test_r1_attention.py": "test_tiny_generate_round14",
    "analyze_probe_r1.py": "plot_override",
    "vr_decoder_probe.py": "def build_candidates",  # the preflight tests load it
    "vr_latent_fix.py": "def smooth_stats",  # round 9a, and the preflight tests load it
    "vr_global_fix.py": "def global_pixel_fix",  # round 12a, and the preflight tests load it
}
SCHED_LOG = os.path.join(OUT_DIR, f"log_{RUN_TAG}_scheduler.txt")
PID_FILE = os.path.join(OUT_DIR, f"{RUN_TAG}_scheduler.pid")
STATE_FILE = os.path.join(OUT_DIR, f"{RUN_TAG}_scheduler_state.json")


def tag(task):
    code, scene = task
    return f"{RUN_TAG}_{code}_{scene}"


def dependency(task):
    """The run whose output this run reads, or None."""
    code, scene = task
    need = EXPERIMENTS[code].get("needs")
    return (need, scene) if need else None


def latents_file(task):
    """Round 9: the final latents an override run saves (--vr_save_latents)."""
    return os.path.join(OUT_DIR, f"{tag(task)}_latents.pt")


def outputs(task):
    """Files that prove a run finished: its video, its latents (override runs), plus any extra outputs."""
    code, scene = task
    extra = EXPERIMENTS[code].get("outputs")
    lat = [latents_file(task)] if EXPERIMENTS[code]["mode"] == "override" else []
    return [os.path.join(OUT_DIR, f"{tag(task)}.mp4")] + lat + (extra(scene) if extra else [])


def is_done(task):
    return all(os.path.isfile(path) for path in outputs(task))


def generate_args(task):
    code, scene = task
    sc, ex = SCENES[scene], EXPERIMENTS[code]
    if scene not in ex.get("only_scenes", SCENES):
        raise ValueError(f"{code} is defined for {ex['only_scenes']} only")
    img1, img2 = (os.path.join(IMAGE_DIR, name) for name in sc["images"])
    # IMG2: plain I2V of Img2 reads the after-Img2 prompt everywhere.
    prompt = SUFFIX_PROMPTS[scene] if ex["mode"] == "img2" else sc["prompt"]
    args = list(COMMON) + ["--base_seed", str(ex.get("seed", SEED)), "--prompt", prompt,
                           "--save_file", os.path.join(OUT_DIR, f"{tag(task)}.mp4")]
    if ex.get("suffix_prompt"):
        args += ["--vr_suffix_prompt", SUFFIX_PROMPTS[scene]]
    if ex.get("reverse_prompt"):
        args += ["--vr_rev_prompt", REVERSE_PROMPTS[scene]]
    if ex.get("relay_prompt") == "land":
        args += ["--vr_relay_prompt", LANDING_PROMPTS[scene]]
    elif ex.get("relay_prompt") == "detailed":
        args += ["--vr_relay_prompt", DETAILED_PROMPTS[scene]]
    elif ex.get("relay_prompt"):
        args += ["--vr_relay_prompt", SUFFIX_PROMPTS[scene]]
    if ex.get("stream_b"):
        # Gordon's lambda: 1.0 for FNR, 0.5 for SAM, faded out over steps 25-30 as in round 2.
        args += ["--vr_stream_b_weight", str(sc["weight"]), "--vr_stream_b_fade_start_step", "25",
                 "--vr_stream_b_fade_end_step", "30"]
    if ex["mode"] == "override":
        args += ["--image", img1, "--image_latent_index", "0", "--image", img2,
                 "--image_latent_index", str(ex.get("anchor", sc["anchor"]))]
    elif ex["mode"] == "paired":
        args += ["--image", img1, "--image_latent_index", "0", "--image", img2,
                 "--image_latent_index", str(ex.get("anchor", sc["anchor"])),
                 "--image_condition_paired_anchor_attention",
                 "--image_condition_paired_stream_weight", str(sc["weight"]),
                 "--image_condition_paired_suffix_fade_start_step", "25",
                 "--image_condition_paired_suffix_fade_end_step", "30",
                 "--image_condition_paired_prefix_route_switch_step", "25"]
    elif ex["mode"] == "img2":
        args += ["--image", img2]
    elif ex["mode"] == "reach":
        anchor = ex.get("anchor", sc["anchor"])
        args += ["--image", img1, "--vr_relay_from_latent", str(anchor + 1), "--vr_reach_target", str(anchor)]
    else:
        args += ["--image", img1]
    if ex.get("needs") == "IMG2":
        args += ["--vr_img2_motion_frames", head_file(scene)]
    if ex.get("context"):
        args += ["--vr_img2_context_video", context_video(scene)]
    if ex["mode"] == "override":
        args += ["--vr_save_latents"]  # round 9: for the decode lab
    if ex["mode"] == "override" and ex.get("decode_fix", True):
        # Round 10: round 9's decode fix inside the run (<tag>_fix.mp4); the main video stays uncorrected.
        args += ["--vr_decode_fix", "meanstd", "--vr_decode_fix_window", sc["fix_window"]]
    return args + (PROBE if ex.get("probe", True) else []) + ex["args"]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def torchrun(script_args, port):
    return [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2",
            "--master_addr=127.0.0.1", f"--master_port={port}"] + script_args


def task_env(pair, lse_backend):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in pair)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["VR_LSE_BACKEND"] = lse_backend
    # torchrun forces OMP_NUM_THREADS=1 when it is unset. With --t5_cpu that makes the
    # T5-XXL prompt encoding single-threaded (tens of minutes). Set it explicitly instead.
    env.setdefault("OMP_NUM_THREADS", str(OMP_THREADS_PER_RANK))
    return env


def gpu_memory_used():
    """{gpu index: MiB used}; empty if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        return {int(i): int(m) for i, m in (line.split(",") for line in out.strip().splitlines())}
    except Exception as e:  # noqa: BLE001
        print(f"Warning: could not read GPU usage ({e}).", flush=True)
        return {}


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# --- preflight ----------------------------------------------------------------------------------
def check_installed():
    problems = []
    for path, marker in INSTALLED_MARKERS.items():
        if not os.path.isfile(path):
            problems.append(f"missing {path}")
            continue
        text = open(path, encoding="utf-8").read()
        if marker not in text:
            problems.append(f"{path} is not the round-14 version (marker '{marker}' not found)")
    for scene in SCENES.values():
        for name in scene["images"]:
            if not os.path.isfile(os.path.join(IMAGE_DIR, name)):
                problems.append(f"missing image {os.path.join(IMAGE_DIR, name)}")
    if not os.path.isdir(CKPT_DIR):
        problems.append(f"missing checkpoint folder {CKPT_DIR}")
    return problems


def missing_contexts(tasks):
    """Two-pass runs whose first-pass video is not on disk."""
    return [f"{tag(t)} needs one of {CONTEXT_VIDEOS[t[1]]} (the first pass); none exists" for t in tasks
            if EXPERIMENTS[t[0]].get("context") and not os.path.isfile(context_video(t[1]))]


def run_preflight(pair):
    os.makedirs(OUT_DIR, exist_ok=True)
    single_log = os.path.join(OUT_DIR, "preflight_single_gpu.txt")
    print(f"Preflight 1/2: attention tests on GPU {pair[0]} (log: {single_log}) ...", flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(pair[0])
    res = subprocess.run([sys.executable, "test_r1_attention.py"], env=env, capture_output=True, text=True,
                         timeout=PREFLIGHT_TIMEOUT_S)
    open(single_log, "w").write(res.stdout + "\n" + res.stderr)
    backend = "flash"
    for line in res.stdout.splitlines():
        if line.startswith("LSE_BACKEND="):
            backend = line.split("=", 1)[1].strip()
    if res.returncode != 0 or "ALL CHECKS PASSED" not in res.stdout:
        fails = [line for line in res.stdout.splitlines() if line.startswith("[FAIL]")]
        sys.exit("[ERROR] preflight 1 failed:\n  " + "\n  ".join(fails or res.stderr.splitlines()[-15:])
                 + f"\nFull log: {single_log}. Nothing was launched.")
    print(f"  passed (group log-sum-exp backend: {backend})", flush=True)
    sp_log = os.path.join(OUT_DIR, "preflight_two_gpu.txt")
    print(f"Preflight 2/2: sequence-parallel tests on GPUs {pair} (log: {sp_log}) ...", flush=True)
    res = subprocess.run(torchrun(["test_r1_attention.py", "--sp"], free_port()), env=task_env(pair, backend),
                         capture_output=True, text=True, timeout=PREFLIGHT_TIMEOUT_S)
    open(sp_log, "w").write(res.stdout + "\n" + res.stderr)
    if res.returncode != 0 or "ALL CHECKS PASSED" not in res.stdout:
        fails = [line for line in res.stdout.splitlines() if line.startswith("[FAIL]")]
        sys.exit("[ERROR] preflight 2 failed:\n  " + "\n  ".join(fails or res.stderr.splitlines()[-15:])
                 + f"\nFull log: {sp_log}. Nothing was launched.")
    print("  passed", flush=True)
    return backend


# --- scheduler ----------------------------------------------------------------------------------
def start(task, pair, lse_backend, attempt=1):
    cmd = torchrun(["generate.py"] + generate_args(task), free_port())
    log_path = os.path.join(OUT_DIR, f"log_{tag(task)}.txt")
    fh = open(log_path, "w" if attempt == 1 else "a")
    if attempt > 1:
        fh.write(f"\n\n===== attempt {attempt} =====\n")
    fh.write(f"GPUs {pair}\nVR_LSE_BACKEND={lse_backend}\n" + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
    fh.flush()
    proc = subprocess.Popen(cmd, env=task_env(pair, lse_backend), stdout=fh, stderr=subprocess.STDOUT,
                            start_new_session=True)
    return proc, fh


def machine_gpus(usage):
    """The GPUs this machine has (from nvidia-smi), or DEFAULT_GPUS when it cannot be read."""
    return sorted(usage) if usage else list(DEFAULT_GPUS)


def check_gpus(gpus, usage):
    """Problems with the GPU list: unknown GPU numbers (needs nvidia-smi), duplicates, fewer than two."""
    problems = []
    if len(set(gpus)) != len(gpus):
        problems.append(f"a GPU is listed twice: {gpus}")
    if len(set(gpus)) < 2:
        problems.append(f"every run needs 2 GPUs; only {sorted(set(gpus))} listed")
    if usage:
        missing = sorted(set(gpus) - set(usage))
        if missing:
            problems.append(f"GPU(s) {missing} do not exist; this machine has GPUs {sorted(usage)} (numbered from 0)")
    return problems


def free_gpus(gpus, usage, taken=(), retired=()):
    """GPUs that can take a run now: listed, not running one of ours, not retired, under BUSY_MB in use."""
    return [g for g in sorted(gpus) if g not in taken and g not in retired and usage.get(g, 0) <= BUSY_MB]


def form_pairs(free):
    """Pair free GPUs lowest numbers first: [1, 2, 5, 6, 7] -> [(1, 2), (5, 6)]."""
    free = sorted(free)
    return [(free[i], free[i + 1]) for i in range(0, len(free) - 1, 2)]


def analyze(paths, compare=False):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [sys.executable, "analyze_probe_r1.py"] + (["--compare", OUT_DIR] if compare else paths)
    try:
        subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    except Exception as e:  # noqa: BLE001
        print(f"[{stamp()}] analysis failed: {e}", flush=True)


def pick_task(pending, pair, task_fail_pairs, ready, other_pairs_possible):
    """The first waiting run that is ready and has not already failed to start on this pair. A run that has may
    retry on it only when no other pair can ever form (e.g. a 2-GPU machine); otherwise it waits for another."""
    fresh = next((t for t in pending if ready(t) and pair not in task_fail_pairs.get(t, ())), None)
    if fresh is not None or other_pairs_possible:
        return fresh
    return next((t for t in pending if ready(t)), None)


def run_scheduler():
    state = json.load(open(STATE_FILE))
    tasks = [tuple(t) for t in state["tasks"]]
    gpus = [int(g) for g in state["gpus"]]
    backend = state["lse_backend"]
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    pending, running = list(tasks), {}
    gpu_fast_fails = {g: 0 for g in gpus}
    task_fail_pairs = {}  # task -> pairs it failed fast on
    attempts = {}
    given_up, failed = [], []
    print(f"[{stamp()}] scheduler pid {os.getpid()}: {len(pending)} runs on GPUs {gpus} (any two free ones), "
          f"VR_LSE_BACKEND={backend}", flush=True)
    while pending or running:
        for pair, (proc, fh, task, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            del running[pair]
            elapsed = time.time() - t0
            print(f"[{stamp()}] GPUs {pair}: finished {tag(task)} (exit code {rc}, {elapsed / 60:.1f} min)",
                  flush=True)
            if rc != 0 and elapsed < FAST_FAIL_S:
                # Failed to start: back in the queue; retire GPUs that keep doing this.
                task_fail_pairs.setdefault(task, []).append(pair)
                for g in pair:
                    gpu_fast_fails[g] += 1
                    if gpu_fast_fails[g] == GPU_FAST_FAIL_LIMIT:
                        print(f"[{stamp()}] GPU {g}: retired after {GPU_FAST_FAIL_LIMIT} runs on it failed within "
                              f"{FAST_FAIL_S} s of starting (see their logs)", flush=True)
                if len(task_fail_pairs[task]) >= TASK_FAST_FAIL_LIMIT:
                    given_up.append(task)
                    print(f"[{stamp()}] giving up {tag(task)}: it failed to start on {task_fail_pairs[task]}",
                          flush=True)
                else:
                    pending.insert(0, task)
                    print(f"[{stamp()}] {tag(task)} failed to start; back in the queue", flush=True)
                continue
            if rc != 0:
                failed.append(task)
            probe = os.path.join(OUT_DIR, f"{tag(task)}_probe.pt")
            if os.path.isfile(probe):
                analyze([probe])
        # A run whose input run is neither queued, running nor finished can never start.
        running_tasks = [entry[2] for entry in running.values()]
        for t in list(pending):
            dep = dependency(t)
            if dep and dep not in pending and dep not in running_tasks and not is_done(dep):
                pending.remove(t)
                given_up.append(t)
                print(f"[{stamp()}] giving up {tag(t)}: it needs the output of {tag(dep)}, which did not finish",
                      flush=True)

        def ready(t):
            dep = dependency(t)
            return dep is None or (dep not in pending and dep not in running_tasks and is_done(dep))

        retired = {g for g in gpus if gpu_fast_fails[g] >= GPU_FAST_FAIL_LIMIT}
        if pending and len(set(gpus) - retired) < 2:
            print(f"[{stamp()}] fewer than two usable GPUs left; not run: {[tag(t) for t in pending]}", flush=True)
            given_up.extend(pending)
            pending = []
        if pending:
            taken = {g for p in running for g in p}
            if len(set(gpus) - taken - retired) >= 2:
                usage = gpu_memory_used()
                other_pairs_possible = len(set(gpus) - retired) > 2
                for pair in form_pairs(free_gpus(gpus, usage, taken, retired)):
                    choice = pick_task(pending, pair, task_fail_pairs, ready, other_pairs_possible)
                    if choice is None:
                        continue
                    pending.remove(choice)
                    attempts[choice] = attempts.get(choice, 0) + 1
                    proc, fh = start(choice, pair, backend, attempts[choice])
                    running[pair] = (proc, fh, choice, time.time())
                    print(f"[{stamp()}] GPUs {pair}: started {tag(choice)} pid {proc.pid} ({len(pending)} waiting)",
                          flush=True)
        time.sleep(POLL_S)
    analyze([], compare=True)
    done = len(tasks) - len(given_up) - len(failed)
    print(f"[{stamp()}] scheduler finished: {done} of {len(tasks)} runs succeeded"
          + (f"; failed during generation: {[tag(t) for t in failed]}" if failed else "")
          + (f"; never started: {[tag(t) for t in given_up]}" if given_up else "")
          + f". Comparison in {OUT_DIR}/{RUN_TAG}_probe_compare.txt", flush=True)
    os.remove(PID_FILE)


def parse_cli(argv):
    opts = dict(dry=False, skip_preflight=False, only=None, scenes=None, gpus=None)
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--dry_run":
            opts["dry"] = True
        elif a == "--skip_preflight":
            opts["skip_preflight"] = True
        elif a == "--pairs":
            sys.exit("--pairs is replaced by --gpus (e.g. --gpus 0,1,2,3): any two free GPUs are paired")
        elif a in ("--only", "--scenes", "--gpus"):
            if i + 1 >= len(argv):
                sys.exit(f"{a} needs a value")
            opts[a[2:]] = argv[i + 1]
            i += 1
        elif a != "--scheduler":
            sys.exit(f"unknown option {a}")
        i += 1
    return opts


def main():
    if "--scheduler" in sys.argv:
        run_scheduler()
        return
    opts = parse_cli(sys.argv[1:])
    tasks = list(TASKS)
    if opts["only"]:
        wanted = [c.strip() for c in opts["only"].split(",")]
        unknown = [c for c in wanted if c not in EXPERIMENTS]
        if unknown:
            sys.exit(f"unknown experiment(s) {unknown}; choose from {list(EXPERIMENTS)}")
        tasks = [(c, s) for c in wanted for s in SCENES
                 if s in EXPERIMENTS[c].get("only_scenes", SCENES)]
    if opts["scenes"]:
        wanted = [s.strip() for s in opts["scenes"].split(",")]
        tasks = [t for t in tasks if t[1] in wanted]
    usage = gpu_memory_used()
    gpus = list(GPUS) if GPUS is not None else machine_gpus(usage)
    if opts["gpus"]:
        try:
            gpus = [int(g) for g in opts["gpus"].split(",") if g.strip()]
        except ValueError:
            sys.exit("--gpus takes GPU numbers separated by commas, e.g. --gpus 0,1,2,3")
    gpu_problems = check_gpus(gpus, usage)
    if gpu_problems and not opts["dry"]:
        sys.exit("[ERROR] bad GPU list:\n  " + "\n  ".join(gpu_problems) + "\nNothing was launched.")

    # A run that reads another run's output gets that run queued first when the output is missing.
    for t in list(tasks):
        dep = dependency(t)
        if dep and dep not in tasks and not is_done(dep):
            tasks.insert(tasks.index(t), dep)
            print(f"Adding {tag(dep)}: {tag(t)} needs its output.")
    # Runs whose outputs already exist are skipped (never overwritten), so rerunning the same command
    # finishes only what is missing.
    done = [t for t in tasks if is_done(t)]
    if done:
        print(f"Skipping {len(done)} run(s) whose outputs already exist in {OUT_DIR}/: "
              + ", ".join(tag(t) for t in done) + "\n")
        tasks = [t for t in tasks if t not in done]
    if not tasks:
        print("Nothing to run.")
        return

    now = form_pairs(free_gpus(gpus, usage))
    print(f"{len(tasks)} runs on GPUs {gpus}, any two free ones per run ({len(now)} pair(s) free now):\n")
    for i, task in enumerate(tasks):
        where = f"GPUs {now[i]}" if i < len(now) else "queued"
        dep = dependency(task)
        wait = f"  (starts after {tag(dep)} has saved {os.path.basename(head_file(task[1]))})" if (
            dep and not is_done(dep)) else ""
        ctx = f"  (first pass: {context_video(task[1])})" if EXPERIMENTS[task[0]].get("context") else ""
        print(f"  {i + 1:2d}. [{where}] {tag(task)}{wait}{ctx}")
        if opts["dry"]:
            cmd = torchrun(["generate.py"] + generate_args(task), "<free port>")
            print("      CUDA_VISIBLE_DEVICES=<pair> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\\n      "
                  + " ".join(shlex.quote(c) for c in cmd) + "\n")
    if opts["dry"]:
        problems = check_installed() + missing_contexts(tasks)
        print("\nInstall check: " + ("OK" if not problems else "\n  " + "\n  ".join(problems)))
        print("GPUs: " + ("OK" if not gpu_problems else "\n  " + "\n  ".join(gpu_problems)))
        print("Dry run: nothing launched.")
        return

    if not os.path.isfile("generate.py"):
        sys.exit("[ERROR] run this from the Wan2.2 repo root (the folder that holds generate.py)")
    problems = check_installed()
    if problems:
        sys.exit("[ERROR] round-14 files are not installed correctly:\n  " + "\n  ".join(problems))
    problems = missing_contexts(tasks)
    if problems:
        sys.exit("[ERROR] first-pass videos are missing (edit CONTEXT_VIDEOS if they are elsewhere):\n  "
                 + "\n  ".join(problems) + "\nNothing was launched.")
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.isfile(PID_FILE):
        sys.exit(f"[ERROR] {PID_FILE} exists: a round-14 scheduler may already be running (delete it if not).")
    usage = gpu_memory_used()
    first = form_pairs(free_gpus(gpus, usage))
    if not first:
        print(f"Fewer than two of GPUs {gpus} are free now (MiB used: {usage}); waiting for two to free up "
              f"(Ctrl-C to stop) ...", flush=True)
        while not first:
            time.sleep(POLL_S)
            usage = gpu_memory_used()
            first = form_pairs(free_gpus(gpus, usage))
        print(f"[{stamp()}] GPUs {first[0]} are free.", flush=True)
    busy = [g for g in gpus if g not in free_gpus(gpus, usage)]
    if busy:
        print(f"Note: GPU(s) {busy} are busy now; the scheduler will use them once they free up.")

    backend = os.environ.get("VR_LSE_BACKEND", "flash")
    if opts["skip_preflight"]:
        print("Preflight skipped (--skip_preflight).")
    else:
        backend = run_preflight(first[0])

    with open(STATE_FILE, "w") as f:
        json.dump(dict(tasks=tasks, gpus=gpus, lse_backend=backend,
                       commands={tag(t): generate_args(t) for t in tasks}), f, indent=1)
    log = open(SCHED_LOG, "a")
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--scheduler"], stdout=log,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"\nScheduler running in the background (pid {proc.pid}). Progress: {SCHED_LOG}")
    print("You can close the terminal.")


if __name__ == "__main__":
    main()
