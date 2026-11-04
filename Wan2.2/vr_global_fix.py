"""
Visual Relay, round 12a: whole-video colour fix lab. VAE only, one GPU, a few minutes per video, no generation.

    python vr_global_fix.py                          # every source in SOURCES whose latent file exists
    python vr_global_fix.py --only fnr-RL2,fnr-RL4 --gpu 0
    python vr_global_fix.py --add fnr-X=fnr:r11_outputs/r11_RL2_fnr_latents.pt

Why. Round 9's decode fix (Lms) corrects only the latents around Img2. In FNR, RL2 and RL4 are too bright over the
whole video (frames 4-12 darker, then up to 200 against the keyframes' 152, more contrast, a golden ground), and that
is made by stream 2 itself (round 11, F60). Here every latent gets a colour target from the two keyframes instead.

Targets. Img1 held for the whole clip and Img2 held for the whole clip are encoded with the same VAE ("stills"), so
each latent position has a reference with the right colour for that position. Latent j's target is the line from
Img1's still to Img2's still, weight w = min(j / a, 1) (Img1 at latent 0, Img2 from Img2's latent a on). Latent 0
(Img1 itself) is never changed. Variants, each decoded in full:
  rt     no correction (the run's own main video)
  Lms    round 9's correction on the latents around Img2 only (what <tag>_fix.mp4 holds)
  Gm     every latent: each channel's spatial mean set to the target's; the spatial pattern is kept
  Gms    the same for each channel's mean and spread (std)
  Gl     the channel means as a coarse 4 x 4 map instead of one number (sky and ground get their own targets), plus
         the spread; only meaningful for a static camera (FNR), since the stills' layout must match the video's
  GP     the Gms idea on pixels, after decoding (each colour channel's mean and spread per frame, targets from the
         decoded stills); a reference
Only colour statistics change: no variant can move, add or remove anything. A real lighting change (SAM's camera
turning towards the lanterns, a scene going from bright to dark) is flattened too, because the target only knows the
two keyframes.

Outputs in r12_outputs/: r12_gfix_<source>_<variant>.mp4 for Gm, Gms, Gl and GP, and r12_gfix_<source>.txt / .json
(brightness, saturation, contrast, sharpness and frame change every 4th frame, a summary per variant, and the target
brightness) and r12_gfix_<source>_sheet.png (every 8th frame of every variant).
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_lab():
    """vr_latent_fix.py (round 9a), next to this file: Lms, the metrics, the sheet, image and video helpers."""
    spec = importlib.util.spec_from_file_location("vr_latent_fix", os.path.join(HERE, "vr_latent_fix.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


L = load_lab()
P = L.P

SOURCES = {
    "fnr-RL2": dict(scene="fnr", paths=["r11_outputs/r11_RL2_fnr_latents.pt"]),
    "fnr-RL4": dict(scene="fnr", paths=["r11_outputs/r11_RL4_fnr_latents.pt"]),
    "fnr-PR-LF": dict(scene="fnr", paths=["r11_outputs/r11_PR-LF_fnr_latents.pt"]),
    "fnr-CL-LF": dict(scene="fnr", paths=["r10_outputs/r10_CL-LF_fnr_latents.pt", "r9_outputs/r9_CL-LF_fnr_latents.pt"]),
    "sam-RL2": dict(scene="sam", paths=["r11_outputs/r11_RL2_sam_latents.pt"]),
    "sam-RL4-DP": dict(scene="sam", paths=["r11_outputs/r11_RL4-DP_sam_latents.pt"]),
}
LOCAL = {"fnr": (10, 11), "sam": (11, 13)}  # round 9's Lms windows (what <tag>_fix.mp4 uses)
GRID = (4, 4)  # Gl: cells of the coarse colour map
OUT_DIR = "r12_outputs"
TAG = "r12_gfix"
STRIDE = 4
SAVED = ("Gm", "Gms", "Gl", "GP")  # rt and Lms are the run's own videos


# --- corrections --------------------------------------------------------------------------------
def key_weights(n, a):
    """Weight of Img2 in the target of latent j = 0..n-1: min(j / a, 1) (0 at Img1, 1 from Img2's latent a on)."""
    if a < 1:
        raise ValueError("Img2's latent must be 1 or later")
    return [min(j / a, 1.0) for j in range(n)]


def still_latents(encode, image, frames):
    """Latents of image [3, H, W] held for all frames: the colour a still shot has at every latent position."""
    return encode([image[:, None].expand(-1, frames, -1, -1).contiguous()])[0]


def lowfreq(x, grid=GRID):
    """x [C, h, w] -> its coarse part: averaged over grid cells, then spread back bilinearly to [C, h, w]."""
    import torch.nn.functional as F
    h, w = x.shape[-2:]
    small = F.interpolate(x[None].float(), size=grid, mode="area")
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)[0]


def global_fix(z, ref1, ref2, a, mode, grid=GRID):
    """z, ref1, ref2 [C, T, h, w]. Every latent j >= 1 of z gets the colour statistics of the line from ref1's latent j
    to ref2's latent j (weight key_weights). 'mean': each channel's spatial mean; 'meanstd': mean and std; 'lowfreq':
    the coarse mean map (lowfreq) and the std of what is left. The spatial pattern (z minus its mean or map, over its
    std) is kept. Latent 0 is returned unchanged."""
    if mode not in ("mean", "meanstd", "lowfreq"):
        raise ValueError(f"unknown mode {mode}")
    if tuple(ref1.shape) != tuple(z.shape) or tuple(ref2.shape) != tuple(z.shape):
        raise ValueError(f"the stills' latents {tuple(ref1.shape)} / {tuple(ref2.shape)} do not match the source's "
                         f"{tuple(z.shape)}")
    out = z.clone()
    for j, w in enumerate(key_weights(z.shape[1], a)):
        if j == 0:
            continue
        cur, r1, r2 = z[:, j].float(), ref1[:, j].float(), ref2[:, j].float()
        if mode == "lowfreq":
            low, l1, l2 = lowfreq(cur, grid), lowfreq(r1, grid), lowfreq(r2, grid)
            res = cur - low
            std_t = (1 - w) * (r1 - l1).std(dim=(1, 2)) + w * (r2 - l2).std(dim=(1, 2))
            res = res / res.std(dim=(1, 2)).clamp(min=1e-6)[:, None, None] * std_t[:, None, None]
            out[:, j] = (res + (1 - w) * l1 + w * l2).to(out.dtype)
            continue
        mean_t = (1 - w) * r1.mean(dim=(1, 2)) + w * r2.mean(dim=(1, 2))
        cur_mean = cur.mean(dim=(1, 2))[:, None, None]
        if mode == "mean":
            out[:, j] = (cur - cur_mean + mean_t[:, None, None]).to(out.dtype)
        else:
            std_t = (1 - w) * r1.std(dim=(1, 2)) + w * r2.std(dim=(1, 2))
            cur_std = cur.std(dim=(1, 2)).clamp(min=1e-6)[:, None, None]
            out[:, j] = ((cur - cur_mean) / cur_std * std_t[:, None, None] + mean_t[:, None, None]).to(out.dtype)
    return out


def global_pixel_fix(video, dec1, dec2, a, stride=STRIDE):
    """video, dec1, dec2 [3, F, H, W] in [-1, 1] (dec1, dec2: the decoded stills). Every frame f gets each colour
    channel's mean and std (0-1 scale) on the line from frame f of dec1 to frame f of dec2, weight min(f / (stride a),
    1); clipped to 0-1."""
    if tuple(dec1.shape) != tuple(video.shape) or tuple(dec2.shape) != tuple(video.shape):
        raise ValueError("the decoded stills do not match the video's size")
    x, r1, r2 = (video.float() + 1) / 2, (dec1.float() + 1) / 2, (dec2.float() + 1) / 2
    out = x.clone()
    for f in range(x.shape[1]):
        w = min(f / (stride * a), 1.0)
        mean_t = (1 - w) * r1[:, f].mean(dim=(1, 2)) + w * r2[:, f].mean(dim=(1, 2))
        std_t = (1 - w) * r1[:, f].std(dim=(1, 2)) + w * r2[:, f].std(dim=(1, 2))
        cur = x[:, f]
        cur_mean = cur.mean(dim=(1, 2))[:, None, None]
        cur_std = cur.std(dim=(1, 2)).clamp(min=1e-6)[:, None, None]
        out[:, f] = ((cur - cur_mean) / cur_std * std_t[:, None, None] + mean_t[:, None, None]).clamp(0, 1)
    return out * 2 - 1


# --- one source ---------------------------------------------------------------------------------
def run_source(label, scene, z, images, encode, decode, save=None):
    """All variants for one source's latents z [C, T, h, w]; images = (img1, img2) [3, H, W] at the video size.
    Returns (results {variant: {frame: metrics}}, sheet {variant: all frames, uint8}, target luma per frame)."""
    a = L.SCENES[scene]["anchor"]
    frames = 1 + STRIDE * (z.shape[1] - 1)
    ref1, ref2 = (still_latents(encode, im, frames) for im in images)
    if tuple(ref1.shape) != tuple(z.shape):
        raise ValueError(f"the keyframes give latents {tuple(ref1.shape)}, the source has {tuple(z.shape)}; "
                         "check the images and the source's scene")
    base = decode([z])[0]
    dec1, dec2 = decode([ref1])[0], decode([ref2])[0]
    lo, hi = LOCAL[scene]
    variants = {"rt": base, f"Lms{lo}-{hi}": decode([L.smooth_stats(z, lo, hi, "meanstd")])[0]}
    for name, mode in (("Gm", "mean"), ("Gms", "meanstd"), ("Gl", "lowfreq")):
        variants[name] = decode([global_fix(z, ref1, ref2, a, mode)])[0]
    variants["GP"] = global_pixel_fix(base, dec1, dec2, a)
    u1, u2 = P.to_uint8(dec1), P.to_uint8(dec2)
    target = {}
    for f in range(u1.shape[0]):
        w = min(f / (STRIDE * a), 1.0)
        target[f] = (1 - w) * P.frame_stats(u1[f])["luma"] + w * P.frame_stats(u2[f])["luma"]
    results, sheet = {}, {}
    for name, video in variants.items():
        u8 = P.to_uint8(video)
        results[name] = L.measure(u8, 0, u8.shape[0] - 1)
        sheet[name] = u8
        if save is not None and name in SAVED:
            save(f"{label}_{name}", video)
    return results, sheet, target


def summary(rows, target):
    """Brightness range and mean distance to the target over frames 1.., mean contrast, mean sharpness, max change."""
    fs = [f for f in sorted(rows) if f >= 1]
    luma = [rows[f]["luma"] for f in fs]
    return dict(luma_min=min(luma), luma_max=max(luma),
                off_target=float(np.mean([abs(rows[f]["luma"] - target[f]) for f in fs])),
                contrast=float(np.mean([rows[f]["contrast"] for f in fs])),
                sharp=float(np.mean([rows[f]["sharp"] for f in fs])),
                max_change=max(rows[f]["change"] for f in fs))


def report(label, scene, path, results, target):
    a = L.SCENES[scene]["anchor"]
    shown = sorted(f for f in target if f % 4 == 0)
    lines = [f"Visual Relay round 12a whole-video colour fix, {label}: {path}; Img2 is latent {a} (frames "
             f"{STRIDE * a - 3}-{STRIDE * a})",
             "rt = none; Lms = round 9's fix around Img2 only; Gm / Gms = every latent's channel means (and stds) set "
             "to the line from Img1's still to Img2's; Gl = the same with a coarse 4x4 mean map; GP = Gms on pixels.",
             ""]
    lines.append(f"target brightness (decoded stills, Img1 -> Img2), frames {shown[0]}..{shown[-1]} every 4:")
    lines.append("  target     " + " ".join("%4.0f" % target[f] for f in shown))
    lines.append("")
    for key, fmt in (("luma", "%4.0f"), ("sat", "%4.0f"), ("contrast", "%4.0f"), ("sharp", "%5.0f"),
                     ("change", "%4.1f")):
        lines.append(f"{key} per frame, every 4th:")
        for name, rows in results.items():
            lines.append(f"  {name:10s} " + " ".join(fmt % rows[f][key] for f in shown))
        lines.append("")
    lines.append("summary over frames 1-%d: brightness range | mean distance to target | contrast | sharpness | "
                 "largest frame change" % max(target))
    for name, rows in results.items():
        s = summary(rows, target)
        lines.append(f"  {name:10s} {s['luma_min']:4.0f}-{s['luma_max']:<4.0f} | {s['off_target']:5.1f} | "
                     f"{s['contrast']:4.1f} | {s['sharp']:6.0f} | {s['max_change']:5.1f}")
    return "\n".join(lines) + "\n"


def parse_add(spec):
    """name=scene:path -> (name, dict)."""
    name, rest = spec.split("=", 1)
    scene, path = rest.split(":", 1)
    if scene not in L.SCENES:
        raise ValueError(f"unknown scene {scene}; choose from {list(L.SCENES)}")
    return name, dict(scene=scene, paths=[path])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default=None, help="comma-separated source names")
    p.add_argument("--add", action="append", default=[], help="name=scene:path to a <tag>_latents.pt")
    p.add_argument("--stream", default="stream1", choices=["stream1", "stream2"],
                   help="stream1 = the main video's latents (default), stream2 = the _s2 video's")
    p.add_argument("--ckpt_dir", default=L.CKPT_DIR)
    p.add_argument("--image_dir", default=L.IMAGE_DIR)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--no_videos", action="store_true", help="tables and sheets only")
    args = p.parse_args(argv)
    sources = dict(SOURCES)
    sources.update(parse_add(s) for s in args.add)
    if args.only:
        wanted = [s.strip() for s in args.only.split(",")]
        unknown = [s for s in wanted if s not in sources]
        if unknown:
            sys.exit(f"unknown source(s) {unknown}; choose from {list(sources)}")
        sources = {k: sources[k] for k in wanted}
    sys.path.insert(0, os.getcwd())
    from wan.configs import WAN_CONFIGS
    from wan.modules.vae2_1 import Wan2_1_VAE
    from wan.utils.utils import save_video
    device = torch.device(f"cuda:{args.gpu}")
    vae = Wan2_1_VAE(vae_pth=os.path.join(args.ckpt_dir, WAN_CONFIGS["i2v-A14B"].vae_checkpoint), device=device)
    os.makedirs(args.out_dir, exist_ok=True)

    def encode(clips):
        return [z.float().cpu() for z in vae.encode([c.to(device) for c in clips])]

    def decode(latents):
        return [v.float().cpu() for v in vae.decode([z.to(device) for z in latents])]

    def save(name, video):
        if not args.no_videos:
            save_video(tensor=video[None], save_file=os.path.join(args.out_dir, f"{TAG}_{name}.mp4"), fps=L.FPS,
                       nrow=1, normalize=True, value_range=(-1, 1))

    done = []
    for name, src in sources.items():
        path = next((q for q in src["paths"] if os.path.isfile(q)), None)
        if path is None:
            print(f"[{name}] skipped: none of {src['paths']} exists", flush=True)
            continue
        kind, z = L.load_source(path, args.stream)
        if kind != "lat":
            sys.exit(f"[{name}] {path} is not a latent file; this lab needs <tag>_latents.pt (--vr_save_latents)")
        size = (z.shape[2] * 8, z.shape[3] * 8)
        sc = L.SCENES[src["scene"]]
        images = tuple(P.load_image(os.path.join(args.image_dir, n), size) for n in sc["images"])
        print(f"[{name}] {path} ...", flush=True)
        with torch.no_grad():
            results, sheet, target = run_source(name, src["scene"], z, images, encode, decode, save=save)
        stem = os.path.join(args.out_dir, f"{TAG}_{name}")
        text = report(name, src["scene"], path, results, target)
        open(stem + ".txt", "w").write(text)
        json.dump(dict(source=name, path=path, stream=args.stream, results=results, target=target,
                       summary={k: summary(v, target) for k, v in results.items()}),
                  open(stem + ".json", "w"), indent=1)
        L.save_sheet(stem + "_sheet.png", sheet, 0, max_cols=11)  # every 8th frame
        print(text, flush=True)
        done.append(name)
    print(f"done: {done}; outputs in {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
