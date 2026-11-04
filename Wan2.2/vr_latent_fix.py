"""
Visual Relay, round 9a: decode lab. VAE only, one GPU, about 10 minutes, no generation.

    python vr_latent_fix.py                     # every source in SOURCES whose file exists
    python vr_latent_fix.py --only fnr-CL-LF --gpu 0
    python vr_latent_fix.py --add sam-X=sam:r9_outputs/r9_JM-4_sam_latents.pt:11-12,11-13

Idea. The dark spot (FNR) and the bright bump (SAM) on Img2's frames are made when Img2's latent is decoded (F38,
F44), after the motion is settled. So correct the latents around Img2 after sampling, before decoding:
  rt      no correction: plain decode of saved latents, or the VAE round trip of a video; the control
  Lm      latents lo..hi: each channel's spatial mean moved onto the straight line between latents lo-1 and hi+1
  Lms     the same for each channel's mean and std
  P       the same correction on pixels, on the rt video: frames 4lo-3..4hi get each colour channel's mean and std
          on the line between frames 4lo-4 and 4hi+1 (a reference for a perfect colour fix)
A source is either a latent file saved by a round 9b run (<tag>_latents.pt, --vr_save_latents; stream 1 = the main
video: the real method) or an mp4, re-encoded (a stand-in while no latents are saved). Windows lo-hi are latent
indices; FNR's Img2 is latent 10 (frames 37-40), SAM's latent 12 (frames 45-48).
For sources marked jm (SAM): the joint Img2 latent (M1-PR's input) as it is, with the long latent's channel means,
and with its means and stds (JM-4's input), each put into the source's latents at Img2's latent and decoded: a
prediction for round 9b's JM-4 run.

Outputs in r9_outputs/: r9_fix_<source>_<variant>.mp4 (all frames), r9_fix_<source>.txt and .json (per-frame
brightness, saturation, contrast, sharpness and frame change around Img2), r9_fix_<source>_sheet.png.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = {
    "fnr-CL-LF": dict(scene="fnr", paths=["r9_outputs/r9_CL-LF_fnr_latents.pt", "r7_outputs/r7_CL-LF_fnr.mp4"],
                      windows=[(10, 10), (10, 11)]),
    "sam-M1-PR": dict(scene="sam", paths=["r9_outputs/r9_M1-PR_sam_latents.pt", "r4_outputs/r4_M1-PR_sam.mp4"],
                      windows=[(11, 12), (11, 13)], jm=True),
    "fnr-LG-LF": dict(scene="fnr", paths=["r8_outputs/r8_LG-LF_fnr.mp4"], windows=[(10, 11), (8, 11)]),
    "fnr-LGG-LF": dict(scene="fnr", paths=["r9_outputs/r9_LGG-LF_fnr_latents.pt"], windows=[(10, 10), (10, 11)]),
    "sam-JM-4": dict(scene="sam", paths=["r9_outputs/r9_JM-4_sam_latents.pt"], windows=[(11, 12), (11, 13)]),
}
SCENES = {"fnr": dict(images=["fnr1.png", "fnr2.png"], anchor=10),
          "sam": dict(images=["Img1.png", "Img2.png"], anchor=12)}
CKPT_DIR = "/home/ycwang/Visual-Relay/Wan2.2-I2V-A14B"
IMAGE_DIR = "/home/ycwang/Visual-Relay/"
OUT_DIR = "r9_outputs"
FPS = 16
STRIDE = 4


def load_probe():
    """vr_decoder_probe.py (round 8a), next to this file: image/video loading, splicing and measuring."""
    spec = importlib.util.spec_from_file_location("vr_decoder_probe", os.path.join(HERE, "vr_decoder_probe.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = load_probe()


# --- corrections --------------------------------------------------------------------------------
def window_frames(lo, hi):
    """First and last RGB frame of latents lo..hi."""
    return 1 + (lo - 1) * STRIDE, hi * STRIDE


def smooth_stats(z, lo, hi, mode):
    """z [C, T, h, w]. Latents lo..hi get each channel's spatial mean (mode 'mean') or mean and std ('meanstd') on
    the straight line between latents lo-1 and hi+1; latent j sits at (j - lo + 1) / (hi - lo + 2) of the way.
    The spatial pattern of each latent is kept."""
    if not 1 <= lo <= hi < z.shape[1] - 1:
        raise ValueError(f"window {lo}-{hi} needs a latent on each side (latents 0..{z.shape[1] - 1})")
    if mode not in ("mean", "meanstd"):
        raise ValueError(f"unknown mode {mode}")
    out = z.clone()
    m0, m1 = z[:, lo - 1].mean(dim=(1, 2)), z[:, hi + 1].mean(dim=(1, 2))
    s0, s1 = z[:, lo - 1].std(dim=(1, 2)), z[:, hi + 1].std(dim=(1, 2))
    n = hi - lo + 2
    for j in range(lo, hi + 1):
        t = (j - lo + 1) / n
        mean_t, std_t = m0 + t * (m1 - m0), s0 + t * (s1 - s0)
        cur = z[:, j]
        cur_mean = cur.mean(dim=(1, 2))[:, None, None]
        if mode == "mean":
            out[:, j] = cur - cur_mean + mean_t[:, None, None]
        else:
            cur_std = cur.std(dim=(1, 2)).clamp(min=1e-6)[:, None, None]
            out[:, j] = (cur - cur_mean) / cur_std * std_t[:, None, None] + mean_t[:, None, None]
    return out


def pixel_fix(video, f0, f1):
    """video [3, F, H, W] in [-1, 1]. Frames f0..f1 get each colour channel's mean and std (0-1 scale) on the line
    between frames f0-1 and f1+1, clipped to 0-1."""
    if not 1 <= f0 <= f1 < video.shape[1] - 1:
        raise ValueError(f"frames {f0}-{f1} need a frame on each side")
    x = (video.float() + 1) / 2
    out = x.clone()
    a, b = x[:, f0 - 1], x[:, f1 + 1]
    m0, m1 = a.mean(dim=(1, 2)), b.mean(dim=(1, 2))
    s0, s1 = a.std(dim=(1, 2)), b.std(dim=(1, 2))
    n = f1 - f0 + 2
    for f in range(f0, f1 + 1):
        t = (f - f0 + 1) / n
        mean_t, std_t = m0 + t * (m1 - m0), s0 + t * (s1 - s0)
        cur = x[:, f]
        cur_mean = cur.mean(dim=(1, 2))[:, None, None]
        cur_std = cur.std(dim=(1, 2)).clamp(min=1e-6)[:, None, None]
        out[:, f] = ((cur - cur_mean) / cur_std * std_t[:, None, None] + mean_t[:, None, None]).clamp(0, 1)
    return out * 2 - 1


def with_stats(latent, ref, mode):
    """latent [C, h, w] with each channel's spatial mean (and std) taken from ref [C, h, w]."""
    m, rm = latent.mean(dim=(1, 2))[:, None, None], ref.mean(dim=(1, 2))[:, None, None]
    if mode == "mean":
        return latent - m + rm
    s = latent.std(dim=(1, 2)).clamp(min=1e-6)[:, None, None]
    return (latent - m) / s * ref.std(dim=(1, 2))[:, None, None] + rm


def jm_candidates(encode, img1, img2, frames, a):
    """Joint Img2 latent (Img1, grey, Img2 x4 at latent a, grey) and the long latent (Img2 held from frame 0), and
    the joint latent with the long latent's means / means and stds."""
    s, e = window_frames(a, a)
    joint = torch.zeros(3, frames, *img2.shape[1:], dtype=img2.dtype, device=img2.device)
    joint[:, 0] = img1
    joint[:, s:e + 1] = img2[:, None]
    j = encode([joint])[0][:, a]
    long = encode([img2[:, None].expand(-1, e + 1, -1, -1).contiguous()])[0][:, a]
    return {"joint4": j, "joint4_m": with_stats(j, long, "mean"), "joint4_ms": with_stats(j, long, "meanstd"),
            "long4": long}


# --- measuring ----------------------------------------------------------------------------------
def frame_metrics(frame, prev=None):
    """Brightness, saturation (HSV S, 0-255), contrast (luma std), sharpness (Laplacian variance) and the mean
    absolute change from the previous frame, for one [H, W, 3] uint8 frame."""
    st = P.frame_stats(frame)
    f = frame.astype(np.float32)
    mx, mn = f.max(axis=2), f.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6) * 255.0, 0.0)
    luma = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    st.update(sat=float(sat.mean()), contrast=float(luma.std()),
              change=float(np.abs(f - prev.astype(np.float32)).mean()) if prev is not None else 0.0)
    return st


def measure(video_u8, first, last):
    """Metrics for frames first..last of [F, H, W, 3] uint8."""
    return {f: frame_metrics(video_u8[f], video_u8[f - 1] if f > 0 else None) for f in range(first, last + 1)}


def load_source(path, stream="stream1"):
    """('lat', latents [C, T, h, w]) from a .pt saved by --vr_save_latents, or ('vid', video [3, F, H, W])."""
    if path.endswith(".pt"):
        data = torch.load(path, map_location="cpu")
        if not isinstance(data, dict) or data.get(stream) is None:
            raise ValueError(f"{path} has no {stream} latents")
        return "lat", data[stream].float()
    return "vid", P.load_video(path)


def run_source(label, scene, kind, data, windows, encode, decode, images=None, jm=False, save=None):
    """All variants for one source. Returns (results, sheet frames, frame range). images: (img1, img2) at the video
    size (needed for jm). save(name, video[3, F, H, W]) writes a variant's video."""
    a = SCENES[scene]["anchor"]
    if kind == "lat":
        z = data
        base = decode([z])[0]
    else:
        z = encode([data.contiguous()])[0]
        base = decode([z])[0]
    lo_all, hi_all = min(w[0] for w in windows), max(w[1] for w in windows)
    first = max(1, window_frames(lo_all, lo_all)[0] - 8)
    last = min(base.shape[1] - 1, window_frames(hi_all, hi_all)[1] + 8)
    variants = {"rt": base}
    for lo, hi in windows:
        for mode, tag in (("mean", "Lm"), ("meanstd", "Lms")):
            variants[f"{tag}{lo}-{hi}"] = decode([smooth_stats(z, lo, hi, mode)])[0]
        f0, f1 = window_frames(lo, hi)
        variants[f"P{lo}-{hi}"] = pixel_fix(base, f0, f1)
    if kind == "vid":
        variants = dict(orig=data, **variants)
    results, sheet = {}, {}
    for name, video in variants.items():
        u8 = P.to_uint8(video)
        results[name] = measure(u8, first, last)
        sheet[name] = u8[first:last + 1]
        if save is not None and name != "orig":
            save(f"{label}_{name}", video)
    jm_rows = {}
    if jm:
        img1, img2 = images
        s, e = window_frames(a, a)
        for name, cand in jm_candidates(encode, img1, img2, base.shape[1], a).items():
            dec = decode([P.splice(z, a, cand)])[0]
            u8 = P.to_uint8(dec)
            jm_rows[name] = {f: frame_metrics(u8[f])["luma"] for f in range(s - 4, min(e + 4, u8.shape[0] - 1) + 1)}
    return results, sheet, (first, last), jm_rows


def report(label, scene, path, kind, results, frange, jm_rows, img2_luma):
    a = SCENES[scene]["anchor"]
    s, e = window_frames(a, a)
    first, last = frange
    head = (f"Visual Relay round 9a decode lab, {label}: {path} ({'saved latents' if kind == 'lat' else 're-encoded video'}"
            f"); Img2 is latent {a}, frames {s}-{e}; Img2 brightness {img2_luma:.0f}")
    lines = [head, "rt = no correction; Lm/Lms lo-hi = latents lo..hi with channel means (and stds) on the line between "
             "their neighbours; P lo-hi = the same on pixels.", ""]
    for key, fmt in (("luma", "%4.0f"), ("sat", "%4.0f"), ("contrast", "%4.0f"), ("sharp", "%5.0f"),
                     ("change", "%4.1f")):
        lines.append(f"{key} per frame {first}..{last}:")
        for name, rows in results.items():
            lines.append(f"  {name:10s} " + " ".join(fmt % rows[f][key] for f in range(first, last + 1)))
        lines.append("")
    if jm_rows:
        lines.append(f"JM check: Img2's latent replaced, brightness of frames {s - 4}..{e + 4}:")
        for name, rows in jm_rows.items():
            lines.append(f"  {name:10s} " + " ".join("%4.0f" % rows[f] for f in sorted(rows)))
        lines.append("")
    return "\n".join(lines) + "\n"


def save_sheet(path, sheet, first, width=150, max_cols=14):
    from PIL import Image, ImageDraw
    rows = []
    for name, frames in sheet.items():
        tiles = []
        step = max(1, int(np.ceil(len(frames) / max_cols)))
        for i in range(0, len(frames), step):
            im = Image.fromarray(frames[i])
            im = im.resize((width, round(im.height * width / im.width)))
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, 90, 11], fill="black")
            d.text((2, 0), f"{name} f{first + i}", fill="white")
            tiles.append(im)
        rows.append(tiles)
    tw, th = rows[0][0].size
    cols = max(len(r) for r in rows)
    canvas = Image.new("RGB", (cols * (tw + 2), len(rows) * (th + 2)), "white")
    for r, tiles in enumerate(rows):
        for c, im in enumerate(tiles):
            canvas.paste(im, (c * (tw + 2), r * (th + 2)))
    canvas.save(path)


def parse_add(spec):
    """name=scene:path:lo-hi,lo-hi -> (name, dict)."""
    name, rest = spec.split("=", 1)
    scene, path, wins = rest.split(":")
    windows = [tuple(int(v) for v in w.split("-")) for w in wins.split(",")]
    return name, dict(scene=scene, paths=[path], windows=windows, jm=scene == "sam")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default=None, help="comma-separated source names")
    p.add_argument("--add", action="append", default=[], help="name=scene:path:lo-hi[,lo-hi]")
    p.add_argument("--stream", default="stream1", choices=["stream1", "stream2"])
    p.add_argument("--ckpt_dir", default=CKPT_DIR)
    p.add_argument("--image_dir", default=IMAGE_DIR)
    p.add_argument("--gpu", type=int, default=0)
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
    os.makedirs(OUT_DIR, exist_ok=True)

    def encode(clips):
        return [z.float() for z in vae.encode([c.to(device) for c in clips])]

    def decode(latents):
        return [v.float().cpu() for v in vae.decode([z.to(device) for z in latents])]

    def save(name, video):
        if not args.no_videos:
            save_video(tensor=video[None], save_file=os.path.join(OUT_DIR, f"r9_fix_{name}.mp4"), fps=FPS, nrow=1,
                       normalize=True, value_range=(-1, 1))

    done = []
    for name, src in sources.items():
        path = next((q for q in src["paths"] if os.path.isfile(q)), None)
        if path is None:
            print(f"[{name}] skipped: none of {src['paths']} exists", flush=True)
            continue
        kind, data = load_source(path, args.stream)
        label = f"{name}-{kind}"
        size = (data.shape[2] * 8, data.shape[3] * 8) if kind == "lat" else tuple(data.shape[2:])
        sc = SCENES[src["scene"]]
        images = tuple(P.load_image(os.path.join(args.image_dir, n), size) for n in sc["images"])
        print(f"[{label}] {path}: windows {src['windows']} ...", flush=True)
        with torch.no_grad():
            results, sheet, frange, jm_rows = run_source(label, src["scene"], kind, data, src["windows"], encode,
                                                         decode, images=images, jm=src.get("jm", False), save=save)
        img2_luma = P.frame_stats(P.to_uint8(images[1][:, None])[0])["luma"]
        stem = os.path.join(OUT_DIR, f"r9_fix_{label}")
        text = report(label, src["scene"], path, kind, results, frange, jm_rows, img2_luma)
        open(stem + ".txt", "w").write(text)
        json.dump(dict(source=name, path=path, kind=kind, frames=list(frange), results=results, jm=jm_rows),
                  open(stem + ".json", "w"), indent=1)
        save_sheet(stem + "_sheet.png", sheet, frange[0])
        print(text, flush=True)
        done.append(label)
    print(f"done: {done}; outputs in {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
