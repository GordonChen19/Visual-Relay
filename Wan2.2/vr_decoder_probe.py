"""
Visual Relay, round 8a: decoder probe. VAE only, one GPU, a few minutes, no generation.

    python vr_decoder_probe.py                       # both scenes, default context videos
    python vr_decoder_probe.py --scenes fnr --gpu 0
    python vr_decoder_probe.py --context fnr=r6_outputs/r6_PR-CL_fnr.mp4

Question (H11): why do Img2's frames come out darker when Img2's latent was encoded on its own? Round 6's PR-RR14
forced that latent exactly and still decoded frames 37-40 darker (150, 143, 140, 137 against Img2's 152), so the dip
is made when the latent is decoded inside the video. Two candidate causes:
  (a) clip start: the latent was encoded at the start of its own short clip, where the causal VAE has no history;
  (b) history mismatch: it was encoded after Img2 (or grey frames), but is decoded after the video's own frames.

Method. The context video (an existing generated video) is encoded into its 21 latents. For each candidate Img2
latent, latent a (FNR 10, SAM 12) is replaced by the candidate and latents 0..a+1 are decoded; frames 4a-7..4a+4
(FNR 33-44) are measured. Each candidate is also decoded inside the clip it was encoded from ("own clip"), which
shows whether the latent itself is dark or only becomes dark in the video's context. Candidates:
  video      the context video's own latent a (round-trip control)
  joint4     joint encoding, Img2 on all 4 frames (M1-PR's input)          jointLF  same, Img2 on frame 4a only (PR-LF)
  alone4     Img2 held 5 frames, encoded alone, latent 1 (PR-CL)           aloneLF  [grey x4, Img2] alone (CL-LF)
  first      Img2 encoded as a first frame (PR-FE)
  long4      Img2 held for the whole clip, latent a (history: Img2 for 36 frames, no clip start)
  longLF     Img2 held to frame 4a-4, grey on 4a-3..4a-1, Img2 on 4a
  ctx4       the video's own frames 0..4a-4, then Img2 x4, latent a         ctxLF    video 0..4a-4, grey x3, Img2
  ctxLFv     the video's own frames 0..4a-1, then Img2 on 4a               ctxLFh   video 0..4a-4, frame 4a-4 x3, Img2
  alone4_b, aloneLF_b   as alone4 / aloneLF with Img2 10% brighter (--gain)
  alone4_m   alone4 with each channel's spatial mean moved to the video's latent a-1 (a single-pass latent fix)
If (a) is the cause, long4 should not dip; if (b), long4 dips but ctx4 does not.

Outputs in r8_outputs/: r8_decoder_probe_<scene>.txt (tables), .json (all numbers), _sheet.png (frames 4a-5..4a+2
per candidate).
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

SCENES = {
    "fnr": dict(images=["fnr1.png", "fnr2.png"], anchor=10,
                context=["r7_outputs/r7_CL-LF_fnr.mp4", "r6_outputs/r6_PR-CL_fnr.mp4"]),
    "sam": dict(images=["Img1.png", "Img2.png"], anchor=12,
                context=["r4_outputs/r4_M1-PR_sam.mp4", "r7_outputs/r7_PR-MV_sam.mp4"]),
}
CKPT_DIR = "/home/ycwang/Visual-Relay/Wan2.2-I2V-A14B"
IMAGE_DIR = "/home/ycwang/Visual-Relay/"
OUT_DIR = "r8_outputs"
MAX_AREA = 832 * 480
STRIDE = 4  # frames per latent (time)


# --- inputs -------------------------------------------------------------------------------------
def working_size(height, width, max_area=MAX_AREA, vae_stride=8, patch=2):
    """The pipeline's working size for an image of this shape (WanI2V.generate)."""
    aspect = height / width
    lat_h = round(np.sqrt(max_area * aspect) // vae_stride // patch * patch)
    lat_w = round(np.sqrt(max_area / aspect) // vae_stride // patch * patch)
    return lat_h * vae_stride, lat_w * vae_stride


def load_image(path, size=None):
    """[3, h, w] in [-1, 1], resized exactly as the pipeline does (bicubic, on the CPU)."""
    import torchvision.transforms.functional as TF
    from PIL import Image
    img = Image.open(path).convert("RGB")
    x = TF.to_tensor(img).sub_(0.5).div_(0.5)
    if size is None:
        size = working_size(x.shape[1], x.shape[2])
    return torch.nn.functional.interpolate(x[None].cpu(), size=size, mode="bicubic")[0]


def load_video(path):
    """[3, F, H, W] in [-1, 1] from an mp4."""
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im[:, :, ::-1].copy())
    if not frames:
        raise RuntimeError(f"no frames read from {path}")
    arr = np.stack(frames).astype(np.float32) / 127.5 - 1.0  # [F, H, W, 3]
    return torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()


def brighten(img, gain):
    """Scale an image in [0, 1] space by gain, clipped."""
    return ((img + 1) / 2 * gain).clamp(0, 1) * 2 - 1


# --- candidates ---------------------------------------------------------------------------------
def anchor_frames(a):
    """First and last RGB frame of latent a."""
    return 1 + (a - 1) * STRIDE, a * STRIDE


def build_candidates(encode, img1, img2, video, a, gain=1.1, prev_latent=None):
    """Candidate Img2 latents for slot a. encode: list of [3, n, H, W] -> list of [C, L, h, w].
    Returns {name: dict(latent=[C, h, w], clip=[C, L, h, w], index=int)} where clip is the latent sequence the
    candidate was taken from (for the own-clip decode) and index its position in it."""
    s, e = anchor_frames(a)
    F = video.shape[1]
    grey = torch.zeros_like(img2)

    def stack(frames):
        return torch.stack(frames, dim=1)

    def take(clip, index):
        z = encode([clip.contiguous()])[0]
        return dict(latent=z[:, index].clone(), clip=z[:, :index + 1].clone(), index=index)

    out = {}
    joint = torch.zeros(3, F, *img2.shape[1:], dtype=img2.dtype, device=img2.device)
    joint[:, 0] = img1
    j4 = joint.clone()
    j4[:, s:e + 1] = img2[:, None]
    out["joint4"] = take(j4, a)
    jl = joint.clone()
    jl[:, e] = img2
    out["jointLF"] = take(jl, a)
    out["alone4"] = take(stack([img2] * 5), 1)
    out["aloneLF"] = take(stack([grey] * 4 + [img2]), 1)
    out["first"] = take(img2[:, None], 0)
    out["long4"] = take(stack([img2] * F), a)
    out["longLF"] = take(stack([img2] * s + [grey] * 3 + [img2]), a)
    ctx = [video[:, i] for i in range(s)]
    out["ctx4"] = take(stack(ctx + [img2] * 4), a)
    out["ctxLF"] = take(stack(ctx + [grey] * 3 + [img2]), a)
    out["ctxLFv"] = take(stack([video[:, i] for i in range(e)] + [img2]), a)
    out["ctxLFh"] = take(stack(ctx + [video[:, s - 1]] * 3 + [img2]), a)
    bright = brighten(img2, gain)
    out["alone4_b"] = take(stack([bright] * 5), 1)
    out["aloneLF_b"] = take(stack([grey] * 4 + [bright]), 1)
    if prev_latent is not None:
        lat = out["alone4"]["latent"]
        moved = lat - lat.mean(dim=(1, 2), keepdim=True) + prev_latent.mean(dim=(1, 2), keepdim=True)
        out["alone4_m"] = dict(latent=moved, clip=None, index=None)
    return out


def splice(context_latents, a, latent):
    """Latents 0..a+1 of the context with latent a replaced (frames up to 4a+4 decode from these)."""
    z = context_latents[:, :a + 2].clone()
    z[:, a] = latent.to(z.dtype)
    return z


# --- measurement --------------------------------------------------------------------------------
def to_uint8(frames):
    """[3, N, H, W] in [-1, 1] -> [N, H, W, 3] uint8."""
    x = ((frames.float().clamp(-1, 1) + 1) * 127.5).round()
    return x.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)


def frame_stats(frame):
    """Brightness (BT.601 luma), channel means and sharpness (Laplacian variance) of one [H, W, 3] uint8 frame."""
    f = frame.astype(np.float32)
    luma = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    lap = (-4 * luma[1:-1, 1:-1] + luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:])
    return dict(luma=float(luma.mean()), r=float(f[..., 0].mean()), g=float(f[..., 1].mean()),
                b=float(f[..., 2].mean()), sharp=float(lap.var()))


def rms(a, b):
    return float(np.sqrt(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean()))


def summarise(frames_u8, first_frame, a, img2_u8):
    """Per-frame stats of decoded frames first_frame.. and the dip of Img2's latent (frames 4a-3..4a) against the
    4 frames before it."""
    s, e = anchor_frames(a)
    rows = {first_frame + i: frame_stats(fr) for i, fr in enumerate(frames_u8)}
    before = [rows[f] for f in range(s - 4, s) if f in rows]
    ref = {k: float(np.mean([r[k] for r in before])) for k in ("luma", "r", "g", "b")}
    last = rows[e]
    return dict(
        luma={f: r["luma"] for f, r in rows.items()},
        sharp={f: r["sharp"] for f, r in rows.items()},
        ref_luma=ref["luma"],
        dip=[rows[f]["luma"] - ref["luma"] for f in range(s, e + 1)],
        rgb_at_last=[last["r"] / max(ref["r"], 1e-6), last["g"] / max(ref["g"], 1e-6), last["b"] / max(ref["b"], 1e-6)],
        rms_img2_first=rms(frames_u8[s - first_frame], img2_u8),
        rms_img2_last=rms(frames_u8[e - first_frame], img2_u8),
    )


def latent_stats(latent, prev, cur):
    """Distance of a candidate's per-channel spatial mean and std from the context latents a-1 and a."""
    m, sd = latent.mean(dim=(1, 2)), latent.std(dim=(1, 2))

    def dist(x, y):
        return float((x - y).pow(2).sum().sqrt())
    return dict(mean_vs_prev=dist(m, prev.mean(dim=(1, 2))), mean_vs_video=dist(m, cur.mean(dim=(1, 2))),
                std_ratio_prev=float((sd / prev.std(dim=(1, 2)).clamp_min(1e-6)).mean()))


def probe_scene(encode, decode, img1, img2, video, a, gain=1.1):
    """All candidates for one scene. Returns (results dict, decoded frames per candidate for the sheet)."""
    s, e = anchor_frames(a)
    first_frame = s - 4
    zc = encode([video.contiguous()])[0]
    img2_u8 = to_uint8(img2[:, None])[0]
    cands = build_candidates(encode, img1, img2, video, a, gain=gain, prev_latent=zc[:, a - 1])
    cands = dict(video=dict(latent=zc[:, a].clone(), clip=zc[:, :a + 1].clone(), index=a), **cands)
    results, sheet = {}, {}
    for name, c in cands.items():
        frames = decode([splice(zc, a, c["latent"])])[0]
        part = to_uint8(frames[:, first_frame:e + 5])
        res = summarise(part, first_frame, a, img2_u8)
        res["latent"] = latent_stats(c["latent"].float(), zc[:, a - 1].float(), zc[:, a].float())
        if c["clip"] is not None:
            own = to_uint8(decode([c["clip"]])[0])
            own_frames = own[-1:] if c["index"] == 0 else own[-STRIDE:]
            res["own_clip_luma"] = [frame_stats(f)["luma"] for f in own_frames]
        results[name] = res
        sheet[name] = part
    return results, sheet


def report(scene, a, results, context_path, img2_luma):
    s, e = anchor_frames(a)
    lines = [f"Visual Relay round 8a decoder probe, {scene}: Img2's latent {a} (frames {s}-{e}), context "
             f"{context_path}; Img2 brightness {img2_luma:.0f}",
             "Each candidate replaces latent a of the context video's latents; latents 0..a+1 are decoded.",
             "dip = brightness of frames "
             f"{s}-{e} minus the mean of frames {s - 4}-{s - 1}; RGB = R, G, B means at frame {e} over frames "
             f"{s - 4}-{s - 1}; own clip = brightness of the latent's frames decoded inside the clip it came from.", "",
             f"{'candidate':10s} | {'dip ' + str(s) + '..' + str(e):>22s} | {'RGB at ' + str(e):>16s} | "
             f"{'sharp ' + str(s) + '..' + str(e):>22s} | {'own clip':>18s} | RMS to Img2 {s}/{e} | "
             f"latent mean vs a-1 / vs video"]
    for name, r in results.items():
        own = " ".join("%.0f" % v for v in r.get("own_clip_luma", [])) or "-"
        dip = " ".join("%+5.0f" % d for d in r["dip"])
        rgb = " ".join("%.2f" % v for v in r["rgb_at_last"])
        sharp = " ".join("%5.0f" % r["sharp"][f] for f in range(s, e + 1))
        lat = r["latent"]
        lines.append("%-10s | %22s | %16s | %22s | %18s | %5.1f / %5.1f | %.2f / %.2f" % (
            name, dip, rgb, sharp, own, r["rms_img2_first"], r["rms_img2_last"], lat["mean_vs_prev"],
            lat["mean_vs_video"]))
    lines += ["", f"brightness per frame {s - 4}..{e + 4}:"]
    for name, r in results.items():
        lines.append(f"{name:10s} " + ' '.join(f"{r['luma'][f]:4.0f}" for f in sorted(r['luma'])))
    lines += ["", "Reading: if long4 does not dip, the clip start (a) is the cause; if long4 dips but ctx4 does not, "
              "the history mismatch (b) is. A candidate whose own-clip brightness matches Img2 but dips in the video is "
              "fine by itself and goes wrong only in the video's context."]
    return "\n".join(lines) + "\n"


def save_sheet(path, sheet, first_frame, columns=8):
    from PIL import Image, ImageDraw
    tiles = []
    for name, frames in sheet.items():
        row = []
        for i in range(min(columns, len(frames))):
            im = Image.fromarray(frames[i + 2] if i + 2 < len(frames) else frames[-1])
            w = 200
            im = im.resize((w, round(im.height * w / im.width)))
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, 96, 11], fill="black")
            d.text((2, 0), f"{name} f{first_frame + i + 2}", fill="white")
            row.append(im)
        tiles.append(row)
    tw, th = tiles[0][0].size
    canvas = Image.new("RGB", (columns * (tw + 2), len(tiles) * (th + 2)), "white")
    for r, row in enumerate(tiles):
        for c, im in enumerate(row):
            canvas.paste(im, (c * (tw + 2), r * (th + 2)))
    canvas.save(path)


# --- main ---------------------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", default="fnr,sam")
    p.add_argument("--context", action="append", default=[], help="scene=path.mp4 (default: see SCENES)")
    p.add_argument("--ckpt_dir", default=CKPT_DIR)
    p.add_argument("--image_dir", default=IMAGE_DIR)
    p.add_argument("--gain", type=float, default=1.1)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args(argv)
    overrides = dict(c.split("=", 1) for c in args.context)
    sys.path.insert(0, os.getcwd())
    from wan.configs import WAN_CONFIGS
    from wan.modules.vae2_1 import Wan2_1_VAE
    device = torch.device(f"cuda:{args.gpu}")
    cfg = WAN_CONFIGS["i2v-A14B"]
    vae = Wan2_1_VAE(vae_pth=os.path.join(args.ckpt_dir, cfg.vae_checkpoint), device=device)
    os.makedirs(OUT_DIR, exist_ok=True)

    def encode(clips):
        return [z.float() for z in vae.encode([c.to(device) for c in clips])]

    def decode(latents):
        return [v.float().cpu() for v in vae.decode([z.to(device) for z in latents])]

    for scene in [s.strip() for s in args.scenes.split(",") if s.strip()]:
        sc = SCENES[scene]
        paths = [overrides[scene]] if scene in overrides else sc["context"]
        context = next((q for q in paths if os.path.isfile(q)), None)
        if context is None:
            print(f"[{scene}] skipped: no context video among {paths}", flush=True)
            continue
        video = load_video(context)
        size = tuple(video.shape[2:])
        img1, img2 = (load_image(os.path.join(args.image_dir, n), size) for n in sc["images"])
        a = sc["anchor"]
        if video.shape[1] < anchor_frames(a)[1] + 5:
            raise RuntimeError(f"{context} has {video.shape[1]} frames; need at least {anchor_frames(a)[1] + 5}")
        print(f"[{scene}] context {context} {tuple(video.shape)}; probing Img2's latent {a} ...", flush=True)
        with torch.no_grad():
            results, sheet = probe_scene(encode, decode, img1, img2, video, a, gain=args.gain)
        img2_luma = frame_stats(to_uint8(img2[:, None])[0])["luma"]
        stem = os.path.join(OUT_DIR, f"r8_decoder_probe_{scene}")
        text = report(scene, a, results, context, img2_luma)
        open(stem + ".txt", "w").write(text)
        json.dump(dict(scene=scene, anchor=a, context=context, gain=args.gain, img2_luma=img2_luma,
                       results=results), open(stem + ".json", "w"), indent=1)
        save_sheet(stem + "_sheet.png", sheet, anchor_frames(a)[0] - 4)
        print(text, flush=True)
        print(f"[{scene}] wrote {stem}.txt, .json, _sheet.png", flush=True)


if __name__ == "__main__":
    main()
