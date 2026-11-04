#!/usr/bin/env python
"""Plots and a compact cross-run table from Visual Relay round-1 probe files.

    python analyze_probe_r1.py r1_outputs/r1_BASE_sam_probe.pt   # one run -> PNG next to it
    python analyze_probe_r1.py --compare r1_outputs               # all *_probe.pt -> r1_probe_compare.txt (+ PNGs)

The compare table is the short, paste-able version of the evidence:
  kappa  = attention mass a prefix frame puts on Img1 (frame 0)
  anchor = mass it puts on the Img2 anchor
  lambda = mean mass on each temporal neighbour (t-1, t+1)
  ell    = sqrt(lambda / kappa): decay length (in latents) of Img2's influence
           in the screened-diffusion toy model; small ell = snap at the end.
All numbers are averaged over the sampled positions, every head and layer.
Runs on CPU; matplotlib is optional (text output is always written).
"""
import argparse
import glob
import math
import os

import torch


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def last_high_step(meta):
    ts, b = meta.get('timesteps'), meta.get('boundary_timestep')
    steps = meta['steps']
    if ts is None or b is None:
        return steps[0]
    high = [s for s in steps if s < len(ts) and ts[s] >= b]
    return high[-1] if high else steps[0]


def mid_low_step(meta):
    ts, b = meta.get('timesteps'), meta.get('boundary_timestep')
    steps = meta['steps']
    low = [s for s in steps if ts is not None and s < len(ts) and ts[s] < b]
    if not low:
        return steps[-1]
    route = meta.get('route_active', {})
    active = [s for s in low if route.get(s, True)]
    pool = active or low
    return pool[len(pool) // 2]


def override_profile(merged, name='diff_rms'):
    """Method 1: [S, F] layer- and head-averaged per-frame statistic."""
    return torch.nanmean(merged['tensors'][name], dim=(1, 3))


def averaged(merged):
    """Layer- and head-averaged [S, Q, K] matrix of the main query frames."""
    tensors = merged['tensors']
    name = 'prefix_eff' if merged['meta'].get('mode') == 'paired' else 'plain'
    return torch.nanmean(tensors[name], dim=(1, 4))


def neighbour_mass(mat_row, t, a=None, frames=None):
    vals = []
    if t - 1 >= 1:
        vals.append(float(mat_row[t - 1]))
    if a is None:
        if t + 1 < frames:
            vals.append(float(mat_row[t + 1]))
    elif t + 1 <= a - 1:
        vals.append(float(mat_row[t + 1]))
    return sum(vals) / len(vals) if vals else float('nan')


def run_name(path):
    base = os.path.basename(path)
    return base[:-len('_probe.pt')] if base.endswith('_probe.pt') else base


def compare(paths, out_txt):
    lines = []
    out = lines.append
    out("Visual Relay probe comparison (layer/head averaged).")
    out("kappa = mass on Img1, anchor = mass on the Img2 anchor, lambda = "
        "mean mass on one temporal neighbour, ell = sqrt(lambda/kappa).")
    out('')
    for path in sorted(paths):
        merged = load(path)
        meta = merged['meta']
        steps = meta['steps']
        mode = meta.get('mode')
        if mode == 'override':
            ov = meta.get('override') or {}
            extra = ''
            if ov.get('mode') == 'route':
                extra += ', route mode (O1 = attn(q2, k2, v1))'
            if ov.get('rerender_from_step') is not None:
                extra += (f", re-rendered from step {ov['rerender_from_step']}"
                          " (probe = the plan)")
            out(f"== {run_name(path)}  (override: stream 2 {ov.get('stream')}, "
                f"min weight {ov.get('min_weight')}, radius {ov.get('radius')}, "
                f"{ov.get('blend')} blend, {ov.get('latent')} latent{extra})")
            if meta.get('probe_errors'):
                out(f"   PROBE ERRORS: {meta['probe_errors']}")
            a = int(meta.get('anchor_index', -1))
            diff = override_profile(merged)
            frames = diff.shape[1]
            out("   RMS(O1 - O2) per latent frame (* = Img2), mean over layers "
                "and heads")
            out("   step " + ''.join(f"{('*' + str(f)) if f == a else f:>6}"
                                    for f in range(frames)))
            for si, step in enumerate(steps):
                out(f"   {step:4d} " + ''.join(
                    f"{float(diff[si, f]):6.3f}" for f in range(frames)))
            out('')
            continue
        mat = averaged(merged)
        chosen = [steps[0], last_high_step(meta), mid_low_step(meta)]
        chosen = list(dict.fromkeys(chosen))
        extras = []
        kg = meta.get('keyframe_guidance')
        if kg and kg[0]:
            extras.append(f"KG k={kg[0]} until {kg[1]}")
        rev = meta.get('reverse_stream')
        if rev:
            extras.append(f"Stream R {rev['mode']} s={rev['strength']} "
                          f"{rev['shape']} until {rev['until_step']}")
        if meta.get('suffix_prompt'):
            extras.append("Stream-B prompt")
        out(f"== {run_name(path)}  ({mode}; variant {meta.get('variant')}"
            + ''.join(f"; {e}" for e in extras) + ")")
        if rev and rev['mode'] == 'attn':
            out("   (Stream R mixes prefix attention outputs after the softmax; "
                "the masses below are the main stream's own.)")
        if meta.get('probe_errors'):
            out(f"   PROBE ERRORS: {meta['probe_errors']}")
        if mode == 'paired':
            a = int(meta['anchor_index'])
            probe_t = sorted({1, max(1, a // 2), a - 1})
            head = ''.join(f"  kap@{t:<2d} anc@{t:<2d}" for t in probe_t)
            out(f"   step  t     {head}   lambda@{a // 2}  ell@{a // 2}")
            for step in chosen:
                si = steps.index(step)
                row = mat[si]
                cells = ''.join(
                    f"  {float(row[t, 0]):6.3f} {float(row[t, a]):6.3f}"
                    for t in probe_t)
                mid = max(1, a // 2)
                lam = neighbour_mass(row[mid], mid, a=a)
                kap = float(row[mid, 0])
                ell = math.sqrt(lam / kap) if kap > 0 and lam == lam else float('nan')
                ts = meta.get('timesteps', [None] * (step + 1))[step]
                out(f"   {step:4d}  {ts!s:>4} {cells}   {lam:8.3f}  {ell:6.2f}")
        else:
            frames = mat.shape[1]
            probe_t = sorted({1, 3, frames // 2, frames - 1})
            out("   step  t    " + ''.join(f"  kap@{t:<3d}" for t in probe_t)
                + f"  lambda@{frames // 2}  ell@{frames // 2}")
            for step in chosen:
                si = steps.index(step)
                row = mat[si]
                mid = frames // 2
                lam = neighbour_mass(row[mid], mid, frames=frames)
                kap = float(row[mid, 0])
                ell = math.sqrt(lam / kap) if kap > 0 else float('nan')
                ts = meta.get('timesteps', [None] * (step + 1))[step]
                out(f"   {step:4d}  {ts!s:>4}" + ''.join(
                    f"  {float(row[t, 0]):8.3f}" for t in probe_t)
                    + f"  {lam:9.3f}  {ell:6.2f}")
        out('')
    with open(out_txt, 'w') as handle:
        handle.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f"written {out_txt}")


def plot_run(path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as error:  # noqa: BLE001
        print(f"matplotlib unavailable ({error}); skipping plots")
        return
    merged = load(path)
    meta = merged['meta']
    steps = meta['steps']
    if meta.get('mode') == 'override':
        plot_override(path, merged, plt)
        return
    mat = averaged(merged)
    paired = meta.get('mode') == 'paired'
    a = int(meta['anchor_index']) if paired else None
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    cmap = plt.get_cmap('viridis')
    q_frames = list(range(1, mat.shape[1]))
    for i, step in enumerate(steps):
        colour = cmap(i / max(1, len(steps) - 1))
        si = steps.index(step)
        axes[0].plot(q_frames, [float(mat[si, t, 0]) for t in q_frames],
                     color=colour, label=f"step {step}")
        if paired:
            axes[0].plot(q_frames, [float(mat[si, t, a]) for t in q_frames],
                         color=colour, linestyle='--')
    axes[0].set_xlabel('query latent frame t (prefix)' if paired else
                       'query latent frame t')
    axes[0].set_ylabel('attention mass')
    axes[0].set_title('Img1 (solid)' + (' vs Img2 anchor (dashed)' if paired
                                        else ''))
    axes[0].legend(fontsize=7, ncol=2)
    step = last_high_step(meta)
    im = axes[1].imshow(mat[steps.index(step)].numpy(), aspect='auto',
                        cmap='magma', vmin=0)
    axes[1].set_title(f"query frame x key frame mass, step {step}")
    axes[1].set_xlabel('key working frame' + (f' (anchor = {a})' if paired
                                              else ''))
    axes[1].set_ylabel('query frame')
    fig.colorbar(im, ax=axes[1])
    fig.suptitle(run_name(path))
    fig.tight_layout()
    png = os.path.splitext(path)[0] + '.png'
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f"written {png}")


def plot_override(path, merged, plt):
    """Gordon's plot per probed step, plus RMS(O1) and RMS(O2)."""
    meta = merged['meta']
    steps = meta['steps']
    a = int(meta.get('anchor_index', -1))
    diff = override_profile(merged)
    o1 = override_profile(merged, 'o1_rms')
    o2 = override_profile(merged, 'o2_rms')
    frames = list(range(diff.shape[1]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    cmap = plt.get_cmap('viridis')
    for i, step in enumerate(steps):
        colour = cmap(i / max(1, len(steps) - 1))
        axes[0].plot(frames, diff[i].tolist(), color=colour,
                     label=f"step {step}")
        axes[1].plot(frames, o1[i].tolist(), color=colour)
        axes[1].plot(frames, o2[i].tolist(), color=colour, linestyle='--')
    for ax in axes:
        ax.axvline(a, color='grey', linewidth=0.8, linestyle=':')
        ax.set_xlabel('latent frame (dotted: Img2)')
    axes[0].set_ylabel('RMS(O1 - O2)')
    axes[0].set_title('difference of the self-attention outputs')
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_ylabel('RMS')
    axes[1].set_title('RMS(O1) solid, RMS(O2) dashed')
    fig.suptitle(run_name(path))
    fig.tight_layout()
    png = os.path.splitext(path)[0] + '.png'
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f"written {png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='*')
    parser.add_argument('--compare', default=None,
                        help='folder with *_probe.pt files')
    args = parser.parse_args()
    if args.compare:
        paths = sorted(glob.glob(os.path.join(args.compare, '*_probe.pt')))
        if not paths:
            print(f"no probe files in {args.compare}")
            return
        prefix = os.path.basename(os.path.normpath(args.compare)).replace(
            '_outputs', '')
        compare(paths, os.path.join(args.compare,
                                    f'{prefix}_probe_compare.txt'))
        for path in paths:
            if not os.path.exists(os.path.splitext(path)[0] + '.png'):
                plot_run(path)
    for path in args.paths:
        plot_run(path)


if __name__ == '__main__':
    main()
