# Visual Relay round 1: read-only attention-mass probe.
"""Measure how much softmax attention each query latent frame puts on each key
latent frame, at a few chosen denoising steps (conditional CFG pass only).

The probe never writes into a tensor the model uses. It recomputes float32
logits for a fixed sample of spatial positions per query frame, so a probed
run produces the same video as an unprobed one. Any probe error disables the
probe and the generation carries on.

Output (rank 0): ``<path>`` (torch file) and ``<path stem>_summary.txt``.
"""
import logging
import math
import os

import torch
import torch.distributed as dist

__all__ = ['ATTN_PROBE', 'AttentionProbe', 'write_probe_summary']


class AttentionProbe:

    head_chunk = 4

    def __init__(self):
        self.reset()

    def reset(self):
        self.enabled = False
        self.steps = frozenset()
        self.samples = 48
        self.seed = 1234
        self.record = False
        self.step = -1
        self.layer = 0
        self.data = {}
        self.meta = {}
        self.route_active = {}
        self.head_offset = 0
        self.sp = False
        self.failed = None
        self._index_cache = {}

    def configure(self, steps, samples=48, seed=1234):
        self.reset()
        self.steps = frozenset(int(s) for s in steps)
        self.samples = max(1, int(samples))
        self.seed = int(seed)
        self.enabled = bool(self.steps)

    def begin_forward(self, step, branch):
        """Call before every DiT forward. Only 'cond' passes are recorded."""
        self.layer = 0
        self.step = int(step)
        self.record = (self.enabled and self.failed is None
                       and branch == 'cond' and self.step in self.steps)

    def next_layer(self):
        layer = self.layer
        self.layer += 1
        return layer

    def spatial_index(self, tokens_per_frame, device):
        """Fixed spatial positions sampled once per run (same for every frame)."""
        key = (int(tokens_per_frame), str(device))
        if key not in self._index_cache:
            generator = torch.Generator(device='cpu')
            generator.manual_seed(self.seed)
            count = min(self.samples, int(tokens_per_frame))
            index = torch.randperm(
                int(tokens_per_frame), generator=generator)[:count]
            self._index_cache[key] = torch.sort(index).values.to(device)
        return self._index_cache[key]

    def store(self, layer, tensors, head_offset, sp, meta):
        entry = self.data.setdefault((self.step, int(layer)), {})
        for name, value in tensors.items():
            entry[name] = value.detach().float().cpu()
        self.head_offset = int(head_offset)
        self.sp = bool(sp)
        if 'route_active' in meta:
            self.route_active[self.step] = bool(meta['route_active'])
        for key, value in meta.items():
            if key != 'route_active':
                self.meta.setdefault(key, value)

    def run(self, fn, *args, **kwargs):
        """Run one probe computation; an error disables the probe only."""
        try:
            with torch.no_grad():
                fn(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 - never stop a generation
            self.failed = f"step {self.step}: {error!r}"
            self.record = False
            logging.warning("Attention probe disabled after an error: %s",
                            self.failed)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def finalize(self, path, extra_meta=None):
        """Merge all ranks' records into ``path``; every rank must call this."""
        if not self.enabled or not path:
            self.reset()
            return None
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        torch.save(
            dict(data=self.data, meta=self.meta, route_active=self.route_active,
                 head_offset=self.head_offset, sp=self.sp, failed=self.failed),
            f"{path}.rank{rank}")
        if dist.is_initialized():
            dist.barrier()
        summary_path = None
        if rank == 0:
            try:
                parts = [
                    torch.load(f"{path}.rank{r}", map_location='cpu',
                               weights_only=False) for r in range(world)
                ]
                merged = _merge_parts(parts, extra_meta or {})
                torch.save(merged, path)
                summary_path = os.path.splitext(path)[0] + '_summary.txt'
                write_probe_summary(merged, summary_path,
                                    source=os.path.basename(path))
                logging.info("Attention probe written to %s and %s", path,
                             summary_path)
            except Exception as error:  # noqa: BLE001
                logging.warning("Attention probe merge failed: %r", error)
            finally:
                for r in range(world):
                    try:
                        os.remove(f"{path}.rank{r}")
                    except OSError:
                        pass
        if dist.is_initialized():
            dist.barrier()
        self.reset()
        return summary_path


ATTN_PROBE = AttentionProbe()


def _merge_parts(parts, extra_meta):
    """Concatenate head shards (Ulysses ranks) into [steps, layers, ..., heads]."""
    if not any(part['sp'] for part in parts):
        parts = parts[:1]  # without sequence parallel every rank holds all heads
    parts = sorted(parts, key=lambda part: part['head_offset'])
    keys = set(parts[0]['data'])
    for part in parts[1:]:
        keys &= set(part['data'])
    steps = sorted({step for step, _ in keys})
    layers = sorted({layer for _, layer in keys})
    names = sorted({name for key in keys for name in parts[0]['data'][key]})
    tensors = {}
    for name in names:
        shape = None
        for key in keys:
            if all(name in part['data'][key] for part in parts):
                shape = torch.cat([part['data'][key][name] for part in parts],
                                  dim=-1).shape
                break
        if shape is None:
            continue
        out = torch.full((len(steps), len(layers), *shape), float('nan'))
        for step, layer in keys:
            if all(name in part['data'][(step, layer)] for part in parts):
                out[steps.index(step), layers.index(layer)] = torch.cat(
                    [part['data'][(step, layer)][name] for part in parts],
                    dim=-1)
        tensors[name] = out
    meta = dict(parts[0]['meta'])
    meta.update(extra_meta)
    meta['steps'] = steps
    meta['layers'] = layers
    route = {}
    for part in parts:
        route.update(part['route_active'])
    meta['route_active'] = route
    meta['probe_errors'] = [part['failed'] for part in parts if part['failed']]
    if tensors:
        meta['heads'] = int(next(iter(tensors.values())).shape[-1])
    return dict(meta=meta, tensors=tensors)


# --------------------------------------------------------------------------
# Text summary
# --------------------------------------------------------------------------
def _fmt(value, width=7):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ' ' * (width - 1) + '-'
    return f"{value:{width}.3f}"


def _step_label(meta, step):
    timesteps = meta.get('timesteps')
    boundary = meta.get('boundary_timestep')
    parts = [f"step {step}"]
    if timesteps is not None and step < len(timesteps):
        t = timesteps[step]
        parts.append(f"t={t}")
        if boundary is not None:
            parts.append('high-noise expert' if t >= boundary else
                         'low-noise expert')
    route = meta.get('route_active', {}).get(step)
    if route is not None and meta.get('mode') == 'paired':
        parts.append('anchor route ON' if route else 'anchor route OFF')
    return ', '.join(parts)


def _prefix_groups(t, a, frames):
    """Disjoint key-frame groups for prefix query frame t (working indices)."""
    prev = [t - 1] if t - 1 >= 1 else []
    if t + 1 <= a - 1:
        nxt = [t + 1]
    elif a + 1 < frames:
        nxt = [a + 1]
    else:
        nxt = []
    taken = set([0, a] + ([t] if t > 0 else []) + prev + nxt)
    return dict(
        Img1=[0],
        anchor=[a],
        self=[t] if t > 0 else [],
        prev=prev,
        next=nxt,
        prefix=[f for f in range(1, a) if f not in taken],
        suffix=[f for f in range(a + 1, frames) if f not in taken],
    )


def _plain_groups(t, frames):
    prev = [t - 1] if t - 1 >= 1 else []
    nxt = [t + 1] if t + 1 < frames else []
    taken = set([0, t] + prev + nxt)
    return dict(
        Img1=[0],
        self=[t] if t > 0 else [],
        prev=prev,
        next=nxt,
        rest=[f for f in range(1, frames) if f not in taken],
    )


def _sum(row, frames):
    if not frames:
        return None
    return float(sum(float(row[f]) for f in frames))


def _key_steps(meta, steps):
    """First probed step, last high-noise probed step, a mid low-noise step."""
    timesteps = meta.get('timesteps')
    boundary = meta.get('boundary_timestep')
    chosen = [steps[0]]
    if timesteps is not None and boundary is not None:
        high = [s for s in steps if s < len(timesteps)
                and timesteps[s] >= boundary]
        low = [s for s in steps if s < len(timesteps)
               and timesteps[s] < boundary]
        if high and high[-1] not in chosen:
            chosen.append(high[-1])
        if low:
            mid = low[len(low) // 2]
            if mid not in chosen:
                chosen.append(mid)
    return chosen


def write_probe_summary(merged, path, source=''):
    meta, tensors = merged['meta'], merged['tensors']
    steps, layers = meta['steps'], meta['layers']
    lines = []
    out = lines.append
    out(f"Visual Relay attention probe {source}")
    if meta.get('mode') != 'override':
        out("Mass = softmax weight one query token puts on ALL keys of a latent "
            "frame. Averaged over the sampled spatial positions of the query "
            "frame, and over all heads and layers unless a table says "
            "otherwise.")
    keys = ['mode', 'anchor_index', 'main_frames', 'suffix_frames',
            'tokens_per_frame', 'samples', 'heads', 'variant',
            'variant_until_step', 'stream_weight', 'route_switch_step',
            'suffix_fade', 'scene_tag']
    out('  ' + '  '.join(f"{k}={meta[k]}" for k in keys if k in meta))
    out(f"  layers recorded: {len(layers)}; steps: "
        + ', '.join(str(s) for s in steps))
    if meta.get('probe_errors'):
        out(f"  PROBE ERRORS: {meta['probe_errors']}")
    out('')
    if meta.get('mode') == 'paired' and 'prefix_eff' in tensors:
        _summary_paired(meta, tensors, steps, layers, out)
    elif meta.get('mode') == 'plain' and 'plain' in tensors:
        _summary_plain(meta, tensors, steps, layers, out)
    elif meta.get('mode') == 'override' and 'diff_rms' in tensors:
        _summary_override(meta, tensors, steps, layers, out)
    else:
        out("No probe records were found.")
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines) + '\n')


def _summary_paired(meta, tensors, steps, layers, out):
    a, frames = int(meta['anchor_index']), int(meta['main_frames'])
    eff = tensors['prefix_eff']  # [S, L, a, F, H]
    nat = tensors.get('prefix_nat')
    variant = meta.get('variant') or {}
    show_nat = variant.get('mass', 'natural') != 'natural'
    eff_avg = torch.nanmean(eff, dim=(1, 4))  # [S, a, F]
    nat_avg = torch.nanmean(nat, dim=(1, 4)) if nat is not None else None
    out("Working frame index: 0 = Img1, a = temporary Img2 anchor, a+1.. = "
        "suffix. Output latent = working index for t < a, index-1 after a.")
    out("[A] Prefix query frame t: mass on key groups (all layers, all heads)")
    out("    prev/next = temporal neighbours (next of t=a-1 is the first "
        "suffix frame, which shares the anchor's RoPE position).")
    header = ("     t    Img1  anchor    self    prev    next  prefix  suffix"
              + ('  anchor(natural)' if show_nat else ''))
    for si, step in enumerate(steps):
        out(f"  {_step_label(meta, step)}")
        out(header)
        for t in range(a):
            groups = _prefix_groups(t, a, frames)
            row = eff_avg[si, t]
            cells = [_fmt(_sum(row, groups[name])) for name in
                     ('Img1', 'anchor', 'self', 'prev', 'next', 'prefix',
                      'suffix')]
            line = f"  {t:4d} " + ' '.join(cells)
            if show_nat and nat_avg is not None:
                line += '  ' + _fmt(float(nat_avg[si, t, a]))
            out(line)
        out('')
    key_steps = _key_steps(meta, steps)
    probe_t = sorted({1, max(1, a // 2), a - 1})
    out("[B] Per-layer Img1 / anchor mass (all heads) at prefix frames "
        + ', '.join(f"t={t}" for t in probe_t))
    for step in key_steps:
        si = steps.index(step)
        out(f"  {_step_label(meta, step)}")
        out("  layer " + ' '.join(f"  Img1@{t:<2d} anch@{t:<2d}" for t in probe_t))
        per_layer = torch.nanmean(eff[si], dim=-1)  # [L, a, F]
        for li, layer in enumerate(layers):
            cells = []
            for t in probe_t:
                cells.append(_fmt(float(per_layer[li, t, 0]), 9))
                cells.append(_fmt(float(per_layer[li, t, a]), 9))
            out(f"  {layer:5d} " + ' '.join(cells))
        out('')
    out("[C] Head spread of the anchor mass (averaged over layers): "
        "min / median / p90 / max over heads, and the 5 strongest heads")
    for step in key_steps:
        si = steps.index(step)
        out(f"  {_step_label(meta, step)}")
        per_head = torch.nanmean(eff[si], dim=0)  # [a, F, H]
        for t in probe_t:
            values = per_head[t, a]
            values = values[~torch.isnan(values)]
            if values.numel() == 0:
                continue
            ordered = torch.sort(values).values
            p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
            top = torch.topk(values, k=min(5, values.numel()))
            top_txt = ', '.join(f"h{int(i)}:{float(v):.3f}"
                                for v, i in zip(top.values, top.indices))
            out(f"    t={t:<3d} {float(ordered[0]):.3f} / "
                f"{float(ordered[len(ordered) // 2]):.3f} / {float(p90):.3f} / "
                f"{float(ordered[-1]):.3f}   top: {top_txt}")
        out('')
    if 'stream_b' in tensors:
        sb = torch.nanmean(tensors['stream_b'], dim=(1, 4))  # [S, nB, nB]
        n_b = sb.shape[1]
        out("[D] Reference sink profile from Stream B (ordinary I2V of Img2): "
            "mass on Img2 (B0) for Stream-B frame j = 1..nB-1")
        out("  step    " + ' '.join(f"{j:7d}" for j in range(1, n_b)))
        for si, step in enumerate(steps):
            out(f"  {step:4d}    " + ' '.join(
                _fmt(float(sb[si, j, 0])) for j in range(1, n_b)))
        out('')
    if 'suffix_main' in tensors:
        sm = torch.nanmean(tensors['suffix_main'], dim=(1, 4))  # [S, nS, F]
        n_s = sm.shape[1]
        out("[E] Main-stream suffix queries (ordinary part of the fusion): "
            "mass on Img1 for working frames a+1..")
        out("  step    " + ' '.join(f"{a + 1 + j:7d}" for j in range(n_s)))
        for si, step in enumerate(steps):
            out(f"  {step:4d}    " + ' '.join(
                _fmt(float(sm[si, j, 0])) for j in range(n_s)))
        out('')
    out("[F] Full prefix matrices (rows: query frame t, columns: key working "
        "frame 0..F-1; column a is the anchor), all layers and heads")
    for si, step in enumerate(steps):
        out(f"  {_step_label(meta, step)}")
        out("     t " + ''.join(
            f"{('A' + str(f)) if f == a else f:>6}" for f in range(frames)))
        for t in range(a):
            out(f"  {t:4d} " + ''.join(
                f"{float(eff_avg[si, t, f]):6.3f}" for f in range(frames)))
        out('')


def _summary_override(meta, tensors, steps, layers, out):
    """Method 1: per-frame RMS(O1 - O2), Gordon's plot, before blending."""
    a = int(meta.get('anchor_index', -1))
    frames = int(meta['main_frames'])
    avg = {name: torch.nanmean(tensors[name], dim=(1, 3))  # [S, F]
           for name in ('o1_rms', 'o2_rms', 'diff_rms', 'cos')}
    out("O1 = stream 1 (Img1 only), O2 = stream 2 (Img1 + Img2) "
        "self-attention outputs before the blend, per latent frame f "
        f"(Img2 at f={a}); RMS over tokens and channels, mean over heads and "
        "layers.")
    out("[A] RMS(O1 - O2) per frame (Gordon's plot)")
    out("  step      " + ''.join(
        f"{('*' + str(f)) if f == a else f:>7}" for f in range(frames)))
    for si, step in enumerate(steps):
        out(f"  {step:4d}      " + ''.join(
            _fmt(float(avg['diff_rms'][si, f])) for f in range(frames)))
    out('')
    out("[B] RMS(O1), RMS(O2) and mean cosine(O1, O2) per frame")
    for si, step in enumerate(steps):
        out(f"  {_step_label(meta, step)}")
        for name, label in (('o1_rms', 'RMS O1'), ('o2_rms', 'RMS O2'),
                            ('cos', 'cosine')):
            out(f"    {label:8s}" + ''.join(
                _fmt(float(avg[name][si, f])) for f in range(frames)))
        out('')
    probe_f = sorted({0, max(0, a - 3), max(0, a - 1), a,
                      min(frames - 1, a + 1), frames - 1})
    out("[C] Per-layer RMS(O1 - O2) at frames "
        + ', '.join(f"f={f}" for f in probe_f))
    per_layer = torch.nanmean(tensors['diff_rms'], dim=3)  # [S, L, F]
    for step in _key_steps(meta, steps):
        si = steps.index(step)
        out(f"  {_step_label(meta, step)}")
        out("  layer " + ''.join(f"{('f=' + str(f)):>9}" for f in probe_f))
        for li, layer in enumerate(layers):
            out(f"  {layer:5d} " + ''.join(
                _fmt(float(per_layer[si, li, f]), 9) for f in probe_f))
        out('')


def _summary_plain(meta, tensors, steps, layers, out):
    mass = tensors['plain']  # [S, L, F, F, H]
    frames = mass.shape[2]
    avg = torch.nanmean(mass, dim=(1, 4))  # [S, F, F]
    out("[A] Plain I2V: query frame t, mass on key groups (all layers, heads)")
    for si, step in enumerate(steps):
        out(f"  {_step_label(meta, step)}")
        out("     t    Img1    self    prev    next    rest")
        for t in range(frames):
            groups = _plain_groups(t, frames)
            row = avg[si, t]
            out(f"  {t:4d} " + ' '.join(
                _fmt(_sum(row, groups[name])) for name in
                ('Img1', 'self', 'prev', 'next', 'rest')))
        out('')
    key_steps = _key_steps(meta, steps)
    probe_t = sorted({1, frames // 2, frames - 1})
    out("[B] Per-layer Img1 mass (all heads) at frames "
        + ', '.join(f"t={t}" for t in probe_t))
    for step in key_steps:
        si = steps.index(step)
        out(f"  {_step_label(meta, step)}")
        per_layer = torch.nanmean(mass[si], dim=-1)  # [L, F, F]
        for li, layer in enumerate(layers):
            out(f"  {layer:5d} " + ' '.join(
                _fmt(float(per_layer[li, t, 0]), 9) for t in probe_t))
        out('')
    out("[F] Full matrices (rows: query frame, columns: key frame)")
    for si, step in enumerate(steps):
        out(f"  {_step_label(meta, step)}")
        out("     t " + ''.join(f"{f:>6}" for f in range(frames)))
        for t in range(frames):
            out(f"  {t:4d} " + ''.join(
                f"{float(avg[si, t, f]):6.3f}" for f in range(frames)))
        out('')
