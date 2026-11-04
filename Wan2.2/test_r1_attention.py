#!/usr/bin/env python
"""Visual Relay round-1 preflight test. Run from the Wan2.2 repo root.

    python test_r1_attention.py
        one GPU: FlashAttention LSE, baseline bit-identity, every round-1
        variant against an independent dense float64 reference, the probe,
        a tiny WanModel forward (paired + plain), round 2's Stream R
        (reverse Img2 stream) against dense references, and a tiny
        end-to-end run of the real generate() loop (fake VAE and T5) for
        plain I2V, BASE, PB, KG, Stream R attn/pred and the old variants.
    torchrun --standalone --nproc_per_node=2 test_r1_attention.py --sp
        two GPUs: the Ulysses (sequence-parallel) path with variants and
        Stream R (3 streams and its plain pass), and the probe's merge.

The references are rebuilt from pre-RoPE tensors (RoPE positions applied from
scratch), so they share no code with the implementation under test. Exit
code 0 means every check passed. ``run_jobs_claude_r1.py`` runs both modes
before launching any generation and parses the ``LSE_BACKEND=`` line.
"""
import argparse
import copy
import math
import os
import sys
import tempfile
import traceback
import types

import torch

FAILURES = []


def check(name, ok, detail=''):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" +
          (f"  ({detail})" if detail else ''), flush=True)
    if not ok:
        FAILURES.append(name)


def max_abs(a, b):
    return float((a.double() - b.double()).abs().max())


def rel_err(a, b):
    num = float((a.double() - b.double()).abs().max())
    den = float(b.double().abs().max()) + 1e-12
    return num / den


# ---------------------------------------------------------------------------
# Imports. On a machine without CUDA (the author's numpy stand-in) the
# FlashAttention kernels are replaced by a naive implementation.
# ---------------------------------------------------------------------------
def import_attention():
    import wan.modules.attention as A
    return A


def naive_flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                          softmax_scale=None, q_scale=None, causal=False,
                          window_size=(-1, -1), deterministic=False,
                          dtype=torch.bfloat16, version=None,
                          return_lse=False):
    assert q_lens is None and not causal
    half_dtypes = (torch.float16, torch.bfloat16)
    out_dtype = q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    qf, kf, vf = half(q).float(), half(k).float(), half(v).float()
    scale = softmax_scale or q.size(-1)**-0.5
    logits = torch.einsum('bqhd,bkhd->bhqk', qf, kf) * scale
    if k_lens is not None:
        for b, n in enumerate(k_lens.tolist()):
            logits[b, :, :, int(n):] = float('-inf')
    lse = torch.logsumexp(logits, dim=-1)
    probs = torch.exp(logits - lse[..., None])
    out = torch.einsum('bhqk,bkhd->bqhd', probs, vf).to(half(q).dtype)
    out = out.to(out_dtype)
    if return_lse:
        return out, lse.permute(0, 2, 1).float()
    return out


# ---------------------------------------------------------------------------
# The paired-anchor attention exactly as it was before round 1.
# ---------------------------------------------------------------------------
A = None  # set in main()

def original_paired_anchor_attention(
    q,
    k,
    v,
    seq_lens,
    grid_sizes,
    freqs,
    anchor_index,
    stream_weight=0.25,
    prefix_anchor_active=True,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
):
    """Fuse an Img1 main stream with an aligned Img2-first suffix stream.

    ``q`` and ``k`` must already have ordinary per-stream RoPE applied. The
    batch contains exactly two samples: a full main sequence followed by a
    suffix whose local frame zero aligns with ``anchor_index`` in the main
    sequence. While the prefix anchor route is active, prefix queries use one
    ordinary softmax over all ordinary main-stream K/V on both sides of the
    temporary anchor plus a hybrid anchor: the history-aware Stream-A anchor
    key selects the target in its joint temporal context, while the
    fresh-cache Stream-B anchor value supplies the image feature. After that
    route is disabled, only the hybrid anchor is removed; ordinary post-anchor
    K/V remain visible.
    Main suffix queries exclude the anchor and use main-stream Q/K/V for every
    ordinary key. A second full attention uses the aligned suffix-stream Q/K/V,
    including Img2 and all later suffix frames. Their outputs use
    ``stream_weight``.
    """
    if window_size != (-1, -1):
        raise ValueError("paired anchor attention requires global attention")
    if freqs is None:
        raise ValueError("paired anchor attention requires RoPE frequencies")
    if q.size(0) != 2 or len(seq_lens) != 2 or len(grid_sizes) != 2:
        raise ValueError(
            "paired anchor attention requires main and suffix streams")
    if not 0.0 <= stream_weight <= 1.0:
        raise ValueError("paired stream weight must be between zero and one")

    main_frames, main_height, main_width = (
        int(value) for value in grid_sizes[0].tolist())
    suffix_frames, suffix_height, suffix_width = (
        int(value) for value in grid_sizes[1].tolist())
    anchor_index = int(anchor_index)
    if not 0 < anchor_index < main_frames - 1:
        raise ValueError("paired anchor must have frames on both sides")
    if (main_height, main_width) != (suffix_height, suffix_width):
        raise ValueError("paired streams must have identical spatial grids")
    if suffix_frames != main_frames - anchor_index:
        raise ValueError("paired Img2 stream must align with the main suffix")

    tokens_per_frame = main_height * main_width
    main_seq_len = main_frames * tokens_per_frame
    suffix_seq_len = suffix_frames * tokens_per_frame
    if (main_seq_len != int(seq_lens[0])
            or suffix_seq_len != int(seq_lens[1])):
        raise ValueError("paired stream grid and sequence lengths disagree")

    main_q = q[0:1, :main_seq_len]
    main_k = k[0:1, :main_seq_len]
    main_v = v[0:1, :main_seq_len]
    suffix_q = q[1:2, :suffix_seq_len]
    suffix_k = k[1:2, :suffix_seq_len]
    suffix_v = v[1:2, :suffix_seq_len]

    # Compress the anchor out of the main stream's ordinary timeline. Frames
    # after the anchor move one RoPE unit to the left, while the anchor key
    # itself stays at its original position for the temporary messenger route
    # itself supplies the history-aware key for prefix attention, paired with
    # the fresh-cache B0 value. A suffix query
    # sees the retained left neighbour through main attention and sees the Img2
    # anchor through the local suffix stream at the same relative offset:
    # q=13 gives k={11, 12} offset -1.
    # Shift one frame at a time to bound the temporary complex tensor size.
    for frame_index in range(anchor_index + 1, main_frames):
        frame_start = frame_index * tokens_per_frame
        frame_end = frame_start + tokens_per_frame
        shifted_q = A._shift_temporal_rope(
            main_q[:, frame_start:frame_end], -1, freqs)
        main_q[:, frame_start:frame_end].copy_(shifted_q)
        del shifted_q
        shifted_k = A._shift_temporal_rope(
            main_k[:, frame_start:frame_end], -1, freqs)
        main_k[:, frame_start:frame_end].copy_(shifted_k)
        del shifted_k

    # The Img2-first suffix remains an independent reference stream.
    suffix_output = A.flash_attention(
        q=suffix_q,
        k=suffix_k,
        v=suffix_v,
        window_size=window_size,
        dtype=dtype)

    anchor_start = anchor_index * tokens_per_frame
    anchor_end = anchor_start + tokens_per_frame
    ordinary_k = torch.cat(
        [main_k[:, :anchor_start], main_k[:, anchor_end:]], dim=1)
    ordinary_v = torch.cat(
        [main_v[:, :anchor_start], main_v[:, anchor_end:]], dim=1)

    # Prefix queries always retain all ordinary Stream-A context on both sides
    # of the temporary anchor. Before the route switch, also include a hybrid
    # anchor: the joint-conditioned, history-aware Stream-A anchor supplies K,
    # while the fresh-cache Stream-B anchor supplies V. Both are already
    # aligned to the anchor for this prefix attention computation. After the
    # switch, remove only this hybrid anchor route.
    if prefix_anchor_active:
        prefix_k = torch.cat([
            main_k[:, :anchor_start],
            main_k[:, anchor_start:anchor_end],
            main_k[:, anchor_end:],
        ], dim=1)
        prefix_v = torch.cat([
            main_v[:, :anchor_start],
            suffix_v[:, :tokens_per_frame],
            main_v[:, anchor_end:],
        ], dim=1)
    else:
        prefix_k = ordinary_k
        prefix_v = ordinary_v
    prefix_output = A.flash_attention(
        q=main_q[:, :anchor_start],
        k=prefix_k,
        v=prefix_v,
        window_size=window_size,
        dtype=dtype)

    # Evolve the joint-conditioned Stream-A anchor with the prefix and itself,
    # but never with the post-anchor suffix. Stream B evolves its independent
    # fresh-cache Img2 anchor through the Img2-first suffix branch.
    messenger_output = A.flash_attention(
        q=main_q[:, anchor_start:anchor_end],
        k=main_k[:, :anchor_end],
        v=main_v[:, :anchor_end],
        window_size=window_size,
        dtype=dtype)

    main_suffix_q = main_q[:, anchor_end:]
    img2_suffix_output = suffix_output[:, tokens_per_frame:suffix_seq_len]
    if main_suffix_q.size(1) != img2_suffix_output.size(1):
        raise ValueError("paired suffix query lengths do not align")

    ordinary_output = A.flash_attention(
        q=main_suffix_q,
        k=ordinary_k,
        v=ordinary_v,
        window_size=window_size,
        dtype=dtype)
    main_suffix_output = ((1.0 - stream_weight) * ordinary_output
                          + stream_weight * img2_suffix_output)

    main_output = torch.cat([
        prefix_output,
        messenger_output,
        main_suffix_output,
    ], dim=1)
    padded_len = q.size(1)
    if main_seq_len < padded_len:
        main_output = torch.cat([
            main_output,
            main_output.new_zeros(
                1, padded_len - main_seq_len, *main_output.shape[2:])
        ], dim=1)
    if suffix_seq_len < padded_len:
        suffix_output = torch.cat([
            suffix_output,
            suffix_output.new_zeros(
                1, padded_len - suffix_seq_len, *suffix_output.shape[2:])
        ], dim=1)
    return torch.cat([main_output, suffix_output], dim=0)



# ---------------------------------------------------------------------------
# Independent dense reference (float64, RoPE from scratch).
# ---------------------------------------------------------------------------
def rope_thetas(head_dim):
    c = head_dim // 2
    t_dim, s_dim = c - 2 * (c // 3), c // 3
    th_t = 1.0 / (10000.0**(torch.arange(0, 2 * t_dim, 2, dtype=torch.float64)
                            / (2 * t_dim)))
    th_s = 1.0 / (10000.0**(torch.arange(0, 2 * s_dim, 2, dtype=torch.float64)
                            / (2 * s_dim)))
    return th_t, th_s


def rope_at(x_pre, tpos, ypos, xpos):
    """x_pre [N, H, D] float64; positions [N] float64 -> rotated [N, H, D]."""
    n, heads, head_dim = x_pre.shape
    th_t, th_s = rope_thetas(head_dim)
    th_t, th_s = th_t.to(x_pre.device), th_s.to(x_pre.device)
    phase = torch.cat([
        tpos[:, None] * th_t[None], ypos[:, None] * th_s[None],
        xpos[:, None] * th_s[None]
    ], dim=1)
    rot = torch.polar(torch.ones_like(phase), phase)
    xc = torch.view_as_complex(
        x_pre.reshape(n, heads, head_dim // 2, 2).contiguous())
    return torch.view_as_real(xc * rot[:, None, :]).flatten(-2)


def token_positions(frames, h, w, frame_to_pos, device):
    tokens = frames * h * w
    idx = torch.arange(tokens, device=device)
    frame = idx // (h * w)
    spatial = idx % (h * w)
    tpos = frame_to_pos.to(device)[frame].double()
    return tpos, (spatial // w).double(), (spatial % w).double()


def make_freqs(head_dim, device):
    try:
        from wan.modules.model import rope_params
    except Exception:  # noqa: BLE001 - numpy stand-in has no torch.nn
        rope_params = _rope_params_copy
    d = head_dim
    return torch.cat([
        rope_params(1024, d - 4 * (d // 6)),
        rope_params(1024, 2 * (d // 6)),
        rope_params(1024, 2 * (d // 6))
    ], dim=1).to(device)


def _rope_params_copy(max_seq_len, dim, theta=10000):
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_model_rope(x, grid_sizes, freqs):
    try:
        from wan.modules.model import rope_apply
    except Exception:  # noqa: BLE001
        rope_apply = _rope_apply_copy
    return rope_apply(x, grid_sizes, freqs)


def _rope_apply_copy(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


def make_inputs(device, dtype, frames=7, anchor=3, h=2, w=3, heads=4,
                head_dim=128, seed=0):
    tpf = h * w
    n_b = frames - anchor
    main_len, suffix_len = frames * tpf, n_b * tpf
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    q_pre = torch.randn(2, main_len, heads, head_dim, generator=gen)
    k_pre = torch.randn(2, main_len, heads, head_dim, generator=gen)
    v = torch.randn(2, main_len, heads, head_dim, generator=gen)
    for t in (q_pre, k_pre, v):
        t[1, suffix_len:] = 0.0  # padding of the shorter Stream-B sample
    q_pre, k_pre, v = q_pre.to(device), k_pre.to(device), v.to(device)
    grid = torch.tensor([[frames, h, w], [n_b, h, w]], dtype=torch.long)
    freqs = make_freqs(head_dim, device)
    q = apply_model_rope(q_pre, grid, freqs).to(dtype)
    k = apply_model_rope(k_pre, grid, freqs).to(dtype)
    v = v.to(dtype)
    return dict(F=frames, a=anchor, h=h, w=w, T=tpf, H=heads, D=head_dim,
                nB=n_b, q=q, k=k, v=v, q_pre=q_pre.double(),
                k_pre=k_pre.double(), v_ref=v.double(), grid=grid,
                freqs=freqs,
                seq_lens=torch.tensor([main_len, suffix_len],
                                      dtype=torch.long), device=device)


def settings_of(variant):
    s = dict(anchor_key='A', mirror=False, mass='natural',
             split_shape='linear', mirror_gain=1.0, mass_cap=0.95)
    s.update(variant or {})
    return s


def dense_prefix(inp, variant):
    """Prefix output, group weights and per-frame masses, float64."""
    s = settings_of(variant)
    frames, a, h, w, tpf, heads, head_dim = (inp[x] for x in
                                             ('F', 'a', 'h', 'w', 'T', 'H',
                                              'D'))
    dev = inp['device']
    scale = head_dim**-0.5
    main_len = frames * tpf
    q_pre, k_pre = inp['q_pre'][0, :main_len], inp['k_pre'][0, :main_len]
    v_main = inp['v_ref'][0, :main_len]
    kb_pre, vb0 = inp['k_pre'][1, :tpf], inp['v_ref'][1, :tpf]
    to_pos = torch.tensor([f if f <= a else f - 1 for f in range(frames)])
    tpos, ypos, xpos = token_positions(frames, h, w, to_pos, dev)
    q_rot, k_rot = rope_at(q_pre, tpos, ypos, xpos), rope_at(
        k_pre, tpos, ypos, xpos)
    y0, x0 = ypos[:tpf], xpos[:tpf]
    anchor_pre = kb_pre if s['anchor_key'] == 'B' else k_pre[a * tpf:(a + 1) *
                                                             tpf]
    n_q = a * tpf
    m_img1 = torch.zeros(n_q, heads, dtype=torch.float64, device=dev)
    m_anchor = torch.zeros_like(m_img1)
    frame_mass = torch.zeros(n_q, heads, frames, dtype=torch.float64,
                             device=dev)
    o_img1 = torch.zeros(n_q, heads, head_dim, dtype=torch.float64,
                         device=dev)
    o_anchor = torch.zeros_like(o_img1)
    o_rest = torch.zeros_like(o_img1)
    for t in range(a):
        rows = slice(t * tpf, (t + 1) * tpf)
        pos = float(2 * t - a) if s['mirror'] else float(a)
        ak = rope_at(anchor_pre,
                     torch.full((tpf,), pos, dtype=torch.float64, device=dev),
                     y0, x0)
        keys = torch.cat([k_rot[:a * tpf], ak, k_rot[(a + 1) * tpf:]])
        vals = torch.cat([v_main[:a * tpf], vb0, v_main[(a + 1) * tpf:]])
        logits = torch.einsum('qhd,khd->hqk', q_rot[rows], keys) * scale
        p = torch.softmax(logits, dim=-1)  # [H, T, F*T]
        pf = p.reshape(heads, tpf, frames, tpf).sum(-1)  # [H, T, F]
        frame_mass[rows] = pf.permute(1, 0, 2)
        mi, ma = pf[:, :, 0], pf[:, :, a]
        m_img1[rows], m_anchor[rows] = mi.t(), ma.t()
        o_img1[rows] = torch.einsum('hqk,khd->qhd', p[..., :tpf] /
                                    mi[..., None], vals[:tpf])
        o_anchor[rows] = torch.einsum(
            'hqk,khd->qhd', p[..., a * tpf:(a + 1) * tpf] / ma[..., None],
            vals[a * tpf:(a + 1) * tpf])
        rest_idx = torch.cat([
            torch.arange(tpf, a * tpf, device=dev),
            torch.arange((a + 1) * tpf, frames * tpf, device=dev)
        ])
        p_rest = p.index_select(2, rest_idx)
        mr = p_rest.sum(-1)
        o_rest[rows] = torch.einsum('hqk,khd->qhd', p_rest / mr[..., None],
                                    vals.index_select(0, rest_idx))
    m_rest = 1.0 - m_img1 - m_anchor
    frame_of_row = torch.arange(n_q, device=dev) // tpf
    if s['mass'] == 'natural':
        w_i, w_a, w_r = m_img1, m_anchor, m_rest
    elif s['mass'] == 'img1_fade':
        x = frame_of_row.double() / a
        lam = x if s['split_shape'] == 'linear' else 0.5 * (
            1.0 - torch.cos(math.pi * x))
        scaled = m_img1 * (1.0 - lam)[:, None]  # Img1 weights times (1-lam)
        norm = scaled + m_anchor + m_rest
        w_i, w_a, w_r = scaled / norm, m_anchor / norm, m_rest / norm
    elif s['mass'] == 'split':
        x = frame_of_row.double() / a
        lam = x if s['split_shape'] == 'linear' else 0.5 * (
            1.0 - torch.cos(math.pi * x))
        lam = lam[:, None]
        w_i, w_a, w_r = (1 - lam) * (m_img1 + m_anchor), lam * (
            m_img1 + m_anchor), m_rest
    else:  # mirror_mass
        share = m_img1 / (m_img1 + m_rest)  # Img1 mass with the anchor removed
        per_frame = share.reshape(a, tpf, heads).mean(1)  # [a, H]
        target = torch.zeros_like(per_frame)
        for t in range(1, a):
            target[t] = per_frame[a - t]
        target = (s['mirror_gain'] * target).clamp(0.0, s['mass_cap'])
        tgt = target[frame_of_row]  # [n_q, H]
        w_i = (1 - tgt) * share
        w_r = (1 - tgt) * (1 - share)
        w_a = tgt
    out = (w_i[..., None] * o_img1 + w_a[..., None] * o_anchor +
           w_r[..., None] * o_rest)
    eff_mass = frame_mass / m_rest[..., None] * w_r[..., None]
    eff_mass[:, :, 0] = w_i
    eff_mass[:, :, a] = w_a
    return dict(out=out, w=(w_i, w_a, w_r), nat_mass=frame_mass,
                eff_mass=eff_mass)


def dense_stream_masses(q_pre, k_pre, frames, h, w, positions, key_mask=None):
    """Per-query-token, per-head, per-key-frame masses [N, H, F]."""
    tpf = h * w
    dev = q_pre.device
    tpos, ypos, xpos = token_positions(frames, h, w, positions, dev)
    qr, kr = rope_at(q_pre, tpos, ypos, xpos), rope_at(k_pre, tpos, ypos, xpos)
    logits = torch.einsum('qhd,khd->hqk', qr, kr) * q_pre.shape[-1]**-0.5
    if key_mask is not None:
        logits = logits.masked_fill(~key_mask[None, None, :], float('-inf'))
    p = torch.softmax(logits, dim=-1)
    heads = q_pre.shape[1]
    return p.reshape(heads, -1, frames, tpf).sum(-1).permute(1, 0, 2)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
VARIANTS = [
    ('FreshKey B', dict(anchor_key='B')),
    ('MirrorSink B+mirror', dict(anchor_key='B', mirror=True)),
    ('Mirror A', dict(anchor_key='A', mirror=True)),
    ('SinkSplit linear A', dict(mass='split', split_shape='linear')),
    ('SinkSplit cosine B+mirror', dict(anchor_key='B', mirror=True,
                                       mass='split', split_shape='cosine')),
    ('MirrorMass A', dict(mass='mirror_mass')),
    ('MirrorMass B gain1.5 cap0.6', dict(anchor_key='B', mass='mirror_mass',
                                         mirror_gain=1.5, mass_cap=0.6)),
    ('Img1Fade linear A', dict(mass='img1_fade')),
    ('Img1Fade cosine B+mirror', dict(anchor_key='B', mirror=True,
                                      mass='img1_fade', split_shape='cosine')),
]


# Round 2: variants that also change Stream-B rows (not in the exact prefix test).
STREAM_VARIANTS = [
    ('SuffixTimeFix', dict(suffix_time_fix=True)),
    ('MirrorSink+SuffixTimeFix', dict(anchor_key='B', mirror=True,
                                      suffix_time_fix=True)),
]


def dense_stream_output(q_pre, k_pre, v, frames, h, w, positions):
    tpos, ypos, xpos = token_positions(frames, h, w, positions, q_pre.device)
    qr, kr = rope_at(q_pre, tpos, ypos, xpos), rope_at(k_pre, tpos, ypos, xpos)
    logits = torch.einsum('qhd,khd->hqk', qr, kr) * q_pre.shape[-1]**-0.5
    return torch.einsum('hqk,khd->qhd', torch.softmax(logits, dim=-1), v)


def test_suffix_time_fix():
    inp = make_inputs(DEVICE, torch.float32, seed=17)
    frames, a, h, w, tpf, heads = (inp[x] for x in
                                   ('F', 'a', 'h', 'w', 'T', 'H'))
    n_b = inp['nB']
    suffix_len = n_b * tpf
    base = run_paired(inp, None)
    out = run_paired(inp, dict(suffix_time_fix=True))
    check("suffix time fix leaves prefix and anchor rows bit-identical",
          torch.equal(out[0, :(a + 1) * tpf], base[0, :(a + 1) * tpf]))
    shifted = torch.tensor([0] + list(range(n_b - 1)))
    ref = dense_stream_output(inp['q_pre'][1, :suffix_len],
                              inp['k_pre'][1, :suffix_len],
                              inp['v_ref'][1, :suffix_len], n_b, h, w, shifted)
    err = rel_err(out[1, :suffix_len], ref)
    unshifted = dense_stream_output(inp['q_pre'][1, :suffix_len],
                                    inp['k_pre'][1, :suffix_len],
                                    inp['v_ref'][1, :suffix_len], n_b, h, w,
                                    torch.arange(n_b))
    err_old = rel_err(base[1, :suffix_len], unshifted)
    check("suffix time fix: Stream B equals dense with positions 0,0,1,..",
          err < 3e-2 and err_old < 3e-2 and
          rel_err(out[1, :suffix_len], base[1, :suffix_len]) > 1e-3,
          f"rel err {err:.1e} (baseline vs unshifted {err_old:.1e})")
    A.ATTN_PROBE.configure([0], samples=4)
    A.ATTN_PROBE.begin_forward(0, 'cond')
    run_paired(inp, dict(suffix_time_fix=True))
    rec = A.ATTN_PROBE.data.get((0, 0))
    idx = A.ATTN_PROBE.spatial_index(tpf, DEVICE)
    sb = dense_stream_masses(inp['q_pre'][1, :suffix_len],
                             inp['k_pre'][1, :suffix_len], n_b, h, w, shifted)
    err_b = max(max_abs(rec['stream_b'][t], sb[t * tpf + idx].mean(0).t().cpu())
                for t in range(n_b)) if rec else float('inf')
    check("suffix time fix: probe Stream-B masses equal dense", err_b < 1e-4,
          f"err {err_b:.1e}")
    A.ATTN_PROBE.reset()


# ---------------------------------------------------------------------------
# Round 2: Stream R (reverse Img2-first stream fused into the prefix)
# ---------------------------------------------------------------------------
REV_VARIANTS = [
    ('StreamR linear s1', dict(rev_strength=1.0, rev_shape='linear')),
    ('StreamR cosine s0.5', dict(rev_strength=0.5, rev_shape='cosine')),
    ('StreamR + SinkSplit', dict(rev_strength=1.0, mass='split')),
]


# Round 3: coupled Stream R (Stream R frame a-t takes the same mixed output).
REV_COUPLED = [
    ('StreamR coupled s0.5', dict(rev_strength=0.5, rev_couple=True)),
    ('StreamR coupled s1 cosine', dict(rev_strength=1.0, rev_shape='cosine',
                                       rev_couple=True)),
]


def expected_rev_weight(t, a, strength, shape):
    x = t / a
    if shape == 'cosine':
        x = 0.5 * (1.0 - math.cos(math.pi * x))
    return strength * x


def make_inputs_rev(device, dtype, seed=0, frames=7, anchor=3, h=2, w=3,
                    heads=4, head_dim=128):
    """make_inputs plus a third sample: Stream R over `anchor` frames."""
    inp = make_inputs(device, dtype, frames=frames, anchor=anchor, h=h, w=w,
                      heads=heads, head_dim=head_dim, seed=seed)
    tpf = h * w
    main_len, rev_len = frames * tpf, anchor * tpf
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed + 100)
    q_r = torch.randn(1, main_len, heads, head_dim, generator=gen)
    k_r = torch.randn(1, main_len, heads, head_dim, generator=gen)
    v_r = torch.randn(1, main_len, heads, head_dim, generator=gen)
    for t in (q_r, k_r, v_r):
        t[0, rev_len:] = 0.0
    q_r, k_r, v_r = q_r.to(device), k_r.to(device), v_r.to(device)
    grid_r = torch.tensor([[anchor, h, w]], dtype=torch.long)
    q_r_rot = apply_model_rope(q_r, grid_r, inp['freqs']).to(dtype)
    k_r_rot = apply_model_rope(k_r, grid_r, inp['freqs']).to(dtype)
    inp.update(
        q3=torch.cat([inp['q'], q_r_rot]), k3=torch.cat([inp['k'], k_r_rot]),
        v3=torch.cat([inp['v'], v_r.to(dtype)]),
        grid3=torch.cat([inp['grid'], grid_r]),
        seq_lens3=torch.tensor([main_len, inp['nB'] * tpf, rev_len],
                               dtype=torch.long),
        q_pre_r=q_r[0, :rev_len].double(), k_pre_r=k_r[0, :rev_len].double(),
        v_r=v_r.to(dtype)[0, :rev_len].double(), rev_len=rev_len)
    return inp


def run_paired3(inp, variant, active=True, weight=0.5):
    return A.paired_anchor_attention(
        q=inp['q3'].clone(), k=inp['k3'].clone(), v=inp['v3'].clone(),
        seq_lens=inp['seq_lens3'], grid_sizes=inp['grid3'],
        freqs=inp['freqs'], anchor_index=inp['a'], stream_weight=weight,
        prefix_anchor_active=active, variant=variant)


def test_reverse_stream():
    saved = A._LSE_BACKEND
    A._LSE_BACKEND = 'torch'
    inp = make_inputs_rev(DEVICE, torch.float32, seed=21)
    frames, a, h, w, tpf = (inp[x] for x in ('F', 'a', 'h', 'w', 'T'))
    rev_len = inp['rev_len']
    ref_r = dense_stream_output(inp['q_pre_r'], inp['k_pre_r'], inp['v_r'],
                                a, h, w, torch.arange(a))
    for name, variant in REV_VARIANTS:
        prefix_only = {k: v for k, v in variant.items()
                       if not k.startswith('rev_')} or None
        two = run_paired(inp, prefix_only)
        out = run_paired3(inp, variant)
        check(f"{name}: three samples returned", out.shape[0] == 3)
        check(f"{name}: Img1, anchor, suffix and Stream-B rows unchanged",
              torch.equal(out[0, :tpf], two[0, :tpf])
              and torch.equal(out[0, a * tpf:], two[0, a * tpf:])
              and torch.equal(out[1], two[1]))
        err_r = rel_err(out[2, :rev_len], ref_r)
        check(f"{name}: Stream R equals dense I2V attention (positions 0..a-1)",
              err_r < 3e-2, f"rel err {err_r:.1e}")
        s = variant.get('rev_strength', 1.0)
        shape = variant.get('rev_shape', 'linear')
        err = 0.0
        for t in range(1, a):
            wt = expected_rev_weight(t, a, s, shape)
            rows = slice(t * tpf, (t + 1) * tpf)
            rev_rows = slice((a - t) * tpf, (a - t + 1) * tpf)
            expect = (1 - wt) * two[0, rows].double() + wt * ref_r[rev_rows]
            err = max(err, max_abs(out[0, rows], expect))
        check(f"{name}: prefix t = (1-w_t) main + w_t StreamR[a-t]",
              err < 3e-2, f"max abs err {err:.1e}")
    two = run_paired(inp, None)
    zero = run_paired3(inp, dict(rev_strength=0.0))
    check("Stream R strength 0 leaves main and Stream B bit-identical",
          torch.equal(zero[0], two[0]) and torch.equal(zero[1], two[1]))
    off = run_paired3(inp, dict(rev_strength=1.0), active=False)
    off_two = run_paired(inp, None, active=False)
    check("Stream R still fuses after the anchor route switch",
          torch.equal(off[0, :tpf], off_two[0, :tpf])
          and not torch.equal(off[0, tpf:a * tpf], off_two[0, tpf:a * tpf]))
    ok = all(abs(A.reverse_stream_weight(t, 6, s, sh)
                 - expected_rev_weight(t, 6, s, sh)) < 1e-12
             for t in range(6) for s in (0.0, 0.5, 1.0)
             for sh in ('linear', 'cosine'))
    check("reverse_stream_weight matches w_t = s * f(t/a)", ok)
    for name, variant in REV_COUPLED:
        uncoupled = dict(variant, rev_couple=False)
        out_c = run_paired3(inp, variant)
        out_u = run_paired3(inp, uncoupled)
        check(f"{name}: main and Stream B rows equal the uncoupled run",
              torch.equal(out_c[0], out_u[0]) and torch.equal(out_c[1],
                                                              out_u[1]))
        ok = torch.equal(out_c[2, :tpf], out_u[2, :tpf])  # R frame 0 kept
        ok = ok and torch.equal(out_c[2, rev_len:], out_u[2, rev_len:])
        for j in range(1, a):  # R frame j = the mixed prefix frame a - j
            ok = ok and torch.equal(out_c[2, j * tpf:(j + 1) * tpf],
                                    out_c[0, (a - j) * tpf:(a - j + 1) * tpf])
        check(f"{name}: Stream R frame j takes prefix frame a-j's mixed "
              "output, frame 0 unchanged", ok)
    bad = 0
    for kwargs in (dict(variant=dict(rev_strength=1.5)),
                   dict(variant=dict(rev_shape='square'))):
        try:
            run_paired3(inp, **kwargs)
        except ValueError:
            bad += 1
    wrong = dict(inp)
    wrong['grid3'] = inp['grid3'].clone()
    wrong['grid3'][2, 0] = a - 1
    wrong['seq_lens3'] = inp['seq_lens3'].clone()
    wrong['seq_lens3'][2] = (a - 1) * tpf
    try:
        run_paired3(wrong, dict(rev_strength=1.0))
    except ValueError:
        bad += 1
    check("Stream R rejects bad strength, shape and frame count", bad == 3)
    A.ATTN_PROBE.configure([0], samples=4)
    A.ATTN_PROBE.begin_forward(0, 'uncond')
    plain = run_paired3(inp, dict(rev_strength=1.0))
    A.ATTN_PROBE.begin_forward(0, 'cond')
    probed = run_paired3(inp, dict(rev_strength=1.0))
    check("probe is read-only with Stream R in the batch",
          torch.equal(plain, probed) and A.ATTN_PROBE.failed is None
          and (0, 0) in A.ATTN_PROBE.data, A.ATTN_PROBE.failed or '')
    A.ATTN_PROBE.reset()
    A._LSE_BACKEND = saved


# ---------------------------------------------------------------------------
# Round 3: Gordon's Method 1, self-attention override
# ---------------------------------------------------------------------------
def expected_override_weights(frames, a, min_weight, radius):
    out = []
    for f in range(frames):
        d = abs(f - a)
        ramp = (0.0 if d == 0 else 1.0) if radius <= 0 else min(d, radius) / radius
        out.append(min_weight + (1.0 - min_weight) * ramp)
    return out


def make_override_inputs(device, dtype, seed=31, frames=6, h=2, w=3, heads=4,
                         head_dim=128):
    tpf = h * w
    seq = frames * tpf
    pad = 4  # padded batch, as in the model
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    q_pre = torch.randn(2, seq + pad, heads, head_dim, generator=gen)
    k_pre = torch.randn(2, seq + pad, heads, head_dim, generator=gen)
    v = torch.randn(2, seq + pad, heads, head_dim, generator=gen)
    for t in (q_pre, k_pre, v):
        t[:, seq:] = 0.0
    q_pre, k_pre, v = q_pre.to(device), k_pre.to(device), v.to(device)
    grid = torch.tensor([[frames, h, w], [frames, h, w]], dtype=torch.long)
    freqs = make_freqs(head_dim, device)
    q = apply_model_rope(q_pre, grid, freqs).to(dtype)
    k = apply_model_rope(k_pre, grid, freqs).to(dtype)
    refs = [dense_stream_output(q_pre[i, :seq].double(), k_pre[i, :seq].double(),
                                v.to(dtype)[i, :seq].double(), frames, h, w,
                                torch.arange(frames)) for i in range(2)]
    return dict(q=q, k=k, v=v.to(dtype), grid=grid, frames=frames, tpf=tpf,
                seq=seq, refs=refs, q_pre=q_pre.double(), k_pre=k_pre.double(),
                seq_lens=torch.tensor([seq, seq], dtype=torch.long))


def run_override(inp, weights, blend='linear', anchor=3):
    return A.override_attention(
        q=inp['q'].clone(), k=inp['k'].clone(), v=inp['v'].clone(),
        seq_lens=inp['seq_lens'], grid_sizes=inp['grid'],
        override=dict(weights=weights, blend=blend, anchor=anchor))


def test_override_attention():
    inp = make_override_inputs(DEVICE, torch.float32)
    frames, tpf, seq = inp['frames'], inp['tpf'], inp['seq']
    o1, o2 = inp['refs']
    ok = all(max(abs(x - y) for x, y in zip(
        A.override_frame_weights(frames, a, mw, r),
        expected_override_weights(frames, a, mw, r))) < 1e-12
        for a in (1, 3) for mw in (0.0, 0.25, 1.0) for r in (0, 2, 3))
    check("override weights: min at Img2, linear back to 1 at the radius", ok)
    full = run_override(inp, [1.0] * frames)
    check("full override: stream 1 output is stream 2's (bit-identical)",
          torch.equal(full[0, :seq], full[1, :seq]))
    err2 = rel_err(full[1, :seq], o2)
    check("override: stream 2 keeps its own attention (dense)", err2 < 3e-2,
          f"rel err {err2:.1e}")
    none = run_override(inp, [0.0] * frames)
    err1 = rel_err(none[0, :seq], o1)
    check("override weight 0: stream 1 keeps its own attention (dense)",
          err1 < 3e-2, f"rel err {err1:.1e}")
    weights = A.override_frame_weights(frames, 3, 0.25, 2)
    lin = run_override(inp, weights, 'linear')
    nrm = run_override(inp, weights, 'norm')
    own = none[0, :seq].double()
    other = full[1, :seq].double()
    err_l = err_n = 0.0
    for f, wf in enumerate(weights):
        rows = slice(f * tpf, (f + 1) * tpf)
        mix = own[rows] + wf * (other[rows] - own[rows])
        err_l = max(err_l, max_abs(lin[0, rows], mix))
        target = ((1 - wf) * own[rows].norm(dim=-1, keepdim=True)
                  + wf * other[rows].norm(dim=-1, keepdim=True))
        want = mix * target / mix.norm(dim=-1, keepdim=True)
        err_n = max(err_n, max_abs(nrm[0, rows], want))
    check("override linear blend: O1 + w_f (O2 - O1) per frame",
          err_l < 1e-5, f"max abs err {err_l:.1e}")
    check("override norm blend: same direction, length (1-w)|O1| + w|O2|",
          err_n < 1e-5, f"max abs err {err_n:.1e}")
    lens = lin[0, :seq].double().norm(dim=-1)
    lens_n = nrm[0, :seq].double().norm(dim=-1)
    check("override norm blend is never shorter than the linear blend",
          bool((lens_n >= lens - 1e-6).all()))
    check("override leaves stream 2 and padding untouched",
          torch.equal(lin[1], full[1]) and torch.equal(lin[0, seq:],
                                                       full[0, seq:]))
    bad = 0
    for weights_bad, kwargs in (([1.0] * (frames - 1), {}),
                                ([1.0] * frames, dict(blend='cubic'))):
        try:
            run_override(inp, weights_bad, **kwargs)
        except ValueError:
            bad += 1
    try:
        A.override_attention(q=inp['q'][:1], k=inp['k'][:1], v=inp['v'][:1],
                             seq_lens=inp['seq_lens'][:1],
                             grid_sizes=inp['grid'][:1],
                             override=dict(weights=[1.0] * frames))
    except ValueError:
        bad += 1
    check("override rejects bad weights, blend and stream count", bad == 3)
    # probe: Gordon's per-frame statistics, read-only
    A.ATTN_PROBE.configure([0], samples=4)
    A.ATTN_PROBE.begin_forward(0, 'cond')
    probed = run_override(inp, weights, 'linear')
    rec = A.ATTN_PROBE.data.get((0, 0))
    check("override probe is read-only", torch.equal(probed, lin)
          and A.ATTN_PROBE.failed is None, A.ATTN_PROBE.failed or '')
    err = float('inf')
    if rec is not None:
        a1 = own.reshape(frames, tpf, *own.shape[1:])
        b1 = other.reshape(frames, tpf, *other.shape[1:])
        want = dict(
            o1_rms=a1.pow(2).mean(dim=(1, 3)).sqrt(),
            o2_rms=b1.pow(2).mean(dim=(1, 3)).sqrt(),
            diff_rms=(a1 - b1).pow(2).mean(dim=(1, 3)).sqrt(),
            cos=torch.nn.functional.cosine_similarity(a1, b1, dim=-1).mean(1))
        err = max(max_abs(rec[name], want[name].cpu()) for name in want)
    check("override probe: per-frame RMS(O1), RMS(O2), RMS(O1-O2), cosine",
          err < 1e-4, f"max abs err {err:.1e}")
    A.ATTN_PROBE.reset()


def test_override_stream_b():
    """Round 5: local profile weights and Stream B mixed into the latents
    after Img2, against dense float64 references."""
    w_local = A.override_frame_weights(7, 3, profile='local', radius=3)
    want_local = [0.0, 1 / 3, 2 / 3, 1.0, 2 / 3, 1 / 3, 0.0]
    check("local profile: w = 1 at Img2, linear to 0 at the radius",
          max(abs(x - y) for x, y in zip(w_local, want_local)) < 1e-12
          and A.override_frame_weights(5, 2, profile='local', radius=0)
          == [0.0, 0.0, 1.0, 0.0, 0.0])
    inp = make_override_inputs(DEVICE, torch.float32, seed=41)
    frames, tpf, seq = inp['frames'], inp['tpf'], inp['seq']
    a = 3
    frames_b = frames - a
    seq_b = frames_b * tpf
    pad = inp['q'].size(1) - seq_b
    gen = torch.Generator(device='cpu')
    gen.manual_seed(43)
    heads, head_dim = inp['q'].size(2), inp['q'].size(3)
    qb_pre = torch.randn(1, seq_b, heads, head_dim, generator=gen)
    kb_pre = torch.randn(1, seq_b, heads, head_dim, generator=gen)
    vb = torch.randn(1, seq_b, heads, head_dim, generator=gen)
    zeros = torch.zeros(1, pad, heads, head_dim)
    qb_pre, kb_pre, vb = (torch.cat([u, zeros], 1).to(DEVICE)
                          for u in (qb_pre, kb_pre, vb))
    h, w = int(inp['grid'][0, 1]), int(inp['grid'][0, 2])
    grid_b = torch.tensor([[frames_b, h, w]], dtype=torch.long)
    freqs = make_freqs(head_dim, DEVICE)
    qb = apply_model_rope(qb_pre, grid_b, freqs)
    kb = apply_model_rope(kb_pre, grid_b, freqs)
    ref_b = dense_stream_output(qb_pre[0, :seq_b].double(),
                                kb_pre[0, :seq_b].double(),
                                vb[0, :seq_b].double(), frames_b, h, w,
                                torch.arange(frames_b))
    o2 = inp['refs'][1]
    grid3 = torch.cat([inp['grid'], grid_b])
    lens3 = torch.tensor([seq, seq, seq_b], dtype=torch.long)

    def run3(lam, weights):
        return A.override_attention(
            q=torch.cat([inp['q'], qb]), k=torch.cat([inp['k'], kb]),
            v=torch.cat([inp['v'], vb]), seq_lens=lens3, grid_sizes=grid3,
            override=dict(weights=weights, blend='linear', anchor=a,
                          stream_b=dict(weight=lam)))
    for lam in (0.5, 1.0):
        out = run3(lam, [1.0] * frames)
        err = 0.0
        for f in range(frames):
            rows = slice(f * tpf, (f + 1) * tpf)
            want = o2[rows]
            if f > a:
                rb = slice((f - a) * tpf, (f - a + 1) * tpf)
                want = want + lam * (ref_b[rb] - want)
            err = max(err, rel_err(out[0, rows], want))
        err_b = rel_err(out[2, :seq_b], ref_b)
        check(f"Stream B (lambda {lam}): latents after Img2 = O2 + lambda "
              "(O_B - O2), Img2's latent and before = O2 (dense)",
              err < 3e-2 and err_b < 3e-2, f"rel err {err:.1e} / {err_b:.1e}")
    two = run_override(inp, [1.0] * frames)
    zero = run3(0.0, [1.0] * frames)
    check("Stream B weight 0 changes nothing for streams 1 and 2",
          max_abs(zero[:2, :seq], two[:, :seq]) < 1e-5)
    # Round 13, Gordon's K_A12 / V_B0: stream 2's queries read Img2's latent
    # with stream 2's keys and Stream B's frame-0 values.
    v2_clean = inp['v'][1, :seq].double().clone()
    v2_clean[a * tpf:(a + 1) * tpf] = vb[0, :tpf].double()
    ref_clean = dense_stream_output(inp['q_pre'][1, :seq], inp['k_pre'][1, :seq],
                                    v2_clean, frames, h, w, torch.arange(frames))
    for scope in ('before', 'all'):
        out = A.override_attention(
            q=torch.cat([inp['q'], qb]), k=torch.cat([inp['k'], kb]),
            v=torch.cat([inp['v'], vb]), seq_lens=lens3, grid_sizes=grid3,
            override=dict(weights=[1.0] * frames, blend='linear', anchor=a,
                          stream_b=dict(weight=0.0), img2_value=scope))
        err = 0.0
        for f in range(frames):
            rows = slice(f * tpf, (f + 1) * tpf)
            clean = f < a or (scope == 'all' and f != a)
            want = ref_clean[rows] if clean else o2[rows]
            err = max(err, rel_err(out[1, rows], want),
                      rel_err(out[0, rows], want))  # w = 1: stream 1 = O2
        err_b = rel_err(out[2, :seq_b], ref_b)
        moved = rel_err(out[1, :tpf], o2[:tpf])
        check(f"clean Img2 values ({scope}): stream 2's queries on "
              f"{'latents before Img2' if scope == 'before' else 'every latent but Img2'}"
              " use stream 2's keys and Stream B's frame-0 values at Img2's "
              "latent; the rest and Stream B are unchanged (dense)",
              err < 3e-2 and err_b < 3e-2 and moved > 1e-3,
              f"rel err {err:.1e} / {err_b:.1e}, change {moved:.1e}")
    bad = 0
    for extra in (dict(img2_value='after', stream_b=dict(weight=0.0)),
                  dict(img2_value='before', stream_b=None)):
        try:
            sb = extra.pop('stream_b')
            A.override_attention(
                q=torch.cat([inp['q'], qb]) if sb else inp['q'],
                k=torch.cat([inp['k'], kb]) if sb else inp['k'],
                v=torch.cat([inp['v'], vb]) if sb else inp['v'],
                seq_lens=lens3 if sb else inp['seq_lens'],
                grid_sizes=grid3 if sb else inp['grid'],
                override=dict(weights=[1.0] * frames, anchor=a, stream_b=sb,
                              **extra))
        except ValueError:
            bad += 1
    check("clean Img2 values: an unknown scope, or no Stream B, is refused",
          bad == 2)
    # Round 14: the single stream (stream 2's tokens) plus Stream B.
    lens2 = torch.tensor([seq, seq_b], dtype=torch.long)
    grid2 = torch.cat([inp['grid'][1:2], grid_b])
    for scope in ('before', 'all'):
        out = A.override_attention(
            q=torch.cat([inp['q'][1:2], qb]), k=torch.cat([inp['k'][1:2], kb]),
            v=torch.cat([inp['v'][1:2], vb]), seq_lens=lens2, grid_sizes=grid2,
            override=dict(single=True, anchor=a, img2_value=scope))
        err = 0.0
        for f in range(frames):
            rows = slice(f * tpf, (f + 1) * tpf)
            clean = f < a or (scope == 'all' and f != a)
            err = max(err, rel_err(out[0, rows],
                                   ref_clean[rows] if clean else o2[rows]))
        err_b = rel_err(out[1, :seq_b], ref_b)
        check(f"single stream + Stream B ({scope}): the video stream reads "
              "Img2's latent with its own keys and Stream B's frame-0 values; "
              "Stream B unchanged (dense)", err < 3e-2 and err_b < 3e-2,
              f"rel err {err:.1e} / {err_b:.1e}")
    bad = 0
    for kwargs in (dict(scope='after'), dict(grid=torch.cat(
            [inp['grid'][1:2], torch.tensor([[frames_b + 1, h, w]])])),
            dict(lens=torch.tensor([seq, seq_b - tpf], dtype=torch.long))):
        try:
            A.override_attention(
                q=torch.cat([inp['q'][1:2], qb]), k=torch.cat([inp['k'][1:2], kb]),
                v=torch.cat([inp['v'][1:2], vb]),
                seq_lens=kwargs.get('lens', lens2),
                grid_sizes=kwargs.get('grid', grid2),
                override=dict(single=True, anchor=a,
                              img2_value=kwargs.get('scope', 'before')))
        except ValueError:
            bad += 1
    check("single stream + Stream B: an unknown scope, or a Stream B that "
          "does not cover latents a..F-1 (grid or length), is refused", bad == 3)
    bad = 0
    for kwargs in (dict(stream_b=None), dict(grid=torch.cat(
            [inp['grid'], torch.tensor([[frames_b + 1, h, w]])]))):
        try:
            A.override_attention(
                q=torch.cat([inp['q'], qb]), k=torch.cat([inp['k'], kb]),
                v=torch.cat([inp['v'], vb]), seq_lens=lens3,
                grid_sizes=kwargs.get('grid', grid3),
                override=dict(weights=[1.0] * frames, anchor=a,
                              stream_b=kwargs.get('stream_b',
                                                  dict(weight=0.5))))
        except ValueError:
            bad += 1
    check("Stream B: a third stream needs Stream B settings and the "
          "latents a..F-1", bad == 2)


def run_route(inp, weights=None, anchor=3, **extra):
    frames = inp['frames']
    return A.override_attention(
        q=inp['q'].clone(), k=inp['k'].clone(), v=inp['v'].clone(),
        seq_lens=inp['seq_lens'], grid_sizes=inp['grid'],
        override=dict(weights=weights or [1.0] * frames, blend='linear',
                      anchor=anchor, mode='route', **extra))


def test_override_route():
    """Round 6, Direction C: stream 1 = attn(q2, k2, v1), O2 at Img2's
    latent, against a dense float64 reference."""
    inp = make_override_inputs(DEVICE, torch.float32, seed=51)
    frames, tpf, seq = inp['frames'], inp['tpf'], inp['seq']
    h, w = int(inp['grid'][0, 1]), int(inp['grid'][0, 2])
    a = 3
    o1, o2 = inp['refs']
    ref_route = dense_stream_output(inp['q_pre'][1, :seq], inp['k_pre'][1, :seq],
                                    inp['v'][0, :seq].double(), frames, h, w,
                                    torch.arange(frames))
    out = run_route(inp)
    rows_a = slice(a * tpf, (a + 1) * tpf)
    err = max(rel_err(out[0, f * tpf:(f + 1) * tpf],
                      ref_route[f * tpf:(f + 1) * tpf])
              for f in range(frames) if f != a)
    check("route: stream 1 = attn(q2, k2, v1) on every latent but Img2's "
          "(dense)", err < 3e-2, f"rel err {err:.1e}")
    check("route: Img2's latent takes O2 (bit-identical to stream 2)",
          torch.equal(out[0, rows_a], out[1, rows_a]))
    err2 = rel_err(out[1, :seq], o2)
    check("route: stream 2 keeps its own attention (dense)", err2 < 3e-2,
          f"rel err {err2:.1e}")
    gap1, gap2 = rel_err(ref_route, o1), rel_err(ref_route, o2)
    check("route test: the routed output differs clearly from O1 and O2",
          gap1 > 0.1 and gap2 > 0.1, f"{gap1:.2f} / {gap2:.2f}")
    out5 = run_route(inp, anchor=5)
    check("route: the O2 copy follows the anchor",
          torch.equal(out5[0, 5 * tpf:6 * tpf], out5[1, 5 * tpf:6 * tpf])
          and rel_err(out5[0, rows_a], ref_route[rows_a]) < 3e-2)
    bad = 0
    for kwargs in (dict(weights=[1.0] * (frames - 1) + [0.5]),
                   dict(anchor=frames),
                   dict(stream_b=dict(weight=0.5))):
        try:
            run_route(inp, **kwargs)
        except ValueError:
            bad += 1
    try:
        A.override_attention(
            q=inp['q'], k=inp['k'], v=inp['v'], seq_lens=inp['seq_lens'],
            grid_sizes=inp['grid'],
            override=dict(weights=[1.0] * frames, anchor=a, mode='mix'))
    except ValueError:
        bad += 1
    check("route rejects w < 1, an anchor outside, Stream B and bad modes",
          bad == 4, f"{bad} of 4")
    A.ATTN_PROBE.configure([0], samples=4)
    A.ATTN_PROBE.begin_forward(0, 'cond')
    probed = run_route(inp)
    rec = A.ATTN_PROBE.data.get((0, 0))
    ok = torch.equal(probed, out) and A.ATTN_PROBE.failed is None
    if rec is not None:
        r1 = ref_route.reshape(frames, tpf, *ref_route.shape[1:])
        want = r1.pow(2).mean(dim=(1, 3)).sqrt()
        ok = ok and max_abs(rec['o1_rms'], want.cpu()) < 1e-3
    check("route probe is read-only and records RMS(O_route) as o1",
          ok and rec is not None, A.ATTN_PROBE.failed or '')
    A.ATTN_PROBE.reset()


def run_paired(inp, variant, active=True, weight=0.5, fn=None):
    fn = fn or A.paired_anchor_attention
    kwargs = dict(q=inp['q'].clone(), k=inp['k'].clone(), v=inp['v'].clone(),
                  seq_lens=inp['seq_lens'], grid_sizes=inp['grid'],
                  freqs=inp['freqs'], anchor_index=inp['a'],
                  stream_weight=weight, prefix_anchor_active=active)
    if fn is A.paired_anchor_attention:
        kwargs['variant'] = variant
    return fn(**kwargs)


def test_flash_lse(device):
    gen = torch.Generator(device='cpu')
    gen.manual_seed(1)
    q = torch.randn(1, 37, 4, 128, generator=gen).to(device, torch.bfloat16)
    k = torch.randn(1, 53, 4, 128, generator=gen).to(device, torch.bfloat16)
    v = torch.randn(1, 53, 4, 128, generator=gen).to(device, torch.bfloat16)
    logits = torch.einsum('bqhd,bkhd->bqhk', q.double(), k.double()) / math.sqrt(128)
    ref_lse = torch.logsumexp(logits, dim=-1)
    ref_out = torch.einsum('bqhk,bkhd->bqhd', torch.softmax(logits, -1),
                           v.double())
    try:
        out, lse = A.flash_attention(q=q, k=k, v=v, return_lse=True)
        ok = (tuple(lse.shape) == (1, 37, 4) and max_abs(lse, ref_lse) < 1e-2
              and max_abs(out, ref_out) < 2e-2)
        detail = (f"lse err {max_abs(lse, ref_lse):.2e}, out err "
                  f"{max_abs(out, ref_out):.2e}, lse shape {tuple(lse.shape)}")
    except Exception as error:  # noqa: BLE001
        ok, detail = False, repr(error)
    # Not fatal: without a usable FlashAttention LSE the runs use the exact
    # (slower) PyTorch path, VR_LSE_BACKEND=torch.
    print(f"[{'PASS' if ok else 'WARN'}] FlashAttention return_lse matches "
          f"dense  ({detail})", flush=True)
    backend = 'flash' if ok else 'torch'
    out, lse = A._torch_attention_lse(q, k, v)
    check('torch LSE fallback matches dense',
          max_abs(lse, ref_lse) < 1e-3 and max_abs(out, ref_out) < 2e-2,
          f"lse err {max_abs(lse, ref_lse):.2e}")
    return backend


def test_baseline_identity():
    for dtype in (torch.bfloat16, torch.float32):
        inp = make_inputs(DEVICE, dtype)
        for active in (True, False):
            new = run_paired(inp, None, active=active)
            old = run_paired(inp, None, active=active,
                             fn=original_paired_anchor_attention)
            check(f"baseline path bit-identical to the pre-round-1 code "
                  f"({dtype}, anchor route {'on' if active else 'off'})",
                  torch.equal(new, old))


def test_probe_read_only():
    inp = make_inputs(DEVICE, torch.bfloat16)
    A.ATTN_PROBE.configure([0], samples=4)
    for name, variant in (('baseline', None),
                          ('SinkSplit', dict(mass='split'))):
        A.ATTN_PROBE.begin_forward(0, 'uncond')  # not recorded
        plain = run_paired(inp, variant)
        A.ATTN_PROBE.begin_forward(0, 'cond')  # recorded
        probed = run_paired(inp, variant)
        check(f"probe does not change outputs ({name})",
              torch.equal(plain, probed) and A.ATTN_PROBE.failed is None,
              A.ATTN_PROBE.failed or '')
    A.ATTN_PROBE.reset()


def test_variants_exact():
    """torch LSE backend + float32 inputs: the merge logic itself."""
    saved = A._LSE_BACKEND
    A._LSE_BACKEND = 'torch'
    inp = make_inputs(DEVICE, torch.float32, seed=3)
    tpf, a = inp['T'], inp['a']
    base = run_paired(inp, None)
    for name, variant in VARIANTS:
        out = run_paired(inp, variant)
        ref = dense_prefix(inp, variant)['out']
        err = max_abs(out[0, :a * tpf], ref)
        same_rest = torch.equal(out[0, a * tpf:], base[0, a * tpf:]) and \
            torch.equal(out[1], base[1])
        check(f"variant {name}: prefix equals dense reference", err < 2e-4,
              f"max abs err {err:.2e}")
        check(f"variant {name}: anchor/suffix/Stream-B rows unchanged",
              same_rest)
    natural = dense_prefix(inp, None)['out']
    check("dense reference reproduces the baseline prefix",
          max_abs(base[0, :a * tpf], natural) < 3e-2,
          f"{max_abs(base[0, :a * tpf], natural):.2e} (bf16 kernels)")
    A._LSE_BACKEND = saved


def test_variants_flash(backend):
    """Production backend + bfloat16 inputs, against the exact torch path."""
    if backend != 'flash':
        print("[SKIP] FlashAttention LSE unavailable; jobs will use "
              "VR_LSE_BACKEND=torch")
        return
    inp = make_inputs(DEVICE, torch.bfloat16, seed=5)
    tpf, a = inp['T'], inp['a']
    for name, variant in VARIANTS:
        A._LSE_BACKEND = 'flash'
        fast = run_paired(inp, variant)
        A._LSE_BACKEND = 'torch'
        exact = run_paired(inp, variant)
        err = max_abs(fast[0, :a * tpf], exact[0, :a * tpf])
        check(f"variant {name}: flash LSE path equals exact path (bf16)",
              err < 2e-2, f"max abs err {err:.2e}")
    A._LSE_BACKEND = 'flash'
    base = run_paired(inp, None)
    same = run_paired(inp, dict(anchor_key='A'))
    check("all-default variant dict runs the baseline path bit-identically",
          torch.equal(base, same))


def test_probe_paired():
    saved = A._LSE_BACKEND
    A._LSE_BACKEND = 'torch'
    inp = make_inputs(DEVICE, torch.float32, seed=7)
    frames, a, h, w, tpf, heads = (inp[x] for x in
                                   ('F', 'a', 'h', 'w', 'T', 'H'))
    n_b = inp['nB']
    cases = [('baseline', None), ('MirrorSink', dict(anchor_key='B',
                                                     mirror=True)),
             ('SinkSplit', dict(mass='split')),
             ('MirrorMass', dict(mass='mirror_mass')),
             ('Img1Fade', dict(mass='img1_fade'))]
    for name, variant in cases:
        A.ATTN_PROBE.configure([0], samples=4, seed=11)
        A.ATTN_PROBE.begin_forward(0, 'cond')
        run_paired(inp, variant)
        rec = A.ATTN_PROBE.data.get((0, 0))
        ok = rec is not None and A.ATTN_PROBE.failed is None
        check(f"probe recorded paired layer ({name})", ok,
              A.ATTN_PROBE.failed or '')
        if not ok:
            continue
        idx = A.ATTN_PROBE.spatial_index(tpf, DEVICE)
        ref = dense_prefix(inp, variant)
        eff = rec['prefix_eff']  # [a, F, H]
        nat = rec['prefix_nat']
        err_eff = err_nat = 0.0
        for t in range(a):
            rows = t * tpf + idx
            r_eff = ref['eff_mass'][rows].mean(0).t()  # [F, H]
            r_nat = ref['nat_mass'][rows].mean(0).t()
            err_eff = max(err_eff, max_abs(eff[t], r_eff.cpu()))
            err_nat = max(err_nat, max_abs(nat[t], r_nat.cpu()))
        sums = eff.sum(1)
        check(f"probe prefix masses equal dense reference ({name})",
              err_eff < 1e-4 and err_nat < 1e-4
              and max_abs(sums, torch.ones_like(sums)) < 1e-4,
              f"eff err {err_eff:.1e}, nat err {err_nat:.1e}")
        # Stream B: ordinary I2V of Img2 (local positions 0..nB-1).
        main_len, suffix_len = frames * tpf, n_b * tpf
        sb = dense_stream_masses(inp['q_pre'][1, :suffix_len],
                                 inp['k_pre'][1, :suffix_len], n_b, h, w,
                                 torch.arange(n_b))
        err_b = max(
            max_abs(rec['stream_b'][t], sb[t * tpf + idx].mean(0).t().cpu())
            for t in range(n_b))
        # Main suffix queries: ordinary keys only (anchor removed).
        to_pos = torch.tensor([f if f <= a else f - 1 for f in range(frames)])
        mask = torch.ones(main_len, dtype=torch.bool, device=DEVICE)
        mask[a * tpf:(a + 1) * tpf] = False
        sm = dense_stream_masses(inp['q_pre'][0, :main_len],
                                 inp['k_pre'][0, :main_len], frames, h, w,
                                 to_pos, key_mask=mask)
        err_s = max(
            max_abs(rec['suffix_main'][t - a - 1],
                    sm[t * tpf + idx].mean(0).t().cpu())
            for t in range(a + 1, frames))
        check(f"probe Stream-B and main-suffix masses equal dense ({name})",
              err_b < 1e-4 and err_s < 1e-4,
              f"stream B err {err_b:.1e}, suffix err {err_s:.1e}")
    # finalize -> merged file + summary
    A.ATTN_PROBE.configure([0, 1], samples=4)
    for step in (0, 1):
        A.ATTN_PROBE.begin_forward(step, 'cond')
        run_paired(inp, dict(mass='split'), active=(step == 0))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'probe.pt')
        A.ATTN_PROBE.finalize(path, extra_meta=dict(
            timesteps=[999, 991], boundary_timestep=900.0,
            variant=dict(mass='split')))
        merged = torch.load(path, map_location='cpu', weights_only=False)
        shape = tuple(merged['tensors']['prefix_eff'].shape)
        summary = open(os.path.join(tmp, 'probe_summary.txt')).read()
        check("probe finalize writes merged tensors and a summary",
              shape == (2, 1, a, frames, heads) and
              '[A] Prefix query frame' in summary and
              'anchor route OFF' in summary,
              f"prefix_eff shape {shape}")
    A._LSE_BACKEND = saved


def test_probe_plain():
    frames, h, w, heads = 5, 2, 3, 4
    tpf = h * w
    gen = torch.Generator(device='cpu')
    gen.manual_seed(13)
    q_pre = torch.randn(1, frames * tpf, heads, 128, generator=gen).to(DEVICE)
    k_pre = torch.randn(1, frames * tpf, heads, 128, generator=gen).to(DEVICE)
    grid = torch.tensor([[frames, h, w]], dtype=torch.long)
    freqs = make_freqs(128, DEVICE)
    q = apply_model_rope(q_pre, grid, freqs)
    k = apply_model_rope(k_pre, grid, freqs)
    A.ATTN_PROBE.configure([3], samples=4)
    A.ATTN_PROBE.begin_forward(3, 'cond')
    A.probe_plain_self_attention(q, k, grid)
    rec = A.ATTN_PROBE.data.get((3, 0))
    ok = rec is not None
    err = float('inf')
    if ok:
        idx = A.ATTN_PROBE.spatial_index(tpf, DEVICE)
        ref = dense_stream_masses(q_pre[0].double(), k_pre[0].double(),
                                  frames, h, w, torch.arange(frames))
        err = max(max_abs(rec['plain'][t], ref[t * tpf + idx].mean(0).t().cpu())
                  for t in range(frames))
    check("plain-I2V probe equals dense reference", ok and err < 1e-4,
          f"err {err:.1e}")
    A.ATTN_PROBE.reset()


# ---------------------------------------------------------------------------
# Tiny WanModel forward (real model code, GPU only)
# ---------------------------------------------------------------------------
def tiny_model(num_heads=2, dim=256, num_layers=2):
    from wan.modules.model import WanModel
    torch.manual_seed(0)
    model = WanModel(model_type='i2v', patch_size=(1, 2, 2), text_len=16,
                     in_dim=36, dim=dim, ffn_dim=2 * dim, freq_dim=256,
                     text_dim=64, out_dim=16, num_heads=num_heads,
                     num_layers=num_layers, window_size=(-1, -1), qk_norm=True,
                     cross_attn_norm=True, eps=1e-6)
    torch.nn.init.normal_(model.head.head.weight, std=0.02)
    return model.eval().requires_grad_(False)


def tiny_inputs(device, frames=7, anchor=3, lat_h=4, lat_w=6, seed=0,
                rev=False):
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    noise = torch.randn(16, frames, lat_h, lat_w, generator=gen).to(device)
    y_main = torch.randn(20, frames, lat_h, lat_w, generator=gen).to(device)
    y_suffix = torch.randn(20, frames - anchor, lat_h, lat_w,
                           generator=gen).to(device)
    ctx = torch.randn(5, 64, generator=gen).to(device)
    tokens = frames * (lat_h // 2) * (lat_w // 2)
    inp = dict(x=[noise, noise[:, anchor:]], y=[y_main, y_suffix],
               context=[ctx, ctx], t=torch.tensor([900.0, 900.0],
                                                  device=device),
               tokens=tokens, anchor=anchor, noise=noise, y_main=y_main,
               ctx=ctx)
    if rev:
        # Stream R: anchor slot, then prefix frames a-1..1 (reverse order).
        y_rev = torch.randn(20, anchor, lat_h, lat_w, generator=gen).to(device)
        ctx_rev = torch.randn(4, 64, generator=gen).to(device)
        inp.update(x=inp['x'] + [noise[:, 1:anchor + 1].flip(1)],
                   y=inp['y'] + [y_rev], context=inp['context'] + [ctx_rev],
                   t=torch.tensor([900.0, 900.0, 900.0], device=device))
    return inp


def tiny_override_inputs(device, frames=7, lat_h=4, lat_w=6, seed=4):
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    noise = torch.randn(16, frames, lat_h, lat_w, generator=gen).to(device)
    y1 = torch.randn(20, frames, lat_h, lat_w, generator=gen).to(device)
    y2 = torch.randn(20, frames, lat_h, lat_w, generator=gen).to(device)
    ctx = torch.randn(5, 64, generator=gen).to(device)
    return dict(x=[noise, noise], y=[y1, y2], context=[ctx, ctx], ctx=ctx,
                t=torch.tensor([900.0, 900.0], device=device),
                tokens=frames * (lat_h // 2) * (lat_w // 2))


def override_forward(model, inp, weights, blend='linear', seq_len=None,
                     mode='blend', both=False):
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
        out = model(inp['x'], t=inp['t'], context=inp['context'],
                    seq_len=seq_len or inp['tokens'], y=inp['y'],
                    override=dict(weights=weights, blend=blend, anchor=3,
                                  mode=mode))
    return out if both else out[0]


def model_forward(model, inp, variant, seq_len=None, active=True):
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
        return model(inp['x'], t=inp['t'], context=inp['context'],
                     seq_len=seq_len or inp['tokens'], y=inp['y'],
                     paired_anchor_index=inp['anchor'],
                     paired_stream_weight=0.5,
                     paired_prefix_anchor_active=active,
                     paired_variant=variant)[0]


def test_tiny_model():
    model = tiny_model().to(DEVICE).to(torch.bfloat16)
    inp = tiny_inputs(DEVICE)
    base = model_forward(model, inp, None)
    nat = model_forward(model, inp, dict(anchor_key='A'))
    err = rel_err(nat, base)
    check("tiny WanModel: all-default variant equals baseline", err < 3e-2,
          f"rel err {err:.2e}")
    for name, variant in VARIANTS + STREAM_VARIANTS:
        out = model_forward(model, inp, variant)
        check(f"tiny WanModel runs variant {name}",
              bool(torch.isfinite(out).all()) and out.shape == base.shape)
    off = model_forward(model, inp, dict(anchor_key='B', mirror=True),
                        active=False)
    off_base = model_forward(model, inp, None, active=False)
    check("tiny WanModel: variant ignored when the anchor route is off",
          torch.equal(off, off_base))
    A.ATTN_PROBE.configure([0], samples=4)
    A.ATTN_PROBE.begin_forward(0, 'cond')
    model_forward(model, inp, dict(mass='split'))
    layers = sorted(layer for _, layer in A.ATTN_PROBE.data)
    check("tiny WanModel: probe records every layer (paired)",
          layers == [0, 1] and A.ATTN_PROBE.failed is None,
          f"layers {layers} {A.ATTN_PROBE.failed or ''}")
    A.ATTN_PROBE.configure([0], samples=4)
    A.ATTN_PROBE.begin_forward(0, 'cond')
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
        model([inp['noise'][:, :6]], t=inp['t'][:1], context=[inp['ctx']],
              seq_len=6 * 6, y=[inp['y_main'][:, :6]])
    rec = A.ATTN_PROBE.data.get((0, 1), {})
    check("tiny WanModel: probe records the plain I2V path",
          'plain' in rec and A.ATTN_PROBE.failed is None)
    A.ATTN_PROBE.reset()
    inp_r = tiny_inputs(DEVICE, rev=True)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
        outs = model(inp_r['x'], t=inp_r['t'], context=inp_r['context'],
                     seq_len=inp_r['tokens'], y=inp_r['y'],
                     paired_anchor_index=inp_r['anchor'],
                     paired_stream_weight=0.5,
                     paired_variant=dict(rev_strength=1.0))
    check("tiny WanModel: three-stream (Stream R) forward",
          len(outs) == 3 and outs[0].shape == base.shape
          and tuple(outs[2].shape) == tuple(inp_r['x'][2].shape)
          and all(bool(torch.isfinite(o).all()) for o in outs))
    zero = model_forward(model, inp_r, dict(rev_strength=0.0))
    err = rel_err(zero, base)
    check("tiny WanModel: Stream R strength 0 matches the two-stream output",
          err < 3e-2, f"rel err {err:.2e}")
    coupled = model_forward(model, inp_r, dict(rev_strength=1.0,
                                               rev_couple=True))
    check("tiny WanModel: coupled Stream R runs and differs from uncoupled",
          bool(torch.isfinite(coupled).all()) and rel_err(
              coupled, model_forward(model, inp_r, dict(rev_strength=1.0)))
          > 1e-4)
    ov_inp = tiny_override_inputs(DEVICE)
    ov_full = override_forward(model, ov_inp, [1.0] * 7)
    ov_none = override_forward(model, ov_inp, [0.0] * 7)
    check("tiny WanModel: override forward runs; full and none differ",
          bool(torch.isfinite(ov_full).all())
          and rel_err(ov_full, ov_none) > 1e-4)
    full = model_forward(model, inp_r, dict(rev_strength=1.0))
    check("tiny WanModel: Stream R strength 1 changes the prediction",
          rel_err(full[:, 1:3], base[:, 1:3]) > 1e-3,
          f"prefix rel diff {rel_err(full[:, 1:3], base[:, 1:3]):.1e}")


# ---------------------------------------------------------------------------
# Tiny end-to-end generate(): the real sampling loop of wan/image2video.py
# with a tiny DiT, a fake VAE and a fake T5 (one GPU, a few seconds).
# ---------------------------------------------------------------------------
def _text_seed(text):
    return sum((i + 1) * ord(c) for i, c in enumerate(text)) % (2**31 - 1)


class FakeT5:

    def __init__(self, dim=64, length=5):
        self.dim, self.length, self.calls = dim, length, []
        self.model = types.SimpleNamespace(to=lambda *a, **k: None,
                                           cpu=lambda *a, **k: None)

    def embed(self, text, device):
        gen = torch.Generator(device='cpu')
        gen.manual_seed(_text_seed(text))
        return torch.randn(self.length, self.dim, generator=gen).to(device)

    def __call__(self, texts, device):
        self.calls.extend(texts)
        return [self.embed(text, device) for text in texts]


class FakeVAE:
    """Frame groups 1, 4, 4, ... -> 16 x 8x-downsampled channels. Causal like
    the real VAE: every latent also carries half the mean of the earlier ones,
    so an Img2 placed mid-video leaks into all later latents."""

    def __init__(self, device):
        gen = torch.Generator(device='cpu')
        gen.manual_seed(5)
        self.proj = torch.randn(16, 3, generator=gen).to(device)

    def encode(self, videos):
        out = []
        for video in videos:
            c, f, h, w = video.shape
            # Frames inside a latent are weighted 1:2:3:4 (frame position
            # matters, as in the real encoder).
            wts = [0.1, 0.2, 0.3, 0.4]
            groups = [video[:, 0]] + [
                sum(wts[k] * video[:, 1 + 4 * i + k] for k in range(4))
                for i in range((f - 1) // 4)
            ]
            g = torch.stack(groups, dim=1)
            g = g.reshape(c, g.shape[1], h // 8, 8, w // 8, 8).mean(5).mean(3)
            # Causal history with decaying weights: a still clip's latents
            # differ by position, and Img2 leaks into the latents after it.
            leaky = [g[:, 0]] + [
                g[:, j] + 0.5 * sum(0.5 ** (j - i) * g[:, i] for i in range(j))
                for j in range(1, g.shape[1])]
            g = torch.stack(leaky, dim=1)
            out.append(torch.einsum('kc,cthw->kthw', self.proj, g.float()))
        return out

    def decode(self, latents):
        return [u[:3].clone() for u in latents]


class SpyModel:

    def __init__(self, model):
        self.model, self.calls = model, []

    def parameters(self):
        return self.model.parameters()

    def __call__(self, x, t, **kwargs):
        out = self.model(x, t=t, **kwargs)
        self.calls.append(dict(kwargs, x=[u.clone() for u in x],
                               t=t.clone(), out=out[0].clone(),
                               outs=[u.clone() for u in out]))
        return out


TINY_PAIRED = dict(
    image_latent_indices=[0, 3],
    image_condition_paired_anchor_attention=True,
    image_condition_paired_stream_weight=0.5,
    image_condition_paired_suffix_fade_start_step=3,
    image_condition_paired_suffix_fade_end_step=5,
    image_condition_paired_prefix_route_switch_step=4)


def tiny_pipeline():
    from wan.image2video import WanI2V
    pipe = object.__new__(WanI2V)
    spy = SpyModel(tiny_model().to(DEVICE).to(torch.bfloat16))
    pipe.__dict__.update(
        device=DEVICE, config=None, rank=0, t5_cpu=False, init_on_cpu=False,
        dit_fsdp=False, _paired_high_noise_model_released=False,
        num_train_timesteps=1000, boundary=0.9, param_dtype=torch.bfloat16,
        text_encoder=FakeT5(), vae_stride=(4, 8, 8), patch_size=(1, 2, 2),
        vae=FakeVAE(DEVICE), low_noise_model=spy, high_noise_model=spy,
        sp_size=1, sample_neg_prompt='negative prompt')
    return pipe, spy


def tiny_images(count):
    import numpy as np
    from PIL import Image
    rng = np.random.RandomState(3)
    return [Image.fromarray(rng.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            for _ in range(count)]


def tiny_generate(paired=True, override=False, pipeline=None, **kwargs):
    pipe, spy = pipeline if pipeline is not None else tiny_pipeline()
    two = paired or override
    images = tiny_images(2 if two else 1)
    extra = dict(TINY_PAIRED) if paired else (
        dict(image_latent_indices=[0, 3]) if override else {})
    extra.update(kwargs)
    video = pipe.generate('main prompt', images if two else images[0],
                          max_area=64 * 64, frame_num=25, shift=3.0,
                          sampling_steps=6, guide_scale=3.5, seed=7,
                          offload_model=False, **extra)
    return video, spy, pipe


def test_tiny_generate():
    try:
        from wan.image2video import fuse_reverse_prediction
    except Exception as error:  # noqa: BLE001
        check("import wan.image2video for the end-to-end test", False,
              repr(error))
        return
    # fuse_reverse_prediction on explicit numbers.
    a = 4
    noise_pred = torch.zeros(2, 7, 1, 1)
    rev_pred = torch.arange(1, a + 1, dtype=torch.float32).view(1, a, 1, 1)
    rev_pred = rev_pred.expand(2, a, 1, 1).clone()
    weights = [expected_rev_weight(t, a, 1.0, 'linear') for t in range(a)]
    fused = fuse_reverse_prediction(noise_pred, rev_pred, weights)
    want = torch.zeros(2, 7, 1, 1)
    for t in range(1, a):
        want[:, t] = weights[t] * (a - t + 1)  # Stream R frame a-t holds a-t+1
    check("fuse_reverse_prediction: latent t gets w_t * StreamR[a-t]",
          max_abs(fused, want) < 1e-6 and torch.equal(noise_pred,
                                                       torch.zeros(2, 7, 1, 1)))

    a, steps, until = 3, 6, 3
    tokens = 16  # 8x8 latent grid, 2x2 patches
    plain, _, _ = tiny_generate(paired=False)
    check("tiny generate: plain I2V runs",
          plain is not None and bool(torch.isfinite(plain).all()))
    base, spy, _ = tiny_generate()
    check("tiny generate: paired baseline runs, 2 passes per step",
          base is not None and bool(torch.isfinite(base).all())
          and len(spy.calls) == 2 * steps
          and all(len(c['x']) == 2 for c in spy.calls))
    pb, spy, pipe = tiny_generate(suffix_prompt='suffix prompt')
    emb = pipe.text_encoder.embed
    check("tiny generate: PB gives Stream B its own prompt",
          torch.equal(spy.calls[0]['context'][1], emb('suffix prompt', DEVICE))
          and torch.equal(spy.calls[0]['context'][0], emb('main prompt',
                                                          DEVICE)))
    kg, spy, _ = tiny_generate(keyframe_guidance_scale=3.0,
                               keyframe_guidance_until_step=until)
    ok = len(spy.calls) == 2 * steps + until
    for s in range(until):  # cond, uncond, Img2-free pass in guided steps
        c = spy.calls[3 * s + 2]
        ok = ok and (c['paired_stream_weight'] == 0.0
                     and not c['paired_prefix_anchor_active']
                     and c['paired_variant'] is None and len(c['x']) == 2)
    check("tiny generate: KG adds one Img2-free pass per guided step",
          ok and not torch.equal(kg, base))

    ra, spy, pipe = tiny_generate(
        suffix_prompt='suffix prompt', reverse_prompt='reverse prompt',
        reverse_stream_mode='attn', reverse_stream_until_step=until)
    three = [c for c in spy.calls if len(c['x']) == 3]
    ok = len(three) == 2 * until and len(spy.calls) == 2 * steps
    for i, c in enumerate(three):
        x0, xr = c['x'][0], c['x'][2]
        ok = ok and xr.shape[1] == a and all(
            torch.equal(xr[:, j], x0[:, a - j]) for j in range(a))
        ok = ok and torch.equal(c['y'][2][:4, 0], torch.ones_like(
            c['y'][2][:4, 0])) and not bool(c['y'][2][:4, 1:].any())
        want_ctx = emb('reverse prompt' if i % 2 == 0 else 'negative prompt',
                       DEVICE)
        ok = ok and torch.equal(c['context'][2], want_ctx)
        ok = ok and c['t'].numel() == 3
        ok = ok and c['paired_variant'] == dict(rev_strength=1.0,
                                                rev_shape='linear')
    check("tiny generate: RA puts Stream R (anchor, a-1..1, Img2 mask, own "
          "prompt) in both CFG passes for steps < until", ok)
    check("tiny generate: RA changes the video", rel_err(ra, pb) > 1e-4,
          f"rel diff {rel_err(ra, pb):.1e}")
    ra0, _, _ = tiny_generate(
        suffix_prompt='suffix prompt', reverse_prompt='reverse prompt',
        reverse_stream_mode='attn', reverse_stream_strength=0.0,
        reverse_stream_until_step=until)
    check("tiny generate: RA strength 0 matches PB", rel_err(ra0, pb) < 5e-2,
          f"rel err {rel_err(ra0, pb):.1e}")

    rc, spy, _ = tiny_generate(
        suffix_prompt='suffix prompt', reverse_prompt='reverse prompt',
        reverse_stream_mode='attn', reverse_stream_until_step=until,
        reverse_stream_couple=True)
    three = [c for c in spy.calls if len(c['x']) == 3]
    check("tiny generate: RC passes rev_couple in both CFG passes",
          len(three) == 2 * until and all(
              c['paired_variant'] == dict(rev_strength=1.0, rev_shape='linear',
                                          rev_couple=True) for c in three)
          and rel_err(rc, ra) > 1e-5,
          f"rel diff vs RA {rel_err(rc, ra):.1e}")
    rc0, _, _ = tiny_generate(
        suffix_prompt='suffix prompt', reverse_prompt='reverse prompt',
        reverse_stream_mode='attn', reverse_stream_strength=0.0,
        reverse_stream_until_step=until, reverse_stream_couple=True)
    check("tiny generate: RC strength 0 matches PB", rel_err(rc0, pb) < 5e-2,
          f"rel err {rel_err(rc0, pb):.1e}")
    import wan.image2video as I2V
    fused_args = []
    real_fuse = I2V.fuse_reverse_prediction

    def spy_fuse(noise_pred, rev_pred, frame_weights):
        fused_args.append((noise_pred.clone(), rev_pred.clone(),
                           list(frame_weights)))
        return real_fuse(noise_pred, rev_pred, frame_weights)

    I2V.fuse_reverse_prediction = spy_fuse
    try:
        ro, spy, _ = tiny_generate(
            suffix_prompt='suffix prompt', reverse_prompt='reverse prompt',
            reverse_stream_mode='pred', reverse_stream_until_step=until)
    finally:
        I2V.fuse_reverse_prediction = real_fuse
    ok = len(fused_args) == until
    g = 3.5
    for s in range(until):  # calls per fused step: cond, uncond, R cond, R uncond
        c, u, rc, ru = (spy.calls[4 * s + i]['out'] for i in range(4))
        main_want = u + g * (c - u)
        rev_want = ru + g * (rc - ru)
        npred, rpred, wts = fused_args[s]
        ok = ok and max_abs(npred, main_want) < 1e-4
        ok = ok and max_abs(rpred, rev_want) < 1e-4
        ok = ok and max(abs(wts[t] - expected_rev_weight(t, a, 1.0, 'linear'))
                        for t in range(a)) < 1e-12
    check("tiny generate: RO fuses the CFG-guided Stream R prediction", ok)
    singles = [(i, c) for i, c in enumerate(spy.calls) if len(c['x']) == 1]
    ok = len(singles) == 2 * until and len(spy.calls) == 2 * steps + 2 * until
    for n, (i, c) in enumerate(singles):
        main = spy.calls[i - 2 - (n % 2)]['x'][0]  # this step's cond pass
        xr = c['x'][0]
        ok = ok and xr.shape[1] == a and c['seq_len'] == a * tokens and all(
            torch.equal(xr[:, j], main[:, a - j]) for j in range(a))
        want_ctx = emb('reverse prompt' if n % 2 == 0 else 'negative prompt',
                       DEVICE)
        ok = ok and torch.equal(c['context'][0], want_ctx)
        ok = ok and 'paired_anchor_index' not in c
    check("tiny generate: RO adds two plain Stream R passes per fused step",
          ok)
    check("tiny generate: RO changes the video", rel_err(ro, pb) > 1e-4,
          f"rel diff {rel_err(ro, pb):.1e}")
    ro0, _, _ = tiny_generate(
        suffix_prompt='suffix prompt', reverse_prompt='reverse prompt',
        reverse_stream_mode='pred', reverse_stream_strength=0.0,
        reverse_stream_until_step=until)
    check("tiny generate: RO strength 0 is bit-identical to PB",
          torch.equal(ro0, pb))

    for name, kwargs in (
            ('SinkSplit variant until step 2',
             dict(image_condition_paired_variant=dict(mass='split'),
                  image_condition_paired_variant_until_step=2)),
            ('TFIX ablation', dict(image_condition_paired_variant=dict(
                suffix_time_fix=True))),
            ('RA cosine + KG', dict(reverse_stream_mode='attn',
                                    reverse_stream_shape='cosine',
                                    reverse_stream_until_step=until,
                                    keyframe_guidance_scale=2.0,
                                    keyframe_guidance_until_step=2))):
        video, _, _ = tiny_generate(**kwargs)
        check(f"tiny generate: {name} runs",
              video is not None and bool(torch.isfinite(video).all()))
    errors = 0
    for paired, kwargs in ((True, dict(reverse_stream_mode='both')),
                           (True, dict(reverse_stream_mode='attn',
                                       reverse_stream_strength=2.0)),
                           (False, dict(reverse_stream_mode='attn')),
                           (True, dict(reverse_stream_mode='pred',
                                       reverse_stream_couple=True))):
        try:
            tiny_generate(paired=paired, **kwargs)
        except ValueError:
            errors += 1
    check("tiny generate: bad Stream R settings are rejected", errors == 4)
    test_tiny_generate_override(steps)
    test_tiny_generate_round4(steps)
    test_tiny_generate_round5(steps)
    test_tiny_generate_round6(steps)
    test_tiny_generate_round7(steps)
    test_tiny_generate_round8(steps)
    test_tiny_generate_round9(steps)
    test_tiny_generate_round10(steps)
    test_tiny_generate_round12(steps)
    test_tiny_generate_round13(steps)
    test_tiny_generate_round14(steps)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'probe.pt')
        tiny_generate(reverse_stream_mode='attn', reverse_stream_until_step=2,
                      probe_steps=[0, 3], probe_samples=4, probe_path=path)
        meta = torch.load(path, map_location='cpu', weights_only=False)['meta']
        check("tiny generate: probe file records the Stream R settings",
              meta.get('reverse_stream', {}).get('mode') == 'attn'
              and meta['steps'] == [0, 3])
    A.ATTN_PROBE.reset()


def test_tiny_generate_override(steps):
    a, frames = 3, 7
    plain, _, _ = tiny_generate(paired=False)
    m1, spy, _ = tiny_generate(paired=False, override=True,
                               override_stream='joint')
    ok = len(spy.calls) == 2 * steps
    for c in spy.calls:
        ok = ok and len(c['x']) == 2 and torch.equal(c['x'][0], c['x'][1])
        m1_mask, m2_mask = c['y'][0][:4], c['y'][1][:4]
        ok = ok and bool(m1_mask[:, 0].all()) and not bool(m1_mask[:, 1:].any())
        ok = ok and bool(m2_mask[:, 0].all()) and bool(m2_mask[:, a].all())
        ok = ok and not bool(m2_mask[:, 1:a].any()) and not bool(
            m2_mask[:, a + 1:].any())
        ok = ok and torch.equal(c['context'][0], c['context'][1])
        ok = ok and c['override']['weights'] == [1.0] * frames
        ok = ok and 'paired_anchor_index' not in c
    joint_y2 = spy.calls[0]['y'][1]
    check("tiny generate: M1 runs two streams on one shared latent, masks "
          "Img1 / Img1+Img2, full-override weights", ok)
    check("tiny generate: M1 changes the video vs plain I2V",
          rel_err(m1, plain) > 1e-4, f"rel diff {rel_err(m1, plain):.1e}")
    sp_video, spy, _ = tiny_generate(paired=False, override=True,
                                     override_stream='splice')
    y1, y2 = spy.calls[0]['y'][0], spy.calls[0]['y'][1]
    ok = torch.equal(y2[4:, a], joint_y2[4:, a])
    ok = ok and torch.equal(y2[4:, :a], y1[4:, :a]) and torch.equal(
        y2[4:, a + 1:], y1[4:, a + 1:])
    leaks = not torch.equal(joint_y2[4:, a + 1:], y1[4:, a + 1:])
    check("tiny generate: M1 splice = Img1-only encoding plus the joint Img2 "
          "latent (the joint encoding leaks Img2 into later latents)",
          ok and leaks)
    weights = expected_override_weights(frames, a, 0.5, 3)
    damp, spy, _ = tiny_generate(paired=False, override=True,
                                 override_stream='joint',
                                 override_min_weight=0.5, override_radius=3,
                                 override_blend='norm')
    check("tiny generate: damped norm override passes w_f and the blend",
          all(max(abs(x - y) for x, y in zip(c['override']['weights'],
                                             weights)) < 1e-12
              and c['override']['blend'] == 'norm' for c in spy.calls)
          and rel_err(damp, m1) > 1e-6)
    own, spy, _ = tiny_generate(paired=False, override=True,
                                override_stream='joint', override_latent='own')
    first, later = spy.calls[0], spy.calls[2 * (steps - 1)]
    check("tiny generate: M1 own latent: same noise at step 0, separate "
          "latents later",
          torch.equal(first['x'][0], first['x'][1])
          and not torch.equal(later['x'][0], later['x'][1])
          and rel_err(own, m1) > 1e-6)
    errors = 0
    for kwargs in (dict(paired=True, override=True, override_stream='joint'),
                   dict(paired=False, override=True, override_stream='both'),
                   dict(paired=False, override=True, override_stream='joint',
                        override_blend='cubic'),
                   dict(paired=False, override=True, override_stream='joint',
                        image_latent_indices=[0, frames])):
        try:
            tiny_generate(**kwargs)
        except ValueError:
            errors += 1
    try:
        tiny_generate(paired=False, override_stream='joint')  # one image
    except ValueError:
        errors += 1
    check("tiny generate: bad override settings are rejected", errors == 5)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'm1_probe.pt')
        tiny_generate(paired=False, override=True, override_stream='joint',
                      override_min_weight=0.5, probe_steps=[0, 3],
                      probe_samples=4, probe_path=path)
        merged = torch.load(path, map_location='cpu', weights_only=False)
        meta, tensors = merged['meta'], merged['tensors']
        summary = open(os.path.join(tmp, 'm1_probe_summary.txt')).read()
        check("tiny generate: M1 probe records Gordon's per-frame difference",
              meta.get('mode') == 'override'
              and tuple(tensors['diff_rms'].shape) == (2, 2, frames, 2)
              and meta['override']['min_weight'] == 0.5
              and "Gordon's plot" in summary,
              f"shape {tuple(tensors['diff_rms'].shape)}")
    A.ATTN_PROBE.reset()


# ---------------------------------------------------------------------------
# Round 4: prompt relay, Img2 span, colour anchor, resampling, stream 2 video
# ---------------------------------------------------------------------------
def relay_forward(model, inp, relay=None, context=None, seq_len=None):
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
        return model(inp['x'], t=inp['t'], context=context or inp['context'],
                     seq_len=seq_len or inp['tokens'], y=inp['y'],
                     override=dict(weights=[1.0] * 7, blend='linear',
                                   anchor=3), relay=relay)


def test_relay_model():
    from wan.modules.model import relay_local_split
    grid = torch.tensor([[7, 2, 3], [7, 2, 3]])
    splits = [relay_local_split(4, grid, start, 21) for start in (0, 21, 42)]
    check("prompt relay: per-rank split of the token sequence",
          splits == [21, 3, 0], f"{splits}")
    try:
        relay_local_split(4, torch.tensor([[7, 2, 3], [7, 3, 2]]), 0, 42)
        mixed = False
    except ValueError:
        mixed = True
    shorter = relay_local_split(4, torch.tensor([[7, 2, 3], [4, 2, 3]]), 0,
                                42) == 24
    check("prompt relay: a shorter stream (Stream B) is fine, a different "
          "frame size is rejected", mixed and shorter)
    # One layer: self-attention comes before cross-attention and the FFN and
    # head are per token, so a frame's output depends on its own prompt only.
    model = tiny_model(num_layers=1).to(DEVICE).to(torch.bfloat16)
    with torch.no_grad():  # make the prompt's effect large against bf16 noise
        model.blocks[0].cross_attn.o.weight.mul_(30.0)
    inp = tiny_override_inputs(DEVICE)
    gen = torch.Generator(device='cpu')
    gen.manual_seed(11)
    ctx_b = torch.randn(4, 64, generator=gen).to(DEVICE)
    out_a = relay_forward(model, inp)
    out_b = relay_forward(model, inp, context=[ctx_b, ctx_b])
    gap = min(rel_err(out_b[i], out_a[i]) for i in range(2))
    check("prompt relay test: the two prompts give clearly different outputs",
          gap > 0.1, f"rel diff {gap:.2e}")
    tol = 0.05 * gap  # bf16 kernels on a token slice vs the whole sequence
    for start in (2, 4):
        out_r = relay_forward(model, inp, relay=dict(context=[ctx_b, ctx_b],
                                                     from_frame=start))
        ok = all(rel_err(out_r[i][:, :start], out_a[i][:, :start]) < tol
                 and rel_err(out_r[i][:, start:], out_b[i][:, start:]) < tol
                 for i in range(2))
        check(f"prompt relay (1 layer): frames < {start} use the main "
              f"prompt, frames >= {start} the relay prompt", ok)
    same = relay_forward(model, inp, relay=dict(context=inp['context'],
                                                from_frame=3))
    check("prompt relay with the main prompt changes nothing",
          all(rel_err(same[i], out_a[i]) < tol for i in range(2)))
    model2 = tiny_model().to(DEVICE).to(torch.bfloat16)
    full = relay_forward(model2, inp, relay=dict(context=[ctx_b, ctx_b],
                                                 from_frame=0))
    ref = relay_forward(model2, inp, context=[ctx_b, ctx_b])
    check("prompt relay from latent 0 equals running on the relay prompt",
          all(rel_err(full[i], ref[i]) < 1e-2 for i in range(2)))


def test_round4_helpers():
    from wan.image2video import (anchor_frame_stats, build_step_plan,
                                 relay_stat_targets)
    plan = build_step_plan(6, (2, 4, 1))
    check("resample plan: steps 0-4, then 2-4 again, then 5",
          [p[0] for p in plan] == [0, 1, 2, 3, 4, 2, 3, 4, 5]
          and [p[2] for p in plan] == [False] * 5 + [True, False, False, False]
          and [p[1] for p in plan] == [True] * 5 + [False] * 3 + [True])
    two = build_step_plan(5, (1, 2, 2))
    check("resample plan: two cycles",
          [p[0] for p in two] == [0, 1, 2, 1, 2, 1, 2, 3, 4]
          and sum(p[2] for p in two) == 2)
    check("resample plan: off without cycles",
          build_step_plan(4, None) == build_step_plan(4, (1, 2, 0))
          == [(i, True, False) for i in range(4)])
    try:
        build_step_plan(5, (3, 2, 1))
        bad = False
    except ValueError:
        bad = True
    check("resample plan: from > to is rejected", bad)
    s1 = (torch.tensor([1.0, -2.0]), torch.tensor([0.5, 1.0]))
    s2 = (torch.tensor([3.0, 2.0]), torch.tensor([1.5, 3.0]))
    mean, std = relay_stat_targets(s1, s2, 7, 4)
    check("colour anchor targets: Img1 at 0, halfway at a/2, Img2 from a on",
          tuple(mean.shape) == (2, 7, 1, 1)
          and max_abs(mean[:, 0, 0, 0], s1[0]) < 1e-6
          and max_abs(mean[:, 2, 0, 0], (s1[0] + s2[0]) / 2) < 1e-6
          and max_abs(mean[:, 4, 0, 0], s2[0]) < 1e-6
          and max_abs(mean[:, 6, 0, 0], s2[0]) < 1e-6
          and max_abs(std[:, 2, 0, 0], (s1[1] + s2[1]) / 2) < 1e-6)
    gen = torch.Generator(device='cpu')
    gen.manual_seed(2)
    sample = torch.randn(2, 7, 4, 5, generator=gen)
    vel = torch.randn(2, 7, 4, 5, generator=gen)
    sigma = 0.6
    x0 = sample - sigma * vel
    new_v = anchor_frame_stats(vel, sample, sigma, mean)
    new_x0 = sample - sigma * new_v
    got = new_x0.mean(dim=(2, 3))
    check("colour anchor (mean): frames 1.. get the target channel means, "
          "frame 0 and the spatial pattern unchanged",
          max_abs(got[:, 1:], mean[:, 1:, 0, 0]) < 1e-5
          and max_abs(new_x0[:, 0], x0[:, 0]) < 1e-6
          and max_abs(new_x0 - new_x0.mean(dim=(2, 3), keepdim=True),
                      x0 - x0.mean(dim=(2, 3), keepdim=True)) < 1e-5)
    half = sample - sigma * anchor_frame_stats(vel, sample, sigma, mean,
                                               strength=0.5)
    check("colour anchor strength 0.5 moves the mean half way",
          max_abs(half.mean(dim=(2, 3))[:, 1:],
                  0.5 * (x0.mean(dim=(2, 3))[:, 1:] + mean[:, 1:, 0, 0]))
          < 1e-5)
    check("colour anchor strength 0 leaves the velocity unchanged",
          max_abs(anchor_frame_stats(vel, sample, sigma, mean, strength=0.0),
                  vel) < 1e-5)
    ms = sample - sigma * anchor_frame_stats(vel, sample, sigma, mean, std)
    check("colour anchor (mean+std): frames 1.. get target means and stds",
          max_abs(ms.mean(dim=(2, 3))[:, 1:], mean[:, 1:, 0, 0]) < 1e-5
          and max_abs(ms.std(dim=(2, 3))[:, 1:], std[:, 1:, 0, 0]) < 1e-4)


def test_route_model():
    """Round 6, Direction C through a 2-layer WanModel: stream 2 is the same
    as with the blend override (it only ever attends to itself), stream 1
    differs from both the blend override and plain I2V."""
    model = tiny_model().to(DEVICE).to(torch.bfloat16)
    inp = tiny_override_inputs(DEVICE)
    blend = override_forward(model, inp, [1.0] * 7, both=True)
    route = override_forward(model, inp, [1.0] * 7, mode='route', both=True)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
        plain = model(inp['x'][:1], t=inp['t'][:1], context=inp['context'][:1],
                      seq_len=inp['tokens'], y=inp['y'][:1])[0]
    same2 = rel_err(route[1], blend[1])
    gap_b, gap_p = rel_err(route[0], blend[0]), rel_err(route[0], plain)
    check("route (model): stream 2 is unchanged, stream 1 differs from the "
          "blend override and from plain I2V",
          same2 < 1e-3 and gap_b > 1e-2 and gap_p > 1e-2,
          f"{same2:.1e} / {gap_b:.1e} / {gap_p:.1e}")


def test_round6_helpers():
    from wan.image2video import pin_latent_prediction
    gen = torch.Generator(device='cpu')
    gen.manual_seed(6)
    sample = torch.randn(3, 7, 4, 5, generator=gen)
    vel = torch.randn(3, 7, 4, 5, generator=gen)
    target = torch.randn(3, 4, 5, generator=gen)
    sigma = 0.7
    x0 = sample - sigma * vel
    new_x0 = sample - sigma * pin_latent_prediction(vel, sample, sigma, 3,
                                                     target)
    check("pin: the predicted clean latent at the pinned frame is the target, "
          "every other frame unchanged",
          max_abs(new_x0[:, 3], target) < 1e-5
          and max_abs(torch.cat([new_x0[:, :3], new_x0[:, 4:]], 1),
                      torch.cat([x0[:, :3], x0[:, 4:]], 1)) < 1e-5)
    check("pin: inputs are left untouched",
          max_abs(sample - sigma * vel, x0) == 0.0)
    try:
        pin_latent_prediction(vel, sample, 0.0, 3, target)
        bad = False
    except ValueError:
        bad = True
    check("pin: sigma 0 is rejected", bad)


def guided(calls, i, stream, gs=3.5):
    cond, uncond = calls[2 * i]['outs'][stream], calls[2 * i + 1]['outs'][stream]
    return uncond + gs * (cond - uncond)


def test_tiny_generate_round4(steps):
    a, frames = 3, 7
    own = dict(paired=False, override=True, override_stream='joint',
               override_latent='own')
    base, spy0, pipe0 = tiny_generate(**own, override_save_stream2=True)
    s2_base = pipe0.last_stream2_video
    check("stream 2 video: saved with an own latent, differs from stream 1",
          s2_base is not None and s2_base.shape == base.shape
          and rel_err(s2_base, base) > 1e-4)
    # Prompt relay.
    pr, spy, pipe = tiny_generate(**own, relay_prompt='relay prompt',
                                  override_save_stream2=True)
    relay_ctx = FakeT5().embed('relay prompt', DEVICE)
    ok = 'relay prompt' in pipe.text_encoder.calls
    for i, c in enumerate(spy.calls):
        if i % 2 == 0:
            r = c.get('relay')
            ok = ok and r is not None and r['from_frame'] == a + 1
            ok = ok and all(torch.equal(u, relay_ctx) for u in r['context'])
        else:
            ok = ok and c.get('relay') is None
    check("prompt relay: conditional passes of both streams read the relay "
          "prompt from the latent after Img2; unconditional passes do not",
          ok and rel_err(pr, base) > 1e-6
          and rel_err(pipe.last_stream2_video, s2_base) > 1e-6)
    _, spy, _ = tiny_generate(**own, relay_prompt='relay prompt',
                              relay_from_latent=5)
    check("prompt relay: --vr_relay_from_latent sets the first relay latent",
          spy.calls[0]['relay']['from_frame'] == 5)
    # Img2 span = one frame.
    sf, spy, pipe = tiny_generate(**own, override_img2_span='frame',
                                  override_save_stream2=True)
    y2 = spy.calls[0]['y'][1]
    m2 = y2[:4]
    ok = bool(m2[0, a].all()) and not bool(m2[1:, a].any())
    ok = ok and bool(m2[:, 0].all()) and not bool(m2[:, 1:a].any())
    ok = ok and not bool(m2[:, a + 1:].any())
    ref_y2 = spy0.calls[0]['y'][1]
    ok = ok and torch.equal(y2[4:, :a], ref_y2[4:, :a])
    ok = ok and not torch.equal(y2[4:, a], ref_y2[4:, a])
    # FakeVAE: latent a = leak + mean of its 4 frames = Img2 / 4 (+ leak).
    check("Img2 span 'frame': stream 2's mask covers only the first frame of "
          "Img2's latent and its encoding holds Img2 once, not 4 times",
          ok and rel_err(sf, base) > 1e-6)
    # Colour anchor on stream 1 only: stream 2 is bit-identical.
    sm, spy, pipe = tiny_generate(**own, stat_anchor='mean',
                                  override_save_stream2=True)
    check("colour anchor (stream 1 only): stream 1 changes, stream 2's video "
          "is bit-identical to the run without it",
          rel_err(sm, base) > 1e-6
          and rel_err(pipe.last_stream2_video, s2_base) < 1e-5)
    smb, spy, pipe = tiny_generate(**own, stat_anchor='meanstd',
                                   stat_streams='both', stat_std_from_step=2,
                                   override_save_stream2=True)
    check("colour anchor on both streams changes stream 2 too",
          rel_err(pipe.last_stream2_video, s2_base) > 1e-6)
    # Resampling: steps 2-4 twice (all low-noise in the tiny schedule).
    rs, spy, pipe = tiny_generate(**own, resample_from_step=2,
                                  resample_to_step=4, resample_cycles=1,
                                  override_save_stream2=True)
    ts = [float(c['t'][0]) for c in spy.calls[::2]]
    ts_base = [float(c['t'][0]) for c in spy0.calls[::2]]
    order = [0, 1, 2, 3, 4, 2, 3, 4, 5]
    ok = len(spy.calls) == 2 * len(order)
    ok = ok and ts == [ts_base[i] for i in order]
    # x0 at step 4 (visit 4); restart latent at visit 5 (step 2).
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    sched = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1,
                                        use_dynamic_shifting=False)
    sched.set_timesteps(steps, device=DEVICE, shift=3.0)
    sig = [float(v) for v in sched.sigmas]  # as the sampler reads them
    x0_1 = spy.calls[8]['x'][0] - sig[4] * guided(spy.calls, 4, 0)
    x0_2 = spy.calls[8]['x'][1] - sig[4] * guided(spy.calls, 4, 1)
    r1, r2 = spy.calls[10]['x'][0], spy.calls[10]['x'][1]
    eps1 = (r1 - (1 - sig[2]) * x0_1) / sig[2]
    eps2 = (r2 - (1 - sig[2]) * x0_2) / sig[2]
    ok = ok and max_abs(eps1, eps2) < 1e-4
    ok = ok and abs(float(eps1.std()) - 1.0) < 0.35
    ok = ok and max_abs(spy.calls[0]['x'][0], spy0.calls[0]['x'][0]) == 0
    check("resampling: steps 2-4 run twice; the restart latent of both "
          "streams is (1 - s_J) x0(step K) + s_J * one shared fresh noise",
          ok and rel_err(rs, base) > 1e-6, f"t {ts}")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'rs_probe.pt')
        tiny_generate(**own, resample_from_step=2, resample_to_step=4,
                      resample_cycles=1, probe_steps=[3], probe_samples=4,
                      probe_path=path)
        merged = torch.load(path, map_location='cpu', weights_only=False)
        ref_path = os.path.join(tmp, 'own_probe.pt')
        tiny_generate(**own, probe_steps=[3], probe_samples=4,
                      probe_path=ref_path)
        ref = torch.load(ref_path, map_location='cpu', weights_only=False)
        # The first visit of step 3 is the same computation as in the run
        # without resampling; the repeated visit is not.
        check("resampling: the probe records only the first visit of a step",
              tuple(merged['tensors']['diff_rms'].shape)[0] == 1
              and torch.equal(merged['tensors']['diff_rms'],
                              ref['tensors']['diff_rms'])
              and merged['meta']['round4']['resample'] == (2, 4, 1),
              f"shape {tuple(merged['tensors']['diff_rms'].shape)}")
    A.ATTN_PROBE.reset()
    errors = 0
    for kwargs in (
            dict(own, resample_from_step=1, resample_to_step=3,
                 resample_cycles=1),                       # crosses experts
            dict(own, resample_from_step=4, resample_to_step=2,
                 resample_cycles=1),
            dict(paired=False, override=True, override_stream='joint',
                 override_save_stream2=True),              # shared latent
            dict(paired=False, override=True, override_stream='joint',
                 stat_anchor='mean', stat_streams='both'),  # shared latent
            dict(own, override_img2_span='both'),
            dict(own, stat_anchor='median'),
            dict(own, relay_prompt='x', relay_from_latent=frames),
            dict(paired=False, relay_prompt='x'),          # no override
            dict(paired=False, stat_anchor='mean'),        # no override
            dict(paired=True, resample_from_step=2, resample_to_step=4,
                 resample_cycles=1)):
        try:
            tiny_generate(**kwargs)
        except ValueError:
            errors += 1
    check("round 4: bad settings are rejected", errors == 10,
          f"{errors} of 10")


def test_tiny_generate_round5(steps):
    a, frames = 3, 7
    own = dict(paired=False, override=True, override_stream='joint',
               override_latent='own', relay_prompt='relay prompt')
    pr, spy0, _ = tiny_generate(**own)
    # Stream B: lambda 0.5 fading out over steps 2-4.
    sb, spy, pipe = tiny_generate(**own, stream_b_weight=0.5,
                                  stream_b_fade_start_step=2,
                                  stream_b_fade_end_step=4,
                                  override_save_stream2=True)
    fake = FakeVAE(DEVICE)
    import numpy as np
    img2 = torch.from_numpy(np.asarray(tiny_images(2)[1])).float()
    img2 = (img2.permute(2, 0, 1) / 255.0 - 0.5) / 0.5
    relay_ctx = FakeT5().embed('relay prompt', DEVICE)
    null_ctx = FakeT5().embed('negative prompt', DEVICE)
    fades = [1.0, 1.0, 1.0, 0.5, 0.0, 0.0]
    ok = len(spy.calls) == 2 * steps
    first_b = None
    for i, c in enumerate(spy.calls):
        step, cond = i // 2, i % 2 == 0
        lam = 0.5 * fades[step]
        if lam > 0:
            ok = ok and len(c['x']) == 3
            ok = ok and torch.equal(c['x'][2], c['x'][0][:, a:])
            ok = ok and abs(c['override']['stream_b']['weight'] - lam) < 1e-9
            ok = ok and torch.equal(c['context'][2],
                                    relay_ctx if cond else null_ctx)
            if cond:
                ok = ok and all(torch.equal(u, relay_ctx)
                                for u in c['relay']['context'])
            yb = c['y'][2]
            ok = ok and bool(yb[:4, 0].all()) and not bool(yb[:4, 1:].any())
            first_b = yb if first_b is None else first_b
        else:
            ok = ok and len(c['x']) == 2 and 'stream_b' not in c['override']
    # Stream B's input: an ordinary Img2-first I2V conditioning.
    want_cond = None
    if first_b is not None:
        h, w = first_b.shape[2] * 8, first_b.shape[3] * 8
        video = torch.zeros(3, 1 + 4 * (frames - a - 1), h, w)
        video[:, 0] = torch.nn.functional.interpolate(
            img2[None], size=(h, w), mode='bicubic')[0]
        want_cond = fake.encode([video.to(DEVICE)])[0]
        ok = ok and rel_err(first_b[4:], want_cond) < 1e-4
    check("Stream B: Img2-first I2V on stream 1's latents a.., relay prompt, "
          "lambda fading 0.5 -> 0 over steps 2-4, then out of the batch",
          ok and rel_err(sb, pr) > 1e-6)
    # Fresh Img2 encoding in stream 2.
    fr, spy, _ = tiny_generate(**dict(own, override_stream='fresh'))
    y1, y2 = spy.calls[0]['y'][0], spy.calls[0]['y'][1]
    ok = bool(y2[:4, a].all()) and not bool(y2[:4, a + 1:].any())
    ok = ok and torch.equal(y2[4:, :a], y1[4:, :a])
    ok = ok and want_cond is not None and rel_err(y2[4:, a:], want_cond) < 1e-4
    check("fresh Img2 encoding: stream 2's latents a.. are an Img2-first I2V "
          "conditioning (one-frame Img2, then zeros), mask on all of latent a",
          ok and rel_err(fr, pr) > 1e-6)
    # Local profile.
    _, spy, _ = tiny_generate(**own, override_profile='local',
                              override_radius=2)
    want = [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0]
    check("local profile reaches the model: w = 1 at Img2, 0 two latents away",
          all(max(abs(x - y) for x, y in zip(c['override']['weights'], want))
              < 1e-12 for c in spy.calls))
    # Handover at step 3.
    ho, spy, _ = tiny_generate(**own, override_handover_step=3)
    after = [1.0 if f == a else 0.0 for f in range(frames)]
    ok = all(c['override']['weights'] == ([1.0] * frames if i // 2 < 3
                                          else after)
             for i, c in enumerate(spy.calls))
    check("handover: full override before step 3, only Img2's latent after",
          ok and rel_err(ho, pr) > 1e-6)
    errors = 0
    for kwargs in (dict(own, stream_b_weight=1.5),
                   dict(own, stream_b_fade_start_step=2,
                        stream_b_fade_end_step=4),
                   dict(own, stream_b_weight=0.5, stream_b_fade_start_step=4,
                        stream_b_fade_end_step=2),
                   dict(own, stream_b_weight=0.5, stream_b_fade_start_step=2),
                   dict(own, override_stream='fresh',
                        override_img2_span='frame'),
                   dict(own, override_profile='wide'),
                   dict(own, override_handover_step=0),
                   dict(own, override_handover_step=steps),
                   dict(paired=False, override_profile='local'),
                   dict(paired=False, override_handover_step=2),
                   dict(paired=False, stream_b_weight=0.5)):
        try:
            tiny_generate(**kwargs)
        except ValueError:
            errors += 1
    check("round 5: bad settings are rejected", errors == 11,
          f"{errors} of 11")


def tiny_img2_static():
    """Img2 held 5 frames through the fake VAE, latent 1 (as the pipeline)."""
    import numpy as np
    img2 = torch.from_numpy(np.asarray(tiny_images(2)[1])).float()
    img2 = (img2.permute(2, 0, 1) / 255.0 - 0.5) / 0.5
    still = torch.nn.functional.interpolate(img2[None], size=(64, 64),
                                            mode='bicubic')[0]
    video = still[:, None].expand(-1, 5, -1, -1).contiguous().to(DEVICE)
    return FakeVAE(DEVICE).encode([video])[0][:, 1].float()


def test_tiny_generate_round6(steps):
    a, frames = 3, 7
    own = dict(paired=False, override=True, override_stream='joint',
               override_latent='own', relay_prompt='relay prompt')
    pr, spy0, pipe0 = tiny_generate(**own, override_save_stream2=True)
    s2_pr = pipe0.last_stream2_video
    z2 = tiny_img2_static()
    relay_ctx = FakeT5().embed('relay prompt', DEVICE)
    main_ctx = FakeT5().embed('main prompt', DEVICE)
    null_ctx = FakeT5().embed('negative prompt', DEVICE)
    # Direction A: clean Img2 latent in stream 2.
    cl, spy, _ = tiny_generate(**dict(own, override_stream='clean'))
    y1, y2 = spy.calls[0]['y'][0], spy.calls[0]['y'][1]
    j1, j2 = spy0.calls[0]['y'][0], spy0.calls[0]['y'][1]
    ok = torch.equal(y1, j1) and torch.equal(y2[:4], j2[:4])
    ok = ok and bool(y2[:4, a].all())
    ok = ok and torch.equal(y2[4:, :a], j2[4:, :a])
    ok = ok and torch.equal(y2[4:, a + 1:], j2[4:, a + 1:])
    ok = ok and rel_err(y2[4:, a], z2) < 1e-5
    ok = ok and rel_err(j2[4:, a], z2) > 1e-2
    check("clean Img2 latent: stream 2 = joint encoding except latent a, "
          "which is Img2 held 5 frames encoded alone (latent 1), mask on all "
          "of it; stream 1 untouched", ok and rel_err(cl, pr) > 1e-6)
    # Direction B: plan, then render from step 3 (a low-noise step).
    S = 3
    import wan.image2video as I2V
    real_pin, pins = I2V.pin_latent_prediction, []

    def spy_pin(velocity, sample, sigma, index, target):
        out = real_pin(velocity, sample, sigma, index, target)
        pins.append(dict(velocity=velocity.clone(), sample=sample.clone(),
                         sigma=sigma, index=index, target=target.clone(),
                         out=out.clone()))
        return out
    I2V.pin_latent_prediction = spy_pin
    try:
        rr, spy, pipe = tiny_generate(**own, rerender_from_step=S,
                                      override_save_stream2=True)
    finally:
        I2V.pin_latent_prediction = real_pin
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    sched = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1,
                                        use_dynamic_shifting=False)
    sched.set_timesteps(steps, device=DEVICE, shift=3.0)
    sig = [float(v) for v in sched.sigmas]
    ts = [float(v) for v in sched.timesteps]
    n1 = 2 * steps
    ok = len(spy.calls) == n1 + 2 * (steps - S)
    for i in range(min(n1, len(spy.calls))):
        ok = ok and all(torch.equal(u, v) for u, v in zip(
            spy.calls[i]['x'], spy0.calls[i]['x']))
    check("plan, then render: pass 1 is the unchanged override run",
          ok, f"{len(spy.calls)} calls")
    check("plan, then render: the plan video is the override run's video and "
          "stream 2's video is unchanged",
          pipe.last_plan_video is not None
          and rel_err(pipe.last_plan_video, pr) < 1e-6
          and rel_err(pipe.last_stream2_video, s2_pr) < 1e-6)
    ok = len(spy.calls) == n1 + 2 * (steps - S)
    for i, c in enumerate(spy.calls[n1:]):
        step, cond = S + i // 2, i % 2 == 0
        ok = ok and len(c['x']) == 1 and len(c['y']) == 1
        ok = ok and torch.equal(c['y'][0], j1) and 'override' not in c
        ok = ok and abs(float(c['t'][0]) - ts[step]) < 1e-3
        ok = ok and torch.equal(c['context'][0], main_ctx if cond else null_ctx)
        if cond:
            r = c.get('relay')
            ok = ok and r is not None and r['from_frame'] == a + 1
            ok = ok and len(r['context']) == 1 and torch.equal(
                r['context'][0], relay_ctx)
        else:
            ok = ok and c.get('relay') is None
    check("plan, then render: pass 2 is one plain I2V stream of Img1 from "
          "step 3 on, relay prompt on the conditional pass only", ok)
    if len(spy.calls) == n1 + 2 * (steps - S) and (
            pipe.last_plan_video is not None):
        # Solver-independent checks (the CPU stub is Euler, the GPU run uses
        # the real UniPC). The fake VAE decodes the first 3 latent channels.
        plan3 = pipe.last_plan_video.clone()
        plan3[:, a] = z2[:3]
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(7)  # the run's seed: the noise, then the re-noise
        shape = spy.calls[0]['x'][0].shape
        torch.randn(*shape, dtype=torch.float32, generator=gen, device=DEVICE)
        renoise = torch.randn(*shape, dtype=torch.float32, generator=gen,
                              device=DEVICE)
        want = (1 - sig[S]) * plan3 + sig[S] * renoise[:3]
        err_start = max_abs(spy.calls[n1]['x'][0][:3], want)
        check("plan, then render: the start is (1 - s_S) x0(plan, Img2's "
              "latent = static Img2) + s_S * fresh noise",
              err_start < 1e-4, f"max abs err {err_start:.1e}")
        ok = len(pins) == steps - S
        for i, p in enumerate(pins[:steps - S]):
            calls = spy.calls[n1 + 2 * i:n1 + 2 * i + 2]
            ok = ok and max_abs(p['velocity'], guided(calls, 0, 0)) < 1e-5
            ok = ok and torch.equal(p['sample'], calls[0]['x'][0])
            ok = ok and abs(p['sigma'] - sig[S + i]) < 1e-9
            ok = ok and p['index'] == a and max_abs(p['target'], z2) < 1e-6
        check("plan, then render: every render step pins Img2's latent of the "
              "guided plain I2V prediction to the static Img2 latent", ok,
              f"{len(pins)} pins")
        # The last step of both solvers returns its predicted clean latent.
        if pins:
            last = pins[-1]
            x0_free = last['sample'] - last['sigma'] * last['velocity']
            rest = [f for f in range(frames) if f != a]
            err_pin = max_abs(rr[:, a], z2[:3])
            err_free = max_abs(rr[:, rest], x0_free[:3, rest])
            check("plan, then render: the solver uses the pinned prediction "
                  "(final Img2 latent = static Img2, the rest = the plain "
                  "I2V prediction)", err_pin < 1e-4 and err_free < 1e-4,
                  f"pin {err_pin:.1e} / free {err_free:.1e}")
    check("plan, then render: the rendered video differs from the plan",
          rel_err(rr, pr) > 1e-6)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'rr_probe.pt')
        tiny_generate(**own, rerender_from_step=S, probe_steps=[4],
                      probe_samples=4, probe_path=path)
        merged = torch.load(path, map_location='cpu', weights_only=False)
        ref_path = os.path.join(tmp, 'pr_probe.pt')
        tiny_generate(**own, probe_steps=[4], probe_samples=4,
                      probe_path=ref_path)
        ref = torch.load(ref_path, map_location='cpu', weights_only=False)
        check("plan, then render: the probe records the plan's step only",
              torch.equal(merged['tensors']['diff_rms'],
                          ref['tensors']['diff_rms'])
              and merged['meta']['override']['rerender_from_step'] == S)
    A.ATTN_PROBE.reset()
    # Direction C: route mode.
    rt, spy, pipe = tiny_generate(**own, override_mode='route',
                                  override_save_stream2=True)
    ok = all(c['override'].get('mode') == 'route' for c in spy.calls)
    check("route mode reaches the model on every pass and changes stream 1's "
          "video; stream 2's own video is unchanged (it attends to itself)",
          ok and rel_err(rt, pr) > 1e-6
          and rel_err(pipe.last_stream2_video, s2_pr) < 1e-5)
    errors = 0
    for kwargs in (dict(own, override_stream='clean',
                        override_img2_span='frame'),
                   dict(own, override_mode='mix'),
                   dict(own, override_mode='route', override_min_weight=0.5),
                   dict(own, override_mode='route', override_profile='local'),
                   dict(own, override_mode='route', override_handover_step=2),
                   dict(own, override_mode='route', stream_b_weight=0.5),
                   dict(own, rerender_from_step=1),        # high-noise step
                   dict(own, rerender_from_step=0),
                   dict(own, rerender_from_step=steps),
                   dict(own, rerender_from_step=3, resample_from_step=2,
                        resample_to_step=4, resample_cycles=1),
                   dict(paired=False, override_mode='route'),
                   dict(paired=False, rerender_from_step=3)):
        try:
            tiny_generate(**kwargs)
        except ValueError:
            errors += 1
    check("round 6: bad settings are rejected", errors == 12,
          f"{errors} of 12")


def test_round7_helpers():
    from wan.image2video import (gaussian_blur_latent, reach_pattern,
                                 reach_response, reach_summary)
    gen = torch.Generator(device='cpu')
    gen.manual_seed(70)
    x = torch.randn(3, 2, 9, 11, generator=gen)
    check("round 7 blur: sigma 0 returns the input", gaussian_blur_latent(x, 0) is x)

    def ref_blur(value, sigma):
        """Dense separable Gaussian, borders clamped (replicate)."""
        r = max(1, int(math.ceil(3 * sigma)))
        wts = [math.exp(-0.5 * (k / sigma)**2) for k in range(-r, r + 1)]
        total = sum(wts)
        wts = [v / total for v in wts]
        H, W = value.shape[-2], value.shape[-1]
        tmp = torch.zeros_like(value)
        for i in range(W):
            acc = torch.zeros_like(value[..., :, 0])
            for k, wk in zip(range(-r, r + 1), wts):
                acc = acc + wk * value[..., :, min(max(i + k, 0), W - 1)]
            tmp[..., :, i] = acc
        out = torch.zeros_like(value)
        for i in range(H):
            acc = torch.zeros_like(value[..., 0, :])
            for k, wk in zip(range(-r, r + 1), wts):
                acc = acc + wk * tmp[..., min(max(i + k, 0), H - 1), :]
            out[..., i, :] = acc
        return out
    errs = [rel_err(gaussian_blur_latent(x, sg), ref_blur(x, sg))
            for sg in (0.7, 1.5, 4.0)]
    # The sampler runs under bfloat16 autocast; the blur must stay float32
    # (on CUDA, conv2d would otherwise be autocast to bfloat16).
    xd = x.to(DEVICE)
    with torch.amp.autocast(DEVICE.type, dtype=torch.bfloat16):
        inside = gaussian_blur_latent(xd, 1.5)
    outside = gaussian_blur_latent(xd, 1.5)
    check("round 7 blur: float32 inside the sampler's bfloat16 autocast "
          "(same result as outside it)",
          inside.dtype == torch.float32
          and max_abs(inside, outside) <= 1e-6 * float(outside.abs().max()),
          f"max abs diff {max_abs(inside, outside):.1e} on {DEVICE.type}")
    flat = gaussian_blur_latent(torch.ones(2, 5, 6) * 3.0, 2.0)
    check("round 7 blur: separable Gaussian with replicate borders (dense "
          "reference); a constant field stays constant",
          max(errs) < 1e-5 and max_abs(flat, torch.ones(2, 5, 6) * 3.0) < 1e-5,
          f"rel errs {[f'{e:.1e}' for e in errs]}")
    p1 = reach_pattern(16, 8, 10, 2.0, 5)
    p2 = reach_pattern(16, 8, 10, 2.0, 5)
    p3 = reach_pattern(16, 8, 10, 2.0, 6)
    white = torch.randn(16, 8, 10, generator=gen)
    white = white / white.pow(2).mean().sqrt()

    def roughness(u):
        return float((u[:, :, 1:] - u[:, :, :-1]).abs().mean())
    check("round 7 nudge pattern: seeded (same seed = same pattern), RMS 1, "
          "blurred (much smoother than white noise)",
          torch.equal(p1, p2) and rel_err(p1, p3) > 0.1
          and abs(float(p1.pow(2).mean().sqrt()) - 1.0) < 1e-5
          and roughness(p1) < 0.3 * roughness(white))
    C, L, H, W = 4, 5, 6, 7
    x = torch.randn(C, L, H, W, generator=gen)
    mix = torch.randn(L, L, generator=gen) * 0.3

    def velocity(z, quad):
        return torch.einsum('kj,cjhw->ckhw', mix, z) + quad * z * z
    sigma, blur, scale = 0.8, 1.0, 0.3
    delta = torch.zeros_like(x)
    delta[:, 2] = scale * torch.randn(C, H, W, generator=gen)

    def per_frame(u):
        return u.pow(2).mean(dim=(0, 2, 3)).sqrt() / scale
    ok = True
    for quad in (0.0, 0.1):
        res = reach_response(x, sigma, scale, delta, velocity(x + delta, quad),
                             velocity(x - delta, quad), velocity(x, quad),
                             blur)
        lin = torch.einsum('kj,cjhw->ckhw', mix, delta) + 2 * quad * x * delta
        first = delta - sigma * lin
        second = -sigma * quad * delta * delta
        ok = ok and rel_err(res['raw'], per_frame(first)) < 1e-4
        ok = ok and rel_err(res['low'], per_frame(
            gaussian_blur_latent(first, blur))) < 1e-4
        ok = ok and rel_err(res['model_low'], per_frame(
            gaussian_blur_latent(-sigma * lin, blur))) < 1e-4
        if quad:
            ok = ok and rel_err(res['second_low'], per_frame(
                gaussian_blur_latent(second, blur))) < 1e-3
        else:
            ok = ok and float(res['second_low'].abs().max()) < 1e-5
    check("round 7 reach response: first-order change of x0 = x - sigma v "
          "(and the model's part), second-order part, per frame, per unit of "
          "nudge, on a linear and a quadratic toy model", ok)
    rows = [dict(low=[0.0] * 5, second_low=[0.0] * 5) for _ in range(5)]
    for j in range(5):
        rows[j]['low'][3] = 0.1 * (j + 1)
    pat = dict(rows=rows, quarter=dict(low=[0.0, 0.0, 0.0, 0.3, 0.0]),
               floor=dict(low=[0.0, 0.0, 0.0, 0.01, 0.0]))
    record = dict(meta=dict(target=3, eps=0.1, blur=4.0, patterns=1,
                            relay_from=4),
                  steps={'5': dict(timestep=950.0, sigma=0.9, nudge_rms=0.1,
                                   patterns=[pat])})
    text = reach_summary(record)
    target_line = [ln for ln in text.splitlines() if '<- target' in ln]
    check("round 7 reach summary: one row per nudged latent, the target's "
          "row is 1.000 of itself, floor and quarter-size control listed",
          len(target_line) == 1 and '1.000' in target_line[0]
          and target_line[0].strip().startswith('3 ')
          and 'step 5' in text and 'floor at target: 0.0100' in text
          and 'x1/4: 0.3000' in text)


def tiny_img_tensor(index):
    import numpy as np
    img = torch.from_numpy(np.asarray(tiny_images(2)[index])).float()
    return ((img.permute(2, 0, 1) / 255.0 - 0.5) / 0.5).to(DEVICE)


def tiny_joint_latents(span, motion=None, frames=25, a=3):
    """The joint encoding (Img1, grey, Img2's latent frames, grey) through
    the fake VAE, built by hand."""
    img1, img2 = tiny_img_tensor(0), tiny_img_tensor(1)
    video = torch.zeros(3, frames, 64, 64).to(DEVICE)
    video[:, 0] = img1
    start = 1 + 4 * (a - 1)
    if span == 'last':
        video[:, start + 3] = img2
    elif span in ('last2', 'last3'):
        for f in range(4 - int(span[-1]), 4):
            video[:, start + f] = img2
    elif span == 'motion':
        video[:, start] = img2
        video[:, start + 1:start + 4] = motion
    else:
        for f in range(4):
            video[:, start + f] = img2
    return FakeVAE(DEVICE).encode([video])[0]


def tiny_own_latent(frames4):
    """Latent 1 of [frame 0, the 4 frames] encoded on its own."""
    clip = torch.cat([frames4[:, :1], frames4], dim=1)
    return FakeVAE(DEVICE).encode([clip])[0][:, 1].float()


def test_tiny_generate_round7(steps):
    from wan.image2video import reach_pattern, reach_response
    a, frames = 3, 7
    own = dict(paired=False, override=True, override_stream='joint',
               override_latent='own', relay_prompt='relay prompt')
    pr, spy0, _ = tiny_generate(**own)
    j1, j2 = spy0.calls[0]['y'][0], spy0.calls[0]['y'][1]
    img2 = tiny_img_tensor(1)
    gen = torch.Generator(device='cpu')
    gen.manual_seed(77)
    motion = (torch.randn(3, 3, 64, 64, generator=gen) * 0.5).to(DEVICE)
    zero = torch.zeros_like(img2)
    cases = [
        ("PR-LF: joint encoding with Img2 on frame 4a only, mask on that "
         "frame only", dict(own, override_img2_span='last'), 'last', False),
        ("CL-LF: as PR-LF, but Img2's latent is [grey x3, Img2] encoded on "
         "its own after a grey frame",
         dict(own, override_stream='clean', override_img2_span='last'),
         'last', True),
        ("PR-MV: joint encoding with Img2 then the 3 motion frames, mask on "
         "all 4", dict(own, override_img2_span='motion',
                       img2_motion_frames=motion), 'motion', False),
        ("CL-MV: as PR-MV, but Img2's latent is [Img2, 3 motion frames] "
         "encoded on its own after Img2",
         dict(own, override_stream='clean', override_img2_span='motion',
              img2_motion_frames=motion), 'motion', True),
    ]
    for name, kwargs, span, clean in cases:
        video, spy, _ = tiny_generate(**kwargs)
        y1, y2 = spy.calls[0]['y'][0], spy.calls[0]['y'][1]
        mask = torch.zeros(4, frames, 8, 8).to(DEVICE)
        mask[:, 0] = 1
        if span == 'last':
            mask[3, a] = 1
        else:
            mask[:, a] = 1
        joint = tiny_joint_latents(span, motion if span == 'motion' else None)
        want = joint.clone()
        if clean:
            if span == 'last':
                frames4 = torch.stack([zero, zero, zero, img2], dim=1)
            else:
                frames4 = torch.cat([img2[:, None], motion], dim=1)
            want[:, a] = tiny_own_latent(frames4)
        ok = torch.equal(y1, j1) and torch.equal(y2[:4], mask)
        ok = ok and rel_err(y2[4:], want) < 1e-5
        ok = ok and all(torch.equal(c['y'][0], y1) and torch.equal(
            c['y'][1], y2) for c in spy.calls)
        ok = ok and rel_err(y2[4:, a], j2[4:, a]) > 1e-3
        if clean:
            ok = ok and rel_err(y2[4:, a], joint[:, a]) > 1e-3
            ok = ok and torch.equal(y2[4:, :a], joint[:, :a])
        check(f"round 7 {name}; stream 1 untouched", ok
              and rel_err(video, pr) > 1e-6)
    errors = 0
    bad = (dict(own, override_img2_span='motion'),
           dict(own, img2_motion_frames=motion),
           dict(own, override_stream='fresh', override_img2_span='last'),
           dict(own, override_stream='clean', override_img2_span='frame'),
           dict(own, override_img2_span='last', rerender_from_step=3),
           dict(own, override_img2_span='motion',
                img2_motion_frames=motion[:, :1]),  # would broadcast
           dict(own, override_img2_span='middle'),
           dict(paired=False, img2_motion_frames=motion),
           dict(paired=False, relay_prompt='x'),
           dict(paired=True, relay_prompt='x', relay_from_latent=4),
           dict(own, reach_test=dict(steps=[0], target=a)),
           dict(paired=True, reach_test=dict(steps=[0], target=a)),
           dict(paired=False, reach_test=dict(steps=[steps], target=a)),
           dict(paired=False, reach_test=dict(steps=[0], target=0)),
           dict(paired=False, reach_test=dict(steps=[0], target=frames)),
           dict(paired=False, reach_test=dict(steps=[0], target=a, eps=0)),
           dict(paired=False, reach_test=dict(steps=[0], target=a),
                resample_from_step=2, resample_to_step=4, resample_cycles=1))
    for kwargs in bad:
        try:
            tiny_generate(**kwargs)
        except ValueError:
            errors += 1
    check("round 7: bad settings are rejected", errors == len(bad),
          f"{errors} of {len(bad)}")
    # Plain I2V with the relay prompt (the reach test's base).
    relay_ctx = FakeT5().embed('relay prompt', DEVICE)
    main_ctx = FakeT5().embed('main prompt', DEVICE)
    plain_relay = dict(paired=False, relay_prompt='relay prompt',
                       relay_from_latent=a + 1)
    base, spyb, _ = tiny_generate(**plain_relay)
    plain, _, _ = tiny_generate(paired=False)
    ok = len(spyb.calls) == 2 * steps and rel_err(base, plain) > 1e-6
    for i, c in enumerate(spyb.calls):
        r = c.get('relay')
        if i % 2 == 0:
            ok = ok and r is not None and r['from_frame'] == a + 1 and len(
                r['context']) == 1 and torch.equal(r['context'][0], relay_ctx)
        else:
            ok = ok and r is None
        ok = ok and 'override' not in c and len(c['x']) == 1
    check("round 7: plain I2V reads the relay prompt from the given latent "
          "on the conditional pass only", ok)
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    sched = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1,
                                        use_dynamic_shifting=False)
    sched.set_timesteps(steps, device=DEVICE, shift=3.0)
    sig = [float(v) for v in sched.sigmas]
    rsteps, pats, eps, blur, seed = [0, 2], 2, 0.1, 1.0, 5
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'r7_REACH_reach.json')
        vid, spy, pipe = tiny_generate(**plain_relay, reach_test=dict(
            steps=rsteps, target=a, eps=eps, blur=blur, patterns=pats,
            seed=seed, path=path))
        check("reach test is read-only: the video is bit-identical to the "
              "same run without it", torch.equal(vid, base))
        rec = pipe.last_reach
        n_extra = len(rsteps) * pats * (frames + 2)
        ok_struct = len(spy.calls) == 2 * steps + n_extra and rec is not None
        ok_vals, idx = True, 0
        for s in range(steps):
            if not ok_struct:
                break
            cond = spy.calls[idx]
            ok_struct = ok_struct and torch.equal(cond['x'][0],
                                                  spyb.calls[2 * s]['x'][0])
            idx += 2
            if s not in rsteps:
                continue
            x_t = cond['x'][0].float()
            rms = eps * float(x_t.pow(2).mean().sqrt())
            srec = rec['steps'][str(s)]
            ok_vals = ok_vals and abs(srec['sigma'] - sig[s]) < 1e-9
            ok_vals = ok_vals and abs(srec['nudge_rms'] - rms) < 1e-6 * rms
            for p in range(pats):
                pattern = reach_pattern(16, 8, 8, blur, seed + p).to(DEVICE)
                prec = srec['patterns'][p]
                for k in range(frames + 2):
                    c = spy.calls[idx]
                    idx += 1
                    j, size = ((k, 1.0) if k < frames else
                               (a - 1, 0.25) if k == frames else (a - 1, 1e-2))
                    delta = torch.zeros_like(x_t)
                    delta[:, j] = size * rms * pattern
                    ok_struct = ok_struct and len(c['x']) == 2
                    ok_struct = ok_struct and max_abs(c['x'][0], x_t + delta) < 1e-6
                    ok_struct = ok_struct and max_abs(c['x'][1], x_t - delta) < 1e-6
                    ok_struct = ok_struct and torch.equal(
                        c['t'], torch.stack([cond['t'][0], cond['t'][0]]))
                    ok_struct = ok_struct and all(
                        torch.equal(u, main_ctx) for u in c['context'])
                    ok_struct = ok_struct and len(c['y']) == 2 and all(
                        torch.equal(u, cond['y'][0]) for u in c['y'])
                    r = c.get('relay')
                    ok_struct = ok_struct and r is not None and r[
                        'from_frame'] == a + 1 and len(r['context']) == 2 and all(
                            torch.equal(u, relay_ctx) for u in r['context'])
                    ok_struct = ok_struct and 'override' not in c
                    norm = size * rms if k == frames else rms
                    want = reach_response(x_t, sig[s], norm, delta,
                                          c['outs'][0], c['outs'][1],
                                          cond['outs'][0], blur)
                    got = (prec['rows'][k] if k < frames else
                           prec['quarter'] if k == frames else prec['floor'])
                    for key, value in want.items():
                        ok_vals = ok_vals and max_abs(
                            torch.tensor(got[key]), value.cpu()) < 1e-5 * (
                                1 + float(value.abs().max()))
        check("reach test: at steps 0 and 2, every latent in turn is nudged "
              "by +- (seeded blurred pattern x eps x RMS(x_t)) in one batch-2 "
              "conditional pass with the relay prompt, then latent a-1 at 1/4 "
              "and 1/100 size; other steps untouched", ok_struct,
              f"{len(spy.calls)} calls")
        check("reach test: the recorded responses are the central "
              "differences of x0 = x - sigma v from those passes", ok_vals)
        import json as _json
        on_disk = _json.load(open(path))
        summary = open(os.path.splitext(path)[0] + '.txt').read()
        check("reach test: rank 0 writes the record (JSON) and its summary",
              on_disk['meta']['target'] == a
              and sorted(on_disk['steps']) == ['0', '2']
              and abs(on_disk['steps']['2']['patterns'][1]['rows'][1]['low'][a]
                      - rec['steps']['2']['patterns'][1]['rows'][1]['low'][a])
              < 1e-9 and '<- target' in summary and 'step 2' in summary)


def tiny_steady_latent(before, frames4, a=3):
    """Round 8: latent a of [frames 0..4a-4, Img2's 4 latent frames] encoded
    as one clip through the fake VAE, built by hand."""
    clip = torch.cat([before, frames4], dim=1)
    return FakeVAE(DEVICE).encode([clip])[0][:, a].float()


def ref_lift(frames, start, end):
    """Each frame's per-channel mean (0-1 scale) moved onto the line from
    start's means to end's by a gain, frame k at (k + 1) / (n + 1); written
    channel by channel, independently of the pipeline."""
    n = frames.shape[1]
    out = []
    for k in range(n):
        t = (k + 1) / (n + 1)
        chans = []
        for c in range(3):
            f = (frames[c, k].float() + 1) / 2
            target = (float(((start[c] + 1) / 2).mean()) * (1 - t)
                      + float(((end[c] + 1) / 2).mean()) * t)
            chans.append((f * (target / float(f.mean()))).clamp(0, 1) * 2 - 1)
        out.append(torch.stack(chans))
    return torch.stack(out, dim=1)


def test_tiny_generate_round8(steps):
    a, frames = 3, 7
    own = dict(paired=False, override=True, override_stream='joint',
               override_latent='own', relay_prompt='relay prompt')
    pr, spy0, _ = tiny_generate(**own)
    j1 = spy0.calls[0]['y'][0]
    img2 = tiny_img_tensor(1)
    gen = torch.Generator(device='cpu')
    gen.manual_seed(88)
    ctx = (torch.rand(3, 25, 64, 64, generator=gen) * 1.6 - 0.8).to(DEVICE)
    # A dark dip on frames 4a-3..4a-1 with bright specks, so lifting the
    # means pushes the specks past white (the lift clips at 0-1).
    ctx[:, 9:12] = ctx[:, 9:12] * 0.3 - 0.5
    ctx[:, 9:12, ::5, ::3] = 0.9
    before_frames = 1 + 4 * (a - 1)
    before_long = img2[:, None].expand(-1, before_frames, -1, -1)
    before_ctx = ctx[:, :before_frames]
    x4 = img2[:, None].expand(-1, 4, -1, -1)
    hold = torch.cat([before_ctx[:, -1:].expand(-1, 3, -1, -1),
                      img2[:, None]], dim=1)
    lifted = ref_lift(ctx[:, 9:12], ctx[:, 8], img2)
    video4 = torch.cat([lifted, img2[:, None]], dim=1)
    # The test's own reference: without clipping, the lifted means sit on
    # the line from frame 8 to Img2.
    probe = (ctx[:, 9:12] * 0.2 + 0.1)
    plift = ref_lift(probe, ctx[:, 8], img2)
    line_ok = True
    for k in range(3):
        t = (k + 1) / 4
        for c in range(3):
            want_mean = (float(((ctx[c, 8] + 1) / 2).mean()) * (1 - t)
                         + float(((img2[c] + 1) / 2).mean()) * t)
            line_ok = line_ok and abs(
                float(((plift[c, k] + 1) / 2).mean()) - want_mean) < 1e-5
    check("round 8 reference: lifted channel means lie on the line from "
          "frame 4a-4 to Img2", line_ok)
    ctxd = dict(own, override_stream='context', img2_context_video=ctx)
    cases = [
        ("LG-4: Img2's latent = latent a of Img2 held from frame 0, all 4 "
         "frames marked", dict(own, override_stream='long'), 'latent',
         before_long, x4),
        ("LG-LF: the same code, mask on frame 4a only",
         dict(own, override_stream='long', override_img2_span='last'),
         'last', before_long, x4),
        ("CX-4: latent a of the context video's frames 0..4a-4, then Img2 "
         "x4, all 4 marked", ctxd, 'latent', before_ctx, x4),
        ("CX-LFi: as CX-4, mask on frame 4a only",
         dict(ctxd, override_img2_span='last'), 'last', before_ctx, x4),
        ("CX-LFh: context frames, frame 4a-4 held x3, then Img2",
         dict(ctxd, override_img2_span='last', img2_fill='hold'), 'last',
         before_ctx, hold),
        ("CX-LFv: context frames, the context's frames 4a-3..4a-1 lifted "
         "onto the line to Img2, then Img2",
         dict(ctxd, override_img2_span='last', img2_fill='video'), 'last',
         before_ctx, video4),
    ]
    codes = {}
    for name, kwargs, span, before, frames4 in cases:
        video, spy, _ = tiny_generate(**kwargs)
        y1, y2 = spy.calls[0]['y'][0], spy.calls[0]['y'][1]
        mask = torch.zeros(4, frames, 8, 8).to(DEVICE)
        mask[:, 0] = 1
        if span == 'last':
            mask[3, a] = 1
        else:
            mask[:, a] = 1
        joint = tiny_joint_latents(span)
        want = joint.clone()
        want[:, a] = tiny_steady_latent(before, frames4)
        codes[name[:6]] = y2[4:, a].clone()
        ok = torch.equal(y1, j1) and torch.equal(y2[:4], mask)
        ok = ok and rel_err(y2[4:], want) < 1e-5
        ok = ok and torch.equal(y2[4:, :a], joint[:, :a])
        ok = ok and torch.equal(y2[4:, a + 1:], joint[:, a + 1:])
        ok = ok and rel_err(y2[4:, a], joint[:, a]) > 1e-3
        ok = ok and all(torch.equal(c['y'][0], y1) and torch.equal(
            c['y'][1], y2) for c in spy.calls)
        check(f"round 8 {name}; the joint encoding elsewhere; stream 1 "
              "untouched", ok and rel_err(video, pr) > 1e-6)
    same = (rel_err(codes['LG-4: '], codes['LG-LF:']) < 1e-6
            and rel_err(codes['CX-4: '], codes['CX-LFi']) < 1e-6)
    distinct = [codes[k] for k in ('LG-LF:', 'CX-LFi', 'CX-LFh', 'CX-LFv')]
    check("round 8: span changes only the mask (LG-4 = LG-LF, CX-4 = "
          "CX-LFi codes); history and fill change the code", same and all(
              rel_err(distinct[i], distinct[k]) > 1e-4 for i in range(4)
              for k in range(i + 1, 4)))
    # Only frames 0..4a-4 (and 4a-3..4a-1 for the video fill) are read; a
    # context video of another size is resized to the working size.
    ok = True
    for fill, need in (('img2', before_frames), ('hold', before_frames),
                       ('video', before_frames + 3)):
        kw = dict(ctxd, override_img2_span='last', img2_fill=fill)
        _, spy_full, _ = tiny_generate(**kw)
        short = ctx[:, :need].clone()
        _, spy_short, _ = tiny_generate(**dict(kw, img2_context_video=short))
        ok = ok and rel_err(spy_full.calls[0]['y'][1],
                            spy_short.calls[0]['y'][1]) < 1e-6
    check("round 8: the context code reads only frames 0..4a-4 (and "
          "4a-3..4a-1 for fill 'video')", ok)
    small = ctx[:, :, ::2, ::2].clone()
    _, spy_small, _ = tiny_generate(**dict(ctxd, img2_context_video=small))
    resized = torch.nn.functional.interpolate(
        small[:, :before_frames].transpose(0, 1).contiguous(), size=(64, 64),
        mode='bicubic').transpose(0, 1).clamp(-1, 1)
    want_small = tiny_steady_latent(resized, x4)
    check("round 8: a context video of another size is resized to the "
          "working size", rel_err(spy_small.calls[0]['y'][1][4:, a],
                                  want_small) < 1e-5)
    # Each bad setting must be refused by its own check (message), not by a
    # later shape error.
    span_msg, pair_msg = 'take span', 'go together'
    fill_msg, size_msg = 'needs the context encoding', 'needs at least'
    shape_msg, plain_msg = 'must be [3, N, H, W]', 'self-attention override'
    bad = ((dict(own, override_stream='long', override_img2_span='frame'),
            span_msg),
           (dict(own, override_stream='long', override_img2_span='motion',
                 img2_motion_frames=x4[:, :3]), span_msg),
           (dict(own, override_stream='context'), pair_msg),
           (dict(own, override_stream='long', img2_context_video=ctx),
            pair_msg),
           (dict(own, override_stream='clean', img2_context_video=ctx),
            pair_msg),
           (dict(own, img2_context_video=ctx), pair_msg),
           (dict(own, override_stream='long', override_img2_span='last',
                 img2_fill='hold'), fill_msg),
           (dict(ctxd, img2_fill='hold'), fill_msg),
           (dict(ctxd, override_img2_span='last', img2_fill='blur'),
            "Img2 fill must be"),
           (dict(ctxd, img2_context_video=ctx[:, :before_frames - 1]),
            size_msg),
           (dict(ctxd, override_img2_span='last', img2_fill='video',
                 img2_context_video=ctx[:, :before_frames + 2]), size_msg),
           (dict(ctxd, img2_context_video=ctx[0]), shape_msg),
           (dict(ctxd, img2_context_video=torch.cat([ctx, ctx[:1]])),
            shape_msg),
           (dict(paired=False, img2_context_video=ctx), plain_msg),
           (dict(paired=False, img2_fill='hold'), plain_msg),
           (dict(own, img2_fill='video'), fill_msg))
    errors, wrong = 0, []
    for kwargs, msg in bad:
        try:
            tiny_generate(**kwargs)
        except ValueError as error:
            if msg in str(error):
                errors += 1
            else:
                wrong.append(str(error)[:60])
    check("round 8: bad settings are rejected, each by its own check",
          errors == len(bad), f"{errors} of {len(bad)}; {wrong[:2]}")


def stats_like(latent, ref, mode):
    """Test reference: latent [C, h, w] with ref's per-channel spatial mean
    (and std), written channel by channel."""
    out = []
    for c in range(latent.shape[0]):
        x, r = latent[c].float(), ref[c].float()
        if mode == 'mean':
            out.append(x - float(x.mean()) + float(r.mean()))
        else:
            out.append((x - float(x.mean())) / float(x.std()) * float(r.std())
                       + float(r.mean()))
    return torch.stack(out)


def test_tiny_generate_round9(steps):
    a, frames = 3, 7
    own = dict(paired=False, override=True, override_stream='joint',
               override_latent='own', relay_prompt='relay prompt')
    pr, spy0, _ = tiny_generate(**own)
    j1, j2 = spy0.calls[0]['y'][0], spy0.calls[0]['y'][1]
    img2 = tiny_img_tensor(1)
    gen = torch.Generator(device='cpu')
    gen.manual_seed(91)
    ctx = (torch.rand(3, 25, 64, 64, generator=gen) * 1.6 - 0.8).to(DEVICE)
    before_frames = 1 + 4 * (a - 1)
    grey3 = torch.zeros(3, 3, 64, 64).to(DEVICE)
    long4 = tiny_steady_latent(img2[:, None].expand(-1, before_frames, -1, -1),
                               img2[:, None].expand(-1, 4, -1, -1))
    cases = [
        ("LGG-LF: Img2 held to frame 4a-4, grey x3, Img2 on 4a, latent a; mask "
         "on frame 4a", dict(own, override_stream='long', override_img2_span='last',
                             img2_fill='grey'),
         tiny_steady_latent(img2[:, None].expand(-1, before_frames, -1, -1),
                            torch.cat([grey3, img2[:, None]], dim=1))),
        ("CX grey: the context's frames 0..4a-4, grey x3, Img2",
         dict(own, override_stream='context', override_img2_span='last',
              img2_fill='grey', img2_context_video=ctx),
         tiny_steady_latent(ctx[:, :before_frames],
                            torch.cat([grey3, img2[:, None]], dim=1))),
    ]
    for name, kwargs, code in cases:
        _, spy, _ = tiny_generate(**kwargs)
        y2 = spy.calls[0]['y'][1]
        joint = tiny_joint_latents('last')
        ok = rel_err(y2[4:, a], code) < 1e-5 and torch.equal(
            y2[4:, :a], joint[:, :a]) and float(y2[3, a].min()) == 1.0
        ok = ok and float(y2[:3, a].max()) == 0.0
        check(f"round 9 {name}", ok)
    for mode in ('mean', 'meanstd'):
        _, spy, _ = tiny_generate(**dict(own, img2_code_stats=mode))
        y2 = spy.calls[0]['y'][1]
        want = stats_like(j2[4:, a], long4, mode)
        ok = rel_err(y2[4:, a], want) < 1e-4 and torch.equal(
            y2[4:, :a], j2[4:, :a]) and torch.equal(y2[4:, a + 1:], j2[4:, a + 1:])
        ok = ok and torch.equal(y2[:4], j2[:4]) and torch.equal(
            spy.calls[0]['y'][0], j1)
        got_mean, ref_mean = y2[4:, a].float().mean(dim=(1, 2)), long4.mean(dim=(1, 2))
        ok = ok and max_abs(got_mean, ref_mean) < 1e-4
        if mode == 'meanstd':
            ok = ok and max_abs(y2[4:, a].float().std(dim=(1, 2)),
                                long4.std(dim=(1, 2))) < 1e-4
        else:
            ok = ok and max_abs(y2[4:, a].float().std(dim=(1, 2)),
                                j2[4:, a].float().std(dim=(1, 2))) < 1e-4
        check(f"round 9 JM ({mode}): the joint Img2 latent keeps its pattern "
              "and takes the long latent's channel "
              f"{'mean' if mode == 'mean' else 'mean and std'}; nothing else "
              "changes", ok)
    # Final latents are kept for the decode lab; they are what was decoded.
    video, spy, pipe = tiny_generate(**dict(own, save_latents=True))
    lat = pipe.last_latents
    ok = lat is not None and lat['anchor'] == a and lat['plan'] is None
    ok = ok and tuple(lat['stream1'].shape) == (16, frames, 8, 8)
    ok = ok and lat['stream1'].device.type == 'cpu' and lat['stream2'] is not None
    ok = ok and torch.equal(video.float().cpu(), lat['stream1'][:3])
    ok = ok and rel_err(video, pr) < 1e-6
    s2 = dict(own, save_latents=True, override_save_stream2=True)
    _, _, pipe2 = tiny_generate(**s2)
    ok = ok and torch.equal(pipe2.last_stream2_video.float().cpu(),
                            pipe2.last_latents['stream2'][:3])
    _, _, pipe3 = tiny_generate(**own)
    ok = ok and pipe3.last_latents is None
    plain, _, pipe4 = tiny_generate(paired=False, save_latents=True)
    ok = ok and pipe4.last_latents['stream2'] is None and torch.equal(
        plain.float().cpu(), pipe4.last_latents['stream1'][:3])
    check("round 9: --vr_save_latents keeps the final latents (stream 1 = the "
          "decoded main video, stream 2 = the s2 video) without changing the "
          "run", ok)
    errors, wrong = 0, []
    bad = ((dict(own, override_stream='clean', override_img2_span='last',
                 img2_fill='grey'), "fill 'grey' needs"),
           (dict(own, override_stream='long', img2_fill='grey'),
            "fill 'grey' needs"),
           (dict(own, img2_code_stats='median'), 'code stats must be'),
           (dict(own, override_img2_span='last', img2_code_stats='mean'),
            "code stats need span 'latent'"),
           (dict(own, override_stream='long', img2_code_stats='mean'),
            "code stats need span 'latent'"),
           (dict(paired=False, img2_code_stats='mean'),
            'self-attention override'),
           (dict(paired=False, img2_fill='grey'), 'self-attention override'))
    for kwargs, msg in bad:
        try:
            tiny_generate(**kwargs)
        except ValueError as error:
            if msg in str(error):
                errors += 1
            else:
                wrong.append(str(error)[:60])
    check("round 9: bad settings are rejected, each by its own check",
          errors == len(bad), f"{errors} of {len(bad)}; {wrong[:2]}")


def test_tiny_generate_round10(steps):
    from wan.image2video import smooth_latent_stats
    a, frames, k = 3, 7, 3
    own = dict(paired=False, override=True, override_stream='joint',
               override_latent='own', relay_prompt='relay prompt')
    pr, spy0, _ = tiny_generate(**own)  # M1-PR
    _, spy_lf, _ = tiny_generate(**dict(own, override_img2_span='last'))
    lf_y2 = spy_lf.calls[0]['y'][1]  # PR-LF's input
    ok = float(lf_y2[3, a].min()) == 1.0 and float(lf_y2[:3, a].max()) == 0.0
    ok = ok and rel_err(lf_y2[4:], tiny_joint_latents('last')) < 1e-5
    ok = ok and not torch.equal(lf_y2, spy0.calls[0]['y'][1])
    rel, spy, _ = tiny_generate(**dict(own, img2_release_step=k))
    per_step = len(spy.calls) // steps
    ok = ok and per_step >= 1 and len(spy.calls) == per_step * steps
    ok = ok and len(spy.calls) == len(spy0.calls)
    for i, call in enumerate(spy.calls):
        want = spy0.calls[i]['y'][1] if i // per_step < k else lf_y2
        ok = ok and torch.equal(call['y'][1], want)
        ok = ok and torch.equal(call['y'][0], spy0.calls[i]['y'][0])
    for i in range(k * per_step):  # before the release: M1-PR, call for call
        ok = ok and rel_err(spy.calls[i]['out'], spy0.calls[i]['out']) < 1e-6
    ok = ok and rel_err(rel, pr) > 1e-4
    check("round 10: --vr_img2_release_step K: stream 2 reads M1-PR's input "
          "(Img2 on all 4 frames) before step K and PR-LF's (joint encoding, "
          "Img2 and the mask on its latent's last frame only) from step K; "
          "stream 1's input and every step before K are unchanged", ok)
    # Decode fix: the main video's final latents, corrected and decoded again.
    for mode, window in (('meanstd', (2, 4)), ('mean', (3, 3))):
        video, _, pipe = tiny_generate(**dict(
            own, save_latents=True, decode_fix=mode, decode_fix_window=window))
        lat = pipe.last_latents['stream1']
        fixed = pipe.last_fixed_video
        want = smooth_latent_stats(lat, window[0], window[1], mode)[:3]
        ok = fixed is not None and rel_err(fixed.float().cpu(), want) < 1e-5
        ok = ok and rel_err(video, pr) < 1e-6  # the main video is unchanged
        lo, hi = window
        ok = ok and rel_err(fixed[:, :lo], video[:, :lo]) < 1e-6
        ok = ok and rel_err(fixed[:, hi + 1:], video[:, hi + 1:]) < 1e-6
        ok = ok and rel_err(fixed[:, lo:hi + 1], video[:, lo:hi + 1]) > 1e-4
        check(f"round 10: --vr_decode_fix {mode} on latents {lo}-{hi}: the "
              "fixed video is the decoded corrected latents, the main video "
              "is unchanged", ok)
    paired, _, pipe = tiny_generate(save_latents=True, decode_fix='meanstd',
                                    decode_fix_window=(2, 4))
    want = smooth_latent_stats(pipe.last_latents['stream1'], 2, 4, 'meanstd')
    ok = rel_err(pipe.last_fixed_video.float().cpu(), want[:3]) < 1e-5
    plain, _, pipe = tiny_generate(paired=False, save_latents=True,
                                   decode_fix='mean', decode_fix_window=(1, 1))
    want = smooth_latent_stats(pipe.last_latents['stream1'], 1, 1, 'mean')
    ok = ok and rel_err(pipe.last_fixed_video.float().cpu(), want[:3]) < 1e-5
    _, _, pipe = tiny_generate(**own)
    ok = ok and pipe.last_fixed_video is None
    check("round 10: the decode fix also works in the paired and plain paths, "
          "and is off by default", ok)
    L = load_lab_module()
    gen = torch.Generator(device='cpu')
    gen.manual_seed(101)
    z = (torch.randn(16, 9, 8, 8, generator=gen) * 0.6).to(DEVICE)
    z[:, 4] += 0.8
    same = all(torch.equal(smooth_latent_stats(z, lo, hi, mode),
                           L.smooth_stats(z, lo, hi, mode))
               for mode in ('mean', 'meanstd') for lo, hi in ((4, 4), (3, 6)))
    check("round 10: the decode fix in the run is the decode lab's correction",
          same)
    errors, wrong = 0, []
    bad = ((dict(own, override_stream='clean', img2_release_step=2),
            'the Img2 release needs'),
           (dict(own, override_img2_span='last', img2_release_step=2),
            'the Img2 release needs'),
           (dict(own, img2_code_stats='mean', img2_release_step=2),
            'the Img2 release needs'),
           (dict(own, img2_release_step=0), 'release step must be'),
           (dict(own, img2_release_step=steps), 'release step must be'),
           (dict(own, img2_release_step=2, stream_b_weight=0.5),
            'does not combine'),
           (dict(paired=False, img2_release_step=2),
            'self-attention override'),
           (dict(own, decode_fix='median', decode_fix_window=(2, 4)),
            'decode fix must be'),
           (dict(own, decode_fix='mean'), 'go together'),
           (dict(own, decode_fix_window=(2, 4)), 'go together'),
           (dict(own, decode_fix='mean', decode_fix_window=(0, 2)),
            'needs a latent on each side'),
           (dict(own, decode_fix='mean', decode_fix_window=(2, frames - 1)),
            'needs a latent on each side'),
           (dict(own, decode_fix='mean', decode_fix_window=(4, 3)),
            'needs a latent on each side'))
    for kwargs, msg in bad:
        pipe, spy = tiny_pipeline()
        try:
            tiny_generate(pipeline=(pipe, spy), **kwargs)
        except ValueError as error:
            if msg in str(error) and not spy.calls:
                errors += 1
            else:
                wrong.append(f"{str(error)[:60]} ({len(spy.calls)} model calls)")
    check("round 10: bad settings are rejected before sampling starts, each "
          "by its own check", errors == len(bad),
          f"{errors} of {len(bad)}; {wrong[:2]}")



def test_tiny_generate_round12(steps):
    a = 3
    own = dict(paired=False, override=True, override_stream='joint',
               override_latent='own', relay_prompt='relay prompt')
    pr, spy0, _ = tiny_generate(**own)  # M1-PR
    m1_y2 = spy0.calls[0]['y'][1]
    _, spy_lf, _ = tiny_generate(**dict(own, override_img2_span='last'))
    lf_y2 = spy_lf.calls[0]['y'][1]  # PR-LF's input
    _, spy_cl, _ = tiny_generate(**dict(own, override_stream='clean',
                                        override_img2_span='last'))
    cl_y2 = spy_cl.calls[0]['y'][1]  # CL-LF's input
    per_step = len(spy0.calls) // steps
    # SR4321: Img2 on the last k frames of its latent, k per step.
    sched = (4, 3, 2, 1)
    out, spy, _ = tiny_generate(**dict(own, img2_span_schedule=sched))
    ok = len(spy.calls) == len(spy0.calls)
    seen = set()
    for i, call in enumerate(spy.calls):
        k = sched[min(i // per_step, len(sched) - 1)]
        y2 = call['y'][1]
        ok = ok and torch.equal(call['y'][0], spy0.calls[i]['y'][0])
        if k == 4:
            ok = ok and torch.equal(y2, m1_y2)
        elif k == 1:
            ok = ok and torch.equal(y2, lf_y2)
        else:
            ok = ok and float(y2[4 - k:4, a].min()) == 1.0
            ok = ok and float(y2[:4 - k, a].max()) == 0.0
            ok = ok and rel_err(y2[4:], tiny_joint_latents(f'last{k}')) < 1e-5
            ok = ok and torch.equal(torch.cat([y2[:4, :a], y2[:4, a + 1:]], 1),
                                    torch.cat([m1_y2[:4, :a], m1_y2[:4, a + 1:]], 1))
        seen.add(k)
    ok = ok and seen == {1, 2, 3, 4}
    for i in range(per_step):  # step 0 is M1-PR, call for call
        ok = ok and rel_err(spy.calls[i]['out'], spy0.calls[i]['out']) < 1e-6
    ok = ok and rel_err(out, pr) > 1e-4
    # A schedule that ends on 4 frames is M1-PR; 4,4,1 is RL2.
    m1_again, _, _ = tiny_generate(**dict(own, img2_span_schedule=(4, 4)))
    ok = ok and rel_err(m1_again, pr) < 1e-6
    rl2, _, _ = tiny_generate(**dict(own, img2_release_step=2))
    sched_rl2, _, _ = tiny_generate(**dict(own, img2_span_schedule=(4, 4, 1)))
    ok = ok and rel_err(sched_rl2, rl2) < 1e-6
    check("round 12: --vr_img2_span_schedule 4,3,2,1: stream 2 gets Img2 on "
          "all 4 frames at step 0 (M1-PR's input), on the last 3 and the last "
          "2 frames (joint encoding, mask on those frames) at steps 1 and 2, "
          "and PR-LF's input from step 3; 4,4 is M1-PR and 4,4,1 is RL2", ok)
    # RLc2: CL-LF's input after the release.
    k = 2
    rlc, spy, _ = tiny_generate(**dict(own, img2_release_step=k,
                                       img2_release_input='own'))
    ok = len(spy.calls) == len(spy0.calls)
    for i, call in enumerate(spy.calls):
        want = m1_y2 if i // per_step < k else cl_y2
        ok = ok and torch.equal(call['y'][1], want)
    ok = ok and not torch.equal(cl_y2, lf_y2)
    ok = ok and torch.equal(cl_y2[:4], lf_y2[:4])
    ok = ok and rel_err(rlc, rl2) > 1e-5  # only Img2's latent, late steps
    joint_default, _, _ = tiny_generate(**dict(own, img2_release_step=k,
                                               img2_release_input='joint'))
    ok = ok and rel_err(joint_default, rl2) < 1e-6
    check("round 12: --vr_img2_release_input own: after the release stream 2 "
          "reads CL-LF's input exactly (Img2's latent encoded on its own); "
          "'joint' is round 10's RL", ok)
    # PR-4C: M1-PR's code with PR-LF's mask.
    pc, spy, _ = tiny_generate(**dict(own, override_img2_mask='last'))
    ok = all(torch.equal(c['y'][1][4:], m1_y2[4:]) for c in spy.calls)
    ok = ok and all(torch.equal(c['y'][1][:4], lf_y2[:4]) for c in spy.calls)
    ok = ok and float(lf_y2[3, a].min()) == 1.0 and float(lf_y2[:3, a].max()) == 0.0
    ok = ok and rel_err(pc, pr) > 1e-4
    check("round 12: --vr_override_img2_mask last: stream 2 keeps M1-PR's "
          "Img2 code (Img2 on all 4 frames) with PR-LF's mask (frame 4a "
          "only), at every step", ok)
    errors, wrong = 0, []
    bad = ((dict(own, img2_span_schedule=(4,)), 'span schedule lists'),
           (dict(own, img2_span_schedule=(4, 5, 1)), 'span schedule lists'),
           (dict(own, img2_span_schedule=(4, 0)), 'span schedule lists'),
           (dict(own, img2_span_schedule=tuple([4] * (steps + 1))),
            'span schedule lists'),
           (dict(own, override_img2_span='last', img2_span_schedule=(4, 1)),
            'span schedule needs'),
           (dict(own, override_stream='clean', img2_span_schedule=(4, 1)),
            'span schedule needs'),
           (dict(own, img2_code_stats='mean', img2_span_schedule=(4, 1)),
            'span schedule needs'),
           (dict(own, img2_release_step=2, img2_span_schedule=(4, 1)),
            'does not combine'),
           (dict(own, stream_b_weight=0.5, img2_span_schedule=(4, 1)),
            'does not combine'),
           (dict(own, img2_release_input='clean', img2_release_step=2),
            'release input must be'),
           (dict(own, img2_release_input='own'), 'needs a release step'),
           (dict(own, override_img2_mask='first'), 'mask must be'),
           (dict(own, override_img2_span='last', override_img2_mask='last'),
            "mask 'last' needs"),
           (dict(own, override_stream='clean', override_img2_mask='last'),
            "mask 'last' needs"),
           (dict(own, img2_release_step=2, override_img2_mask='last'),
            "mask 'last' needs"),
           (dict(own, img2_span_schedule=(4, 1), override_img2_mask='last'),
            "mask 'last' needs"),
           (dict(paired=False, img2_span_schedule=(4, 1)),
            'self-attention override'),
           (dict(paired=False, override_img2_mask='last'),
            'self-attention override'),
           (dict(paired=False, img2_release_input='own'),
            'self-attention override'))
    for kwargs, msg in bad:
        pipe, spy = tiny_pipeline()
        try:
            tiny_generate(pipeline=(pipe, spy), **kwargs)
        except ValueError as error:
            if msg in str(error) and not spy.calls:
                errors += 1
            else:
                wrong.append(f"{str(error)[:60]} ({len(spy.calls)} model calls)")
    check("round 12: bad settings are rejected before sampling starts, each "
          "by its own check", errors == len(bad),
          f"{errors} of {len(bad)}; {wrong[:2]}")


def test_tiny_generate_round13(steps):
    a, k = 3, 2
    own = dict(paired=False, override=True, override_stream='joint',
               override_latent='own', relay_prompt='relay prompt')
    pr, spy0, _ = tiny_generate(**own)  # M1-PR
    m1_y2 = spy0.calls[0]['y'][1]
    _, spy_lf, _ = tiny_generate(**dict(own, override_img2_span='last'))
    lf_y2 = spy_lf.calls[0]['y'][1]
    _, spy_b, _ = tiny_generate(**dict(own, stream_b_weight=0.5))
    y_b = spy_b.calls[0]['y'][2]  # Stream B's Img2-first input
    rl2, spy_rl2, _ = tiny_generate(**dict(own, img2_release_step=k))
    per_step = len(spy0.calls) // steps
    # RL2-VB: Stream B runs every step with weight 0 and supplies the values.
    vb, spy, _ = tiny_generate(**dict(own, img2_release_step=k,
                                      override_img2_value='before'))
    ok = len(spy.calls) == len(spy0.calls)
    for i, call in enumerate(spy.calls):
        ov = call['override']
        ok = ok and len(call['x']) == 3 and len(call['y']) == 3
        ok = ok and torch.equal(call['y'][1], m1_y2 if i // per_step < k else lf_y2)
        ok = ok and torch.equal(call['y'][2], y_b)
        ok = ok and torch.equal(call['x'][2], call['x'][1][:, a:])
        ok = ok and ov.get('img2_value') == 'before'
        ok = ok and float(ov['stream_b']['weight']) == 0.0
    ok = ok and rel_err(vb, rl2) > 1e-5
    vba, spy, _ = tiny_generate(**dict(own, img2_release_step=k,
                                       override_img2_value='all'))
    ok = ok and all(c['override'].get('img2_value') == 'all' for c in spy.calls)
    ok = ok and rel_err(vba, vb) > 1e-6
    check("round 13: --vr_override_img2_value: Stream B (Img2-first input, "
          "stream 2's latents a..) runs at every step with weight 0, the "
          "attention gets the scope, the release still switches stream 2's "
          "input at step K", ok)
    # RL2-SP: Gordon's surgical swap, before and after the release.
    sp, spy, _ = tiny_generate(**dict(own, override_stream='splice',
                                      img2_release_step=k))
    ok = len(spy.calls) == len(spy0.calls)
    for i, call in enumerate(spy.calls):
        y1, y2 = call['y'][0], call['y'][1]
        want = m1_y2 if i // per_step < k else lf_y2
        ok = ok and torch.equal(y2[:4], want[:4])  # the masks
        ok = ok and torch.equal(y2[4:, a], want[4:, a])  # Img2's latent
        ok = ok and torch.equal(y2[4:, :a], y1[4:, :a])  # Img1-only pass
        ok = ok and torch.equal(y2[4:, a + 1:], y1[4:, a + 1:])
    ok = ok and not torch.equal(lf_y2[4:, a + 1:], spy.calls[-1]['y'][0][4:, a + 1:])
    ok = ok and rel_err(sp, rl2) > 1e-5
    _, spy, _ = tiny_generate(**dict(own, override_stream='splice',
                                     img2_span_schedule=(4, 2, 1)))
    y1 = spy.calls[0]['y'][0]
    ok = ok and all(torch.equal(c['y'][1][4:, a + 1:], y1[4:, a + 1:])
                    for c in spy.calls)
    ok = ok and float(spy.calls[per_step]['y'][1][2:4, a].min()) == 1.0
    ok = ok and float(spy.calls[per_step]['y'][1][:2, a].max()) == 0.0
    check("round 13: the splice encoding with the release (and a span "
          "schedule): every hint latent but Img2's comes from the Img1-only "
          "pass, Img2's latent and the masks follow M1-PR then PR-LF", ok)
    # RL2-G: both.
    g, spy, _ = tiny_generate(**dict(own, override_stream='splice',
                                     img2_release_step=k,
                                     override_img2_value='before'))
    ok = all(len(c['x']) == 3 and c['override'].get('img2_value') == 'before'
             and torch.equal(c['y'][1][4:, :a], c['y'][0][4:, :a])
             for c in spy.calls)
    ok = ok and rel_err(g, sp) > 1e-5 and rel_err(g, vb) > 1e-5
    check("round 13: splice + clean Img2 values together (Gordon's fix as he "
          "ran it, on RL2)", ok)
    errors, wrong = 0, []
    bad = ((dict(own, override_img2_value='after'), 'clean Img2 values must'),
           (dict(own, override_stream='clean', override_img2_value='before'),
            'clean Img2 values need'),
           (dict(own, override_mode='route', override_img2_value='before'),
            'clean Img2 values need'),
           (dict(own, stream_b_weight=0.5, override_img2_value='before'),
            'clean Img2 values need'),
           (dict(own, rerender_from_step=3, override_img2_value='before'),
            'clean Img2 values need'),
           (dict(own, override_stream='clean', img2_release_step=2),
            'the Img2 release needs'),
           (dict(paired=False, override_img2_value='before'),
            'self-attention override'))
    for kwargs, msg in bad:
        pipe, spy = tiny_pipeline()
        try:
            tiny_generate(pipeline=(pipe, spy), **kwargs)
        except ValueError as error:
            if msg in str(error) and not spy.calls:
                errors += 1
            else:
                wrong.append(f"{str(error)[:60]} ({len(spy.calls)} model calls)")
    check("round 13: bad settings are rejected before sampling starts, each "
          "by its own check", errors == len(bad),
          f"{errors} of {len(bad)}; {wrong[:2]}")


def test_tiny_generate_round14(steps):
    a, k = 3, 2
    own = dict(paired=False, override=True, override_stream='joint',
               override_latent='own', relay_prompt='relay prompt')
    per_step = None
    rows = []
    for name, extra in (('M1-PR', {}), ('PR-LF', dict(override_img2_span='last')),
                        ('RL2', dict(img2_release_step=k)),
                        ('SR421', dict(img2_span_schedule=(4, 2, 1))),
                        ('RL2-G', dict(img2_release_step=k, override_stream='splice',
                                       override_img2_value='before'))):
        two, spy2, pipe2 = tiny_generate(**dict(own, override_save_stream2=True,
                                                **extra))
        s2_video = pipe2.last_stream2_video
        one, spy1, pipe1 = tiny_generate(**dict(own, override_single=True, **extra))
        per_step = len(spy1.calls) // steps
        ok = len(spy1.calls) == len(spy2.calls)
        gordon = 'override_img2_value' in extra
        # The inputs (y) and step 0's latent (the same noise) must be identical
        # on every device. After step 0 the latents are identical on the CPU;
        # on a GPU the bfloat16 kernels round differently for a batch of 1 and
        # a batch of 2 (cuBLAS / FlashAttention pick other tilings), so there
        # they must agree to bfloat16 accuracy, the video's tolerance.
        exact = DEVICE.type == 'cpu'
        lat_err, inputs_ok = 0.0, True
        for i, (c1, c2) in enumerate(zip(spy1.calls, spy2.calls)):
            ok = ok and len(c1['x']) == (2 if gordon else 1)
            inputs_ok = inputs_ok and torch.equal(c1['y'][0], c2['y'][1])  # stream 2's input
            if exact or i < per_step:
                ok = ok and torch.equal(c1['x'][0], c2['x'][1])  # stream 2's latent
            if i >= per_step:
                lat_err = max(lat_err, rel_err(c1['x'][0], c2['x'][1]))
            if gordon:
                inputs_ok = inputs_ok and torch.equal(c1['y'][1], c2['y'][2])
                ok = ok and torch.equal(c1['x'][1], c1['x'][0][:, a:])
                ok = ok and c1['override'] == dict(single=True, anchor=a,
                                                   img2_value='before')
            else:
                ok = ok and c1.get('override') is None
            ok = ok and len(c1['context']) == len(c1['x'])
        err = rel_err(one, s2_video)
        ok = (ok and inputs_ok and lat_err < 1e-2 and err < 1e-2
              and pipe1.last_latents is None)
        rows.append(f"{name} {err:.0e}")
        check(f"round 14: --vr_override_single ({name}): stream 2 alone, step "
              "by step the same input and latent as the two-stream run's stream "
              f"2{', plus Stream B for the clean values' if gordon else ''}; "
              "the video is the old _s2 video", ok,
              f"inputs {'identical' if inputs_ok else 'DIFFER'}; latents after "
              f"step 0 rel err {lat_err:.1e}{' (CPU: must be 0)' if exact else ''}; "
              f"video rel err {err:.1e}")
    _, _, pipe = tiny_generate(**dict(own, override_single=True, save_latents=True))
    ok = pipe.last_latents is not None and pipe.last_latents['stream2'] is None
    check("round 14: the single stream saves its latents as stream 1 (the "
          "video), no stream 2", ok)
    errors, wrong = 0, []
    bad = ((dict(own, override_single=True, override_mode='route'),
            'single stream does not combine'),
           (dict(own, override_single=True, stream_b_weight=0.5),
            'single stream does not combine'),
           (dict(own, override_single=True, rerender_from_step=3),
            'single stream does not combine'),
           (dict(own, override_single=True, override_save_stream2=True),
            'single stream does not combine'),
           (dict(own, override_single=True, override_handover_step=2),
            'single stream does not combine'),
           (dict(paired=False, override_single=True), 'self-attention override'))
    for kwargs, msg in bad:
        pipe, spy = tiny_pipeline()
        try:
            tiny_generate(pipeline=(pipe, spy), **kwargs)
        except ValueError as error:
            if msg in str(error) and not spy.calls:
                errors += 1
            else:
                wrong.append(f"{str(error)[:60]} ({len(spy.calls)} model calls)")
    check("round 14: bad single-stream settings are rejected before sampling "
          "starts", errors == len(bad), f"{errors} of {len(bad)}; {wrong[:2]}")

class FakeVAEDecoding(FakeVAE):
    """FakeVAE that also decodes to frames: latent 0 -> 1 frame, latent k -> 4
    frames (a small ramp inside the latent), channels 0-2, 8x nearest-up."""

    def decode(self, latents):
        out = []
        for u in latents:
            frames = [u[:3, 0]] + [u[:3, k] * (1.0 - 0.05 * i)
                                   for k in range(1, u.shape[1]) for i in range(4)]
            v = torch.stack(frames, dim=1)
            v = v.repeat_interleave(8, dim=2).repeat_interleave(8, dim=3)
            out.append(v.clamp(-1, 1))
        return out


def load_probe_module():
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'vr_decoder_probe.py')
    spec = importlib.util.spec_from_file_location('vr_decoder_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_round8_probe():
    P = load_probe_module()
    a, frames = 3, 25
    vae = FakeVAEDecoding(DEVICE)
    img1, img2 = tiny_img_tensor(0), tiny_img_tensor(1)
    gen = torch.Generator(device='cpu')
    gen.manual_seed(81)
    video = (torch.rand(3, frames, 64, 64, generator=gen) * 1.6 - 0.8).to(DEVICE)
    zc = vae.encode([video])[0]
    cands = P.build_candidates(vae.encode, img1, img2, video, a, gain=1.1,
                               prev_latent=zc[:, a - 1])
    # The candidates that reproduce a run's input equal the pipeline's own.
    own = dict(paired=False, override=True, override_stream='joint',
               override_latent='own', relay_prompt='relay prompt')
    runs = {'joint4': dict(own), 'jointLF': dict(own, override_img2_span='last'),
            'alone4': dict(own, override_stream='clean'),
            'aloneLF': dict(own, override_stream='clean',
                            override_img2_span='last'),
            'first': dict(own, override_stream='fresh')}
    errs = {}
    for name, kwargs in runs.items():
        _, spy, _ = tiny_generate(**kwargs)
        errs[name] = rel_err(cands[name]['latent'], spy.calls[0]['y'][1][4:, a])
    check("round 8 probe: joint4, jointLF, alone4, aloneLF and first are "
          "exactly the Img2 latents of M1-PR, PR-LF, PR-CL, CL-LF and PR-FE",
          max(errs.values()) < 1e-5,
          ' '.join(f"{k} {v:.0e}" for k, v in errs.items()))
    s, e = P.anchor_frames(a)
    grey = torch.zeros_like(img2)

    def lat(frames_list, index):
        return vae.encode([torch.stack(frames_list, dim=1)])[0][:, index]
    ctx = [video[:, i] for i in range(s)]
    bright = P.brighten(img2, 1.1)
    want = {
        'long4': lat([img2] * frames, a),
        'longLF': lat([img2] * s + [grey] * 3 + [img2], a),
        'ctx4': lat(ctx + [img2] * 4, a),
        'ctxLF': lat(ctx + [grey] * 3 + [img2], a),
        'ctxLFv': lat([video[:, i] for i in range(e)] + [img2], a),
        'ctxLFh': lat(ctx + [video[:, s - 1]] * 3 + [img2], a),
        'alone4_b': lat([bright] * 5, 1),
        'aloneLF_b': lat([grey] * 4 + [bright], 1)}
    ok = all(rel_err(cands[k]['latent'], v) < 1e-5 for k, v in want.items())
    ok = ok and float(bright.max()) <= 1.0 and rel_err(
        (bright + 1) / 2, ((img2 + 1) / 2 * 1.1).clamp(0, 1)) < 1e-6
    m = cands['alone4_m']['latent']
    base = cands['alone4']['latent']
    ok = ok and max_abs(m.mean(dim=(1, 2)), zc[:, a - 1].mean(dim=(1, 2))) < 1e-5
    ok = ok and max_abs(m - m.mean(dim=(1, 2), keepdim=True),
                        base - base.mean(dim=(1, 2), keepdim=True)) < 1e-5
    for k, c in cands.items():
        if c['clip'] is not None:
            ok = ok and c['clip'].shape[1] == c['index'] + 1 and torch.equal(
                c['clip'][:, c['index']], c['latent'])
    check("round 8 probe: long, in-context, brightened and mean-moved "
          "candidates are built from the clips the plan names; each keeps "
          "its own clip for the own-clip decode", ok)
    z = P.splice(zc, a, cands['ctx4']['latent'])
    rest = [i for i in range(a + 2) if i != a]
    check("round 8 probe: splice replaces only latent a and keeps latents "
          "0..a+1 of the video",
          z.shape[1] == a + 2 and torch.equal(z[:, a], cands['ctx4']['latent'])
          and all(torch.equal(z[:, i], zc[:, i]) for i in rest))
    import numpy as np
    flat = np.zeros((4, 6, 3), np.uint8)
    flat[...] = (100, 150, 200)
    st = P.frame_stats(flat)
    seq = []
    for f in range(s - 4, e + 5):
        v = 100 if f < s else (120 if f > e else 100 - 10 * (f - s + 1))
        fr = np.zeros((8, 8, 3), np.uint8)
        fr[...] = (v, v, v)  # grey: brightness = v
        seq.append(fr)
    summ = P.summarise(np.stack(seq), s - 4, a, seq[0])
    check("round 8 probe: brightness (BT.601), channel means, sharpness and "
          "the dip against the 4 frames before Img2's latent",
          abs(st['luma'] - 140.75) < 1e-3 and st['sharp'] == 0.0
          and [round(d, 3) for d in summ['dip']] == [-10.0, -20.0, -30.0, -40.0]
          and max(abs(x - 0.6) for x in summ['rgb_at_last']) < 1e-6
          and abs(summ['rms_img2_first'] - P.rms(seq[s - (s - 4)], seq[0])) < 1e-9)
    results, sheet = P.probe_scene(vae.encode, vae.decode, img1, img2, video, a)
    names = ['video'] + list(cands)
    ok = list(results) == names and set(sheet) == set(names)
    for k, r in results.items():
        ok = ok and len(r['dip']) == 4 and sorted(r['luma']) == list(
            range(s - 4, e + 5))
        if k in ('alone4_m',):
            ok = ok and 'own_clip_luma' not in r
            ok = ok and r['latent']['mean_vs_prev'] < 1e-4
        else:
            ok = ok and len(r['own_clip_luma']) == (1 if k == 'first' else 4)
    # One candidate by hand: decode the spliced latents, measure frame e.
    dec = vae.decode([P.splice(zc, a, cands['aloneLF']['latent'])])[0]
    by_hand = P.frame_stats(P.to_uint8(dec[:, e:e + 1])[0])['luma']
    ok = ok and abs(results['aloneLF']['luma'][e] - by_hand) < 1e-6
    text = P.report('fnr', a, results, 'ctx.mp4', 152.0)
    ok = ok and all(n in text for n in names) and 'Reading:' in text
    with tempfile.TemporaryDirectory() as tmp:
        P.save_sheet(os.path.join(tmp, 's.png'), sheet, s - 4)
        ok = ok and os.path.getsize(os.path.join(tmp, 's.png')) > 0
    check("round 8 probe: every candidate decoded inside the video and in "
          "its own clip, measured on frames 4a-7..4a+4, reported", ok)
    # End to end through main() with a small mp4 and a fake Wan VAE.
    try:
        import cv2
    except Exception:  # noqa: BLE001
        print("[INFO] no OpenCV: round 8 probe main() not run")
        return
    import types
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmp:
        mp4 = os.path.join(tmp, 'ctx.mp4')
        writer = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*'mp4v'), 16,
                                 (64, 64))
        rng = np.random.RandomState(5)
        for _ in range(frames):
            writer.write(rng.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        writer.release()
        for i, im in enumerate(tiny_images(2)):
            im.save(os.path.join(tmp, f'i{i}.png'))
        try:
            readable = P.load_video(mp4).shape[1] == frames
        except RuntimeError:
            readable = False
        if not readable:
            print("[INFO] OpenCV cannot round-trip mp4 here: main() not run")
            return
        fake_cfg = types.ModuleType('wan.configs')
        fake_cfg.WAN_CONFIGS = {'i2v-A14B': types.SimpleNamespace(
            vae_checkpoint='vae.pth')}
        fake_vae = types.ModuleType('wan.modules.vae2_1')
        fake_vae.Wan2_1_VAE = lambda vae_pth, device: FakeVAEDecoding(DEVICE)
        saved = {k: sys.modules.get(k) for k in ('wan.configs',
                                                 'wan.modules.vae2_1')}
        sys.modules['wan.configs'] = fake_cfg
        sys.modules['wan.modules.vae2_1'] = fake_vae
        P.SCENES = {'fnr': dict(images=['i0.png', 'i1.png'], anchor=a,
                                context=[mp4]),
                    'sam': dict(images=['i0.png', 'i1.png'], anchor=a,
                                context=[os.path.join(tmp, 'missing.mp4')])}
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            P.main(['--scenes', 'fnr,sam', '--image_dir', tmp,
                    '--ckpt_dir', tmp])
            txt = open(os.path.join('r8_outputs', 'r8_decoder_probe_fnr.txt')).read()
            import json as _json
            js = _json.load(open(os.path.join('r8_outputs',
                                              'r8_decoder_probe_fnr.json')))
            ok = ('aloneLF' in txt and 'ctx4' in txt
                  and set(js['results']) == set(names)
                  and os.path.isfile(os.path.join('r8_outputs',
                                                  'r8_decoder_probe_fnr_sheet.png'))
                  and not os.path.exists(os.path.join(
                      'r8_outputs', 'r8_decoder_probe_sam.txt')))
        finally:
            os.chdir(cwd)
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
    check("round 8 probe: main() reads the context mp4 and the images, "
          "writes the table, the JSON and the sheet; a scene without its "
          "context video is skipped", ok)


def load_lab_module():
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'vr_latent_fix.py')
    spec = importlib.util.spec_from_file_location('vr_latent_fix', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_round9_lab():
    import numpy as np
    L = load_lab_module()
    gen = torch.Generator(device='cpu')
    gen.manual_seed(93)
    z = (torch.randn(16, 7, 8, 8, generator=gen) * 0.5).to(DEVICE)
    z[:, 3] += 0.7  # a bright latent between its neighbours
    ok = True
    for mode in ('mean', 'meanstd'):
        for lo, hi in ((3, 3), (2, 4)):
            out = L.smooth_stats(z, lo, hi, mode)
            n = hi - lo + 2
            for j in range(z.shape[1]):
                if not lo <= j <= hi:
                    ok = ok and torch.equal(out[:, j], z[:, j])
                    continue
                t = (j - lo + 1) / n
                for c in range(16):
                    m_t = (float(z[c, lo - 1].mean()) * (1 - t)
                           + float(z[c, hi + 1].mean()) * t)
                    ok = ok and abs(float(out[c, j].mean()) - m_t) < 1e-5
                    d_in = z[c, j] - z[c, j].mean()
                    d_out = out[c, j] - out[c, j].mean()
                    if mode == 'mean':
                        ok = ok and max_abs(d_in, d_out) < 1e-5
                    else:
                        s_t = (float(z[c, lo - 1].std()) * (1 - t)
                               + float(z[c, hi + 1].std()) * t)
                        ok = ok and abs(float(out[c, j].std()) - s_t) < 1e-4
                        ok = ok and max_abs(d_in / float(z[c, j].std()),
                                            d_out / float(out[c, j].std())) < 1e-4
    raised = 0
    for args in ((0, 2, 'mean'), (3, 6, 'mean'), (3, 3, 'median')):
        try:
            L.smooth_stats(z, *args)
        except ValueError:
            raised += 1
    check("round 9 lab: Lm/Lms put each channel's mean (and std) of latents "
          "lo..hi on the line between their neighbours, keep each latent's "
          "pattern, leave other latents untouched; bad windows refused",
          ok and raised == 3)
    vid = (torch.rand(3, 13, 16, 16, generator=gen) * 0.6 - 0.3).to(DEVICE)
    vid[:, 5:8] -= 0.25
    out = L.pixel_fix(vid, 5, 7)
    ok = all(torch.equal(out[:, f], vid[:, f].float()) or max_abs(
        out[:, f], vid[:, f]) < 1e-6 for f in range(13) if not 5 <= f <= 7)
    for f in range(5, 8):
        t = (f - 4) / 4
        for c in range(3):
            x0, x1 = (vid[c, 4] + 1) / 2, (vid[c, 8] + 1) / 2
            y = (out[c, f] + 1) / 2
            ok = ok and abs(float(y.mean()) - (float(x0.mean()) * (1 - t)
                                               + float(x1.mean()) * t)) < 1e-4
            ok = ok and abs(float(y.std()) - (float(x0.std()) * (1 - t)
                                              + float(x1.std()) * t)) < 1e-4
    check("round 9 lab: P puts each colour channel's mean and std of the "
          "window's frames on the line between the frames around it", ok)
    # The lab's JM latent is exactly round 9b's JM-4 input.
    vae = FakeVAEDecoding(DEVICE)
    img1, img2 = tiny_img_tensor(0), tiny_img_tensor(1)
    a = 3
    own = dict(paired=False, override=True, override_stream='joint',
               override_latent='own', relay_prompt='relay prompt')
    cands = L.jm_candidates(vae.encode, img1, img2, 25, a)
    _, spy_j, _ = tiny_generate(**own)
    _, spy_ms, _ = tiny_generate(**dict(own, img2_code_stats='meanstd'))
    _, spy_m, _ = tiny_generate(**dict(own, img2_code_stats='mean'))
    _, spy_l, _ = tiny_generate(**dict(own, override_stream='long'))
    errs = [rel_err(cands['joint4'], spy_j.calls[0]['y'][1][4:, a]),
            rel_err(cands['joint4_ms'], spy_ms.calls[0]['y'][1][4:, a]),
            rel_err(cands['joint4_m'], spy_m.calls[0]['y'][1][4:, a]),
            rel_err(cands['long4'], spy_l.calls[0]['y'][1][4:, a])]
    check("round 9 lab: the JM latents equal the pipeline's M1-PR, JM-4 "
          "(meanstd), JM mean and LG inputs", max(errs) < 1e-4,
          ' '.join(f"{e:.0e}" for e in errs))
    fr = np.zeros((4, 4, 3), np.uint8)
    fr[..., 0], fr[..., 1], fr[..., 2] = 200, 100, 50
    prev = np.zeros_like(fr)
    m = L.frame_metrics(fr, prev)
    ok = abs(m['sat'] - 150 / 200 * 255) < 1e-3 and m['contrast'] == 0.0
    ok = ok and abs(m['change'] - (200 + 100 + 50) / 3) < 1e-4
    check("round 9 lab: saturation (HSV S), contrast and frame change", ok)
    # run_source on saved latents and on a video.
    zl = vae.encode([(torch.rand(3, 25, 64, 64, generator=gen) * 1.6
                      - 0.8).to(DEVICE)])[0]
    L.SCENES = dict(fnr=dict(images=['i0.png', 'i1.png'], anchor=a),
                    sam=dict(images=['i0.png', 'i1.png'], anchor=a))
    saved = []
    res, sheet, frange, jm_rows = L.run_source(
        'x-lat', 'sam', 'lat', zl, [(2, 3), (3, 4)], vae.encode, vae.decode,
        images=(img1, img2), jm=True, save=lambda n, v: saved.append((n, v)))
    names = ['rt', 'Lm2-3', 'Lms2-3', 'P2-3', 'Lm3-4', 'Lms3-4', 'P3-4']
    ok = list(res) == names and [n for n, _ in saved] == [
        f'x-lat_{n}' for n in names] and list(sheet) == names
    ok = ok and frange == (max(1, L.window_frames(2, 2)[0] - 8),
                           min(24, L.window_frames(4, 4)[1] + 8))
    vids = dict(saved)
    ok = ok and torch.equal(vids['x-lat_rt'], vae.decode([zl])[0])
    ok = ok and torch.equal(vids['x-lat_Lms3-4'],
                            vae.decode([L.smooth_stats(zl, 3, 4, 'meanstd')])[0])
    ok = ok and max_abs(vids['x-lat_P2-3'], L.pixel_fix(vae.decode([zl])[0],
                                                        5, 12)) < 1e-6
    ok = ok and set(jm_rows) == {'joint4', 'joint4_m', 'joint4_ms', 'long4'}
    dec = vae.decode([P_splice(zl, a, cands['joint4_ms'])])[0]
    ok = ok and abs(jm_rows['joint4_ms'][12] - L.frame_metrics(
        L.P.to_uint8(dec)[12])['luma']) < 1e-6
    ok = ok and len(next(iter(res.values()))) == frange[1] - frange[0] + 1
    vres, _, _, _ = L.run_source('x-vid', 'fnr', 'vid', vae.decode([zl])[0],
                                 [(3, 3)], vae.encode, vae.decode)
    ok = ok and list(vres) == ['orig', 'rt', 'Lm3-3', 'Lms3-3', 'P3-3']
    text = L.report('x-lat', 'sam', 'p.pt', 'lat', res, frange, jm_rows, 50.0)
    ok = ok and 'Lms3-4' in text and 'JM check' in text and 'saved latents' in text
    check("round 9 lab: every variant decoded from the corrected latents (or "
          "fixed on pixels), measured, saved; the JM latents decoded in the "
          "source's latents", ok)
    # main(): a saved-latents source, a missing source; fake Wan modules.
    import types
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'r9_outputs'))
        torch.save(dict(stream1=zl.cpu(), stream2=None, plan=None, anchor=a),
                   os.path.join(tmp, 'r9_outputs', 'lat.pt'))
        torch.save(dict(stream1=None, stream2=None, plan=None, anchor=a),
                   os.path.join(tmp, 'r9_outputs', 'later.pt'))
        for i, im in enumerate(tiny_images(2)):
            im.save(os.path.join(tmp, f'i{i}.png'))
        fake_cfg = types.ModuleType('wan.configs')
        fake_cfg.WAN_CONFIGS = {'i2v-A14B': types.SimpleNamespace(
            vae_checkpoint='vae.pth')}
        fake_vae = types.ModuleType('wan.modules.vae2_1')
        fake_vae.Wan2_1_VAE = lambda vae_pth, device: FakeVAEDecoding(DEVICE)
        fake_utils = types.ModuleType('wan.utils.utils')
        written = []
        fake_utils.save_video = lambda tensor, save_file, **k: written.append(
            (save_file, tuple(tensor.shape)))
        keys = ('wan.configs', 'wan.modules.vae2_1', 'wan.utils.utils')
        saved_mods = {k: sys.modules.get(k) for k in keys}
        sys.modules.update({'wan.configs': fake_cfg,
                            'wan.modules.vae2_1': fake_vae,
                            'wan.utils.utils': fake_utils})
        L.SOURCES = {'s-lat': dict(scene='sam', paths=['r9_outputs/lat.pt',
                                                       'missing.pt',
                                                       'r9_outputs/later.pt'],
                                   windows=[(2, 3)], jm=True),
                     's-none': dict(scene='fnr', paths=['nothing.mp4'],
                                    windows=[(3, 3)])}
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            L.main(['--image_dir', tmp, '--ckpt_dir', tmp])
            out = os.path.join('r9_outputs', 'r9_fix_s-lat-lat')
            js = __import__('json').load(open(out + '.json'))
            ok = (os.path.isfile(out + '.txt') and os.path.isfile(out + '_sheet.png')
                  and js['kind'] == 'lat' and js['path'] == 'r9_outputs/lat.pt'
                  and set(js['jm']) == {
                      'joint4', 'joint4_m', 'joint4_ms', 'long4'}
                  and sorted(os.path.basename(f) for f, _ in written) == sorted(
                      f'r9_fix_s-lat-lat_{n}.mp4' for n in ['rt', 'Lm2-3', 'Lms2-3', 'P2-3'])
                  and all(sh == (1, 3, 25, 64, 64) for _, sh in written)
                  and not any('s-none' in f for f in os.listdir('r9_outputs')))
            name, spec = L.parse_add('q=fnr:r9_outputs/x.pt:10-10,10-11')
            ok = ok and name == 'q' and spec['windows'] == [(10, 10), (10, 11)] \
                and spec['jm'] is False
        finally:
            os.chdir(cwd)
            for k, v in saved_mods.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
    check("round 9 lab: main() takes the first existing file of a source, "
          "decodes saved latents, writes every variant's video, the table, "
          "the JSON and the sheet; a source with no file is skipped", ok)


def P_splice(z, a, latent):
    out = z[:, :a + 2].clone()
    out[:, a] = latent.to(out.dtype)
    return out


def load_global_fix_module():
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'vr_global_fix.py')
    spec = importlib.util.spec_from_file_location('vr_global_fix', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_round12_global_fix():
    G = load_global_fix_module()
    a, frames = 3, 25
    vae = FakeVAEDecoding(DEVICE)
    img1, img2 = tiny_img_tensor(0), tiny_img_tensor(1)
    gen = torch.Generator(device='cpu')
    gen.manual_seed(121)
    z = vae.encode([(torch.rand(3, frames, 64, 64, generator=gen) * 1.6
                     - 0.8).to(DEVICE)])[0]
    z[:, 4:] = z[:, 4:] * 1.3 + 0.4  # a drift after Img2's latent
    ref1 = G.still_latents(vae.encode, img1, frames)
    ref2 = G.still_latents(vae.encode, img2, frames)
    wts = G.key_weights(7, a)
    ok = wts == [0.0, 1 / 3, 2 / 3, 1.0, 1.0, 1.0, 1.0]
    ok = ok and tuple(ref1.shape) == tuple(z.shape)
    ok = ok and rel_err(ref1[:, 2], ref1[:, 2]) == 0.0
    for mode in ('mean', 'meanstd'):
        out = G.global_fix(z, ref1, ref2, a, mode)
        ok = ok and torch.equal(out[:, 0], z[:, 0])
        for j in range(1, 7):
            w = wts[j]
            m_t = (1 - w) * ref1[:, j].mean(dim=(1, 2)) + w * ref2[:, j].mean(
                dim=(1, 2))
            ok = ok and max_abs(out[:, j].mean(dim=(1, 2)), m_t) < 1e-4
            if mode == 'meanstd':
                s_t = (1 - w) * ref1[:, j].std(dim=(1, 2)) + w * ref2[:, j].std(
                    dim=(1, 2))
                ok = ok and max_abs(out[:, j].std(dim=(1, 2)), s_t) < 1e-4
            else:
                ok = ok and max_abs(out[:, j].std(dim=(1, 2)),
                                    z[:, j].std(dim=(1, 2))) < 1e-4
            pat_in = (z[:, j] - z[:, j].mean(dim=(1, 2))[:, None, None]) / z[
                :, j].std(dim=(1, 2))[:, None, None]
            pat_out = (out[:, j] - out[:, j].mean(dim=(1, 2))[:, None, None]) / out[
                :, j].std(dim=(1, 2))[:, None, None]
            ok = ok and max_abs(pat_in, pat_out) < 1e-3
    check("round 12a: Gm / Gms set every latent's channel means (and stds) on "
          "the line from Img1's still to Img2's (weight min(j/a, 1)), keep the "
          "spatial pattern, leave latent 0 alone", ok)
    # lowfreq: an offset map that is constant per channel is removed exactly.
    offset = torch.randn(16, 1, 1, 1, generator=gen).to(DEVICE)
    shifted = ref2 + offset
    out = G.global_fix(shifted, ref2, ref2, a, 'lowfreq')
    ok = rel_err(out[:, 1:], ref2[:, 1:]) < 1e-4 and torch.equal(
        out[:, 0], shifted[:, 0])
    low = G.lowfreq(z[:, 2])
    ok = ok and tuple(low.shape) == tuple(z[:, 2].shape)
    flat = torch.full((16, 8, 8), 0.3).to(DEVICE)
    ok = ok and max_abs(G.lowfreq(flat), flat) < 1e-6
    out = G.global_fix(z, ref1, ref2, a, 'lowfreq')
    ok = ok and rel_err(out[:, 3], z[:, 3]) > 1e-3
    for j in (1, 3, 5):
        w = wts[j]
        l1, l2 = G.lowfreq(ref1[:, j]), G.lowfreq(ref2[:, j])
        s_t = (1 - w) * (ref1[:, j] - l1).std(dim=(1, 2)) + w * (
            ref2[:, j] - l2).std(dim=(1, 2))
        rest = out[:, j] - ((1 - w) * l1 + w * l2)
        ok = ok and max_abs(rest.std(dim=(1, 2)), s_t) < 1e-4
        ok = ok and max_abs(rest / s_t[:, None, None], (
            z[:, j] - G.lowfreq(z[:, j])) / (z[:, j] - G.lowfreq(z[:, j])).std(
            dim=(1, 2))[:, None, None]) < 1e-3
    check("round 12a: Gl swaps each latent's coarse colour map for the stills' "
          "and keeps the rest (a constant colour offset is removed exactly)",
          ok)
    # Pixel version: every frame's colour means and stds on the stills' line.
    gen.manual_seed(122)
    video = (torch.rand(3, frames, 64, 64, generator=gen) * 0.6 - 0.1).to(DEVICE)
    dec1, dec2 = vae.decode([ref1])[0], vae.decode([ref2])[0]
    out = G.global_pixel_fix(video, dec1, dec2, a)
    ok = tuple(out.shape) == tuple(video.shape)
    for f in (0, 5, 12, 20):
        w = min(f / (4 * a), 1.0)
        r1, r2, o = (dec1[:, f] + 1) / 2, (dec2[:, f] + 1) / 2, (out[:, f] + 1) / 2
        m_t = (1 - w) * r1.mean(dim=(1, 2)) + w * r2.mean(dim=(1, 2))
        s_t = (1 - w) * r1.std(dim=(1, 2)) + w * r2.std(dim=(1, 2))
        clipped = bool(((o <= 0) | (o >= 1)).any())
        ok = ok and (clipped or (max_abs(o.mean(dim=(1, 2)), m_t) < 1e-4
                                 and max_abs(o.std(dim=(1, 2)), s_t) < 1e-4))
    ok = ok and float(out.min()) >= -1.0 and float(out.max()) <= 1.0
    spike = torch.full_like(video, -0.5)
    spike[:, :, 0, 0] = 0.9  # one bright pixel: scaled to the stills' spread it overshoots
    out = G.global_pixel_fix(spike, dec1, dec2, a)
    ok = ok and float(out.max()) == 1.0 and float(out.min()) >= -1.0
    check("round 12a: GP sets every frame's colour means and stds on the line "
          "from the decoded stills (weight min(f/4a, 1)), clipped to range", ok)
    # run_source: every variant decoded, measured, the new ones saved.
    G.L.SCENES = dict(fnr=dict(images=['i0.png', 'i1.png'], anchor=a),
                      sam=dict(images=['i0.png', 'i1.png'], anchor=a))
    G.LOCAL = dict(fnr=(2, 3), sam=(2, 4))
    saved = []

    def decode(latents):  # not affine, so a pixel fix after Gms is not GP
        return [(v + 1) ** 2 / 2 - 1 for v in vae.decode(latents)]
    dec1, dec2 = decode([ref1])[0], decode([ref2])[0]
    res, sheet, target = G.run_source(
        'x', 'fnr', z, (img1, img2), vae.encode, decode,
        save=lambda n, v: saved.append((n, v)))
    names = ['rt', 'Lms2-3', 'Gm', 'Gms', 'Gl', 'GP']
    vids = dict(saved)
    ok = list(res) == names and list(sheet) == names
    ok = ok and [n for n, _ in saved] == ['x_Gm', 'x_Gms', 'x_Gl', 'x_GP']
    ok = ok and torch.equal(vids['x_Gms'], decode([G.global_fix(
        z, ref1, ref2, a, 'meanstd')])[0])
    ok = ok and torch.equal(vids['x_Gl'], decode([G.global_fix(
        z, ref1, ref2, a, 'lowfreq')])[0])
    ok = ok and max_abs(vids['x_GP'], G.global_pixel_fix(
        decode([z])[0], dec1, dec2, a)) < 1e-6
    ok = ok and max_abs(vids['x_GP'], G.global_pixel_fix(
        vids['x_Gms'], dec1, dec2, a)) > 1e-3
    ok = ok and len(res['rt']) == frames and sorted(target) == list(range(frames))
    u1, u2 = L_P_uint8(G, dec1), L_P_uint8(G, dec2)
    t5 = (1 - 5 / 12) * G.P.frame_stats(u1[5])['luma'] + 5 / 12 * G.P.frame_stats(
        u2[5])['luma']
    ok = ok and abs(target[5] - t5) < 1e-6
    s_rt, s_gms = G.summary(res['rt'], target), G.summary(res['Gms'], target)
    ok = ok and s_gms['off_target'] < s_rt['off_target']
    text = G.report('x', 'fnr', 'p.pt', res, target)
    ok = ok and all(text.count(f'  {n:10s} ') == 6 for n in names)
    ok = ok and 'target' in text and 'summary' in text
    try:
        G.run_source('y', 'fnr', z, (img1[:, :32, :32], img2[:, :32, :32]),
                     vae.encode, vae.decode)
        ok = False
    except ValueError as error:
        ok = ok and 'the keyframes give latents' in str(error)
    name, spec = G.parse_add('q=fnr:r11_outputs/x.pt')
    ok = ok and name == 'q' and spec == dict(scene='fnr', paths=['r11_outputs/x.pt'])
    try:
        G.parse_add('q=moon:x.pt')
        ok = False
    except ValueError:
        pass
    check("round 12a: run_source decodes rt, Lms and the 4 whole-video "
          "variants, saves only the new ones, measures every frame against the "
          "stills' target; the report has every variant; a size mismatch is "
          "refused", ok)
    # main(): fake Wan modules, one latent source and one missing.
    import types
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'r11_outputs'))
        torch.save(dict(stream1=z.cpu(), stream2=(z * 0.5).cpu(), plan=None,
                        anchor=a), os.path.join(tmp, 'r11_outputs', 'lat.pt'))
        torch.save(dict(stream1=(z * 0.2).cpu(), stream2=None, plan=None,
                        anchor=a), os.path.join(tmp, 'r11_outputs', 'later.pt'))
        for i, im in enumerate(tiny_images(2)):
            im.save(os.path.join(tmp, f'i{i}.png'))
        fake_cfg = types.ModuleType('wan.configs')
        fake_cfg.WAN_CONFIGS = {'i2v-A14B': types.SimpleNamespace(
            vae_checkpoint='vae.pth')}
        fake_vae = types.ModuleType('wan.modules.vae2_1')
        fake_vae.Wan2_1_VAE = lambda vae_pth, device: FakeVAEDecoding(DEVICE)
        fake_utils = types.ModuleType('wan.utils.utils')
        written = []
        fake_utils.save_video = lambda tensor, save_file, **k: written.append(
            (save_file, tuple(tensor.shape)))
        keys = ('wan.configs', 'wan.modules.vae2_1', 'wan.utils.utils')
        saved_mods = {k: sys.modules.get(k) for k in keys}
        sys.modules.update({'wan.configs': fake_cfg,
                            'wan.modules.vae2_1': fake_vae,
                            'wan.utils.utils': fake_utils})
        G.SOURCES = {'f-lat': dict(scene='fnr', paths=['missing.pt',
                                                       'r11_outputs/lat.pt',
                                                       'r11_outputs/later.pt']),
                     'f-none': dict(scene='fnr', paths=['nothing.pt'])}
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            G.main(['--image_dir', tmp, '--ckpt_dir', tmp, '--gpu', '0'])
            out = os.path.join('r12_outputs', 'r12_gfix_f-lat')
            js = __import__('json').load(open(out + '.json'))
            ok = (os.path.isfile(out + '.txt') and os.path.isfile(out + '_sheet.png')
                  and js['path'] == 'r11_outputs/lat.pt' and js['stream'] == 'stream1'
                  and set(js['summary']) == set(names)
                  and sorted(os.path.basename(f) for f, _ in written) == sorted(
                      f'r12_gfix_f-lat_{n}.mp4' for n in ('Gm', 'Gms', 'Gl', 'GP'))
                  and all(sh == (1, 3, frames, 64, 64) for _, sh in written)
                  and not any('f-none' in f for f in os.listdir('r12_outputs')))
            rt1 = js['results']['rt']['3']['luma']
            G.main(['--image_dir', tmp, '--ckpt_dir', tmp, '--stream', 'stream2',
                    '--out_dir', 'other', '--no_videos', '--only', 'f-lat'])
            js2 = __import__('json').load(open(os.path.join(
                'other', 'r12_gfix_f-lat.json')))
            ok = ok and js2['stream'] == 'stream2' and js2['results']['rt']['3'][
                'luma'] != rt1 and len(written) == 4
        finally:
            os.chdir(cwd)
            for k, v in saved_mods.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
    check("round 12a: main() takes the first existing latent file, writes the "
          "4 new videos, the table, the JSON and the sheet; --stream stream2, "
          "--out_dir and --no_videos work; a source with no file is skipped", ok)


def L_P_uint8(G, video):
    return G.P.to_uint8(video)


def test_sequence_parallel():
    import torch.distributed as dist
    from wan.distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
    world, rank = dist.get_world_size(), dist.get_rank()
    model = tiny_model(num_heads=4, dim=512).to(DEVICE).to(torch.bfloat16)
    reference = copy.deepcopy(model)
    for block in model.blocks:
        block.self_attn.forward = types.MethodType(sp_attn_forward,
                                                   block.self_attn)
    model.forward = types.MethodType(sp_dit_forward, model)
    inp = tiny_inputs(DEVICE)
    seq_len = int(math.ceil(inp['tokens'] / world)) * world
    for name, variant in [('baseline', None)] + VARIANTS + STREAM_VARIANTS:
        out_sp = model_forward(model, inp, variant, seq_len=seq_len)
        out_ref = model_forward(reference, inp, variant, seq_len=seq_len)
        err = rel_err(out_sp, out_ref)
        check(f"Ulysses path equals single-GPU path ({name})", err < 3e-2,
              f"rel err {err:.2e}")
    inp_r = tiny_inputs(DEVICE, rev=True)
    for name, variant in REV_VARIANTS + REV_COUPLED:
        out_sp = model_forward(model, inp_r, variant, seq_len=seq_len)
        out_ref = model_forward(reference, inp_r, variant, seq_len=seq_len)
        err = rel_err(out_sp, out_ref)
        check(f"Ulysses path equals single-GPU path ({name}, 3 streams)",
              err < 3e-2, f"rel err {err:.2e}")
    ov_inp = tiny_override_inputs(DEVICE)
    ov_len = int(math.ceil(ov_inp['tokens'] / world)) * world
    for name, weights, blend, mode in (
            ('full', [1.0] * 7, 'linear', 'blend'),
            ('damped linear', [1.0, 1.0, 0.67, 0.25, 0.67, 1.0, 1.0], 'linear',
             'blend'),
            ('damped norm', [1.0, 1.0, 0.67, 0.25, 0.67, 1.0, 1.0], 'norm',
             'blend'),
            ('route', [1.0] * 7, 'linear', 'route')):
        out_sp = override_forward(model, ov_inp, weights, blend, ov_len, mode)
        out_ref = override_forward(reference, ov_inp, weights, blend, ov_len,
                                   mode)
        err = rel_err(out_sp, out_ref)
        check(f"Ulysses path equals single-GPU path (override {name})",
              err < 3e-2, f"rel err {err:.2e}")
    gen = torch.Generator(device='cpu')
    gen.manual_seed(11)
    ctx_b = torch.randn(4, 64, generator=gen).to(DEVICE)
    for start in (2, 4):
        relay = dict(context=[ctx_b, ctx_b], from_frame=start)
        out_sp = relay_forward(model, ov_inp, relay, seq_len=ov_len)[0]
        out_ref = relay_forward(reference, ov_inp, relay, seq_len=ov_len)[0]
        err = rel_err(out_sp, out_ref)
        check(f"Ulysses path equals single-GPU path (prompt relay from "
              f"latent {start})", err < 3e-2, f"rel err {err:.2e}")
    # Round 7: a plain batch-2 pass with the relay prompt (the reach test's
    # nudge pair), no override.
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
        outs = [net(ov_inp['x'], t=ov_inp['t'], context=ov_inp['context'],
                    seq_len=ov_len, y=ov_inp['y'],
                    relay=dict(context=[ctx_b, ctx_b], from_frame=4))
                for net in (model, reference)]
    err = max(rel_err(outs[0][i], outs[1][i]) for i in range(2))
    check("Ulysses path equals single-GPU path (plain batch-2 pass with "
          "prompt relay, round 7)", err < 3e-2, f"rel err {err:.2e}")
    # Round 5: Stream B as a third, shorter override stream.
    gen = torch.Generator(device='cpu')
    gen.manual_seed(12)
    x_b = ov_inp['x'][0][:, 3:]
    y_b = torch.randn(20, x_b.size(1), x_b.size(2), x_b.size(3),
                      generator=gen).to(DEVICE)
    inp_b = dict(ov_inp, x=ov_inp['x'] + [x_b], y=ov_inp['y'] + [y_b],
                 context=ov_inp['context'] + [ctx_b],
                 t=torch.tensor([900.0, 900.0, 900.0], device=DEVICE))
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
        outs = []
        for net in (model, reference):
            outs.append(net(inp_b['x'], t=inp_b['t'],
                            context=inp_b['context'], seq_len=ov_len,
                            y=inp_b['y'],
                            override=dict(weights=[1.0] * 7, blend='linear',
                                          anchor=3,
                                          stream_b=dict(weight=0.5)),
                            relay=dict(context=[ctx_b] * 3,
                                       from_frame=4))[0])
    err = rel_err(outs[0], outs[1])
    check("Ulysses path equals single-GPU path (override + Stream B + "
          "prompt relay, 3 streams)", err < 3e-2, f"rel err {err:.2e}")
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
        outs = []
        for net in (model, reference):
            outs.append(net(inp_b['x'], t=inp_b['t'],
                            context=inp_b['context'], seq_len=ov_len,
                            y=inp_b['y'],
                            override=dict(weights=[1.0] * 7, blend='linear',
                                          anchor=3, stream_b=dict(weight=0.0),
                                          img2_value='before'),
                            relay=dict(context=[ctx_b] * 3,
                                       from_frame=4))[0])
    err = rel_err(outs[0], outs[1])
    check("Ulysses path equals single-GPU path (clean Img2 values from "
          "Stream B, round 13)", err < 3e-2, f"rel err {err:.2e}")
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
        outs = []
        for net in (model, reference):
            outs.append(net(inp_b['x'][1:], t=inp_b['t'][1:],
                            context=inp_b['context'][1:], seq_len=ov_len,
                            y=inp_b['y'][1:],
                            override=dict(single=True, anchor=3,
                                          img2_value='before'),
                            relay=dict(context=[ctx_b] * 2,
                                       from_frame=4))[0])
    err = rel_err(outs[0], outs[1])
    check("Ulysses path equals single-GPU path (single stream + Stream B "
          "values, round 14)", err < 3e-2, f"rel err {err:.2e}")
    # Stream R prediction mode: a plain I2V pass over the anchor latents.
    plain_x = [inp_r['x'][2]]
    plain_len = int(math.ceil(
        inp_r['anchor'] * 6 / world)) * world  # 6 tokens per 4x6 frame
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
        out_sp = model(plain_x, t=inp_r['t'][:1], context=[inp_r['ctx']],
                       seq_len=plain_len, y=[inp_r['y'][2]])[0]
        out_ref = reference(plain_x, t=inp_r['t'][:1], context=[inp_r['ctx']],
                            seq_len=plain_len, y=[inp_r['y'][2]])[0]
    err = rel_err(out_sp, out_ref)
    check("Ulysses path equals single-GPU path (Stream R plain pass)",
          err < 3e-2, f"rel err {err:.2e}")
    A.ATTN_PROBE.configure([0], samples=4)
    A.ATTN_PROBE.begin_forward(0, 'cond')
    model_forward(model, inp, dict(anchor_key='B', mirror=True),
                  seq_len=seq_len)
    path = os.path.join(tempfile.gettempdir(), 'vr_r1_sp_probe_test.pt')
    A.ATTN_PROBE.finalize(path, extra_meta=dict(timesteps=[999],
                                                boundary_timestep=900.0))
    if rank == 0:
        merged = torch.load(path, map_location='cpu', weights_only=False)
        shape = tuple(merged['tensors']['prefix_eff'].shape)
        sums = merged['tensors']['prefix_eff'].sum(3)
        check("Ulysses probe merge has all heads and normalised masses",
              shape[-1] == 4 and shape[1] == 2 and
              max_abs(sums, torch.ones_like(sums)) < 1e-3,
              f"shape {shape}")
        os.remove(path)
        summary = os.path.splitext(path)[0] + '_summary.txt'
        if os.path.exists(summary):
            os.remove(summary)


def main():
    global A, DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument('--sp', action='store_true')
    args = parser.parse_args()
    cuda = torch.cuda.is_available()
    if cuda:
        # Exact float32 references: never let TF32 round the test matmuls.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.sp:
        import torch.distributed as dist
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group('nccl')
        DEVICE = torch.device(f'cuda:{local_rank}')
    else:
        DEVICE = torch.device('cuda' if cuda else 'cpu')
    A = import_attention()
    if not cuda:
        A.flash_attention = naive_flash_attention
        A.FLASH_ATTN_2_AVAILABLE = True
        print("[INFO] no CUDA: FlashAttention replaced by a naive kernel")
    try:
        if args.sp:
            A._LSE_BACKEND = os.environ.get('VR_LSE_BACKEND', 'flash')
            test_sequence_parallel()
        else:
            backend = test_flash_lse(DEVICE) if cuda else 'flash'
            A._LSE_BACKEND = backend
            test_baseline_identity()
            test_probe_read_only()
            test_variants_exact()
            test_variants_flash(backend)
            test_probe_paired()
            test_probe_plain()
            test_suffix_time_fix()
            test_reverse_stream()
            test_override_attention()
            test_override_stream_b()
            test_override_route()
            test_round4_helpers()
            test_round6_helpers()
            test_round7_helpers()
            test_round8_probe()
            test_round9_lab()
            test_round12_global_fix()
            if cuda or os.environ.get('VR_TEST_MODEL_ON_CPU'):
                test_tiny_model()
                test_relay_model()
                test_route_model()
                test_tiny_generate()
            print(f"LSE_BACKEND={backend}")
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        FAILURES.append('exception')
    if args.sp:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()
    print(f"{len(FAILURES)} failed check(s)" if FAILURES else
          "ALL CHECKS PASSED", flush=True)
    sys.exit(1 if FAILURES else 0)


DEVICE = None

if __name__ == '__main__':
    main()
