# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import os

import torch

from .attn_probe import ATTN_PROBE

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'paired_anchor_attention',
    'attention',
    'ATTN_PROBE',
    'VARIANT_DEFAULTS',
    'probe_plain_self_attention',
    'reverse_stream_weight',
    'override_attention',
    'override_frame_weights',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
    return_lse=False,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    return_lse:     bool. Also return per-query, per-head softmax log-sum-exp.
                    This does not materialize the attention map.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if return_lse and dropout_p != 0:
        raise ValueError(
            "return_lse is supported only with zero attention dropout")

    softmax_lse = None
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        result = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)
        x = result[0].unflatten(0, (b, lq))
        if return_lse:
            softmax_lse = result[1]
    else:
        assert FLASH_ATTN_2_AVAILABLE
        result = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            return_attn_probs=return_lse)
        if return_lse:
            x, softmax_lse, _ = result
        else:
            x = result
        x = x.unflatten(0, (b, lq))

    # output
    x = x.type(out_dtype)
    if not return_lse:
        return x

    # FlashAttention versions expose LSE as either [H, B*Lq] or [B, H, Lq].
    # Normalize it to [B, Lq, H] so it broadcasts over value features.
    if softmax_lse.ndim == 2:
        if softmax_lse.size(1) != b * lq:
            raise RuntimeError("unexpected FlashAttention LSE shape")
        softmax_lse = softmax_lse.unflatten(1, (b, lq)).permute(1, 2, 0)
    elif softmax_lse.ndim == 3:
        if softmax_lse.size(0) != b:
            raise RuntimeError("unexpected FlashAttention LSE batch shape")
        softmax_lse = softmax_lse.permute(0, 2, 1)
    else:
        raise RuntimeError("unexpected FlashAttention LSE rank")
    return x, softmax_lse.float()


def paired_anchor_attention(
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
    variant=None,
    probe_head_offset=0,
    probe_sp=False,
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

    Visual Relay round 1: ``variant`` (a dict, see ``VARIANT_DEFAULTS``)
    changes only how PREFIX queries see the anchor while the anchor route is
    active. ``variant=None`` runs the unchanged baseline code below.

    Visual Relay round 2, Stream R: an optional third sample is an ordinary
    Img2-first I2V stream that runs BACKWARD over the prefix: local frame 0 is
    the anchor slot (Img2 conditioning) and local frame j is main frame a - j
    (j = 1..a-1). It attends only to itself, like Stream B. Prefix query
    frame t (1..a-1) then gets (1 - w_t) * main + w_t * (Stream R frame a - t)
    with w_t = rev_strength * f(t / a), f linear or cosine: the mirror image
    of how Stream B is fused into the suffix. With rev_couple (round 3),
    Stream R frame a - t takes that same mixed output, so the two streams
    share one trajectory.
    """
    if window_size != (-1, -1):
        raise ValueError("paired anchor attention requires global attention")
    if freqs is None:
        raise ValueError("paired anchor attention requires RoPE frequencies")
    n_streams = q.size(0)
    if (n_streams not in (2, 3) or len(seq_lens) != n_streams
            or len(grid_sizes) != n_streams):
        raise ValueError(
            "paired anchor attention requires main and suffix streams "
            "(plus an optional reverse Stream R)")
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

    rev_seq_len = 0
    if n_streams == 3:
        rev_frames, rev_height, rev_width = (
            int(value) for value in grid_sizes[2].tolist())
        if (rev_frames != anchor_index
                or (rev_height, rev_width) != (main_height, main_width)):
            raise ValueError(
                "Stream R must cover the anchor plus prefix frames 1..a-1")
        rev_seq_len = rev_frames * tokens_per_frame
        if rev_seq_len != int(seq_lens[2]):
            raise ValueError("Stream R grid and sequence length disagree")

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
        shifted_q = _shift_temporal_rope(
            main_q[:, frame_start:frame_end], -1, freqs)
        main_q[:, frame_start:frame_end].copy_(shifted_q)
        del shifted_q
        shifted_k = _shift_temporal_rope(
            main_k[:, frame_start:frame_end], -1, freqs)
        main_k[:, frame_start:frame_end].copy_(shifted_k)
        del shifted_k

    if variant is not None and _variant_settings(variant)['suffix_time_fix']:
        # Ablation of Gordon's offset (not a fix): B1 shares Img2's position.
        for frame_index in range(1, suffix_frames):
            frame_start = frame_index * tokens_per_frame
            frame_end = frame_start + tokens_per_frame
            shifted_q = _shift_temporal_rope(
                suffix_q[:, frame_start:frame_end], -1, freqs)
            suffix_q[:, frame_start:frame_end].copy_(shifted_q)
            del shifted_q
            shifted_k = _shift_temporal_rope(
                suffix_k[:, frame_start:frame_end], -1, freqs)
            suffix_k[:, frame_start:frame_end].copy_(shifted_k)
            del shifted_k

    # The Img2-first suffix remains an independent reference stream.
    suffix_output = flash_attention(
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

    variant_anchor_k = None
    mirror_target = None
    if (prefix_anchor_active and variant is not None
            and _prefix_variant_active(_variant_settings(variant))):
        # Visual Relay round 1: decomposed prefix route (see helpers below).
        settings = _variant_settings(variant)
        variant_anchor_k, variant_anchor_v = _variant_anchor_kv(
            main_k, suffix_k, suffix_v, anchor_start, anchor_end,
            tokens_per_frame, anchor_index, freqs, settings)
        prefix_output, mirror_target = _prefix_variant_attention(
            main_q[:, :anchor_start], main_k, main_v, variant_anchor_k,
            variant_anchor_v, anchor_index, tokens_per_frame, freqs,
            settings, dtype)
        del variant_anchor_v
    else:
        # Prefix queries always retain all ordinary Stream-A context on both
        # sides of the temporary anchor. Before the route switch, also include
        # a hybrid anchor: the joint-conditioned, history-aware Stream-A anchor
        # supplies K, while the fresh-cache Stream-B anchor supplies V. Both
        # are already aligned to the anchor for this prefix attention
        # computation. After the switch, remove only this hybrid anchor route.
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
        prefix_output = flash_attention(
            q=main_q[:, :anchor_start],
            k=prefix_k,
            v=prefix_v,
            window_size=window_size,
            dtype=dtype)

    rev_output = None
    if n_streams == 3:
        # Stream R: ordinary self-attention of the reverse Img2-first stream,
        # then fused into the prefix with location-proportional weights.
        rev_output = flash_attention(
            q=q[2:3, :rev_seq_len],
            k=k[2:3, :rev_seq_len],
            v=v[2:3, :rev_seq_len],
            window_size=window_size,
            dtype=dtype)
        prefix_output = _mix_reverse_stream(
            prefix_output, rev_output, anchor_index, tokens_per_frame,
            _variant_settings(variant))

    # Evolve the joint-conditioned Stream-A anchor with the prefix and itself,
    # but never with the post-anchor suffix. Stream B evolves its independent
    # fresh-cache Img2 anchor through the Img2-first suffix branch.
    messenger_output = flash_attention(
        q=main_q[:, anchor_start:anchor_end],
        k=main_k[:, :anchor_end],
        v=main_v[:, :anchor_end],
        window_size=window_size,
        dtype=dtype)

    main_suffix_q = main_q[:, anchor_end:]
    img2_suffix_output = suffix_output[:, tokens_per_frame:suffix_seq_len]
    if main_suffix_q.size(1) != img2_suffix_output.size(1):
        raise ValueError("paired suffix query lengths do not align")

    ordinary_output = flash_attention(
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

    if ATTN_PROBE.record:
        # Read-only measurement; never changes the outputs above.
        probe_layer = ATTN_PROBE.next_layer()
        ATTN_PROBE.run(
            _probe_paired,
            probe_layer,
            main_q=main_q,
            main_k=main_k,
            anchor_k=(variant_anchor_k if variant_anchor_k is not None else
                      main_k[:, anchor_start:anchor_end]),
            suffix_q=suffix_q,
            suffix_k=suffix_k,
            anchor_index=anchor_index,
            tokens_per_frame=tokens_per_frame,
            main_frames=main_frames,
            suffix_frames=suffix_frames,
            prefix_anchor_active=prefix_anchor_active,
            settings=_variant_settings(
                variant if prefix_anchor_active else None),
            mirror_target=mirror_target,
            freqs=freqs,
            head_offset=probe_head_offset,
            sp=probe_sp)
    if rev_output is None:
        return torch.cat([main_output, suffix_output], dim=0)
    if rev_seq_len < padded_len:
        rev_output = torch.cat([
            rev_output,
            rev_output.new_zeros(
                1, padded_len - rev_seq_len, *rev_output.shape[2:])
        ], dim=1)
    return torch.cat([main_output, suffix_output, rev_output], dim=0)


def reverse_stream_weight(frame, anchor_index, strength=1.0, shape='linear'):
    """Stream R weight w_t of prefix frame t (0 at Img1, -> strength at Img2)."""
    x = float(frame) / float(anchor_index)
    if shape == 'cosine':
        x = 0.5 * (1.0 - math.cos(math.pi * x))
    return float(strength) * x


def _mix_reverse_stream(prefix_output, rev_output, anchor_index,
                        tokens_per_frame, settings):
    """Prefix frame t <- (1 - w_t) main + w_t Stream-R frame a - t, t=1..a-1.

    With settings['rev_couple'] the Stream-R frame a - t is set to the same
    mixed value (in place in ``rev_output``); Stream R's anchor frame 0 is
    never changed.
    """
    frame = tokens_per_frame
    for t in range(1, anchor_index):
        weight = reverse_stream_weight(t, anchor_index,
                                       settings['rev_strength'],
                                       settings['rev_shape'])
        if weight == 0.0:
            continue
        rows = slice(t * frame, (t + 1) * frame)
        rev_rows = slice((anchor_index - t) * frame,
                         (anchor_index - t + 1) * frame)
        mixed = ((1.0 - weight) * prefix_output[:, rows].float()
                 + weight * rev_output[:, rev_rows].float())
        prefix_output[:, rows] = mixed.to(prefix_output.dtype)
        if settings['rev_couple']:
            # Coupled: Stream R's own token takes the same mixed output.
            rev_output[:, rev_rows] = mixed.to(rev_output.dtype)
        del mixed
    return prefix_output


# ---------------------------------------------------------------------------
# Visual Relay round 3: Gordon's Method 1, self-attention output override.
# ---------------------------------------------------------------------------
# The batch holds two streams of the same length that share the noisy latent
# and the prompt: stream 1 is plain I2V (Img1 + zeros, mask on Img1 only) and
# stream 2 also has Img2 at latent a (mask on Img1 and Img2). Each stream
# attends only to itself. Stream 1 then replaces its own self-attention output
# O1 by a blend with stream 2's O2, frame by frame:
#     linear: O = O1 + w_f (O2 - O1)
#     norm:   same direction, but each token's per-head length is set to
#             (1 - w_f)|O1| + w_f|O2| (a straight blend of two vectors that
#             point different ways is shorter than either)
# w_f = 1 everywhere is Gordon's full override. Stream 2 keeps its own O2.
# Cross-attention and the FFN of stream 1 run on its own hidden state.
def override_frame_weights(frames, anchor, min_weight=1.0, radius=3,
                           profile='damp'):
    """w_f per latent frame.
    'damp' (round 3, Task 1): min_weight at the Img2 latent a, rising
    linearly to 1 at distance ``radius`` latents and beyond.
    'local' (round 5, damping the other way): 1 at a, falling linearly to 0
    at distance ``radius``, so stream 1 keeps its own attention far away."""
    if profile not in ('damp', 'local'):
        raise ValueError("override profile must be 'damp' or 'local'")
    weights = []
    for frame in range(int(frames)):
        distance = abs(frame - int(anchor))
        if profile == 'local':
            if radius <= 0:
                weights.append(1.0 if distance == 0 else 0.0)
            else:
                weights.append(max(0.0, 1.0 - distance / float(radius)))
            continue
        if radius <= 0:
            ramp = 0.0 if distance == 0 else 1.0
        else:
            ramp = min(distance, radius) / float(radius)
        weights.append(float(min_weight) + (1.0 - float(min_weight)) * ramp)
    return weights


def override_attention(q, k, v, seq_lens, grid_sizes, override,
                       window_size=(-1, -1), dtype=torch.bfloat16,
                       probe_head_offset=0, probe_sp=False):
    """Method 1. ``override`` = dict(weights=[w_f per latent frame],
    blend='linear'|'norm', anchor=a, stream_b=None|dict(weight=lam),
    mode='blend'|'route'). Returns every stream's output, stream 1 replaced
    by the blend ('route', round 6: by attn(q2, k2, v1), O2 at latent a).

    Round 5, optional third sequence (Stream B): an ordinary Img2-first I2V
    over latents a..F-1, whose frame j is latent a + j. After the blend,
    stream 1's latents after Img2 become base + lam (O_B - base), as in the
    paired design's suffix.

    Round 13, ``img2_value`` 'before' | 'all' (Gordon's K_A12 / V_B0): in
    stream 2, the queries of latents 0..a-1 ('before'), or of every latent
    but a ('all'), attend with stream 2's keys as usual, but the values of
    Img2's latent a are Stream B's frame 0 (Img2 encoded as a first frame,
    no earlier frames in its VAE pass). Latent a's own queries keep stream
    2's values. Needs Stream B (weight 0 if it only supplies the values)."""
    if override.get('single'):
        return _single_stream_clean_values(q, k, v, seq_lens, grid_sizes,
                                           override, window_size, dtype)
    stream_b = override.get('stream_b')
    streams = q.size(0)
    if streams not in (2, 3) or len(seq_lens) != streams or len(
            grid_sizes) != streams:
        raise ValueError("override attention takes two streams, or three "
                         "with Stream B")
    if (streams == 3) != (stream_b is not None):
        raise ValueError("a third override stream must be Stream B")
    if not torch.equal(grid_sizes[0], grid_sizes[1]):
        raise ValueError("override streams must have identical grids")
    frames, height, width = (int(value) for value in grid_sizes[0].tolist())
    tokens_per_frame = height * width
    seq_len = frames * tokens_per_frame
    if int(seq_lens[0]) != seq_len or int(seq_lens[1]) != seq_len:
        raise ValueError("override stream grid and sequence lengths disagree")
    if stream_b is not None:
        anchor_b = int(override['anchor'])
        frames_b = frames - anchor_b
        if (grid_sizes[2].tolist() != [frames_b, height, width]
                or int(seq_lens[2]) != frames_b * tokens_per_frame):
            raise ValueError("Stream B must cover latents a..F-1 on the same "
                             "spatial grid")
    weights = list(override['weights'])
    if len(weights) != frames:
        raise ValueError("override needs one weight per latent frame")
    blend = override.get('blend', 'linear')
    if blend not in ('linear', 'norm'):
        raise ValueError("override blend must be 'linear' or 'norm'")
    mode = override.get('mode', 'blend')
    if mode not in ('blend', 'route'):
        raise ValueError("override mode must be 'blend' or 'route'")
    if mode == 'route':
        # Round 6, Direction C: stream 1 keeps its own values but attends
        # with stream 2's queries and keys (stream 2 decides where each token
        # looks, stream 1 what it sees there): O_route = attn(q2, k2, v1).
        # Img2's latent a takes O2, the only way Img2 reaches stream 1.
        # One call of the usual two-stream size: row 0 = (q2, k2, v1),
        # row 1 = (q2, k2, v2) = O2. O1 itself is never computed.
        if stream_b is not None:
            raise ValueError("route mode does not take Stream B")
        if any(float(weight) != 1.0 for weight in weights):
            raise ValueError("route mode needs w = 1 on every latent")
        anchor = int(override['anchor'])
        if not 0 <= anchor < frames:
            raise ValueError("route mode needs Img2's latent inside the video")
        out = flash_attention(
            q=q[1:2].expand(2, -1, -1, -1), k=k[1:2].expand(2, -1, -1, -1),
            v=v, k_lens=seq_lens, window_size=window_size, dtype=dtype)
        if ATTN_PROBE.record:
            # 'o1' of this probe is O_route (stream 1's routed output).
            probe_layer = ATTN_PROBE.next_layer()
            ATTN_PROBE.run(_probe_override, probe_layer, out[0:1, :seq_len],
                           out[1:2, :seq_len], frames, tokens_per_frame,
                           anchor, probe_head_offset, probe_sp)
        rows = slice(anchor * tokens_per_frame,
                     (anchor + 1) * tokens_per_frame)
        out[0:1, rows] = out[1:2, rows]
        return out
    out = flash_attention(
        q=q, k=k, v=v, k_lens=seq_lens, window_size=window_size, dtype=dtype)
    img2_value = override.get('img2_value')
    if img2_value is not None:
        if stream_b is None:
            raise ValueError(
                "clean Img2 values need Stream B (its frame 0 gives them)")
        _clean_img2_values(out, q, k, v, 1, 2, seq_lens, frames,
                           tokens_per_frame, int(override['anchor']),
                           img2_value, window_size, dtype)
    if ATTN_PROBE.record:
        probe_layer = ATTN_PROBE.next_layer()
        ATTN_PROBE.run(_probe_override, probe_layer, out[0:1, :seq_len],
                       out[1:2, :seq_len], frames, tokens_per_frame,
                       int(override.get('anchor', -1)), probe_head_offset,
                       probe_sp)
    for frame, weight in enumerate(weights):
        weight = float(weight)
        if weight == 0.0:
            continue
        rows = slice(frame * tokens_per_frame, (frame + 1) * tokens_per_frame)
        if weight == 1.0:
            out[0:1, rows] = out[1:2, rows]
            continue
        own = out[0:1, rows].float()
        other = out[1:2, rows].float()
        mixed = own + weight * (other - own)
        if blend == 'norm':
            target = ((1.0 - weight) * own.norm(dim=-1, keepdim=True)
                      + weight * other.norm(dim=-1, keepdim=True))
            mixed = mixed * (target / mixed.norm(
                dim=-1, keepdim=True).clamp_min(1e-6))
        out[0:1, rows] = mixed.to(out.dtype)
        del own, other, mixed
    if stream_b is not None:
        lam = float(stream_b['weight'])
        for frame in range(anchor_b + 1, frames if lam != 0.0 else 0):
            rows = slice(frame * tokens_per_frame,
                         (frame + 1) * tokens_per_frame)
            rows_b = slice((frame - anchor_b) * tokens_per_frame,
                           (frame - anchor_b + 1) * tokens_per_frame)
            base = out[0:1, rows].float()
            mixed = base + lam * (out[2:3, rows_b].float() - base)
            out[0:1, rows] = mixed.to(out.dtype)
            del base, mixed
    return out



def _clean_img2_values(out, q, k, v, main, b, seq_lens, frames,
                       tokens_per_frame, anchor, scope, window_size, dtype):
    """Round 13, Gordon's K_A12 / V_B0, in place on ``out``: the queries of
    stream ``main`` on latents 0..a-1 ('before') or on every latent but a
    ('all') attend with that stream's keys, but the values of Img2's latent a
    are Stream B's (row ``b``) frame 0. Latent a's own queries are unchanged."""
    if scope not in ('before', 'all'):
        raise ValueError("override Img2 values must be 'before' or 'all'")
    if not 0 < anchor < frames:
        raise ValueError("clean Img2 values need Img2 inside the video")
    seq_len = frames * tokens_per_frame
    img2_rows = slice(anchor * tokens_per_frame,
                      (anchor + 1) * tokens_per_frame)
    m = slice(main, main + 1)
    v_clean = v[m].clone()
    v_clean[:, img2_rows] = v[b:b + 1, :tokens_per_frame]
    if scope == 'before':
        rows_q = slice(0, anchor * tokens_per_frame)
        out[m, rows_q] = flash_attention(
            q=q[m, rows_q], k=k[m], v=v_clean, k_lens=seq_lens[m],
            window_size=window_size, dtype=dtype)
    else:
        own_img2 = out[m, img2_rows].clone()
        out[m, :seq_len] = flash_attention(
            q=q[m, :seq_len], k=k[m], v=v_clean, k_lens=seq_lens[m],
            window_size=window_size, dtype=dtype)
        out[m, img2_rows] = own_img2
    return out


def _single_stream_clean_values(q, k, v, seq_lens, grid_sizes, override,
                                window_size, dtype):
    """Round 14: one video stream plus Stream B (Img2-first I2V over latents
    a..F-1, only for its frame-0 values): ordinary attention for both, then
    Gordon's clean Img2 values for the video stream (row 0)."""
    if q.size(0) != 2 or len(seq_lens) != 2 or len(grid_sizes) != 2:
        raise ValueError("single-stream override takes the video stream and "
                         "Stream B")
    frames, height, width = (int(value) for value in grid_sizes[0].tolist())
    tokens_per_frame = height * width
    anchor = int(override['anchor'])
    if (grid_sizes[1].tolist() != [frames - anchor, height, width]
            or int(seq_lens[0]) != frames * tokens_per_frame
            or int(seq_lens[1]) != (frames - anchor) * tokens_per_frame):
        raise ValueError("Stream B must cover latents a..F-1 on the same "
                         "spatial grid")
    out = flash_attention(
        q=q, k=k, v=v, k_lens=seq_lens, window_size=window_size, dtype=dtype)
    return _clean_img2_values(out, q, k, v, 0, 1, seq_lens, frames,
                              tokens_per_frame, anchor,
                              override.get('img2_value'), window_size, dtype)


def _probe_override(layer, own, other, frames, tokens_per_frame, anchor,
                    head_offset, sp):
    """Per latent frame and head: RMS of O1, O2 and O1 - O2, mean cosine."""
    heads = own.size(2)
    stats = {name: torch.zeros(frames, heads, device=own.device)
             for name in ('o1_rms', 'o2_rms', 'diff_rms', 'cos')}
    for frame in range(frames):
        rows = slice(frame * tokens_per_frame, (frame + 1) * tokens_per_frame)
        a = own[0, rows].float()  # [T, H, D]
        b = other[0, rows].float()
        stats['o1_rms'][frame] = a.pow(2).mean(dim=(0, 2)).sqrt()
        stats['o2_rms'][frame] = b.pow(2).mean(dim=(0, 2)).sqrt()
        stats['diff_rms'][frame] = (a - b).pow(2).mean(dim=(0, 2)).sqrt()
        stats['cos'][frame] = torch.nn.functional.cosine_similarity(
            a, b, dim=-1).mean(dim=0)
    ATTN_PROBE.store(layer, stats, head_offset=head_offset, sp=sp,
                     meta=dict(mode='override', main_frames=frames,
                               anchor_index=anchor,
                               tokens_per_frame=tokens_per_frame))


@torch.amp.autocast('cuda', enabled=False)
def _shift_temporal_rope(x, position_delta, freqs):
    """Shift only temporal RoPE on an already-rotated token tensor."""
    if position_delta == 0:
        return x
    original_dtype = x.dtype
    complex_dim = x.size(-1) // 2
    temporal_dim = complex_dim - 2 * (complex_dim // 3)
    x_complex = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], complex_dim, 2))
    base_phase = torch.angle(freqs[1, :temporal_dim]).to(
        device=x.device, dtype=torch.float32)
    multiplier = torch.polar(
        torch.ones_like(base_phase),
        float(position_delta) * base_phase,
    ).to(x_complex.dtype).view(*([1] * (x.ndim - 1)), temporal_dim)
    shifted_temporal = x_complex[..., :temporal_dim] * multiplier
    shifted = torch.cat(
        [shifted_temporal, x_complex[..., temporal_dim:]], dim=-1)
    return torch.view_as_real(shifted).flatten(-2).to(original_dtype)


# ---------------------------------------------------------------------------
# Visual Relay round 1: variants of the prefix -> Img2 anchor route.
# ---------------------------------------------------------------------------
# Prefix queries (main-stream latent frames 0..a-1, a = anchor index) attend
# three disjoint key groups: Img1 (frame 0), the anchor (one frame of Img2
# tokens) and all other ordinary frames. Each group is attended separately
# with its log-sum-exp, then the groups are merged with explicit weights. With
# the natural weights this is exactly one softmax over all keys.
#
#   anchor_key  'A': history-aware Stream-A anchor key K_A (baseline).
#               'B': fresh-cache Stream-B frame-0 key K_B0 moved to RoPE
#                    position a, so the anchor's K and V come from one token.
#   mirror      True: for prefix frame t the anchor key sits at RoPE position
#               2t - a, so the query sees Img2 at temporal offset -(a - t):
#               on its LEFT like a first frame, at its true distance.
#   mass        'natural': one softmax, no reweighting.
#               'split': keep each query's natural image mass
#                   m = m(Img1) + m(anchor), give lambda(t) * m to the anchor
#                   and (1 - lambda(t)) * m to Img1 (lambda = t/a or cosine).
#               'mirror_mass': anchor mass of frame t = mean Img1 mass of frame
#                   a - t in the same head and layer (anchor excluded), times
#                   mirror_gain, capped at mass_cap; Img1 and the other frames
#                   share the rest in their natural ratio.
#               'img1_fade': no extra Img2 pull; Img1's logits get the bias
#                   log(1 - lambda(t)), one softmax over all groups after that.
VARIANT_DEFAULTS = dict(
    anchor_key='A',
    mirror=False,
    mass='natural',
    split_shape='linear',
    mirror_gain=1.0,
    mass_cap=0.95,
    suffix_time_fix=False,
    rev_strength=1.0,
    rev_shape='linear',
    rev_couple=False,
)
# suffix_time_fix: ABLATION ONLY, not a fix. Gordon's design deliberately
# has the first suffix frame see Img2 one step to its left through Stream B
# (q=13 sees k={11, 12} at offset -1, comment above). This flag instead moves
# Stream B frames 1.. one RoPE unit left, so B1 shares Img2's position.
# rev_strength / rev_shape (round 2): Stream R fusion weights; they only
# matter when a third (Stream R) sample is in the batch.
# rev_couple (round 3): Stream R frame a - t ALSO receives the mixed output,
# so Stream R sees Img1's side through the main stream and both streams
# follow one trajectory (round 2's uncoupled Stream R never saw Img1).


def _prefix_variant_active(settings):
    return (settings['anchor_key'] != 'A' or settings['mirror']
            or settings['mass'] != 'natural')

# 'flash': group log-sum-exp from FlashAttention (default).
# 'torch': exact chunked PyTorch fallback (slower); set VR_LSE_BACKEND=torch.
_LSE_BACKEND = os.environ.get('VR_LSE_BACKEND', 'flash').strip().lower()


def _variant_settings(variant):
    settings = dict(VARIANT_DEFAULTS)
    if variant:
        unknown = set(variant) - set(VARIANT_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown paired variant keys: {sorted(unknown)}")
        settings.update(variant)
    if settings['anchor_key'] not in ('A', 'B'):
        raise ValueError("variant anchor_key must be 'A' or 'B'")
    if settings['mass'] not in ('natural', 'split', 'mirror_mass',
                                'img1_fade'):
        raise ValueError("variant mass must be 'natural', 'split', "
                         "'mirror_mass' or 'img1_fade'")
    if settings['split_shape'] not in ('linear', 'cosine'):
        raise ValueError("variant split_shape must be 'linear' or 'cosine'")
    if settings['rev_shape'] not in ('linear', 'cosine'):
        raise ValueError("variant rev_shape must be 'linear' or 'cosine'")
    settings['rev_strength'] = float(settings['rev_strength'])
    if not 0.0 <= settings['rev_strength'] <= 1.0:
        raise ValueError("variant rev_strength must be in [0, 1]")
    settings['mirror'] = bool(settings['mirror'])
    settings['suffix_time_fix'] = bool(settings['suffix_time_fix'])
    settings['rev_couple'] = bool(settings['rev_couple'])
    return settings


def _attention_lse(q, k, v, dtype=torch.bfloat16):
    """Attention over one key group. Returns (out [B,Lq,H,D], lse [B,Lq,H])."""
    if _LSE_BACKEND == 'torch' or not (FLASH_ATTN_2_AVAILABLE
                                       or FLASH_ATTN_3_AVAILABLE):
        return _torch_attention_lse(q, k, v)
    return flash_attention(q=q, k=k, v=v, dtype=dtype, return_lse=True)


def _torch_attention_lse(q, k, v, chunk=1024):
    """Exact softmax attention + natural-log LSE (scale 1/sqrt(D)), chunked."""
    batch, q_len, heads, head_dim = q.shape
    scale = head_dim**-0.5
    out = torch.empty(
        batch, q_len, heads, v.size(-1), dtype=q.dtype, device=q.device)
    lse = torch.empty(batch, q_len, heads, dtype=torch.float32,
                      device=q.device)
    for b in range(batch):
        for h in range(heads):
            k_h = k[b, :, h].float()
            v_h = v[b, :, h].float()
            for start in range(0, q_len, chunk):
                end = min(start + chunk, q_len)
                logits = (q[b, start:end, h].float() @ k_h.t()) * scale
                row_lse = torch.logsumexp(logits, dim=-1)
                weights = torch.exp(logits - row_lse[:, None])
                out[b, start:end, h] = (weights @ v_h).to(out.dtype)
                lse[b, start:end, h] = row_lse
                del logits, weights
    return out, lse


def _variant_anchor_kv(main_k, suffix_k, suffix_v, anchor_start, anchor_end,
                       tokens_per_frame, anchor_index, freqs, settings):
    """Anchor key at RoPE position a (K_A or shifted K_B0) and value V_B0."""
    if settings['anchor_key'] == 'B':
        anchor_k = _shift_temporal_rope(suffix_k[:, :tokens_per_frame],
                                        anchor_index, freqs)
    else:
        anchor_k = main_k[:, anchor_start:anchor_end]
    return anchor_k, suffix_v[:, :tokens_per_frame]


def _split_fraction(frame_idx, anchor_index, shape):
    x = frame_idx.float() / float(anchor_index)
    if shape == 'cosine':
        return 0.5 * (1.0 - torch.cos(math.pi * x))
    return x


def _mirror_mass_target(lse_img1, lse_rest, anchor_index, tokens_per_frame,
                        settings):
    """Per prefix frame t and head: mean Img1 mass of frame a-t (t >= 1)."""
    lse_img1 = lse_img1.float()
    lse_rest = lse_rest.float()
    img1_mass = torch.exp(lse_img1 - torch.logaddexp(lse_img1, lse_rest))
    batch, _, heads = img1_mass.shape
    frame_mean = img1_mass.reshape(batch, anchor_index, tokens_per_frame,
                                   heads).mean(dim=2)  # [B, a, H]
    target = torch.zeros_like(frame_mean)
    if anchor_index > 1:
        target[:, 1:] = frame_mean[:, 1:].flip(1)
    gain = float(settings['mirror_gain'])
    cap = float(settings['mass_cap'])
    return (gain * target).clamp(0.0, cap)


def _prefix_group_weights(lse_img1, lse_anchor, lse_rest, frame_idx,
                          anchor_index, settings, mirror_target=None):
    """Weights of the Img1 / anchor / rest groups, each [B, L, H], summing to 1.

    lse_*: [B, L, H] group log-sum-exps; frame_idx: [L] prefix frame of every
    query; mirror_target: [B, a, H] (only for 'mirror_mass').
    """
    lse_img1 = lse_img1.float()
    lse_anchor = lse_anchor.float()
    lse_rest = lse_rest.float()
    mode = settings['mass']
    if mode == 'mirror_mass':
        lse_ir = torch.logaddexp(lse_img1, lse_rest)
        target = mirror_target.index_select(1, frame_idx)  # [B, L, H]
        w_img1 = (1.0 - target) * torch.exp(lse_img1 - lse_ir)
        w_rest = (1.0 - target) * torch.exp(lse_rest - lse_ir)
        return w_img1, target, w_rest
    if mode == 'img1_fade':
        keep = 1.0 - _split_fraction(frame_idx, anchor_index,
                                     settings['split_shape'])
        lse_img1 = lse_img1 + torch.log(keep.clamp_min(1e-6)).to(
            lse_img1).view(1, -1, 1)
    lse_all = torch.logsumexp(
        torch.stack([lse_img1, lse_anchor, lse_rest], dim=0), dim=0)
    w_img1 = torch.exp(lse_img1 - lse_all)
    w_anchor = torch.exp(lse_anchor - lse_all)
    w_rest = torch.exp(lse_rest - lse_all)
    if mode in ('natural', 'img1_fade'):
        return w_img1, w_anchor, w_rest
    lam = _split_fraction(frame_idx, anchor_index,
                          settings['split_shape']).to(w_rest).view(1, -1, 1)
    image_mass = w_img1 + w_anchor
    return (1.0 - lam) * image_mass, lam * image_mass, w_rest


def _prefix_variant_attention(prefix_q, main_k, main_v, anchor_k, anchor_v,
                              anchor_index, tokens_per_frame, freqs, settings,
                              dtype):
    """Prefix attention as an explicit merge of three key groups.

    Returns (output shaped like prefix_q, mirror_target or None).
    """
    a, frame = anchor_index, tokens_per_frame
    out_img1, lse_img1 = _attention_lse(prefix_q, main_k[:, :frame],
                                        main_v[:, :frame], dtype)
    rest_k = torch.cat([main_k[:, frame:a * frame],
                        main_k[:, (a + 1) * frame:]], dim=1)
    rest_v = torch.cat([main_v[:, frame:a * frame],
                        main_v[:, (a + 1) * frame:]], dim=1)
    out_rest, lse_rest = _attention_lse(prefix_q, rest_k, rest_v, dtype)
    del rest_k, rest_v
    if settings['mirror']:
        out_anchor = torch.empty_like(out_img1)
        lse_anchor = torch.empty_like(lse_img1)
        for t in range(a):
            rows = slice(t * frame, (t + 1) * frame)
            # Anchor key from position a to 2t - a: offset (t - key) = a - t.
            mirrored_k = _shift_temporal_rope(anchor_k, 2 * (t - a), freqs)
            out_t, lse_t = _attention_lse(prefix_q[:, rows], mirrored_k,
                                          anchor_v, dtype)
            out_anchor[:, rows] = out_t
            lse_anchor[:, rows] = lse_t
            del mirrored_k, out_t, lse_t
    else:
        out_anchor, lse_anchor = _attention_lse(prefix_q, anchor_k, anchor_v,
                                                dtype)
    mirror_target = None
    if settings['mass'] == 'mirror_mass':
        mirror_target = _mirror_mass_target(lse_img1, lse_rest, a, frame,
                                            settings)
    frame_idx = torch.arange(
        a, device=prefix_q.device).repeat_interleave(frame)
    w_img1, w_anchor, w_rest = _prefix_group_weights(
        lse_img1, lse_anchor, lse_rest, frame_idx, a, settings, mirror_target)
    output = torch.empty_like(out_rest)
    for t in range(a):  # merge one frame at a time to bound float32 temporaries
        rows = slice(t * frame, (t + 1) * frame)
        merged = out_rest[:, rows].float() * w_rest[:, rows, :, None]
        merged += out_img1[:, rows].float() * w_img1[:, rows, :, None]
        merged += out_anchor[:, rows].float() * w_anchor[:, rows, :, None]
        output[:, rows] = merged.to(output.dtype)
        del merged
    return output, mirror_target


# ---------------------------------------------------------------------------
# Visual Relay round 1: attention-mass probe hooks (read-only).
# ---------------------------------------------------------------------------
def _frame_lse(q_rows, bank, n_frames, tokens_per_frame, scale):
    """q_rows [h, n, D], bank [h, n_frames*T, D] -> per-frame LSE [h, n, F]."""
    logits = torch.bmm(q_rows, bank.transpose(1, 2)) * scale
    return logits.reshape(q_rows.size(0), q_rows.size(1), n_frames,
                          tokens_per_frame).logsumexp(-1)


def _probe_paired(layer, main_q, main_k, anchor_k, suffix_q, suffix_k,
                  anchor_index, tokens_per_frame, main_frames, suffix_frames,
                  prefix_anchor_active, settings, mirror_target, freqs,
                  head_offset, sp):
    a, frame = anchor_index, tokens_per_frame
    heads, head_dim = main_q.size(2), main_q.size(3)
    scale = head_dim**-0.5
    device = main_q.device
    idx = ATTN_PROBE.spatial_index(frame, device)
    n = idx.numel()
    natural = settings['mass'] == 'natural' or not prefix_anchor_active
    prefix_eff = torch.zeros(a, main_frames, heads, device=device)
    prefix_nat = torch.zeros(a, main_frames, heads, device=device)
    suffix_main = torch.zeros(main_frames - a - 1, main_frames, heads,
                              device=device)
    stream_b = torch.zeros(suffix_frames, suffix_frames, heads, device=device)
    others = torch.tensor([f for f in range(main_frames) if f not in (0, a)],
                          device=device)
    chunk = ATTN_PROBE.head_chunk
    for h0 in range(0, heads, chunk):
        h1 = min(h0 + chunk, heads)
        bank = main_k[0, :, h0:h1].float().permute(1, 0, 2).contiguous()
        for t in range(a):
            q_rows = main_q[0, t * frame + idx, h0:h1].float().permute(1, 0, 2)
            frame_lse = _frame_lse(q_rows, bank, main_frames, frame, scale)
            if prefix_anchor_active:
                key = anchor_k
                if settings['mirror']:
                    key = _shift_temporal_rope(anchor_k, 2 * (t - a), freqs)
                key = key[0, :, h0:h1].float().permute(1, 0, 2)
                frame_lse[:, :, a] = (torch.bmm(q_rows, key.transpose(1, 2))
                                      * scale).logsumexp(-1)
            else:
                frame_lse[:, :, a] = float('-inf')
            nat = torch.softmax(frame_lse, dim=-1)  # [h, n, F]
            if natural:
                eff = nat
            else:
                lse_rest = frame_lse.index_select(2, others).logsumexp(-1)
                target = (None if mirror_target is None else
                          mirror_target[:, :, h0:h1])
                w_img1, w_anchor, w_rest = _prefix_group_weights(
                    frame_lse[:, :, 0].t()[None], frame_lse[:, :, a].t()[None],
                    lse_rest.t()[None],
                    torch.full((n,), t, dtype=torch.long, device=device), a,
                    settings, target)
                eff = torch.exp(frame_lse - lse_rest[..., None]) * \
                    w_rest[0].t()[..., None]
                eff[:, :, 0] = w_img1[0].t()
                eff[:, :, a] = w_anchor[0].t()
            prefix_eff[t, :, h0:h1] = eff.mean(1).t()
            prefix_nat[t, :, h0:h1] = nat.mean(1).t()
        for t in range(a + 1, main_frames):
            q_rows = main_q[0, t * frame + idx, h0:h1].float().permute(1, 0, 2)
            frame_lse = _frame_lse(q_rows, bank, main_frames, frame, scale)
            frame_lse[:, :, a] = float('-inf')  # suffix queries skip the anchor
            suffix_main[t - a - 1, :, h0:h1] = torch.softmax(
                frame_lse, dim=-1).mean(1).t()
        del bank
        bank = suffix_k[0, :, h0:h1].float().permute(1, 0, 2).contiguous()
        for t in range(suffix_frames):
            q_rows = suffix_q[0, t * frame + idx, h0:h1].float().permute(
                1, 0, 2)
            frame_lse = _frame_lse(q_rows, bank, suffix_frames, frame, scale)
            stream_b[t, :, h0:h1] = torch.softmax(frame_lse,
                                                  dim=-1).mean(1).t()
        del bank
    ATTN_PROBE.store(
        layer,
        dict(prefix_eff=prefix_eff, prefix_nat=prefix_nat,
             suffix_main=suffix_main, stream_b=stream_b),
        head_offset=head_offset,
        sp=sp,
        meta=dict(mode='paired', anchor_index=a, main_frames=main_frames,
                  suffix_frames=suffix_frames, tokens_per_frame=frame,
                  samples=n, route_active=bool(prefix_anchor_active)))


def probe_plain_self_attention(q, k, grid_sizes, head_offset=0, sp=False):
    """Probe hook for ordinary (single-image) self-attention; read-only."""
    if not ATTN_PROBE.record:
        return
    layer = ATTN_PROBE.next_layer()
    ATTN_PROBE.run(_probe_plain, layer, q=q, k=k, grid_sizes=grid_sizes,
                   head_offset=head_offset, sp=sp)


def _probe_plain(layer, q, k, grid_sizes, head_offset, sp):
    frames, height, width = (int(v) for v in grid_sizes[0].tolist())
    frame = height * width
    heads, head_dim = q.size(2), q.size(3)
    scale = head_dim**-0.5
    device = q.device
    idx = ATTN_PROBE.spatial_index(frame, device)
    mass = torch.zeros(frames, frames, heads, device=device)
    chunk = ATTN_PROBE.head_chunk
    for h0 in range(0, heads, chunk):
        h1 = min(h0 + chunk, heads)
        bank = k[0, :frames * frame, h0:h1].float().permute(1, 0,
                                                            2).contiguous()
        for t in range(frames):
            q_rows = q[0, t * frame + idx, h0:h1].float().permute(1, 0, 2)
            frame_lse = _frame_lse(q_rows, bank, frames, frame, scale)
            mass[t, :, h0:h1] = torch.softmax(frame_lse, dim=-1).mean(1).t()
        del bank
    ATTN_PROBE.store(
        layer, dict(plain=mass), head_offset=head_offset, sp=sp,
        meta=dict(mode='plain', main_frames=frames, tokens_per_frame=frame,
                  samples=idx.numel()))


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
