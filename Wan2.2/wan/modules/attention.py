# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

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
):
    """Fuse an Img1 main stream with an aligned Img2-first suffix stream.

    ``q`` and ``k`` must already have ordinary per-stream RoPE applied. The
    batch contains exactly two samples: a full main sequence followed by a
    suffix whose local frame zero aligns with ``anchor_index`` in the main
    sequence. While the prefix anchor route is active, prefix queries use one
    ordinary softmax over all ordinary main-stream K/V on both sides of the
    temporary anchor plus a hybrid anchor using the history-aware Stream-A
    anchor key with the fresh-cache Stream-B anchor value. After that route is
    disabled, only the hybrid anchor is removed; ordinary post-anchor K/V
    remain visible.
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
        shifted_q = _shift_temporal_rope(
            main_q[:, frame_start:frame_end], -1, freqs)
        main_q[:, frame_start:frame_end].copy_(shifted_q)
        del shifted_q
        shifted_k = _shift_temporal_rope(
            main_k[:, frame_start:frame_end], -1, freqs)
        main_k[:, frame_start:frame_end].copy_(shifted_k)
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

    # Prefix queries always retain all ordinary Stream-A context on both sides
    # of the temporary anchor. Before the route switch, also include one hybrid
    # anchor: history-aware A12 supplies K while fresh-cache B0 supplies V.
    # After the switch, remove only that hybrid anchor.
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
    return torch.cat([main_output, suffix_output], dim=0)


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
