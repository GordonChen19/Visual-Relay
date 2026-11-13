# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from contextlib import contextmanager
from contextvars import ContextVar

import torch

_joint_attention_samples = ContextVar('joint_attention_samples', default=None)
_joint_alpha_samples = ContextVar('joint_alpha_samples', default=None)


@contextmanager
def collect_joint_attention_shares(enabled=False):
    """Collect conditional J attention only within the current model call."""
    samples = []
    token = _joint_attention_samples.set(samples if enabled else None)
    try:
        yield samples
    finally:
        _joint_attention_samples.reset(token)


@contextmanager
def collect_joint_adaptive_alpha_stats(enabled=False):
    """Collect per-layer adaptive-alpha summaries for one model call."""
    samples = []
    token = _joint_alpha_samples.set(samples if enabled else None)
    try:
        yield samples
    finally:
        _joint_alpha_samples.reset(token)


def _record_joint_attention_shares(q, k, anchor_index, tokens_per_frame,
                                   suffix_q=None, suffix_k=None,
                                   prefix_anchor_visible=True,
                                   suffix_anchor_transfer=None):
    """Exact group probabilities, chunked to avoid a full attention matrix."""
    samples = _joint_attention_samples.get()
    if samples is None:
        return
    sums = []
    count = 0
    for query_frame, targets in (
            (anchor_index - 1, (anchor_index, anchor_index + 1)),
            (anchor_index + 1, (anchor_index, anchor_index - 1))):
        query = q[:, query_frame * tokens_per_frame:
                  (query_frame + 1) * tokens_per_frame]
        memory = k
        if query_frame > anchor_index and suffix_q is not None:
            query = suffix_q[:, :tokens_per_frame]
            memory = suffix_k
        if query_frame < anchor_index and not prefix_anchor_visible:
            # The anchor was removed from prefix K/V. The original first
            # suffix frame consequently occupies ``anchor_index`` in this
            # compact diagnostic memory.
            targets = (None, anchor_index)
        group_sums = q.new_zeros(2, dtype=torch.float32)
        with torch.autocast(device_type=q.device.type, enabled=False):
            keys = memory.transpose(1, 2).float().transpose(-2, -1)
            for start in range(0, query.size(1), 16):
                logits = torch.matmul(
                    query[:, start:start + 16].transpose(1, 2).float(),
                    keys) * (q.size(-1) ** -0.5)
                normalizer = torch.logsumexp(logits, dim=-1)
                transfer_weights = None
                if query_frame > anchor_index and suffix_anchor_transfer is not None:
                    boundary = anchor_index * tokens_per_frame
                    end = boundary + tokens_per_frame
                    group_lse = torch.stack([
                        torch.logsumexp(logits[..., :boundary], dim=-1),
                        torch.logsumexp(logits[..., boundary:end], dim=-1),
                        torch.logsumexp(logits[..., end:], dim=-1),
                    ], dim=-1)
                    original_weights, transfer_weights = _joint_suffix_transfer_weights(
                        group_lse, *suffix_anchor_transfer)
                for index, frame in enumerate(targets):
                    if frame is None:
                        continue
                    group_lse = torch.logsumexp(
                        logits[..., frame * tokens_per_frame:
                               (frame + 1) * tokens_per_frame], dim=-1)
                    probability = (group_lse - normalizer).exp()
                    if transfer_weights is not None:
                        if frame == anchor_index:
                            probability = transfer_weights[..., 1]
                        elif frame < anchor_index:
                            probability = probability * (
                                transfer_weights[..., 0]
                                / original_weights[..., 0].clamp_min(1e-30))
                    group_sums[index] += probability.sum()
        sums.append(group_sums)
        count = query.size(0) * query.size(1) * query.size(2)
    samples.append(torch.cat([*sums, q.new_tensor([count], dtype=torch.float32)]))

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
    'collect_joint_adaptive_alpha_stats',
    'collect_joint_attention_shares',
    'joint_img2_guidance_attention',
    'attention',
]


def _joint_suffix_transfer_weights(group_lse, ratio, max_delta):
    """Move a bounded probability share from prefix to anchor, keeping suffix fixed."""
    if not math.isfinite(ratio) or ratio < 0:
        raise ValueError("suffix anchor transfer ratio must be finite and nonnegative")
    if not math.isfinite(max_delta) or not 0 <= max_delta <= 1:
        raise ValueError("suffix anchor transfer cap must be in [0, 1]")
    weights = group_lse.float().softmax(dim=-1)
    prefix, anchor, suffix = weights.unbind(dim=-1)
    delta = ((ratio * prefix - anchor) / (1.0 + ratio)).clamp(
        min=0.0, max=max_delta)
    delta = torch.minimum(delta, prefix)
    corrected = torch.stack([prefix - delta, anchor + delta, suffix], dim=-1)
    return weights, corrected


def _joint_first_suffix_attention(q, k, v, anchor_start, anchor_end,
                                  transfer, window_size, dtype):
    """Exactly aggregate one softmax with probability-preserving group biases.

    Separate FlashAttention calls expose each group's LSE and normalized
    spatial output. Their corrected mixture is algebraically the attention
    obtained by applying the proposed per-query/head biases to the full logits.
    """
    outputs, evidence = [], []
    for start, end in ((0, anchor_start), (anchor_start, anchor_end),
                       (anchor_end, k.size(1))):
        output, lse = flash_attention(
            q=q, k=k[:, start:end], v=v[:, start:end],
            window_size=window_size, dtype=dtype, return_lse=True)
        outputs.append(output)
        evidence.append(lse)
    _, weights = _joint_suffix_transfer_weights(
        torch.stack(evidence, dim=-1), *transfer)
    return (torch.stack([output.float() for output in outputs], dim=-2)
            * weights.unsqueeze(-1)).sum(dim=-2).to(outputs[0].dtype)


def joint_img2_guidance_attention(
    q,
    k,
    v,
    seq_lens,
    grid_sizes,
    anchor_index,
    img2_weight=0.25,
    rope_compression=False,
    adaptive_pair_attention=False,
    adaptive_pair_margin=1.05,
    adaptive_pair_alpha_max=0.7,
    compression_ratio=1.0,
    anchor_rope_offset=None,
    suffix_anchor_transfer=None,
    velocity_only=False,
    freqs=None,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
):
    """Guide a native joint Img1/Img2 stream with an Img2-first suffix.

    Stream 0 is a full, jointly VAE-conditioned timeline. Stream 1 is a
    standard Img2-first I2V timeline aligned to J. By default, both streams use
    ordinary bidirectional attention. Optional shifted-anchor RoPE compression
    gives prefix and suffix queries distinct anchor-key coordinates and limits
    the anchor query to itself/future frames. A zero compression_ratio retains
    that routing mask but uses native coordinates for every query and key.
    B always retains ordinary RoPE.
    """
    if q.size(0) != 2 or k.size(0) != 2 or v.size(0) != 2:
        raise ValueError("joint Img2 guidance requires exactly J and B streams")
    if window_size != (-1, -1):
        raise ValueError("joint Img2 guidance requires global self-attention")
    if not 0.0 <= img2_weight <= 1.0:
        raise ValueError("joint Img2 guidance weight must be in [0, 1]")
    if not 0.0 <= compression_ratio <= 1.0:
        raise ValueError("joint RoPE compression ratio must be in [0, 1]")
    backward_pair_rope = anchor_rope_offset == -1.0
    if suffix_anchor_transfer is not None and not (rope_compression and velocity_only):
        raise ValueError("suffix anchor transfer requires shifted-anchor RoPE routing")
    if anchor_rope_offset is not None:
        if (not rope_compression
                or not (backward_pair_rope
                        or 0.0 <= anchor_rope_offset <= 1.0)):
            raise ValueError(
                "fractional anchor RoPE requires compressed routing and offset in [0, 1]")

    frames, height, width = (
        int(value) for value in grid_sizes[0].tolist())
    b_frames, b_height, b_width = (
        int(value) for value in grid_sizes[1].tolist())
    if (b_height, b_width) != (height, width):
        raise ValueError("joint and Img2 streams must share a spatial grid")
    if not 0 < anchor_index < frames:
        raise ValueError("joint Img2 anchor must be inside the timeline")
    if anchor_rope_offset is not None and anchor_index >= frames - 1:
        raise ValueError("fractional anchor RoPE needs a following J frame")
    shifted_anchor = b_frames == frames - anchor_index + 1
    if not shifted_anchor and b_frames != frames - anchor_index:
        raise ValueError("Img2 stream does not align with joint anchor/suffix")

    tokens_per_frame = height * width
    joint_seq_len = frames * tokens_per_frame
    img2_seq_len = b_frames * tokens_per_frame
    if (int(seq_lens[0]) != joint_seq_len
            or int(seq_lens[1]) != img2_seq_len):
        raise ValueError("joint Img2 guidance sequence lengths disagree")

    aligned_start = anchor_index * tokens_per_frame
    anchor_end = aligned_start + tokens_per_frame
    if rope_compression:
        if not (shifted_anchor or velocity_only):
            raise ValueError(
                "joint RoPE compression requires velocity-only B guidance")
        if freqs is None:
            raise ValueError("joint RoPE compression requires RoPE frequencies")

        joint_q = q[0:1, :joint_seq_len]
        joint_k = k[0:1, :joint_seq_len]
        joint_v = v[0:1, :joint_seq_len]
        # A zero ratio is the ordinary-RoPE ablation: preserve all routing,
        # including the anchor's self/future mask, without rotating Q/K.
        # At ratio 1, prefix queries retain their native coordinates. Compress every
        # post-anchor key left by one so the anchor and first suffix frame
        # share the anchor coordinate: [..., J9@9, J10@10, J11@10, J12@11].
        if backward_pair_rope:
            prefix_k = torch.cat([
                joint_k[:, :aligned_start],
                _shift_temporal_rope(
                    joint_k[:, aligned_start:], -1.0, freqs),
            ], dim=1)
        elif anchor_rope_offset is not None:
            # For prefix/suffix queries, J9/J11 are neighbours while J10
            # sits at the requested fraction between them. The anchor query
            # below keeps native self/future offsets and its existing mask.
            prefix_k = torch.cat([
                joint_k[:, :aligned_start],
                _shift_temporal_rope(
                    joint_k[:, aligned_start:anchor_end],
                    anchor_rope_offset - 1.0, freqs),
                _shift_temporal_rope(joint_k[:, anchor_end:], -1.0, freqs),
            ], dim=1)
        else:
            prefix_k = torch.cat([
                joint_k[:, :anchor_end],
                _shift_temporal_rope(
                    joint_k[:, anchor_end:], -compression_ratio, freqs),
            ], dim=1) if compression_ratio else joint_k
        # Suffix queries read all J frames using compressed RoPE. Shift both
        # suffix queries and anchor/suffix keys left by one, colocating J9/J10
        # while preserving relative offsets within the suffix. J11 is then
        # one temporal step from both J9 and J10 at ratio 1.
        suffix_shift = (0.0 if backward_pair_rope else
                        1.0 if anchor_rope_offset is not None else
                        compression_ratio)
        compressed_suffix_q = (_shift_temporal_rope(
            joint_q[:, anchor_end:], -suffix_shift, freqs)
            if suffix_shift else joint_q[:, anchor_end:])
        suffix_k = (joint_k if backward_pair_rope else
                    prefix_k if anchor_rope_offset is not None else
                    torch.cat([
                        joint_k[:, :aligned_start],
                        _shift_temporal_rope(
                            joint_k[:, aligned_start:], -compression_ratio,
                            freqs),
                    ], dim=1) if compression_ratio else joint_k)
        _record_joint_attention_shares(
            joint_q, prefix_k, anchor_index, tokens_per_frame,
            compressed_suffix_q, suffix_k, True, suffix_anchor_transfer)
        # Prefix queries read all J memory, including the anchor and its
        # colocated first suffix neighbour, with no mass correction.
        prefix_output = flash_attention(
            q=joint_q[:, :aligned_start],
            k=prefix_k,
            v=joint_v,
            window_size=window_size,
            dtype=dtype)
        if backward_pair_rope:
            anchor_q = _shift_temporal_rope(
                joint_q[:, aligned_start:anchor_end], -1.0, freqs)
            anchor_k = torch.cat([
                joint_k[:, :aligned_start - tokens_per_frame],
                _shift_temporal_rope(
                    joint_k[:, aligned_start - tokens_per_frame:],
                    -1.0, freqs),
            ], dim=1)
            anchor_output = flash_attention(
                q=anchor_q,
                k=anchor_k,
                v=joint_v,
                window_size=window_size,
                dtype=dtype)
        else:
            anchor_output = flash_attention(
                q=joint_q[:, aligned_start:anchor_end],
                k=joint_k[:, aligned_start:],
                v=joint_v[:, aligned_start:],
                window_size=window_size,
                dtype=dtype)
        if suffix_anchor_transfer is not None:
            first_suffix = _joint_first_suffix_attention(
                compressed_suffix_q[:, :tokens_per_frame], suffix_k, joint_v,
                aligned_start, anchor_end, suffix_anchor_transfer, window_size, dtype)
            if compressed_suffix_q.size(1) > tokens_per_frame:
                remaining_suffix = flash_attention(
                    q=compressed_suffix_q[:, tokens_per_frame:],
                    k=suffix_k, v=joint_v, window_size=window_size, dtype=dtype)
                suffix_output = torch.cat([first_suffix, remaining_suffix], dim=1)
            else:
                suffix_output = first_suffix
        else:
            suffix_output = flash_attention(
                q=compressed_suffix_q,
                k=suffix_k,
                v=joint_v,
                window_size=window_size,
                dtype=dtype)
        joint_output = torch.cat(
            [prefix_output, anchor_output, suffix_output], dim=1)
    else:
        _record_joint_attention_shares(
            q[0:1, :joint_seq_len], k[0:1, :joint_seq_len],
            anchor_index, tokens_per_frame)
        joint_output = flash_attention(
            q=q[0:1, :joint_seq_len],
            k=k[0:1, :joint_seq_len],
            v=v[0:1, :joint_seq_len],
            window_size=window_size,
            dtype=dtype)
    if shifted_anchor or velocity_only:
        img2_output = flash_attention(
            q=q[1:2, :img2_seq_len],
            k=k[1:2, :img2_seq_len],
            v=v[1:2, :img2_seq_len],
            window_size=window_size,
            dtype=dtype)
        # Shifted guidance is velocity-only. The pipeline consumes B's final
        # prediction after the DiT without replacing J attention outputs.
        joint_output = torch.cat([
            joint_output, torch.zeros_like(q[0:1, joint_seq_len:])
        ], dim=1)
        img2_output = torch.cat([
            img2_output, torch.zeros_like(q[1:2, img2_seq_len:])
        ], dim=1)
        return torch.cat([joint_output, img2_output], dim=0)

    joint_anchor_output = flash_attention(
        q=q[0:1, aligned_start:anchor_end],
        k=k[0:1, aligned_start:joint_seq_len],
        v=v[0:1, aligned_start:joint_seq_len],
        window_size=window_size,
        dtype=dtype)
    joint_output = torch.cat([
        joint_output[:, :aligned_start], joint_anchor_output,
        joint_output[:, anchor_end:]
    ], dim=1)
    img2_output = flash_attention(
        q=q[1:2, :img2_seq_len],
        k=k[1:2, :img2_seq_len],
        v=v[1:2, :img2_seq_len],
        window_size=window_size,
        dtype=dtype)
    guided_tail = (
        (1.0 - img2_weight) * joint_output[:, aligned_start:]
        + img2_weight * img2_output)
    joint_output = torch.cat([
        joint_output[:, :aligned_start], guided_tail
    ], dim=1)

    joint_output = torch.cat([
        joint_output, torch.zeros_like(q[0:1, joint_seq_len:])
    ], dim=1)
    img2_output = torch.cat([
        img2_output, torch.zeros_like(q[1:2, img2_seq_len:])
    ], dim=1)
    return torch.cat([joint_output, img2_output], dim=0)


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
