# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

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
    'begin_value_rms_diagnostics',
    'end_value_rms_diagnostics',
    'flash_attention',
    'paired_anchor_attention',
    'self_attention_guidance_attention',
    'temporary_anchor_rope',
    'attention',
]


_VALUE_RMS_DIAGNOSTICS = None


def begin_value_rms_diagnostics(sampling_step):
    """Accumulate aligned M/J value magnitudes for one model evaluation."""
    global _VALUE_RMS_DIAGNOSTICS
    _VALUE_RMS_DIAGNOSTICS = {
        'sampling_step': int(sampling_step),
        'layers': 0,
        'm_sq': None,
        'j_sq': None,
        'delta_sq': None,
        'anchor_sq': None,
    }


def end_value_rms_diagnostics():
    """Return layer-averaged per-frame RMS metrics and clear the collector."""
    global _VALUE_RMS_DIAGNOSTICS
    state = _VALUE_RMS_DIAGNOSTICS
    _VALUE_RMS_DIAGNOSTICS = None
    if state is None or state['layers'] == 0:
        return None
    layers = float(state['layers'])
    m_rms = (state['m_sq'] / layers).sqrt().detach().float().cpu()
    j_rms = (state['j_sq'] / layers).sqrt().detach().float().cpu()
    delta_rms = (state['delta_sq'] / layers).sqrt().detach().float().cpu()
    anchor_rms = (state['anchor_sq'] / layers).sqrt().detach().float().cpu()
    return {
        'sampling_step': state['sampling_step'],
        'layers': state['layers'],
        'm_rms': m_rms.tolist(),
        'j_rms': j_rms.tolist(),
        'delta_rms': delta_rms.tolist(),
        'j_over_m': (j_rms / m_rms.clamp_min(1e-8)).tolist(),
        'j_anchor_rms': float(anchor_rms.item()),
    }


def _record_value_rms(v, frames, tokens_per_frame, anchor_index):
    state = _VALUE_RMS_DIAGNOSTICS
    if state is None or v.size(0) < 3:
        return
    seq_len = frames * tokens_per_frame
    m = v[0, :seq_len].reshape(
        frames, tokens_per_frame, v.size(2), v.size(3))
    j = v[1, :seq_len].reshape(
        frames, tokens_per_frame, v.size(2), v.size(3))
    m = torch.cat([m[:anchor_index], m[anchor_index + 1:]], dim=0)
    j_anchor = j[anchor_index]
    j = torch.cat([j[:anchor_index], j[anchor_index + 1:]], dim=0)
    reduce_dims = (1, 2, 3)
    m_sq = m.float().square().mean(dim=reduce_dims)
    j_sq = j.float().square().mean(dim=reduce_dims)
    delta_sq = (j.float() - m.float()).square().mean(dim=reduce_dims)
    anchor_sq = j_anchor.float().square().mean()
    for name, value in (
            ('m_sq', m_sq), ('j_sq', j_sq),
            ('delta_sq', delta_sq), ('anchor_sq', anchor_sq)):
        state[name] = value if state[name] is None else state[name] + value
    state['layers'] += 1


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


def _duplicate_position_attention(
    q,
    ordinary_k,
    ordinary_v,
    duplicate_k,
    duplicate_v,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
):
    """Attend to one ordinary group and one duplicated temporal position.

    ``duplicate_k``/``duplicate_v`` contain two frame representations that
    intentionally share one temporal RoPE coordinate. Subtracting ``log(2)``
    from that group's LSE is exactly equivalent to applying the same negative
    logit bias to every token in the pair. It removes the pair's automatic
    factor-of-two softmax-mass advantage without changing the learned
    preference between its two representations.
    """
    if ordinary_k.size(1) == 0 or duplicate_k.size(1) == 0:
        raise ValueError(
            "duplicate-position attention requires two non-empty groups")
    ordinary_output, ordinary_lse = flash_attention(
        q=q,
        k=ordinary_k,
        v=ordinary_v,
        window_size=window_size,
        dtype=dtype,
        return_lse=True)
    duplicate_output, duplicate_lse = flash_attention(
        q=q,
        k=duplicate_k,
        v=duplicate_v,
        window_size=window_size,
        dtype=dtype,
        return_lse=True)
    duplicate_lse = duplicate_lse - math.log(2.0)
    duplicate_weight = torch.sigmoid(
        duplicate_lse - ordinary_lse).unsqueeze(-1)
    merged = (
        (1.0 - duplicate_weight) * ordinary_output.float()
        + duplicate_weight * duplicate_output.float())
    return merged.to(ordinary_output.dtype)


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
    prefix_future_eta=0.0,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
):
    """B/C routing with a joint-conditioned first-suffix proxy H.

    C is Img1-only and its temporary anchor is hidden. H is a second
    representation of C's first retained suffix frame, initialized from the
    same noisy latent but conditioned by the joint Img1/Img2 VAE encoding.
    Prefix queries use H K/V in place of regular C at that aligned frame.
    Retained C suffix queries use ordinary C and blend with aligned B output.
    H receives B1's attention update, but its final prediction is discarded.
    ``prefix_anchor_active`` is retained for API compatibility and ignored.
    """
    if window_size != (-1, -1):
        raise ValueError("paired anchor attention requires global attention")
    if freqs is None:
        raise ValueError("paired anchor attention requires RoPE frequencies")
    if q.size(0) != 3 or len(seq_lens) != 3 or len(grid_sizes) != 3:
        raise ValueError(
            "paired anchor attention requires B, C, and H streams")
    if not 0.0 <= stream_weight <= 1.0:
        raise ValueError("paired stream weight must be between zero and one")

    suffix_frames, suffix_height, suffix_width = (
        int(value) for value in grid_sizes[0].tolist())
    joint_frames, joint_height, joint_width = (
        int(value) for value in grid_sizes[1].tolist())
    proxy_frames, proxy_height, proxy_width = (
        int(value) for value in grid_sizes[2].tolist())
    anchor_index = int(anchor_index)
    if not 0 < anchor_index < joint_frames - 1:
        raise ValueError("paired anchor must follow a non-empty prefix")
    if (suffix_height, suffix_width) != (joint_height, joint_width):
        raise ValueError("paired streams must have identical spatial grids")
    if ((proxy_height, proxy_width) != (joint_height, joint_width)
            or proxy_frames != 1):
        raise ValueError("paired H proxy must contain one aligned frame")
    if suffix_frames != joint_frames - anchor_index:
        raise ValueError("paired B stream must align with the C suffix")

    tokens_per_frame = joint_height * joint_width
    suffix_seq_len = suffix_frames * tokens_per_frame
    joint_seq_len = joint_frames * tokens_per_frame
    proxy_seq_len = tokens_per_frame
    if (suffix_seq_len != int(seq_lens[0])
            or joint_seq_len != int(seq_lens[1])
            or proxy_seq_len != int(seq_lens[2])):
        raise ValueError("paired stream grid and sequence lengths disagree")

    suffix_q = q[0:1, :suffix_seq_len]
    suffix_k = k[0:1, :suffix_seq_len]
    suffix_v = v[0:1, :suffix_seq_len]
    joint_q = q[1:2, :joint_seq_len]
    joint_k = k[1:2, :joint_seq_len]
    joint_v = v[1:2, :joint_seq_len]
    proxy_k = k[2:3, :proxy_seq_len]
    proxy_v = v[2:3, :proxy_seq_len]

    # Compress retained C suffix positions left across the hidden transport
    # slot. H begins with local position zero, so shift its key to the same
    # compressed global position as regular C[anchor+1]. Prefix attention uses
    # H at that coordinate and omits the regular copy, avoiding duplicate
    # softmax mass at one temporal position.
    anchor_frame_start = anchor_index * tokens_per_frame
    anchor_frame_end = anchor_frame_start + tokens_per_frame
    compressed_suffix_q = _shift_temporal_rope(
        joint_q[:, anchor_frame_end:joint_seq_len], -1, freqs)
    compressed_suffix_k = _shift_temporal_rope(
        joint_k[:, anchor_frame_end:joint_seq_len], -1, freqs)
    positioned_joint_suffix_k = compressed_suffix_k
    joint_suffix_v = joint_v[:, anchor_frame_end:joint_seq_len]
    positioned_proxy_k = _shift_temporal_rope(
        proxy_k, anchor_index, freqs)

    # B remains an independent Img2-first reference stream and evolves normally.
    suffix_output = flash_attention(
        q=suffix_q,
        k=suffix_k,
        v=suffix_v,
        window_size=window_size,
        dtype=dtype)

    anchor_start = anchor_frame_start
    joint_suffix_query_start = anchor_start + tokens_per_frame
    img2_suffix_output = suffix_output[:, tokens_per_frame:suffix_seq_len]

    # Prefix queries read C prefix, H in place of the first regular suffix
    # frame, and the remaining ordinary C suffix frames. The transport anchor
    # and regular C[anchor+1] are both absent from this route.
    if not 0.0 <= prefix_future_eta <= 1.0:
        raise ValueError("prefix future eta must be in [0, 1]")
    if prefix_future_eta > 0:
        # Frame-dependent grouping: own/earlier prefix versus later prefix,
        # the H proxy, and ordinary C frames after the proxied frame.
        prefix_parts = []
        for frame in range(anchor_index):
            start = frame * tokens_per_frame
            end = start + tokens_per_frame
            query = joint_q[:, start:end]
            past_output, past_lse = flash_attention(
                q=query, k=joint_k[:, :end], v=joint_v[:, :end],
                window_size=window_size, dtype=dtype, return_lse=True)
            future_output, future_lse = flash_attention(
                q=query,
                k=torch.cat([joint_k[:, end:anchor_start],
                             positioned_proxy_k,
                             positioned_joint_suffix_k[:, tokens_per_frame:]],
                            dim=1),
                v=torch.cat([joint_v[:, end:anchor_start],
                             proxy_v,
                             joint_suffix_v[:, tokens_per_frame:]], dim=1),
                window_size=window_size, dtype=dtype, return_lse=True)
            weight = torch.sigmoid(future_lse - past_lse)
            weight = weight + prefix_future_eta * weight * (1.0 - weight)
            # LSE layout [batch, query, head]; use float32 for the merge.
            weight = weight.unsqueeze(-1)
            merged = ((1.0 - weight) * past_output.float()
                      + weight * future_output.float())
            prefix_parts.append(merged.to(past_output.dtype))
        prefix_output = torch.cat(prefix_parts, dim=1)
    else:
        prefix_k = torch.cat([
            joint_k[:, :anchor_start],
            positioned_proxy_k,
            positioned_joint_suffix_k[:, tokens_per_frame:],
        ], dim=1)
        prefix_v = torch.cat([
            joint_v[:, :anchor_start],
            proxy_v,
            joint_suffix_v[:, tokens_per_frame:],
        ], dim=1)
        prefix_output = flash_attention(
            q=joint_q[:, :anchor_start],
            k=prefix_k,
            v=prefix_v,
            window_size=window_size,
            dtype=dtype)

    # Give the temporary C anchor B0's complete Img2-first attention update.
    # C10 therefore has no independent attention call. Its C residual state,
    # cross-attention, and FFN remain in the primary stream, and its final
    # velocity is replaced by B0 after the model call.
    anchor_output = suffix_output[:, :tokens_per_frame]

    # H is aligned with B1, the first retained suffix frame. Reusing B1's
    # complete attention update lets the proxy evolve with Img2-first suffix
    # context at every layer without treating B0 (the removed anchor) as if it
    # were the same frame.
    proxy_output = suffix_output[:, tokens_per_frame:2 * tokens_per_frame]

    # Each retained C-suffix query attends to every ordinary C frame, with the
    # temporary anchor excluded. B independently supplies its Img2-first suffix
    # update. Mixing completed outputs preserves the requested C/B routing
    # without exposing the C continuity branch directly to the anchor.
    ordinary_joint_k = torch.cat([
        joint_k[:, :anchor_start],
        compressed_suffix_k,
    ], dim=1)
    ordinary_joint_v = torch.cat([
        joint_v[:, :anchor_start],
        joint_v[:, joint_suffix_query_start:joint_seq_len],
    ], dim=1)
    continuity_suffix_output = flash_attention(
        q=compressed_suffix_q,
        k=ordinary_joint_k,
        v=ordinary_joint_v,
        window_size=window_size,
        dtype=dtype)
    if continuity_suffix_output.size(1) != img2_suffix_output.size(1):
        raise ValueError("paired suffix query lengths do not align")
    joint_suffix_output = (
        (1.0 - stream_weight) * continuity_suffix_output
        + stream_weight * img2_suffix_output)

    joint_output = torch.cat([
        prefix_output,
        anchor_output,
        joint_suffix_output,
    ], dim=1)
    padded_len = q.size(1)
    if suffix_seq_len < padded_len:
        suffix_output = torch.cat([
            suffix_output,
            suffix_output.new_zeros(
                1, padded_len - suffix_seq_len, *suffix_output.shape[2:])
        ], dim=1)
    if joint_seq_len < padded_len:
        joint_output = torch.cat([
            joint_output,
            joint_output.new_zeros(
                1, padded_len - joint_seq_len, *joint_output.shape[2:])
        ], dim=1)
    if proxy_seq_len < padded_len:
        proxy_output = torch.cat([
            proxy_output,
            proxy_output.new_zeros(
                1, padded_len - proxy_seq_len, *proxy_output.shape[2:])
        ], dim=1)
    return torch.cat([suffix_output, joint_output, proxy_output], dim=0)


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


def temporary_anchor_rope(x, grid_sizes, freqs, anchor_index):
    """Place a temporary anchor halfway between its retained neighbours.

    ``x`` has already received ordinary integer-position RoPE. The temporary
    anchor is shifted left by half a latent frame and every later frame is
    shifted left by one latent frame. Consequently the retained timeline stays
    contiguous after the anchor is removed, while both aligned ablation
    streams retain exactly the same token coordinates.
    """
    if anchor_index is None:
        return x
    output = []
    for batch_index, (frames, height, width) in enumerate(
            grid_sizes.tolist()):
        if not 0 < anchor_index < frames - 1:
            raise ValueError(
                "temporary anchor must have retained frames on both sides")
        tokens_per_frame = height * width
        anchor_start = anchor_index * tokens_per_frame
        anchor_end = anchor_start + tokens_per_frame
        sequence_end = frames * tokens_per_frame
        sample = x[batch_index:batch_index + 1]
        shifted = torch.cat([
            sample[:, :anchor_start],
            _shift_temporal_rope(sample[:, anchor_start:anchor_end], -0.5,
                                 freqs),
            _shift_temporal_rope(sample[:, anchor_end:sequence_end], -1,
                                 freqs),
            sample[:, sequence_end:],
        ], dim=1)
        output.append(shifted)
    return torch.cat(output, dim=0)


def self_attention_guidance_attention(
    q,
    k,
    v,
    seq_lens,
    grid_sizes,
    freqs,
    anchor_index,
    joint_weight=0.5,
    img2_weight=0.25,
    joint_rms_cap=None,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
):
    """Guide a retained M timeline with an expanded joint J timeline.

    M and J both have 22 slots, but M's temporary anchor is excluded from all
    attention and exists only to carry B0's scheduler velocity. J has one
    joint multi-image anchor at ``anchor_index``. Both J-prefix and J-suffix
    queries see that same evolving anchor, and the anchor attends to itself
    plus J's suffix. Removing J's anchor
    aligns its ordinary outputs with M's retained outputs, after which M receives
    ``O_M + joint_weight * (O_J - O_M)`` at every retained position.
    """
    if q.size(0) < 1 or k.size(0) != q.size(0) or v.size(0) != q.size(0):
        raise ValueError(
            "self-attention guidance requires at least the primary stream")
    if window_size != (-1, -1):
        raise ValueError(
            "self-attention guidance requires global self-attention")
    if not 0.0 <= joint_weight <= 1.0:
        raise ValueError("joint attention weight must be in [0, 1]")
    if not 0.0 <= img2_weight <= 1.0:
        raise ValueError("Img2 attention weight must be in [0, 1]")
    if joint_rms_cap is not None and joint_rms_cap <= 0.0:
        raise ValueError("joint RMS cap must be positive")
    frames, height, width = (
        int(value) for value in grid_sizes[0].tolist())
    if not 0 < anchor_index < frames - 1:
        raise ValueError(
            "self-attention-guidance anchor must have frames on both sides")
    tokens_per_frame = height * width
    seq_len = frames * tokens_per_frame

    anchor_start = anchor_index * tokens_per_frame
    anchor_end = anchor_start + tokens_per_frame
    if q.size(0) == 1:
        retained_q = torch.cat([q[:, :anchor_start], q[:, anchor_end:seq_len]], dim=1)
        retained_k = torch.cat([k[:, :anchor_start], k[:, anchor_end:seq_len]], dim=1)
        retained_v = torch.cat([v[:, :anchor_start], v[:, anchor_end:seq_len]], dim=1)
        output = flash_attention(
            q=retained_q,
            k=retained_k,
            v=retained_v,
            window_size=window_size,
            dtype=dtype)
        return torch.cat([
            output[:, :anchor_start],
            torch.zeros_like(q[:, anchor_start:anchor_end]),
            output[:, anchor_start:],
            torch.zeros_like(q[:, seq_len:]),
        ], dim=1)

    if q.size(0) not in (2, 3):
        raise ValueError(
            "self-attention guidance requires M/B or M/J/B streams")
    suffix_frames = frames - anchor_index
    suffix_seq_len = suffix_frames * tokens_per_frame

    # After the configured J stop step, retain standard M and Img2-first B.
    # M's prefix remains untouched while B continues to guide M's suffix.
    if q.size(0) == 2:
        if (int(seq_lens[0]) != seq_len
                or int(seq_lens[1]) != suffix_seq_len):
            raise ValueError(
                "late self-attention-guidance M/B lengths disagree")
        m_retained_output = flash_attention(
            q=torch.cat([
                q[0:1, :anchor_start], q[0:1, anchor_end:seq_len]
            ], dim=1),
            k=torch.cat([
                k[0:1, :anchor_start], k[0:1, anchor_end:seq_len]
            ], dim=1),
            v=torch.cat([
                v[0:1, :anchor_start], v[0:1, anchor_end:seq_len]
            ], dim=1),
            window_size=window_size,
            dtype=dtype)
        b_output = flash_attention(
            q=q[1:2, :suffix_seq_len],
            k=k[1:2, :suffix_seq_len],
            v=v[1:2, :suffix_seq_len],
            window_size=window_size,
            dtype=dtype)
        m_prefix_output = m_retained_output[:, :anchor_start]
        m_suffix_output = m_retained_output[:, anchor_start:]
        b_suffix_output = b_output[:, tokens_per_frame:suffix_seq_len]
        guided_m_suffix_output = (
            (1.0 - img2_weight) * m_suffix_output
            + img2_weight * b_suffix_output)
        guided_m_retained_output = torch.cat([
            m_prefix_output, guided_m_suffix_output
        ], dim=1)
        guided_m_output = torch.cat([
            guided_m_retained_output[:, :anchor_start],
            torch.zeros_like(q[0:1, anchor_start:anchor_end]),
            guided_m_retained_output[:, anchor_start:],
        ], dim=1)
        guided_m_output = torch.cat([
            guided_m_output, torch.zeros_like(q[0:1, seq_len:])
        ], dim=1)
        b_output = torch.cat([
            b_output, torch.zeros_like(q[1:2, suffix_seq_len:])
        ], dim=1)
        return torch.cat([guided_m_output, b_output], dim=0)

    joint_frames, joint_height, joint_width = (
        int(value) for value in grid_sizes[1].tolist())
    if ((joint_height, joint_width) != (height, width)
            or joint_frames != frames):
        raise ValueError(
            "expanded M and J must share one token grid")
    joint_seq_len = joint_frames * tokens_per_frame
    if (int(seq_lens[0]) != seq_len
            or int(seq_lens[1]) != joint_seq_len
            or int(seq_lens[2]) != suffix_seq_len):
        raise ValueError(
            "self-attention-guidance M/J/B sequence lengths disagree")
    _record_value_rms(v, frames, tokens_per_frame, anchor_index)

    # M's temporary anchor is absent from every retained query's Q/K/V set.
    # Its neighbours are adjacent because temporary_anchor_rope shifted all
    # post-anchor retained positions left by one.
    m_retained_output = flash_attention(
        q=torch.cat([q[0:1, :anchor_start], q[0:1, anchor_end:seq_len]], dim=1),
        k=torch.cat([k[0:1, :anchor_start], k[0:1, anchor_end:seq_len]], dim=1),
        v=torch.cat([v[0:1, :anchor_start], v[0:1, anchor_end:seq_len]], dim=1),
        window_size=window_size,
        dtype=dtype)

    # J-prefix sees its own joint anchor. Its compressed anchor key is moved
    # onto the first suffix coordinate for the prefix-facing route. Treat the
    # anchor and first suffix frame as two representations of that one
    # temporal position, and subtract log(2) from their group evidence so the
    # duplicated coordinate does not receive twice the softmax odds merely
    # because it contains two frame-sized K/V sets.
    joint_prefix_anchor_k = _shift_temporal_rope(
        k[1:2, anchor_start:anchor_end], 0.5, freqs)
    first_suffix_end = anchor_end + tokens_per_frame
    j_prefix_output = _duplicate_position_attention(
        q=q[1:2, :anchor_start],
        ordinary_k=torch.cat([
            k[1:2, :anchor_start],
            k[1:2, first_suffix_end:joint_seq_len],
        ], dim=1),
        ordinary_v=torch.cat([
            v[1:2, :anchor_start],
            v[1:2, first_suffix_end:joint_seq_len],
        ], dim=1),
        duplicate_k=torch.cat([
            joint_prefix_anchor_k,
            k[1:2, anchor_end:first_suffix_end],
        ], dim=1),
        duplicate_v=torch.cat([
            v[1:2, anchor_start:anchor_end],
            v[1:2, anchor_end:first_suffix_end],
        ], dim=1),
        window_size=window_size,
        dtype=dtype)

    # J-suffix uses the same history-aware joint anchor as J-prefix. Move its
    # key onto the suffix-facing coordinate shared with the final prefix
    # position. Apply the same duplicate-position correction to the final
    # prefix frame plus anchor pair. J ordinary suffix coordinates are already
    # shifted left once.
    positioned_joint_anchor_k = _shift_temporal_rope(
        k[1:2, anchor_start:anchor_end], -0.5, freqs)
    last_prefix_start = anchor_start - tokens_per_frame
    j_suffix_output = _duplicate_position_attention(
        q=q[1:2, anchor_end:joint_seq_len],
        ordinary_k=torch.cat([
            k[1:2, :last_prefix_start],
            k[1:2, anchor_end:joint_seq_len],
        ], dim=1),
        ordinary_v=torch.cat([
            v[1:2, :last_prefix_start],
            v[1:2, anchor_end:joint_seq_len],
        ], dim=1),
        duplicate_k=torch.cat([
            k[1:2, last_prefix_start:anchor_start],
            positioned_joint_anchor_k,
        ], dim=1),
        duplicate_v=torch.cat([
            v[1:2, last_prefix_start:anchor_start],
            v[1:2, anchor_start:anchor_end],
        ], dim=1),
        window_size=window_size,
        dtype=dtype)

    # The sole multi-image anchor evolves from itself and J's suffix. For this
    # anchor-query route, place both its query and self key at the final-prefix
    # coordinate. The first compressed suffix key is therefore exactly one
    # latent position away while the anchor's self offset remains zero.
    positioned_joint_anchor_q = _shift_temporal_rope(
        q[1:2, anchor_start:anchor_end], -0.5, freqs)
    j_anchor_output = flash_attention(
        q=positioned_joint_anchor_q,
        k=torch.cat([
            positioned_joint_anchor_k,
            k[1:2, anchor_end:joint_seq_len],
        ], dim=1),
        v=v[1:2, anchor_start:joint_seq_len],
        window_size=window_size,
        dtype=dtype)

    # B is an independent standard Img2-first I2V suffix.
    b_output = flash_attention(
        q=q[2:3, :suffix_seq_len],
        k=k[2:3, :suffix_seq_len],
        v=v[2:3, :suffix_seq_len],
        window_size=window_size,
        dtype=dtype)

    j_aligned_output = torch.cat([j_prefix_output, j_suffix_output], dim=1)
    if j_aligned_output.size(1) != m_retained_output.size(1):
        raise ValueError("removing J's anchor did not align J with M")
    b_suffix_output = b_output[:, tokens_per_frame:suffix_seq_len]
    m_prefix_output = m_retained_output[:, :anchor_start]
    m_suffix_output = m_retained_output[:, anchor_start:]
    if b_suffix_output.size(1) != m_suffix_output.size(1):
        raise ValueError("B1: did not align with M's retained suffix")
    def inject_joint_residual(m_output, j_output, frame_count):
        delta = j_output - m_output
        if joint_rms_cap is None:
            return m_output + joint_weight * delta
        shape = (m_output.size(0), frame_count, tokens_per_frame,
                 m_output.size(2), m_output.size(3))
        m_frames = m_output.reshape(shape)
        delta_frames = delta.reshape(shape)
        # One adaptive scale per frame and attention head. Spatial tokens and
        # head channels share the scale so their internal structure is kept.
        m_rms = m_frames.float().square().mean(
            dim=(2, 4), keepdim=True).sqrt()
        delta_rms = delta_frames.float().square().mean(
            dim=(2, 4), keepdim=True).sqrt()
        cap_scale = (
            float(joint_rms_cap) * m_rms
            / delta_rms.clamp_min(1e-8))
        alpha_eff = torch.minimum(
            torch.full_like(cap_scale, float(joint_weight)), cap_scale)
        return (m_frames + alpha_eff.to(delta_frames.dtype) *
                delta_frames).reshape_as(m_output)

    guided_m_prefix_output = inject_joint_residual(
        m_prefix_output, j_prefix_output, anchor_index)
    # Before the J handoff, suffix guidance is exclusively M/J. B is still
    # evaluated so B0 can supply the anchor velocity, but B1: does not enter
    # the layer-wise suffix update until the two-stream M/B handoff route.
    guided_m_suffix_output = inject_joint_residual(
        m_suffix_output, j_suffix_output, frames - anchor_index - 1)
    guided_m_retained_output = torch.cat([
        guided_m_prefix_output, guided_m_suffix_output
    ], dim=1)
    guided_m_output = torch.cat([
        guided_m_retained_output[:, :anchor_start],
        torch.zeros_like(q[0:1, anchor_start:anchor_end]),
        guided_m_retained_output[:, anchor_start:],
    ], dim=1)
    j_valid_output = torch.cat([
        j_prefix_output, j_anchor_output, j_suffix_output
    ], dim=1)
    guided_m_output = torch.cat([
        guided_m_output, torch.zeros_like(q[0:1, seq_len:])
    ], dim=1)
    j_output = torch.cat([
        j_valid_output, torch.zeros_like(q[1:2, joint_seq_len:])
    ], dim=1)
    b_output = torch.cat([
        b_output, torch.zeros_like(q[2:3, suffix_seq_len:])
    ], dim=1)
    return torch.cat([guided_m_output, j_output, b_output], dim=0)


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
