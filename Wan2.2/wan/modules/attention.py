# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import json
import logging
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

from ..utils.frame_conditioning import (blocked_frames_for_query,
                                        build_attention_segments,
                                        build_anchor_query_positions,
                                        build_compressed_position_map,
                                        build_sacrificial_fractional_position_map)

__all__ = [
    'flash_attention',
    'segmented_flash_attention',
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
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
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
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
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
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def _direction_statistics(frame_probabilities, query_frame):
    """Summarize per-head temporal attention as percentages."""
    sections = {
        "previous": frame_probabilities[:, :query_frame].sum(dim=1),
        "current": frame_probabilities[:, query_frame],
        "future": frame_probabilities[:, query_frame + 1:].sum(dim=1),
    }
    result = {}
    for name, values in sections.items():
        values = values * 100
        result[f"{name}_percent"] = round(values.mean().item(), 6)
        result[f"{name}_percent_std_across_heads"] = round(
            values.std(unbiased=False).item(), 6)
    return result


def log_frame_attention_probe(q, k, v, seq_lens, grid_sizes,
                              conditioned_frame_indices, probe,
                              anchor_block_radius=0,
                              query_frame_override=None,
                              key_frame_overrides=None):
    """Measure one query patch without materializing full attention weights.

    FlashAttention remains the source of the model output. This diagnostic
    recomputes only ``num_heads x num_key_tokens`` logits for one selected
    query patch, reduces them to temporal-frame masses, and logs JSON.
    """
    batch_index = int(probe.get("batch_index", 0))
    if batch_index < 0 or batch_index >= q.size(0):
        raise ValueError("attention probe batch index is out of range")

    frames, height, width = grid_sizes[batch_index].tolist()
    tokens_per_frame = height * width
    seq_len = frames * tokens_per_frame
    if seq_len != int(seq_lens[batch_index]):
        raise ValueError(
            "grid size and sequence length disagree for attention probe")

    query_frame = int(probe["query_frame"])
    if query_frame < 0 or query_frame >= frames:
        raise ValueError(
            f"attention probe query frame must be in [0, {frames})")

    query_row = probe.get("query_row")
    query_col = probe.get("query_col")
    query_row = height // 2 if query_row is None else int(query_row)
    query_col = width // 2 if query_col is None else int(query_col)
    if query_row < 0 or query_row >= height:
        raise ValueError(
            f"attention probe query row must be in [0, {height})")
    if query_col < 0 or query_col >= width:
        raise ValueError(
            f"attention probe query column must be in [0, {width})")

    conditioned = ([] if conditioned_frame_indices is None else
                   conditioned_frame_indices[batch_index])
    blocked = blocked_frames_for_query(
        frames,
        conditioned,
        query_frame,
        anchor_block_radius=anchor_block_radius)
    blocked_set = set(blocked)
    anchor = None
    for start, end, segment_anchor, _ in build_attention_segments(
            frames,
            conditioned,
            anchor_block_radius=anchor_block_radius):
        if start <= query_frame < end:
            anchor = segment_anchor
            break

    query_spatial_index = query_row * width + query_col
    query_index = query_frame * tokens_per_frame + query_spatial_index
    half_dtypes = (torch.float16, torch.bfloat16)
    probe_dtype = v.dtype if v.dtype in half_dtypes else torch.bfloat16
    if query_frame_override is None:
        query = q[batch_index, query_index]
    else:
        query = query_frame_override[query_spatial_index]
    query = query.to(probe_dtype).float()
    num_heads, head_dim = query.shape
    scale = 1.0 / math.sqrt(head_dim)

    # Reducing each frame with logsumexp and then applying a frame softmax is
    # exactly equivalent to summing token-level softmax probabilities.
    frame_log_mass = torch.full(
        (num_heads, frames),
        -torch.inf,
        dtype=torch.float32,
        device=q.device)
    for frame in range(frames):
        if frame in blocked_set:
            continue
        start = frame * tokens_per_frame
        end = start + tokens_per_frame
        if key_frame_overrides is not None and frame in key_frame_overrides:
            keys = key_frame_overrides[frame]
        else:
            keys = k[batch_index, start:end]
        keys = keys.to(probe_dtype).float()
        logits = torch.einsum("hd,khd->hk", query, keys) * scale
        frame_log_mass[:, frame] = torch.logsumexp(logits, dim=1)

    frame_probabilities = torch.softmax(frame_log_mass, dim=1)
    frame_percent = frame_probabilities.mean(dim=0) * 100
    frame_percent_std = frame_probabilities.std(
        dim=0, unbiased=False) * 100
    payload = {
        "type": "self_attention_frame_probe",
        "cfg_branch": "conditional",
        "sampling_step": int(probe["sampling_step"]),
        "timestep": float(probe["timestep"]),
        "layer": int(probe["layer"]),
        "query_frame": query_frame,
        "query_row": query_row,
        "query_col": query_col,
        "num_heads": num_heads,
        "assigned_anchor": anchor,
        "temporal_rope_direction": 1,
        "conditioned_frames": [int(frame) for frame in conditioned],
        "blocked_frames": [int(frame) for frame in blocked],
        "frame_percent": [round(value, 6) for value in frame_percent.tolist()],
        "frame_percent_std_across_heads": [
            round(value, 6) for value in frame_percent_std.tolist()
        ],
    }
    payload.update(_direction_statistics(frame_probabilities, query_frame))
    serialized = json.dumps(payload, separators=(",", ":"))
    logging.info("ATTENTION_PROBE %s", serialized)

    output_path = probe.get("output_path")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as output_file:
            output_file.write(serialized + "\n")


@torch.amp.autocast('cuda', enabled=False)
def rope_apply_3d(x, grid_sizes, freqs, temporal_position_maps=None):
    """Apply Wan 3D RoPE, optionally remapping temporal positions per sample."""
    num_heads, complex_dim = x.size(2), x.size(3) // 2
    temporal_dim = complex_dim - 2 * (complex_dim // 3)
    temporal_freqs, height_freqs, width_freqs = freqs.split(
        [temporal_dim, complex_dim // 3, complex_dim // 3], dim=1)

    output = []
    for batch_idx, (frames, height, width) in enumerate(grid_sizes.tolist()):
        seq_len = frames * height * width
        x_sample = torch.view_as_complex(
            x[batch_idx, :seq_len].to(torch.float64).reshape(
                seq_len, num_heads, -1, 2))

        if temporal_position_maps is None:
            sample_temporal_freqs = temporal_freqs[:frames]
        else:
            positions = temporal_position_maps[batch_idx]
            if len(positions) != frames:
                raise ValueError(
                    "temporal RoPE position map must match frame count")
            positions = torch.as_tensor(
                positions, dtype=torch.float64, device=temporal_freqs.device)
            if torch.all(positions == positions.round()):
                sample_temporal_freqs = temporal_freqs.index_select(
                    0, positions.to(torch.long))
            else:
                # ``temporal_freqs[p]`` is exp(i * p * omega). The base
                # frequencies are below pi, so the phase at p=1 recovers
                # omega without wrapping and supports continuous positions.
                base_phase = torch.angle(temporal_freqs[1])
                phase = positions[:, None] * base_phase[None, :]
                sample_temporal_freqs = torch.polar(
                    torch.ones_like(phase), phase)

        multipliers = torch.cat([
            sample_temporal_freqs.view(frames, 1, 1, -1).expand(
                frames, height, width, -1),
            height_freqs[:height].view(1, height, 1, -1).expand(
                frames, height, width, -1),
            width_freqs[:width].view(1, 1, width, -1).expand(
                frames, height, width, -1),
        ], dim=-1).reshape(seq_len, 1, -1)

        x_sample = torch.view_as_real(x_sample * multipliers).flatten(2)
        x_sample = torch.cat([x_sample, x[batch_idx, seq_len:]])
        output.append(x_sample)

    # Multi-image inference currently uses batch size one. Avoid materializing
    # a redundant full-size float64 batch copy before converting the rotated
    # activations back to float32. ``unsqueeze`` is a view, and for one sample
    # this is numerically identical to ``torch.stack(output).float()``.
    if len(output) == 1:
        return output[0].float().unsqueeze(0)
    return torch.stack(output).float()


def _shift_temporal_rope(x, position_delta, freqs):
    """Shift only the temporal RoPE phase of an already rotated tensor.

    ``x`` may contain any number of spatial tokens from one temporal frame.
    Spatial RoPE is left unchanged. This is used for a temporary anchor's
    query-specific integer coordinate without materializing a second rotated
    full-sequence Q/K tensor.
    """
    if position_delta == 0:
        return x
    complex_dim = x.size(-1) // 2
    temporal_dim = complex_dim - 2 * (complex_dim // 3)
    temporal_freqs = freqs[:, :temporal_dim]
    base_phase = torch.angle(temporal_freqs[1]).to(x.device)
    multiplier = torch.polar(
        torch.ones_like(base_phase),
        float(position_delta) * base_phase,
    ).view(*([1] * (x.ndim - 1)), temporal_dim)
    x_complex = torch.view_as_complex(
        x.to(torch.float64).reshape(*x.shape[:-1], -1, 2))
    shifted_temporal = x_complex[..., :temporal_dim] * multiplier
    shifted = torch.cat(
        [shifted_temporal, x_complex[..., temporal_dim:]], dim=-1)
    return torch.view_as_real(shifted).flatten(-2).float()


def segmented_flash_attention(
    q,
    k,
    v,
    seq_lens,
    grid_sizes,
    conditioned_frame_indices,
    sacrificial_frame_indices,
    freqs,
    frame_attention_mode=None,
    local_anchor_indices=None,
    anchor_block_radius=0,
    attention_probe=None,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
):
    """Apply frame-dependent visibility for image-conditioned attention.

    Queries are divided into anchor-started temporal intervals. Ordinary
    queries see all ordinary frames and their permitted image anchors. Every
    query interval uses forward temporal RoPE. Retained frames use the compact
    post-removal timeline, while an isolated removable anchor occupies the
    midpoint between its retained neighbors. Each interval gets a separate
    FlashAttention call.
    """
    if window_size != (-1, -1):
        raise ValueError(
            "Segmented image conditioning requires global self-attention")

    if frame_attention_mode == "compressed_global":
        if sacrificial_frame_indices is None:
            raise ValueError(
                "compressed-global attention requires temporary anchors")
        padded_len = q.size(1)
        outputs = []
        for batch_idx, (frames, height, width) in enumerate(
                grid_sizes.tolist()):
            tokens_per_frame = height * width
            seq_len = frames * tokens_per_frame
            if seq_len != int(seq_lens[batch_idx]):
                raise ValueError(
                    "grid size and sequence length disagree for compressed "
                    "global attention")
            anchors = tuple(sacrificial_frame_indices[batch_idx])
            positions = build_compressed_position_map(frames, anchors)
            sample_grid = grid_sizes[batch_idx:batch_idx + 1]
            query = rope_apply_3d(
                q[batch_idx:batch_idx + 1], sample_grid, freqs,
                temporal_position_maps=[positions])[:, :seq_len]
            key = rope_apply_3d(
                k[batch_idx:batch_idx + 1], sample_grid, freqs,
                temporal_position_maps=[positions])[:, :seq_len]
            value = v[batch_idx:batch_idx + 1, :seq_len]
            keep = torch.ones(seq_len, dtype=torch.bool, device=k.device)
            for anchor in anchors:
                start = anchor * tokens_per_frame
                keep[start:start + tokens_per_frame] = False
            sample = flash_attention(
                q=query,
                k=key[:, keep],
                v=value[:, keep],
                window_size=window_size,
                dtype=dtype)
            if seq_len < padded_len:
                sample = torch.cat([
                    sample,
                    sample.new_zeros(
                        1, padded_len - seq_len, *sample.shape[2:])
                ], dim=1)
            outputs.append(sample)
        return torch.cat(outputs, dim=0)

    if frame_attention_mode == "local_anchor":
        if local_anchor_indices is None:
            raise ValueError("local-anchor attention requires anchor indices")
        if sacrificial_frame_indices is None:
            raise ValueError(
                "local-anchor attention requires temporary anchors")
        padded_len = q.size(1)
        rotated_q = rope_apply_3d(q, grid_sizes, freqs)
        rotated_k = rope_apply_3d(k, grid_sizes, freqs)
        outputs = []
        for batch_idx, (frames, height, width) in enumerate(
                grid_sizes.tolist()):
            tokens_per_frame = height * width
            seq_len = frames * tokens_per_frame
            if seq_len != int(seq_lens[batch_idx]):
                raise ValueError(
                    "grid size and sequence length disagree for local-anchor "
                    "attention")
            sample_anchors = local_anchor_indices[batch_idx]
            if len(sample_anchors) != 1:
                raise ValueError(
                    "each local attention view requires exactly one anchor")
            anchor = int(sample_anchors[0])
            if anchor <= 0 or anchor >= frames:
                raise ValueError("local attention anchor is out of range")

            prefix_end = anchor * tokens_per_frame
            prefix = flash_attention(
                q=rotated_q[batch_idx:batch_idx + 1, :prefix_end],
                k=rotated_k[batch_idx:batch_idx + 1, :seq_len],
                v=v[batch_idx:batch_idx + 1, :seq_len],
                window_size=window_size,
                dtype=dtype)

            keep = torch.zeros(seq_len, dtype=torch.bool, device=k.device)
            keep[prefix_end:] = True
            for other_anchor in sacrificial_frame_indices[batch_idx]:
                if other_anchor == anchor:
                    continue
                start = other_anchor * tokens_per_frame
                keep[start:start + tokens_per_frame] = False
            suffix = flash_attention(
                q=rotated_q[batch_idx:batch_idx + 1, prefix_end:seq_len],
                k=rotated_k[batch_idx:batch_idx + 1, :seq_len][:, keep],
                v=v[batch_idx:batch_idx + 1, :seq_len][:, keep],
                window_size=window_size,
                dtype=dtype)
            sample = torch.cat([prefix, suffix], dim=1)
            if seq_len < padded_len:
                sample = torch.cat([
                    sample,
                    sample.new_zeros(
                        1, padded_len - seq_len, *sample.shape[2:])
                ], dim=1)
            outputs.append(sample)
        return torch.cat(outputs, dim=0)

    if frame_attention_mode is not None:
        raise ValueError(f"unknown frame attention mode: {frame_attention_mode}")
    padded_len = q.size(1)
    segment_specs = []
    for batch_idx, (frames, height, width) in enumerate(grid_sizes.tolist()):
        tokens_per_frame = height * width
        seq_len = frames * tokens_per_frame
        if seq_len != int(seq_lens[batch_idx]):
            raise ValueError(
                "grid size and sequence length disagree for segmented attention"
            )

        conditioned = conditioned_frame_indices[batch_idx]
        sacrificial = ()
        if sacrificial_frame_indices is not None:
            sample_sacrificial = sacrificial_frame_indices[batch_idx]
            if sample_sacrificial:
                sacrificial = tuple(
                    sorted(int(frame) for frame in sample_sacrificial))
                if len(sacrificial) != len(set(sacrificial)):
                    raise ValueError(
                        "sacrificial frame indices must be unique")
                if not set(sacrificial).issubset(conditioned):
                    raise ValueError(
                        "sacrificial frames must be image-conditioned frames")
        segments = build_attention_segments(
            frames,
            conditioned,
            anchor_block_radius=anchor_block_radius)
        segment_specs.append((frames, tokens_per_frame, seq_len, conditioned,
                              sacrificial, segments))

    sample_outputs = [[None] * len(spec[-1]) for spec in segment_specs]
    has_sacrificial = any(spec[4] for spec in segment_specs)
    temporal_position_maps = None
    anchor_query_position_maps = None
    if has_sacrificial:
        temporal_position_maps = [
            build_sacrificial_fractional_position_map(frames, sacrificial)
            if sacrificial else tuple(range(frames))
            for frames, _, _, _, sacrificial, _ in segment_specs
        ]
        anchor_query_position_maps = [
            build_anchor_query_positions(frames, sacrificial)
            if sacrificial else {}
            for frames, _, _, _, sacrificial, _ in segment_specs
        ]
    rotated_q = rope_apply_3d(
        q, grid_sizes, freqs,
        temporal_position_maps=temporal_position_maps)
    # The gathered, unrotated query is no longer needed. Release it before
    # rotating K so the two large complex128 RoPE operations do not also
    # overlap with both raw gathered tensors.
    del q
    rotated_k = rope_apply_3d(
        k, grid_sizes, freqs,
        temporal_position_maps=temporal_position_maps)
    del k

    for batch_idx, (frames, tokens_per_frame, seq_len, conditioned,
                    sacrificial, segments) in enumerate(segment_specs):
        for segment_idx, (start, end, segment_anchor,
                          blocked_frames) in enumerate(segments):
            query = rotated_q[
                batch_idx:batch_idx + 1,
                start * tokens_per_frame:end * tokens_per_frame]
            key = rotated_k[batch_idx:batch_idx + 1, :seq_len]
            value = v[batch_idx:batch_idx + 1, :seq_len]
            probe_query_override = None
            probe_key_frame_overrides = None

            if blocked_frames:
                keep = torch.ones(
                    seq_len, dtype=torch.bool, device=rotated_k.device)
                for frame in blocked_frames:
                    frame_start = frame * tokens_per_frame
                    keep[frame_start:frame_start + tokens_per_frame] = False
                key = key[:, keep]
                value = value[:, keep]

            # Ordinary queries use the shared fractional compressed timeline.
            # A temporary-anchor query instead behaves like a local I2V first
            # frame: move only its Q and self-K from the fractional midpoint
            # to the integer coordinate immediately before the first retained
            # frame on its right. Future retained K positions remain on the
            # compact timeline, giving offsets 1, 2, 3, ... from the anchor.
            if (sacrificial and end == start + 1
                    and start in sacrificial):
                fractional_position = temporal_position_maps[batch_idx][start]
                anchor_position = anchor_query_position_maps[batch_idx][start]
                position_delta = anchor_position - fractional_position
                query = _shift_temporal_rope(query, position_delta, freqs)
                # The anchor query hides every earlier frame, so its own K/V
                # frame is the first frame in the filtered key sequence.
                corrected_self_key = _shift_temporal_rope(
                    key[:, :tokens_per_frame], position_delta, freqs)
                key[:, :tokens_per_frame] = corrected_self_key
                probe_query_override = query[0]
                probe_key_frame_overrides = {
                    start: corrected_self_key[0]
                }
                del corrected_self_key

            sample_outputs[batch_idx][segment_idx] = flash_attention(
                q=query,
                k=key,
                v=value,
                window_size=window_size,
                dtype=dtype)

            probe_batch = (None if attention_probe is None else int(
                attention_probe.get("batch_index", 0)))
            probe_frame = (None if attention_probe is None else int(
                attention_probe["query_frame"]))
            if (batch_idx == probe_batch and start <= probe_frame < end):
                log_frame_attention_probe(
                    rotated_q, rotated_k, v, seq_lens, grid_sizes,
                    conditioned_frame_indices, attention_probe,
                    anchor_block_radius=anchor_block_radius,
                    query_frame_override=probe_query_override,
                    key_frame_overrides=probe_key_frame_overrides)

    del rotated_q, rotated_k

    outputs = []
    for batch_idx, (_, _, seq_len, _, _, _) in enumerate(segment_specs):
        if any(output is None for output in sample_outputs[batch_idx]):
            raise RuntimeError("segmented attention did not fill every query")
        sample = torch.cat(sample_outputs[batch_idx], dim=1)

        if seq_len < padded_len:
            sample = torch.cat([
                sample,
                sample.new_zeros(1, padded_len - seq_len, *sample.shape[2:])
            ], dim=1)
        outputs.append(sample)

    return torch.cat(outputs, dim=0)


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
