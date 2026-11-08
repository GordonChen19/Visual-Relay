# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.distributed as dist

from ..modules.attention import (cross_tail_bridge_attention,
                                 cross_tail_bridge_rope,
                                 duplicate_anchor_rope, flash_attention,
                                 joint_img2_guidance_attention,
                                 one_way_img2_attention,
                                 paired_anchor_attention,
                                 self_attention_guidance_attention,
                                 temporary_anchor_rope)
from .util import all_to_all


def distributed_attention(
        q,
        k,
        v,
        seq_lens,
        window_size=(-1, -1),
        grid_sizes=None,
        freqs=None,
        paired_anchor_index=None,
        paired_stream_weight=0.25,
        paired_prefix_anchor_active=True,
        paired_prefix_future_eta=0.0,
        one_way_img2_anchor_index=None,
        one_way_img2_weight=0.25,
        one_way_prefix_reads_aux_anchor=True,
        one_way_aux_anchor_prior=0.5,
        one_way_guided_frames=None,
        one_way_shared_anchor_update=False,
        joint_img2_guidance_anchor_index=None,
        joint_img2_guidance_weight=0.25,
        joint_img2_guidance_rope_compression=False,
        joint_img2_guidance_adaptive_pair_attention=False,
        joint_img2_guidance_adaptive_pair_margin=1.05,
        joint_img2_guidance_adaptive_pair_alpha_max=0.7,
        joint_img2_guidance_compression_ratio=1.0,
        joint_img2_guidance_anchor_rope_offset=None,
        joint_img2_guidance_velocity_only=False,
        cross_tail_bridge_anchor_index=None,
        cross_tail_bridge_tail_frames=4,
        cross_tail_bridge_memory_prior=0.2,
        cross_tail_bridge_control_memory_mass=True,
        attention_guidance_anchor_index=None,
        self_attention_guidance_scale=0.5,
        self_attention_guidance_img2_weight=0.25,
        self_attention_guidance_rms_cap=None,
        self_attention_guidance_anchor_prior=None,
        self_attention_guidance_joint_query_single_memory_alpha=None,
        self_attention_guidance_selective_suffix_memory_alpha=None,
):
    """
    Performs distributed attention based on DeepSpeed Ulysses attention mechanism.
    please refer to https://arxiv.org/pdf/2309.14509

    Args:
        q:           [B, Lq // p, Nq, C1].
        k:           [B, Lk // p, Nk, C1].
        v:           [B, Lk // p, Nk, C2]. Nq must be divisible by Nk.
        seq_lens:    [B], length of each sequence in batch
        window_size: (left right). If not (-1, -1), apply sliding window local attention.
    """
    if not dist.is_initialized():
        raise ValueError("distributed group should be initialized.")
    b = q.shape[0]

    # gather q/k/v sequence
    q = all_to_all(q, scatter_dim=2, gather_dim=1)
    k = all_to_all(k, scatter_dim=2, gather_dim=1)
    v = all_to_all(v, scatter_dim=2, gather_dim=1)

    if one_way_img2_anchor_index is not None:
        if grid_sizes is None or freqs is None:
            raise ValueError(
                "one-way Img2 RoPE requires grid sizes and frequencies")
        q = torch.cat([
            duplicate_anchor_rope(
                q[:1], grid_sizes[:1], freqs,
                one_way_img2_anchor_index), q[1:]
        ], dim=0)
        k = torch.cat([
            duplicate_anchor_rope(
                k[:1], grid_sizes[:1], freqs,
                one_way_img2_anchor_index), k[1:]
        ], dim=0)

    if attention_guidance_anchor_index is not None:
        if grid_sizes is None or freqs is None:
            raise ValueError(
                "attention-guidance RoPE requires grid sizes and frequencies")
        # Expanded M/J use compressed retained-frame coordinates. Local B
        # keeps ordinary first-frame RoPE.
        expanded_count = 2 if q.size(0) >= 3 else 1
        q = torch.cat([
            temporary_anchor_rope(
                q[:expanded_count], grid_sizes[:expanded_count], freqs,
                attention_guidance_anchor_index), q[expanded_count:]
        ], dim=0)
        k = torch.cat([
            temporary_anchor_rope(
                k[:expanded_count], grid_sizes[:expanded_count], freqs,
                attention_guidance_anchor_index), k[expanded_count:]
        ], dim=0)

    if cross_tail_bridge_anchor_index is not None:
        if grid_sizes is None or freqs is None:
            raise ValueError(
                "cross-tail bridge RoPE requires grid sizes and frequencies")
        q = cross_tail_bridge_rope(
            q, grid_sizes, freqs, cross_tail_bridge_anchor_index)
        k = cross_tail_bridge_rope(
            k, grid_sizes, freqs, cross_tail_bridge_anchor_index)

    # apply attention
    if cross_tail_bridge_anchor_index is not None:
        if grid_sizes is None:
            raise ValueError(
                "cross-tail distributed attention requires grid sizes")
        x = cross_tail_bridge_attention(
            q=q,
            k=k,
            v=v,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            tail_frames=cross_tail_bridge_tail_frames,
            cross_memory_prior=cross_tail_bridge_memory_prior,
            control_memory_mass=cross_tail_bridge_control_memory_mass,
            window_size=window_size,
        )
    elif joint_img2_guidance_anchor_index is not None:
        if grid_sizes is None:
            raise ValueError(
                "joint Img2 distributed attention requires grid sizes")
        x = joint_img2_guidance_attention(
            q=q,
            k=k,
            v=v,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            anchor_index=joint_img2_guidance_anchor_index,
            img2_weight=joint_img2_guidance_weight,
            rope_compression=joint_img2_guidance_rope_compression,
            adaptive_pair_attention=(
                joint_img2_guidance_adaptive_pair_attention),
            adaptive_pair_margin=joint_img2_guidance_adaptive_pair_margin,
            adaptive_pair_alpha_max=(
                joint_img2_guidance_adaptive_pair_alpha_max),
            compression_ratio=joint_img2_guidance_compression_ratio,
            anchor_rope_offset=joint_img2_guidance_anchor_rope_offset,
            velocity_only=joint_img2_guidance_velocity_only,
            freqs=freqs,
            window_size=window_size,
        )
    elif one_way_img2_anchor_index is not None:
        if grid_sizes is None:
            raise ValueError(
                "one-way Img2 distributed attention requires grid sizes")
        x = one_way_img2_attention(
            q=q,
            k=k,
            v=v,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            anchor_index=one_way_img2_anchor_index,
            img2_weight=one_way_img2_weight,
            prefix_reads_aux_anchor=one_way_prefix_reads_aux_anchor,
            aux_anchor_prior=one_way_aux_anchor_prior,
            guided_frames=one_way_guided_frames,
            shared_anchor_update=one_way_shared_anchor_update,
            window_size=window_size,
        )
    elif paired_anchor_index is not None:
        if grid_sizes is None:
            raise ValueError(
                "paired distributed attention requires grid sizes")
        x = paired_anchor_attention(
            q=q,
            k=k,
            v=v,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            anchor_index=paired_anchor_index,
            stream_weight=paired_stream_weight,
            prefix_anchor_active=paired_prefix_anchor_active,
            prefix_future_eta=paired_prefix_future_eta,
            window_size=window_size,
        )
    elif attention_guidance_anchor_index is not None:
        x = self_attention_guidance_attention(
            q=q,
            k=k,
            v=v,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            anchor_index=attention_guidance_anchor_index,
            joint_weight=self_attention_guidance_scale,
            img2_weight=self_attention_guidance_img2_weight,
            joint_rms_cap=self_attention_guidance_rms_cap,
            anchor_prior=self_attention_guidance_anchor_prior,
            joint_query_single_memory_alpha=(
                self_attention_guidance_joint_query_single_memory_alpha),
            selective_suffix_memory_alpha=(
                self_attention_guidance_selective_suffix_memory_alpha),
            window_size=window_size,
        )
    else:
        x = flash_attention(
            q,
            k,
            v,
            k_lens=seq_lens,
            window_size=window_size,
        )

    # scatter q/k/v sequence
    x = all_to_all(x, scatter_dim=1, gather_dim=2)
    return x
