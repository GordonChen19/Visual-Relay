# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.distributed as dist

from ..modules.attention import (flash_attention,
                                 joint_img2_guidance_attention)
from .util import all_to_all


def distributed_attention(
        q,
        k,
        v,
        seq_lens,
        window_size=(-1, -1),
        grid_sizes=None,
        freqs=None,
        joint_img2_guidance_anchor_index=None,
        joint_img2_guidance_weight=0.25,
        joint_img2_guidance_rope_compression=False,
        joint_img2_guidance_adaptive_pair_attention=False,
        joint_img2_guidance_adaptive_pair_margin=1.05,
        joint_img2_guidance_adaptive_pair_alpha_max=0.7,
        joint_img2_guidance_compression_ratio=1.0,
        joint_img2_guidance_anchor_rope_offset=None,
        joint_img2_guidance_suffix_anchor_transfer=None,
        joint_img2_guidance_velocity_only=False,
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

    # apply attention
    if joint_img2_guidance_anchor_index is not None:
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
            suffix_anchor_transfer=joint_img2_guidance_suffix_anchor_transfer,
            velocity_only=joint_img2_guidance_velocity_only,
            freqs=freqs,
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
