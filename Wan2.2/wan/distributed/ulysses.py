# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.distributed as dist

from ..modules.attention import (flash_attention, paired_anchor_attention,
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
        attention_guidance_anchor_index=None,
        self_attention_guidance_scale=0.5,
        self_attention_guidance_img2_weight=0.25,
        self_attention_guidance_rms_cap=None,
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

    # apply attention
    if paired_anchor_index is not None:
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
