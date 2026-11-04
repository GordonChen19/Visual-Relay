# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.distributed as dist

from ..modules.attention import (
    ATTN_PROBE,
    flash_attention,
    override_attention,
    paired_anchor_attention,
    probe_plain_self_attention,
)
from .util import all_to_all, get_rank


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
        paired_variant=None,
        override=None,
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

    # After the all-to-all this rank holds the full sequence for one
    # contiguous block of heads (rank r: heads [r * H_local, (r + 1) * H_local)).
    probe_head_offset = get_rank() * q.size(2)

    # apply attention
    if override is not None:
        if grid_sizes is None:
            raise ValueError("override attention requires grid sizes")
        x = override_attention(
            q,
            k,
            v,
            seq_lens,
            grid_sizes,
            override,
            window_size=window_size,
            probe_head_offset=probe_head_offset,
            probe_sp=True,
        )
    elif paired_anchor_index is None:
        x = flash_attention(
            q,
            k,
            v,
            k_lens=seq_lens,
            window_size=window_size,
        )
        if ATTN_PROBE.record and grid_sizes is not None:
            probe_plain_self_attention(
                q, k, grid_sizes, head_offset=probe_head_offset, sp=True)
    else:
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
            window_size=window_size,
            variant=paired_variant,
            probe_head_offset=probe_head_offset,
            probe_sp=True,
        )

    # scatter q/k/v sequence
    x = all_to_all(x, scatter_dim=1, gather_dim=2)
    return x
