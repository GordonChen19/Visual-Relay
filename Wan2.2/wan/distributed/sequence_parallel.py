# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.cuda.amp as amp

from ..modules.model import sinusoidal_embedding_1d
from .ulysses import distributed_attention
from .util import gather_forward, get_rank, get_world_size


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        sp_size = get_world_size()
        sp_rank = get_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


def sp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
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
    self_attention_guidance_scale=None,
    self_attention_guidance_img2_weight=None,
    self_attention_guidance_rms_cap=None,
    self_attention_guidance_anchor_prior=None,
    self_attention_guidance_joint_query_single_memory_alpha=None,
    self_attention_guidance_selective_suffix_memory_alpha=None,
    attention_guidance_anchor_index=None,
    temporal_prompt_relay=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == 'i2v':
        assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    tokenwise_timesteps = t.dim() != 1
    with torch.amp.autocast('cuda', dtype=torch.float32):
        if not tokenwise_timesteps:
            # The timestep is constant across every token in a sample. Keep a
            # singleton token dimension and broadcast it after sequence
            # sharding instead of constructing a multi-GiB repeated tensor.
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())[:, None]
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        else:
            bt, token_count = t.shape
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(
                    self.freq_dim, t).unflatten(
                        0, (bt, token_count)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    if temporal_prompt_relay is not None:
        temporal_prompt_relay = dict(temporal_prompt_relay)
        temporal_prompt_relay['query_token_offset'] = (
            get_rank() * x.size(1))
    if tokenwise_timesteps:
        e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
        e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
        paired_anchor_index=paired_anchor_index,
        paired_stream_weight=paired_stream_weight,
        paired_prefix_anchor_active=paired_prefix_anchor_active,
        paired_prefix_future_eta=paired_prefix_future_eta,
        one_way_img2_anchor_index=one_way_img2_anchor_index,
        one_way_img2_weight=one_way_img2_weight,
        one_way_prefix_reads_aux_anchor=one_way_prefix_reads_aux_anchor,
        one_way_aux_anchor_prior=one_way_aux_anchor_prior,
        one_way_guided_frames=one_way_guided_frames,
        one_way_shared_anchor_update=one_way_shared_anchor_update,
        joint_img2_guidance_anchor_index=joint_img2_guidance_anchor_index,
        joint_img2_guidance_weight=joint_img2_guidance_weight,
        joint_img2_guidance_rope_compression=(
            joint_img2_guidance_rope_compression),
        joint_img2_guidance_adaptive_pair_attention=(
            joint_img2_guidance_adaptive_pair_attention),
        joint_img2_guidance_adaptive_pair_margin=(
            joint_img2_guidance_adaptive_pair_margin),
        joint_img2_guidance_adaptive_pair_alpha_max=(
            joint_img2_guidance_adaptive_pair_alpha_max),
        joint_img2_guidance_compression_ratio=(
            joint_img2_guidance_compression_ratio),
        joint_img2_guidance_anchor_rope_offset=(
            joint_img2_guidance_anchor_rope_offset),
        joint_img2_guidance_velocity_only=joint_img2_guidance_velocity_only,
        cross_tail_bridge_anchor_index=cross_tail_bridge_anchor_index,
        cross_tail_bridge_tail_frames=cross_tail_bridge_tail_frames,
        cross_tail_bridge_memory_prior=cross_tail_bridge_memory_prior,
        cross_tail_bridge_control_memory_mass=(
            cross_tail_bridge_control_memory_mass),
        self_attention_guidance_scale=self_attention_guidance_scale,
        self_attention_guidance_img2_weight=(
            self_attention_guidance_img2_weight),
        self_attention_guidance_rms_cap=self_attention_guidance_rms_cap,
        self_attention_guidance_anchor_prior=(
            self_attention_guidance_anchor_prior),
        self_attention_guidance_joint_query_single_memory_alpha=(
            self_attention_guidance_joint_query_single_memory_alpha),
        self_attention_guidance_selective_suffix_memory_alpha=(
            self_attention_guidance_selective_suffix_memory_alpha),
        attention_guidance_anchor_index=attention_guidance_anchor_index,
        temporal_prompt_relay=temporal_prompt_relay)

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = gather_forward(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def sp_attn_forward(self,
                    x,
                    seq_lens,
                    grid_sizes,
                    freqs,
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
                    dtype=torch.bfloat16):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)

    x = distributed_attention(
        half(q),
        half(k),
        half(v),
        seq_lens,
        window_size=self.window_size,
        grid_sizes=grid_sizes,
        freqs=freqs,
        paired_anchor_index=paired_anchor_index,
        paired_stream_weight=paired_stream_weight,
        paired_prefix_anchor_active=paired_prefix_anchor_active,
        paired_prefix_future_eta=paired_prefix_future_eta,
        one_way_img2_anchor_index=one_way_img2_anchor_index,
        one_way_img2_weight=one_way_img2_weight,
        one_way_prefix_reads_aux_anchor=one_way_prefix_reads_aux_anchor,
        one_way_aux_anchor_prior=one_way_aux_anchor_prior,
        one_way_guided_frames=one_way_guided_frames,
        one_way_shared_anchor_update=one_way_shared_anchor_update,
        joint_img2_guidance_anchor_index=joint_img2_guidance_anchor_index,
        joint_img2_guidance_weight=joint_img2_guidance_weight,
        joint_img2_guidance_rope_compression=(
            joint_img2_guidance_rope_compression),
        joint_img2_guidance_adaptive_pair_attention=(
            joint_img2_guidance_adaptive_pair_attention),
        joint_img2_guidance_adaptive_pair_margin=(
            joint_img2_guidance_adaptive_pair_margin),
        joint_img2_guidance_adaptive_pair_alpha_max=(
            joint_img2_guidance_adaptive_pair_alpha_max),
        joint_img2_guidance_compression_ratio=(
            joint_img2_guidance_compression_ratio),
        joint_img2_guidance_anchor_rope_offset=(
            joint_img2_guidance_anchor_rope_offset),
        joint_img2_guidance_velocity_only=joint_img2_guidance_velocity_only,
        cross_tail_bridge_anchor_index=cross_tail_bridge_anchor_index,
        cross_tail_bridge_tail_frames=cross_tail_bridge_tail_frames,
        cross_tail_bridge_memory_prior=cross_tail_bridge_memory_prior,
        cross_tail_bridge_control_memory_mass=(
            cross_tail_bridge_control_memory_mass),
        attention_guidance_anchor_index=attention_guidance_anchor_index,
        self_attention_guidance_scale=self_attention_guidance_scale,
        self_attention_guidance_img2_weight=(
            self_attention_guidance_img2_weight),
        self_attention_guidance_rms_cap=self_attention_guidance_rms_cap,
        self_attention_guidance_anchor_prior=(
            self_attention_guidance_anchor_prior),
        self_attention_guidance_joint_query_single_memory_alpha=(
            self_attention_guidance_joint_query_single_memory_alpha),
        self_attention_guidance_selective_suffix_memory_alpha=(
            self_attention_guidance_selective_suffix_memory_alpha),
    )

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x
