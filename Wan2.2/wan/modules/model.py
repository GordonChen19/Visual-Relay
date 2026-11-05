# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import (cross_tail_bridge_attention, cross_tail_bridge_rope,
                        duplicate_anchor_rope, flash_attention,
                        joint_img2_guidance_attention, one_way_img2_attention,
                        paired_anchor_attention,
                        self_attention_guidance_attention,
                        temporary_anchor_rope)

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self,
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
                self_attention_guidance_selective_suffix_memory_alpha=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        if one_way_img2_anchor_index is not None:
            # M contains [joint auxiliary anchor, ordinary Img1-only anchor].
            # Preserve the native M timeline by assigning both the same
            # integer coordinate. B retains ordinary local Img2-first RoPE.
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
            # Expanded M/J both contain the temporary slot and use compressed
            # retained-frame coordinates. B keeps ordinary local RoPE.
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
            q = cross_tail_bridge_rope(
                q, grid_sizes, freqs, cross_tail_bridge_anchor_index)
            k = cross_tail_bridge_rope(
                k, grid_sizes, freqs, cross_tail_bridge_anchor_index)
        if cross_tail_bridge_anchor_index is not None:
            x = cross_tail_bridge_attention(
                q=q,
                k=k,
                v=v,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                tail_frames=cross_tail_bridge_tail_frames,
                cross_memory_prior=cross_tail_bridge_memory_prior,
                control_memory_mass=cross_tail_bridge_control_memory_mass,
                window_size=self.window_size)
        elif joint_img2_guidance_anchor_index is not None:
            x = joint_img2_guidance_attention(
                q=q,
                k=k,
                v=v,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                anchor_index=joint_img2_guidance_anchor_index,
                img2_weight=joint_img2_guidance_weight,
                rope_compression=joint_img2_guidance_rope_compression,
                freqs=freqs,
                window_size=self.window_size)
        elif one_way_img2_anchor_index is not None:
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
                window_size=self.window_size)
        elif paired_anchor_index is not None:
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
                window_size=self.window_size)
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
                window_size=self.window_size)
        else:
            x = flash_attention(
                q=q,
                k=k,
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    @staticmethod
    def _relay_query_frames(query_indices, relay_config):
        """Map flattened video-token indices to retained latent-frame indices."""
        tokens_per_frame = int(relay_config['tokens_per_frame'])
        frames = torch.div(
            query_indices, tokens_per_frame, rounding_mode='floor')
        duplicate_index = relay_config.get('duplicate_frame_index')
        if duplicate_index is not None:
            duplicate_index = int(duplicate_index)
            # Expanded one-way M is [..., A10, M10, M11, ...]. A10 and M10
            # both represent retained frame 10; later expanded positions are
            # shifted back by one retained frame.
            frames = torch.where(frames > duplicate_index, frames - 1,
                                 frames)
        return frames.float()

    def _temporal_relay_attention(self, q, k, v, relay_config,
                                  chunk_size=16):
        """Cross-attention with Prompt-Relay costs on local-prompt tokens."""
        batch, query_len, heads, dim = q.shape
        if batch != 1:
            raise ValueError("temporal relay attention expects one stream")
        key_len = k.size(1)
        scale = dim**-0.5
        query_offset = int(relay_config.get('query_token_offset', 0))
        segments = relay_config.get('segments', ())
        output = torch.empty_like(q)

        q_heads = q.transpose(1, 2)
        k_heads = k.transpose(1, 2)
        v_heads = v.transpose(1, 2)
        k_transposed = k_heads.transpose(-2, -1)
        for start in range(0, query_len, chunk_size):
            end = min(start + chunk_size, query_len)
            logits = torch.matmul(
                q_heads[:, :, start:end].float(),
                k_transposed.float()) * scale
            query_indices = torch.arange(
                query_offset + start,
                query_offset + end,
                device=q.device,
                dtype=torch.long)
            query_frames = self._relay_query_frames(query_indices,
                                                     relay_config)
            for segment in segments:
                token_start = int(segment['token_start'])
                token_end = min(int(segment['token_end']), key_len)
                if token_start >= token_end:
                    continue
                midpoint = float(segment['midpoint'])
                window = float(segment['window'])
                sigma = float(segment['sigma'])
                distance = (query_frames - midpoint).abs()
                cost = torch.relu(distance - window).square().div_(
                    2.0 * sigma * sigma)
                logits[:, :, :, token_start:token_end].sub_(
                    cost[None, None, :, None])
            probabilities = torch.softmax(logits, dim=-1)
            chunk = torch.matmul(probabilities, v_heads.float())
            output[:, start:end].copy_(
                chunk.transpose(1, 2).to(dtype=output.dtype))
        return output

    def forward(self,
                x,
                context,
                context_lens,
                temporal_prompt_relay=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # Compute ordinary cross-attention for every stream except the single
        # M stream selected by the optional Prompt-Relay configuration.
        if temporal_prompt_relay is None:
            x = flash_attention(q, k, v, k_lens=context_lens)
        else:
            stream_index = int(temporal_prompt_relay.get('stream_index', 0))
            if not 0 <= stream_index < b:
                raise ValueError("temporal relay stream index is out of range")
            outputs = []
            for index in range(b):
                if index == stream_index:
                    outputs.append(self._temporal_relay_attention(
                        q[index:index + 1], k[index:index + 1],
                        v[index:index + 1], temporal_prompt_relay))
                else:
                    outputs.append(flash_attention(
                        q[index:index + 1],
                        k[index:index + 1],
                        v[index:index + 1],
                        k_lens=None))
            x = torch.cat(outputs, dim=0)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
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
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            paired_anchor_index=paired_anchor_index,
            paired_stream_weight=paired_stream_weight,
            paired_prefix_anchor_active=paired_prefix_anchor_active,
            paired_prefix_future_eta=paired_prefix_future_eta,
            one_way_img2_anchor_index=one_way_img2_anchor_index,
            one_way_img2_weight=one_way_img2_weight,
            one_way_prefix_reads_aux_anchor=(
                one_way_prefix_reads_aux_anchor),
            one_way_aux_anchor_prior=one_way_aux_anchor_prior,
            one_way_guided_frames=one_way_guided_frames,
            one_way_shared_anchor_update=one_way_shared_anchor_update,
            joint_img2_guidance_anchor_index=(
                joint_img2_guidance_anchor_index),
            joint_img2_guidance_weight=joint_img2_guidance_weight,
            joint_img2_guidance_rope_compression=(
                joint_img2_guidance_rope_compression),
            cross_tail_bridge_anchor_index=cross_tail_bridge_anchor_index,
            cross_tail_bridge_tail_frames=cross_tail_bridge_tail_frames,
            cross_tail_bridge_memory_prior=cross_tail_bridge_memory_prior,
            cross_tail_bridge_control_memory_mass=(
                cross_tail_bridge_control_memory_mass),
            attention_guidance_anchor_index=attention_guidance_anchor_index,
            self_attention_guidance_scale=(
                0.5 if self_attention_guidance_scale is None else
                self_attention_guidance_scale),
            self_attention_guidance_img2_weight=(
                0.25 if self_attention_guidance_img2_weight is None else
                self_attention_guidance_img2_weight),
            self_attention_guidance_rms_cap=(
                self_attention_guidance_rms_cap),
            self_attention_guidance_anchor_prior=(
                self_attention_guidance_anchor_prior),
            self_attention_guidance_joint_query_single_memory_alpha=(
                self_attention_guidance_joint_query_single_memory_alpha),
            self_attention_guidance_selective_suffix_memory_alpha=(
                self_attention_guidance_selective_suffix_memory_alpha))
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(
                self.norm3(x),
                context,
                context_lens,
                temporal_prompt_relay=temporal_prompt_relay)
            y = self.ffn(
                self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        # initialize weights
        self.init_weights()

    def forward(
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
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
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
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with torch.amp.autocast('cuda', dtype=torch.float32):
            if t.dim() == 1:
                # Every token in a sample has the same diffusion timestep.
                # Compute its modulation once and rely on broadcasting in the
                # transformer blocks and output head. Expanding before the two
                # MLPs would materialize B * seq_len * 6 * dim float32 values,
                # which is several GiB for multi-stream 480p inference.
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t).float())[:, None]
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            else:
                # Preserve support for callers that intentionally provide a
                # distinct timestep for every token.
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
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

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
            one_way_prefix_reads_aux_anchor=(
                one_way_prefix_reads_aux_anchor),
            one_way_aux_anchor_prior=one_way_aux_anchor_prior,
            one_way_guided_frames=one_way_guided_frames,
            one_way_shared_anchor_update=one_way_shared_anchor_update,
            joint_img2_guidance_anchor_index=(
                joint_img2_guidance_anchor_index),
            joint_img2_guidance_weight=joint_img2_guidance_weight,
            joint_img2_guidance_rope_compression=(
                joint_img2_guidance_rope_compression),
            cross_tail_bridge_anchor_index=cross_tail_bridge_anchor_index,
            cross_tail_bridge_tail_frames=cross_tail_bridge_tail_frames,
            cross_tail_bridge_memory_prior=cross_tail_bridge_memory_prior,
            cross_tail_bridge_control_memory_mass=(
                cross_tail_bridge_control_memory_mass),
            self_attention_guidance_scale=self_attention_guidance_scale,
            self_attention_guidance_img2_weight=(
                self_attention_guidance_img2_weight),
            self_attention_guidance_rms_cap=(
                self_attention_guidance_rms_cap),
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

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
