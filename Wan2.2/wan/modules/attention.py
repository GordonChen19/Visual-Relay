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
                                   prefix_anchor_visible=True):
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
                for index, frame in enumerate(targets):
                    if frame is None:
                        continue
                    group_lse = torch.logsumexp(
                        logits[..., frame * tokens_per_frame:
                               (frame + 1) * tokens_per_frame], dim=-1)
                    group_sums[index] += (group_lse - normalizer).exp().sum()
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
    'cross_tail_bridge_attention',
    'cross_tail_bridge_rope',
    'one_way_img2_attention',
    'joint_img2_guidance_attention',
    'duplicate_anchor_rope',
    'paired_anchor_attention',
    'self_attention_guidance_attention',
    'temporary_anchor_rope',
    'attention',
]


def _two_source_prior_attention(
    q,
    native_k,
    native_v,
    cross_k,
    cross_v,
    cross_prior,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
):
    """Merge native and cross memories as one length-normalized softmax.

    This is mathematically equivalent to concatenating both memories and
    adding ``log(1-alpha)-log(N_native)`` and
    ``log(alpha)-log(N_cross)`` to their logits.  Equal per-token evidence
    therefore gives the cross bank total mass ``alpha`` regardless of either
    bank's token count.  Separate FlashAttention calls are only an efficient
    way to obtain the two group outputs and log-normalizers without
    materializing the attention matrix.
    """
    if not 0.0 <= cross_prior <= 1.0:
        raise ValueError("cross-memory prior must be in [0, 1]")
    if native_k.size(1) == 0 or cross_k.size(1) == 0:
        raise ValueError("native and cross memories must both be non-empty")
    if cross_prior == 0.0:
        return flash_attention(
            q=q, k=native_k, v=native_v, window_size=window_size,
            dtype=dtype)
    if cross_prior == 1.0:
        return flash_attention(
            q=q, k=cross_k, v=cross_v, window_size=window_size,
            dtype=dtype)

    native_output, native_lse = flash_attention(
        q=q, k=native_k, v=native_v, window_size=window_size,
        dtype=dtype, return_lse=True)
    cross_output, cross_lse = flash_attention(
        q=q, k=cross_k, v=cross_v, window_size=window_size,
        dtype=dtype, return_lse=True)
    native_lse = (native_lse + math.log1p(-float(cross_prior))
                  - math.log(native_k.size(1)))
    cross_lse = (cross_lse + math.log(float(cross_prior))
                 - math.log(cross_k.size(1)))
    group_weights = torch.stack([native_lse, cross_lse], dim=-1).softmax(
        dim=-1).unsqueeze(-1)
    group_outputs = torch.stack(
        [native_output.float(), cross_output.float()], dim=-2)
    return (group_weights * group_outputs).sum(dim=-2).to(native_output.dtype)


def cross_tail_bridge_attention(
    q,
    k,
    v,
    seq_lens,
    grid_sizes,
    tail_frames,
    cross_memory_prior=0.2,
    control_memory_mass=True,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
):
    """Couple M/B windows while evolving an independent Img2-first D stream.

    M ends after ``tail_frames`` shared frames while B begins at the Img2
    anchor and runs to the end of the output. Thus M's final
    ``tail_frames`` frames are temporally aligned with B's first
    ``tail_frames`` frames. ``tail_frames`` is also the safe cross-stream
    distance ``s``.

    Every query reads its complete native stream. A cross-stream key is visible
    when either (1) its globally aligned temporal distance from the query is at
    least ``s`` or (2) its distance from its own first-frame anchor is at least
    ``s``. First-frame anchors may therefore guide sufficiently distant M
    queries, including M0 reading B0 when their global distance is safe. B's
    complete overlap B0:B(s-1) remains native-only. B's non-overlapping tail
    reads complete native B memory plus only M's pre-anchor prefix
    M0:M(anchor-1).

    D performs ordinary native-only bidirectional self-attention and never
    enters M/B memory. Each native-plus-cross route is one prior-weighted
    mathematical softmax; the native-only B overlap uses ordinary B
    self-attention.
    Both outputs are computed from the same input-layer Q/K/V, so the coupling
    is simultaneous rather than sequential within a DiT layer.

    When ``control_memory_mass`` is false, native and cross K/V are directly
    concatenated and passed through an ordinary softmax. The cross-memory
    prior and length normalization are then deliberately disabled.
    """
    if q.size(0) != 3 or k.size(0) != 3 or v.size(0) != 3:
        raise ValueError(
            "cross-tail bridge requires exactly M, B, and D streams")
    if window_size != (-1, -1):
        raise ValueError("cross-tail bridge requires global self-attention")
    if not 0.0 <= cross_memory_prior <= 1.0:
        raise ValueError("cross-tail memory prior must be in [0, 1]")

    m_frames, height, width = (
        int(value) for value in grid_sizes[0].tolist())
    b_frames, b_height, b_width = (
        int(value) for value in grid_sizes[1].tolist())
    d_frames, d_height, d_width = (
        int(value) for value in grid_sizes[2].tolist())
    if ((height, width) != (b_height, b_width)
            or (height, width) != (d_height, d_width)):
        raise ValueError("cross-tail M/B/D streams must share a spatial grid")
    if d_frames != b_frames:
        raise ValueError("cross-tail B and D streams must have equal lengths")
    tail_frames = int(tail_frames)
    if not 1 <= tail_frames < min(m_frames, b_frames):
        raise ValueError(
            "cross-tail overlap must be smaller than both stream lengths")

    tokens_per_frame = height * width
    m_seq_len = m_frames * tokens_per_frame
    b_seq_len = b_frames * tokens_per_frame
    d_seq_len = d_frames * tokens_per_frame
    if (int(seq_lens[0]) != m_seq_len
            or int(seq_lens[1]) != b_seq_len
            or int(seq_lens[2]) != d_seq_len):
        raise ValueError("cross-tail sequence lengths disagree with grids")
    m_anchor_frame = m_frames - tail_frames

    m_q, b_q = q[0:1, :m_seq_len], q[1:2, :b_seq_len]
    m_k, b_k = k[0:1, :m_seq_len], k[1:2, :b_seq_len]
    m_v, b_v = v[0:1, :m_seq_len], v[1:2, :b_seq_len]
    d_q = q[2:3, :d_seq_len]
    d_k = k[2:3, :d_seq_len]
    d_v = v[2:3, :d_seq_len]

    def routed_attention(query, native_k, native_v, cross_k, cross_v):
        if not control_memory_mass:
            return flash_attention(
                q=query,
                k=torch.cat([native_k, cross_k], dim=1),
                v=torch.cat([native_v, cross_v], dim=1),
                window_size=window_size,
                dtype=dtype)
        return _two_source_prior_attention(
            query, native_k, native_v, cross_k, cross_v,
            cross_memory_prior, window_size, dtype)

    def selected_memory(memory, frame_indices):
        return torch.cat([
            memory[:, frame * tokens_per_frame:(frame + 1)
                   * tokens_per_frame]
            for frame in frame_indices
        ], dim=1)

    def safe_cross_frames(query_global_frame, cross_frames,
                          cross_global_start):
        return tuple(
            frame for frame in range(cross_frames)
            if (abs(query_global_frame - (cross_global_start + frame))
                >= tail_frames or frame >= tail_frames))

    def grouped_routed_outputs(query, query_frames, query_global_start,
                               native_k, native_v, cross_k, cross_v,
                               cross_frames, cross_global_start):
        groups = []
        for frame in range(query_frames):
            allowed = safe_cross_frames(
                query_global_start + frame, cross_frames, cross_global_start)
            if groups and groups[-1][2] == allowed:
                groups[-1] = (groups[-1][0], frame + 1, allowed)
            else:
                groups.append((frame, frame + 1, allowed))
        outputs = []
        for frame_start, frame_end, allowed in groups:
            if not allowed:
                raise ValueError(
                    "safe cross-tail routing produced empty cross memory")
            outputs.append(routed_attention(
                query[:, frame_start * tokens_per_frame:
                      frame_end * tokens_per_frame],
                native_k,
                native_v,
                selected_memory(cross_k, allowed),
                selected_memory(cross_v, allowed)))
        return torch.cat(outputs, dim=1)

    m_output = grouped_routed_outputs(
        m_q, m_frames, 0, m_k, m_v, b_k, b_v, b_frames,
        m_anchor_frame)
    b_overlap_tokens = tail_frames * tokens_per_frame
    b_overlap_output = flash_attention(
        q=b_q[:, :b_overlap_tokens],
        k=b_k,
        v=b_v,
        window_size=window_size,
        dtype=dtype)
    b_rest_output = routed_attention(
        b_q[:, b_overlap_tokens:], b_k, b_v,
        m_k[:, :m_anchor_frame * tokens_per_frame],
        m_v[:, :m_anchor_frame * tokens_per_frame])
    b_output = torch.cat([b_overlap_output, b_rest_output], dim=1)
    # D is an independent, ordinary Img2-first I2V stream. It never reads M
    # or B and does not contribute layer-wise memory to either stream; only
    # its final D0 velocity is consumed by the sampling loop.
    d_output = flash_attention(
        q=d_q,
        k=d_k,
        v=d_v,
        window_size=window_size,
        dtype=dtype)

    m_output = torch.cat([
        m_output,
        torch.zeros_like(q[0:1, m_seq_len:])
    ], dim=1)
    b_output = torch.cat([
        b_output,
        torch.zeros_like(q[1:2, b_seq_len:])
    ], dim=1)
    d_output = torch.cat([
        d_output,
        torch.zeros_like(q[2:3, d_seq_len:])
    ], dim=1)
    return torch.cat([m_output, b_output, d_output], dim=0)


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
    freqs=None,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
):
    """Guide a native joint Img1/Img2 stream with an Img2-first suffix.

    Stream 0 is a full, jointly VAE-conditioned timeline. Stream 1 is a
    standard Img2-first I2V timeline aligned to J. By default, both streams use
    ordinary bidirectional attention. Optional shifted-anchor RoPE compression
    gives prefix and suffix queries distinct anchor-key coordinates and limits
    the anchor query to itself/future frames. B always retains ordinary RoPE.
    """
    if q.size(0) != 2 or k.size(0) != 2 or v.size(0) != 2:
        raise ValueError("joint Img2 guidance requires exactly J and B streams")
    if window_size != (-1, -1):
        raise ValueError("joint Img2 guidance requires global self-attention")
    if not 0.0 <= img2_weight <= 1.0:
        raise ValueError("joint Img2 guidance weight must be in [0, 1]")
    if not 0.0 <= compression_ratio <= 1.0:
        raise ValueError("joint RoPE compression ratio must be in [0, 1]")

    frames, height, width = (
        int(value) for value in grid_sizes[0].tolist())
    b_frames, b_height, b_width = (
        int(value) for value in grid_sizes[1].tolist())
    if (b_height, b_width) != (height, width):
        raise ValueError("joint and Img2 streams must share a spatial grid")
    if not 0 < anchor_index < frames:
        raise ValueError("joint Img2 anchor must be inside the timeline")
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
        if not shifted_anchor:
            raise ValueError(
                "joint RoPE compression requires the shifted B alignment")
        if freqs is None:
            raise ValueError("joint RoPE compression requires RoPE frequencies")

        joint_q = q[0:1, :joint_seq_len]
        joint_k = k[0:1, :joint_seq_len]
        joint_v = v[0:1, :joint_seq_len]
        # Prefix queries retain their native coordinates. Compress every
        # post-anchor key left by one so the anchor and first suffix frame
        # share the anchor coordinate: [..., J9@9, J10@10, J11@10, J12@11].
        prefix_k = torch.cat([
            joint_k[:, :anchor_end],
            _shift_temporal_rope(joint_k[:, anchor_end:], -1, freqs),
        ], dim=1)
        # Suffix queries read all J frames using compressed RoPE. Shift both
        # suffix queries and anchor/suffix keys left by one, colocating J9/J10
        # while preserving relative offsets within the suffix. J11 is then
        # one temporal step from both J9 and J10 in either ablation.
        compressed_suffix_q = _shift_temporal_rope(
            joint_q[:, anchor_end:], -1.0, freqs)
        suffix_k = torch.cat([
            joint_k[:, :aligned_start],
            _shift_temporal_rope(
                joint_k[:, aligned_start:], -1.0, freqs),
        ], dim=1)
        _record_joint_attention_shares(
            joint_q, prefix_k, anchor_index, tokens_per_frame,
            compressed_suffix_q, suffix_k)
        # Prefix queries read all J memory, including the anchor and its
        # colocated first suffix neighbour, with no mass correction.
        prefix_output = flash_attention(
            q=joint_q[:, :aligned_start],
            k=prefix_k,
            v=joint_v,
            window_size=window_size,
            dtype=dtype)
        anchor_output = flash_attention(
            q=joint_q[:, aligned_start:anchor_end],
            k=joint_k[:, aligned_start:],
            v=joint_v[:, aligned_start:],
            window_size=window_size,
            dtype=dtype)
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
    if shifted_anchor:
        img2_output = flash_attention(
            q=q[1:2, :img2_seq_len],
            k=k[1:2, :img2_seq_len],
            v=v[1:2, :img2_seq_len],
            window_size=window_size,
            dtype=dtype)
        # B0 aligns with J(anchor-1), making B1 an ordinary four-frame latent
        # at J's anchor. J and B retain independent ordinary bidirectional
        # attention. B1 is consumed only as final anchor velocity.
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


def one_way_img2_attention(
    q,
    k,
    v,
    seq_lens,
    grid_sizes,
    anchor_index,
    img2_weight=0.25,
    prefix_reads_aux_anchor=True,
    aux_anchor_prior=0.5,
    guided_frames=None,
    shared_anchor_update=False,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
):
    """Apply one-way Img2-first attention-output guidance to an Img1 stream.

    Sample zero is an expanded Img1-first stream ``M`` containing an auxiliary
    joint anchor immediately before the ordinary Img1-only anchor frame.
    Sample one is an independently evolving Img2-first stream ``B`` whose
    first latent is aligned with ``M[anchor_index]``.  Both streams perform
    ordinary full bidirectional self-attention over their own tokens.  Only
    the completed attention outputs are coupled:

    Prefix queries read both the internally consistent auxiliary A10 and the
    ordinary retained M10, plus the prefix and retained suffix. A10 and M10
    share one temporal coordinate, so they receive complementary configurable
    priors before their exact group-normalized merge. Their combined evidence
    is therefore exactly one frame's evidence rather than twice that amount.
    A10 uses its own query and attends only to the prefix plus itself.
    Retained M10/suffix queries read only the ordinary 21-frame M memory with
    A10 removed. By default, M10 keeps its own output and B1: guides the whole
    post-anchor suffix. When ``guided_frames`` is supplied, it counts from the
    retained anchor: B0:B(n-1) guide exactly M10:M(10+n-1), and later retained
    frames keep their own M outputs.

    When ``shared_anchor_update`` is enabled, the completed retained-anchor
    output (including any B0 interpolation) is also used as A10's
    self-attention output. A10 and M10 therefore receive the same attention
    increment while their residual states and later sublayers remain distinct.

    ``O_B`` is returned unchanged, so information flows strictly B -> M.
    This function performs no RoPE modification or temporal compression.
    """
    if q.size(0) != 2 or k.size(0) != 2 or v.size(0) != 2:
        raise ValueError("one-way Img2 guidance requires exactly M/B streams")
    if window_size != (-1, -1):
        raise ValueError("one-way Img2 guidance requires global attention")
    if not 0.0 <= img2_weight <= 1.0:
        raise ValueError("one-way Img2 guidance weight must be in [0, 1]")
    if not 0.0 < aux_anchor_prior < 1.0:
        raise ValueError(
            "one-way auxiliary-anchor prior must be strictly between 0 and 1")

    main_frames, height, width = (
        int(value) for value in grid_sizes[0].tolist())
    suffix_frames, suffix_height, suffix_width = (
        int(value) for value in grid_sizes[1].tolist())
    if (height, width) != (suffix_height, suffix_width):
        raise ValueError("M/B spatial grids must match")
    if not 0 < anchor_index < main_frames:
        raise ValueError("one-way Img2 anchor must be inside the M timeline")
    overlap_frames = main_frames - anchor_index - 1
    if suffix_frames < overlap_frames:
        raise ValueError(
            "B must cover the auxiliary anchor and retained M suffix")
    if guided_frames is not None:
        guided_frames = int(guided_frames)
        if not 1 <= guided_frames <= overlap_frames:
            raise ValueError(
                "one-way guided-frame count must fit the retained "
                "anchor/suffix overlap")

    tokens_per_frame = height * width
    main_seq_len = main_frames * tokens_per_frame
    suffix_seq_len = suffix_frames * tokens_per_frame
    overlap_seq_len = overlap_frames * tokens_per_frame
    anchor_start = anchor_index * tokens_per_frame
    anchor_end = anchor_start + tokens_per_frame
    retained_anchor_end = anchor_end + tokens_per_frame
    if (int(seq_lens[0]) != main_seq_len
            or int(seq_lens[1]) != suffix_seq_len):
        raise ValueError("one-way Img2 M/B sequence lengths disagree")

    # The retained anchor and suffix remain an ordinary fully bidirectional
    # Img1 stream: they read every retained M frame but never the auxiliary
    # joint anchor.
    retained_main_k = torch.cat([
        k[0:1, :anchor_start], k[0:1, anchor_end:main_seq_len]
    ], dim=1)
    retained_main_v = torch.cat([
        v[0:1, :anchor_start], v[0:1, anchor_end:main_seq_len]
    ], dim=1)
    if prefix_reads_aux_anchor:
        # Default route: prefix queries read both matched A10 and M10. Since
        # they share one RoPE coordinate, split one position's prior equally
        # so their combined evidence does not receive a factor-of-two bonus.
        prefix_output = _shared_position_prior_attention(
            q=q[0:1, :anchor_start],
            ordinary_k=torch.cat([
                k[0:1, :anchor_start],
                k[0:1, retained_anchor_end:main_seq_len]
            ], dim=1),
            ordinary_v=torch.cat([
                v[0:1, :anchor_start],
                v[0:1, retained_anchor_end:main_seq_len]
            ], dim=1),
            anchor_k=k[0:1, anchor_start:anchor_end],
            anchor_v=v[0:1, anchor_start:anchor_end],
            neighbour_k=k[0:1, anchor_end:retained_anchor_end],
            neighbour_v=v[0:1, anchor_end:retained_anchor_end],
            anchor_prior=aux_anchor_prior,
            window_size=window_size,
            dtype=dtype)
    else:
        # Controlled ablation: prefix queries use the ordinary retained
        # 21-frame M memory, including M10, and cannot read auxiliary A10.
        prefix_output = flash_attention(
            q=q[0:1, :anchor_start],
            k=retained_main_k,
            v=retained_main_v,
            window_size=window_size,
            dtype=dtype)
    retained_anchor_suffix_output = flash_attention(
        q=q[0:1, anchor_end:main_seq_len],
        k=retained_main_k,
        v=retained_main_v,
        window_size=window_size,
        dtype=dtype)
    # Keep the jointly conditioned auxiliary anchor integrated with the
    # prefix throughout the DiT pass. It uses its own query and can read the
    # preceding main-stream prefix plus itself, but not retained M10 or the
    # post-anchor suffix.
    auxiliary_anchor_output = flash_attention(
        q=q[0:1, anchor_start:anchor_end],
        k=k[0:1, :anchor_end],
        v=v[0:1, :anchor_end],
        window_size=window_size,
        dtype=dtype)
    img2_output = flash_attention(
        q=q[1:2, :suffix_seq_len],
        k=k[1:2, :suffix_seq_len],
        v=v[1:2, :suffix_seq_len],
        window_size=window_size,
        dtype=dtype)

    # The legacy route leaves the retained anchor untouched and guides the
    # entire post-anchor suffix with B1:. The localized route instead begins
    # at the retained anchor and applies B0:B(n-1) to exactly n aligned M
    # frames, leaving every later retained frame unchanged.
    retained_anchor_output = retained_anchor_suffix_output[
        :, :tokens_per_frame]
    retained_suffix_output = retained_anchor_suffix_output[:,
                                                            tokens_per_frame:]
    img2_suffix_output = img2_output[:,:overlap_seq_len][:,
                                                        tokens_per_frame:]
    if retained_suffix_output.size(1) != img2_suffix_output.size(1):
        raise RuntimeError("one-way M/B post-anchor suffixes do not align")
    if guided_frames is None:
        guided_retained_anchor = retained_anchor_output
        guided_retained_suffix = (
            (1.0 - img2_weight) * retained_suffix_output
            + img2_weight * img2_suffix_output)
    else:
        guided_seq_len = guided_frames * tokens_per_frame
        guided_aligned_output = (
            (1.0 - img2_weight)
            * retained_anchor_suffix_output[:, :guided_seq_len]
            + img2_weight * img2_output[:, :guided_seq_len])
        guided_retained_anchor = guided_aligned_output[:, :tokens_per_frame]
        guided_retained_suffix = torch.cat([
            guided_aligned_output[:, tokens_per_frame:],
            retained_anchor_suffix_output[:, guided_seq_len:]
        ], dim=1)
    if shared_anchor_update:
        auxiliary_anchor_output = guided_retained_anchor
    guided_main = torch.cat([
        prefix_output,
        auxiliary_anchor_output,
        guided_retained_anchor,
        guided_retained_suffix,
        torch.zeros_like(q[0:1, main_seq_len:]),
    ], dim=1)
    padded_img2 = torch.cat([
        img2_output,
        torch.zeros_like(q[1:2, suffix_seq_len:]),
    ], dim=1)
    return torch.cat([guided_main, padded_img2], dim=0)


def duplicate_anchor_rope(x, grid_sizes, freqs, anchor_index):
    """Give an auxiliary anchor and retained anchor the same temporal index.

    ``x`` has already received ordinary RoPE for an expanded timeline ordered
    as ``[..., auxiliary_anchor, retained_anchor, retained_suffix...]``. The
    auxiliary anchor keeps its integer position and every later frame shifts
    left by one. The retained 21-frame timeline therefore preserves its normal
    coordinates while the auxiliary anchor shares the retained anchor's
    temporal position. No frame is placed at a fractional coordinate.
    """
    if anchor_index is None:
        return x
    output = []
    for batch_index, (frames, height, width) in enumerate(
            grid_sizes.tolist()):
        if not 0 < anchor_index < frames - 1:
            raise ValueError(
                "duplicate anchor must have retained frames on both sides")
        tokens_per_frame = height * width
        anchor_end = (anchor_index + 1) * tokens_per_frame
        sequence_end = frames * tokens_per_frame
        sample = x[batch_index:batch_index + 1]
        output.append(torch.cat([
            sample[:, :anchor_end],
            _shift_temporal_rope(sample[:, anchor_end:sequence_end], -1,
                                 freqs),
            sample[:, sequence_end:],
        ], dim=1))
    return torch.cat(output, dim=0)


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


def _legacy_paired_anchor_attention(
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
    """Update retained C queries from mass-normalized C/B memory.

    B is standard Img2-first I2V. C is standard Img1-first I2V with a hidden
    transport slot. B0 maintains that slot but is excluded from retained C
    attention. Aligned C-suffix and B1: representations are alternatives in
    one exact softmax; ``stream_weight`` is the B-source prior alpha.
    """
    del prefix_anchor_active, prefix_future_eta
    if window_size != (-1, -1):
        raise ValueError("paired anchor attention requires global attention")
    if freqs is None:
        raise ValueError("paired anchor attention requires RoPE frequencies")
    if q.size(0) != 2 or len(seq_lens) != 2 or len(grid_sizes) != 2:
        raise ValueError("paired memory attention requires B and C streams")
    if not 0.0 <= stream_weight <= 1.0:
        raise ValueError("paired memory alpha must be between zero and one")

    suffix_frames, suffix_height, suffix_width = (
        int(value) for value in grid_sizes[0].tolist())
    main_frames, main_height, main_width = (
        int(value) for value in grid_sizes[1].tolist())
    anchor_index = int(anchor_index)
    if not 0 < anchor_index < main_frames - 1:
        raise ValueError("paired anchor must have retained frames on both sides")
    if (suffix_height, suffix_width) != (main_height, main_width):
        raise ValueError("paired streams must have identical spatial grids")
    if suffix_frames != main_frames - anchor_index:
        raise ValueError("B must align with the hidden anchor and C suffix")

    tokens_per_frame = main_height * main_width
    suffix_seq_len = suffix_frames * tokens_per_frame
    main_seq_len = main_frames * tokens_per_frame
    if (suffix_seq_len != int(seq_lens[0])
            or main_seq_len != int(seq_lens[1])):
        raise ValueError("paired stream sequence lengths disagree")

    b_q = q[0:1, :suffix_seq_len]
    b_k = k[0:1, :suffix_seq_len]
    b_v = v[0:1, :suffix_seq_len]
    c_q = q[1:2, :main_seq_len]
    c_k = k[1:2, :main_seq_len]
    c_v = v[1:2, :main_seq_len]

    # B remains a normal Img2-first stream, including direct access to B0.
    b_output = flash_attention(
        q=b_q, k=b_k, v=b_v, window_size=window_size, dtype=dtype)

    anchor_start = anchor_index * tokens_per_frame
    anchor_end = anchor_start + tokens_per_frame

    # Remove C's hidden transport slot and compress the retained suffix by one
    # temporal position. Exported B1: keys are shifted from local B positions
    # to the same compressed global coordinates as the aligned C suffix.
    compressed_c_suffix_q = _shift_temporal_rope(
        c_q[:, anchor_end:main_seq_len], -1, freqs)
    compressed_c_suffix_k = _shift_temporal_rope(
        c_k[:, anchor_end:main_seq_len], -1, freqs)
    c_suffix_v = c_v[:, anchor_end:main_seq_len]
    aligned_b_suffix_k = _shift_temporal_rope(
        b_k[:, tokens_per_frame:suffix_seq_len], anchor_index - 1, freqs)
    aligned_b_suffix_v = b_v[:, tokens_per_frame:suffix_seq_len]
    if aligned_b_suffix_k.size(1) != compressed_c_suffix_k.size(1):
        raise ValueError("B1: did not align with retained C suffix")

    retained_c_q = torch.cat([
        c_q[:, :anchor_start], compressed_c_suffix_q
    ], dim=1)

    group_outputs = []
    group_lses = []

    def add_group(group_k, group_v, log_prior=0.0):
        output, lse = flash_attention(
            q=retained_c_q,
            k=group_k,
            v=group_v,
            window_size=window_size,
            dtype=dtype,
            return_lse=True)
        group_outputs.append(output.float())
        group_lses.append(lse + log_prior)

    # Prefix memory is unique and therefore receives no source prior.
    add_group(c_k[:, :anchor_start], c_v[:, :anchor_start])
    if stream_weight < 1.0:
        add_group(
            compressed_c_suffix_k,
            c_suffix_v,
            math.log1p(-float(stream_weight)))
    if stream_weight > 0.0:
        add_group(
            aligned_b_suffix_k,
            aligned_b_suffix_v,
            math.log(float(stream_weight)))

    weights = torch.stack(group_lses, dim=-1).softmax(dim=-1).unsqueeze(-1)
    retained_c_output = (
        weights * torch.stack(group_outputs, dim=-2)).sum(dim=-2).to(
            c_q.dtype)

    # B0 is not C memory, but its complete update maintains the discarded
    # transport latent so B remains a coherent Img2-first suffix at the next
    # denoising step.
    c_output = torch.cat([
        retained_c_output[:, :anchor_start],
        b_output[:, :tokens_per_frame],
        retained_c_output[:, anchor_start:],
    ], dim=1)

    padded_len = q.size(1)
    if suffix_seq_len < padded_len:
        b_output = torch.cat([
            b_output,
            b_output.new_zeros(
                1, padded_len - suffix_seq_len, *b_output.shape[2:])
        ], dim=1)
    if main_seq_len < padded_len:
        c_output = torch.cat([
            c_output,
            c_output.new_zeros(
                1, padded_len - main_seq_len, *c_output.shape[2:])
        ], dim=1)
    return torch.cat([b_output, c_output], dim=0)


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


def cross_tail_bridge_rope(x, grid_sizes, freqs, anchor_index):
    """Place Img2-first B0/D0 at M's global Img2-anchor coordinate.

    M spans global positions from zero through the shared overlap. B spans
    the anchor through the output end. Shifting all B positions by the anchor
    makes M's final overlap and B's initial overlap use identical temporal
    coordinates. D is native-only, so its constant offset does not change its
    relative temporal geometry. This is an offset, not temporal compression.
    """
    if anchor_index is None:
        return x
    if x.size(0) != 3 or grid_sizes.size(0) != 3:
        raise ValueError("cross-tail RoPE requires exactly M, B, and D streams")
    output = [x[0:1]]
    for stream_index in (1, 2):
        frames, height, width = (
            int(value) for value in grid_sizes[stream_index].tolist())
        sequence_end = frames * height * width
        sample = x[stream_index:stream_index + 1]
        output.append(torch.cat([
            _shift_temporal_rope(
                sample[:, :sequence_end], int(anchor_index), freqs),
            sample[:, sequence_end:],
        ], dim=1))
    return torch.cat(output, dim=0)


def _shared_position_prior_attention(
    q,
    ordinary_k,
    ordinary_v,
    anchor_k,
    anchor_v,
    neighbour_k,
    neighbour_v,
    anchor_prior,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
    return_lse=False,
):
    """Exactly merge three groups into one prior-weighted softmax.

    Anchor and neighbour priors sum to one, preventing two representations at
    one temporal RoPE coordinate from receiving an automatic factor-of-two
    mass advantage.
    """
    if not 0.0 < anchor_prior < 1.0:
        raise ValueError("anchor prior must be strictly between zero and one")
    if (ordinary_k.size(1) == 0 or anchor_k.size(1) == 0
            or neighbour_k.size(1) == 0):
        raise ValueError("shared-position attention groups must be non-empty")

    ordinary_output, ordinary_lse = flash_attention(
        q=q, k=ordinary_k, v=ordinary_v, window_size=window_size,
        dtype=dtype, return_lse=True)
    anchor_output, anchor_lse = flash_attention(
        q=q, k=anchor_k, v=anchor_v, window_size=window_size,
        dtype=dtype, return_lse=True)
    neighbour_output, neighbour_lse = flash_attention(
        q=q, k=neighbour_k, v=neighbour_v, window_size=window_size,
        dtype=dtype, return_lse=True)

    group_lse = torch.stack([
        ordinary_lse,
        anchor_lse + math.log(float(anchor_prior)),
        neighbour_lse + math.log1p(-float(anchor_prior)),
    ], dim=-1)
    group_weights = group_lse.softmax(dim=-1).unsqueeze(-1)
    group_outputs = torch.stack([
        ordinary_output.float(), anchor_output.float(),
        neighbour_output.float()
    ], dim=-2)
    output = (group_weights * group_outputs).sum(dim=-2).to(
        ordinary_output.dtype)
    if return_lse:
        return output, torch.logsumexp(group_lse, dim=-1)
    return output


def _adaptive_shared_position_attention(
    q,
    ordinary_k,
    ordinary_v,
    anchor_k,
    anchor_v,
    neighbour_k,
    neighbour_v,
    margin=1.05,
    alpha_max=0.7,
    control_strength=1.0,
    suppress_anchor_with_control=False,
    diagnostic_route=None,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
):
    """Control a shared-position anchor/neighbour pair.

    ``anchor_lse`` and ``neighbour_lse`` are the log evidences of the two
    colocated frame groups.  Their exact equality threshold is
    ``sigmoid(neighbour_lse - anchor_lse)``.  Applying the margin and cap to
    that tensor independently preserves spatial/head-specific correspondence
    instead of allowing unrelated queries to cancel in one layer mean.

    For prefix queries, that adaptive prior continues to favour the anchor.
    For suffix queries, ``suppress_anchor_with_control`` instead interprets
    ``control_strength`` as a deterministic anchor-suppression alpha: zero
    leaves both groups untouched and one removes the anchor group entirely.
    """
    if margin < 1.0:
        raise ValueError("adaptive pair margin must be at least one")
    if not 0.5 <= alpha_max < 1.0:
        raise ValueError("adaptive pair alpha max must be in [0.5, 1)")
    if not 0.0 <= control_strength <= 1.0:
        raise ValueError("adaptive pair control strength must be in [0, 1]")

    ordinary_output, ordinary_lse = flash_attention(
        q=q, k=ordinary_k, v=ordinary_v, window_size=window_size,
        dtype=dtype, return_lse=True)
    anchor_output, anchor_lse = flash_attention(
        q=q, k=anchor_k, v=anchor_v, window_size=window_size,
        dtype=dtype, return_lse=True)
    neighbour_output, neighbour_lse = flash_attention(
        q=q, k=neighbour_k, v=neighbour_v, window_size=window_size,
        dtype=dtype, return_lse=True)

    with torch.no_grad():
        alpha_min = torch.sigmoid(
            neighbour_lse.detach() - anchor_lse.detach())
        alpha = (alpha_min * float(margin)).clamp(
            min=1.0e-4, max=float(alpha_max))
        if suppress_anchor_with_control:
            # As suffix RoPE compresses J9/J10 into one temporal position,
            # fade duplicate J10 memory out and leave J9 untouched. At zero
            # compression this is exactly the original, uncontrolled
            # attention; at full compression suffix queries cannot read J10.
            suppression_alpha = float(control_strength)
            anchor_weight = torch.full_like(alpha, 1.0 - suppression_alpha)
            neighbour_weight = torch.ones_like(alpha)
            diagnostic_alpha = torch.full_like(alpha, suppression_alpha)
            capped = torch.full_like(
                alpha, suppression_alpha >= 1.0 - 1.0e-6,
                dtype=torch.bool)
        else:
            anchor_weight = ((1.0 - float(control_strength))
                             + float(control_strength) * alpha)
            neighbour_weight = ((1.0 - float(control_strength))
                                + float(control_strength) * (1.0 - alpha))
            diagnostic_alpha = (
                anchor_weight / (anchor_weight + neighbour_weight))
            capped = alpha >= (float(alpha_max) - 1.0e-6)

        alpha_samples = _joint_alpha_samples.get()
        if alpha_samples is not None and diagnostic_route is not None:
            anchor_wins = (
                anchor_lse.detach() + torch.log(anchor_weight)
                >= neighbour_lse.detach() + torch.log(neighbour_weight))
            alpha_samples.append(torch.stack([
                alpha.new_tensor(float(diagnostic_route)),
                diagnostic_alpha.sum(),
                alpha.new_tensor(alpha.numel()),
                capped.sum(),
                anchor_wins.sum(),
            ]).float())

    group_lse = torch.stack([
        ordinary_lse,
        anchor_lse + torch.log(anchor_weight),
        neighbour_lse + torch.log(neighbour_weight),
    ], dim=-1)
    group_weights = group_lse.softmax(dim=-1).unsqueeze(-1)
    group_outputs = torch.stack([
        ordinary_output.float(), anchor_output.float(),
        neighbour_output.float()
    ], dim=-2)
    return (group_weights * group_outputs).sum(dim=-2).to(
        ordinary_output.dtype)


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
    anchor_prior=None,
    joint_query_single_memory_alpha=None,
    selective_suffix_memory_alpha=None,
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
    When ``joint_query_single_memory_alpha`` is set, J prefix and suffix
    queries also interrogate retained Img1-only M K/V. J-memory and M-memory
    outputs are merged from their exact log-normalizers with complementary
    source priors, and those results directly update M's aligned prefix and
    suffix without a second M/J output interpolation. Anchor routing remains
    unchanged.
    When ``selective_suffix_memory_alpha`` is set, M-prefix memory is omitted.
    J-prefix queries see unique J prefix, the J anchor and aligned J/M suffix;
    J-suffix queries see the same memories. At the prefix-facing shared RoPE
    position, the anchor receives prior ``pi`` and the aligned J/M first-suffix
    pair jointly receives ``1-pi``; alpha then divides that neighbour mass
    between J and M. This prevents J-anchor/J-suffix/M-suffix duplication from
    creating three positions' worth of evidence. A neutral ``pi=0.5`` is used
    when no explicit anchor prior is supplied.
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
    if anchor_prior is not None and not 0.0 < anchor_prior < 1.0:
        raise ValueError("anchor prior must be strictly between zero and one")
    if (joint_query_single_memory_alpha is not None
            and not 0.0 <= joint_query_single_memory_alpha <= 1.0):
        raise ValueError(
            "joint-query single-memory alpha must be in [0, 1]")
    if (selective_suffix_memory_alpha is not None
            and not 0.0 <= selective_suffix_memory_alpha <= 1.0):
        raise ValueError(
            "selective suffix-memory alpha must be in [0, 1]")
    if (joint_query_single_memory_alpha is not None
            and selective_suffix_memory_alpha is not None):
        raise ValueError(
            "full-memory and selective-suffix memory modes are exclusive")
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

    # M's temporary anchor is absent from every retained query's Q/K/V set.
    # Its neighbours are adjacent because temporary_anchor_rope shifted all
    # post-anchor retained positions left by one.
    m_retained_q = torch.cat([
        q[0:1, :anchor_start], q[0:1, anchor_end:seq_len]
    ], dim=1)
    m_retained_k = torch.cat([
        k[0:1, :anchor_start], k[0:1, anchor_end:seq_len]
    ], dim=1)
    m_retained_v = torch.cat([
        v[0:1, :anchor_start], v[0:1, anchor_end:seq_len]
    ], dim=1)
    m_retained_output = flash_attention(
        q=m_retained_q,
        k=m_retained_k,
        v=m_retained_v,
        window_size=window_size,
        dtype=dtype)

    use_split_memory = joint_query_single_memory_alpha is not None
    use_selective_suffix_memory = selective_suffix_memory_alpha is not None
    single_memory_alpha = (
        None if joint_query_single_memory_alpha is None else
        float(joint_query_single_memory_alpha))

    def merge_joint_and_m_memory(j_memory_result, j_query):
        """Merge J and retained-M memories under one exact biased softmax."""
        if not use_split_memory:
            return j_memory_result
        if single_memory_alpha == 0.0:
            return j_memory_result[0]
        m_memory_output, m_memory_lse = flash_attention(
            q=j_query,
            k=m_retained_k,
            v=m_retained_v,
            window_size=window_size,
            dtype=dtype,
            return_lse=True)
        if single_memory_alpha == 1.0:
            return m_memory_output
        j_memory_output, j_memory_lse = j_memory_result
        source_lse = torch.stack([
            j_memory_lse + math.log1p(-single_memory_alpha),
            m_memory_lse + math.log(single_memory_alpha),
        ], dim=-1)
        source_weights = source_lse.softmax(dim=-1).unsqueeze(-1)
        source_outputs = torch.stack([
            j_memory_output.float(), m_memory_output.float()
        ], dim=-2)
        return (source_weights * source_outputs).sum(dim=-2).to(
            j_memory_output.dtype)

    def grouped_attention(query, groups):
        """Evaluate disjoint K/V groups as one exact prior-biased softmax."""
        group_outputs = []
        group_lses = []
        for group_k, group_v, log_prior in groups:
            group_output, group_lse = flash_attention(
                q=query,
                k=group_k,
                v=group_v,
                window_size=window_size,
                dtype=dtype,
                return_lse=True)
            group_outputs.append(group_output.float())
            group_lses.append(group_lse + float(log_prior))
        if len(group_outputs) == 1:
            return group_outputs[0].to(query.dtype)
        stacked_lse = torch.stack(group_lses, dim=-1)
        weights = stacked_lse.softmax(dim=-1).unsqueeze(-1)
        outputs = torch.stack(group_outputs, dim=-2)
        return (weights * outputs).sum(dim=-2).to(query.dtype)

    # J-prefix sees its joint anchor moved onto the first suffix coordinate.
    # In selective mode, hierarchical priors make the anchor and the aligned
    # J/M first-suffix pair collectively occupy one shared temporal position.
    joint_prefix_anchor_k = _shift_temporal_rope(
        k[1:2, anchor_start:anchor_end], 0.5, freqs)
    j_prefix_q = q[1:2, :anchor_start]
    if use_selective_suffix_memory:
        selective_alpha = float(selective_suffix_memory_alpha)
        selective_anchor_prior = (
            0.5 if anchor_prior is None else float(anchor_prior))
        first_suffix_end = anchor_end + tokens_per_frame
        prefix_groups = [(
            k[1:2, :anchor_start], v[1:2, :anchor_start], 0.0)]
        prefix_groups.append((
            joint_prefix_anchor_k,
            v[1:2, anchor_start:anchor_end],
            math.log(selective_anchor_prior)))
        if selective_alpha < 1.0:
            prefix_groups.append((
                k[1:2, anchor_end:first_suffix_end],
                v[1:2, anchor_end:first_suffix_end],
                math.log1p(-selective_anchor_prior)
                + math.log1p(-selective_alpha)))
            if first_suffix_end < joint_seq_len:
                prefix_groups.append((
                    k[1:2, first_suffix_end:joint_seq_len],
                    v[1:2, first_suffix_end:joint_seq_len],
                    math.log1p(-selective_alpha)))
        if selective_alpha > 0.0:
            prefix_groups.append((
                k[0:1, anchor_end:first_suffix_end],
                v[0:1, anchor_end:first_suffix_end],
                math.log1p(-selective_anchor_prior)
                + math.log(selective_alpha)))
            if first_suffix_end < seq_len:
                prefix_groups.append((
                    k[0:1, first_suffix_end:seq_len],
                    v[0:1, first_suffix_end:seq_len],
                    math.log(selective_alpha)))
        j_prefix_output = grouped_attention(j_prefix_q, prefix_groups)
    else:
        if anchor_prior is None:
            j_prefix_result = flash_attention(
                q=j_prefix_q,
                k=torch.cat([
                    k[1:2, :anchor_start], joint_prefix_anchor_k,
                    k[1:2, anchor_end:joint_seq_len],
                ], dim=1),
                v=v[1:2, :joint_seq_len],
                window_size=window_size,
                dtype=dtype,
                return_lse=use_split_memory)
        else:
            first_suffix_end = anchor_end + tokens_per_frame
            j_prefix_result = _shared_position_prior_attention(
                q=j_prefix_q,
                ordinary_k=torch.cat([
                    k[1:2, :anchor_start],
                    k[1:2, first_suffix_end:joint_seq_len],
                ], dim=1),
                ordinary_v=torch.cat([
                    v[1:2, :anchor_start],
                    v[1:2, first_suffix_end:joint_seq_len],
                ], dim=1),
                anchor_k=joint_prefix_anchor_k,
                anchor_v=v[1:2, anchor_start:anchor_end],
                neighbour_k=k[1:2, anchor_end:first_suffix_end],
                neighbour_v=v[1:2, anchor_end:first_suffix_end],
                anchor_prior=anchor_prior,
                window_size=window_size,
                dtype=dtype,
                return_lse=use_split_memory)
        j_prefix_output = merge_joint_and_m_memory(
            j_prefix_result, j_prefix_q)

    # J-suffix uses the same history-aware joint anchor as J-prefix. Move its
    # key onto the suffix-facing coordinate shared with the final prefix
    # position. All visible J keys participate in one ordinary softmax. J
    # ordinary suffix coordinates are already shifted left once.
    positioned_joint_anchor_k = _shift_temporal_rope(
        k[1:2, anchor_start:anchor_end], -0.5, freqs)
    j_suffix_q = q[1:2, anchor_end:joint_seq_len]
    if use_selective_suffix_memory:
        suffix_groups = []
        if anchor_prior is None:
            suffix_groups.append((
                torch.cat([
                    k[1:2, :anchor_start], positioned_joint_anchor_k
                ], dim=1),
                torch.cat([
                    v[1:2, :anchor_start],
                    v[1:2, anchor_start:anchor_end]
                ], dim=1),
                0.0))
        else:
            last_prefix_start = anchor_start - tokens_per_frame
            suffix_groups.extend([
                (k[1:2, :last_prefix_start],
                 v[1:2, :last_prefix_start], 0.0),
                (positioned_joint_anchor_k,
                 v[1:2, anchor_start:anchor_end],
                 math.log(float(anchor_prior))),
                (k[1:2, last_prefix_start:anchor_start],
                 v[1:2, last_prefix_start:anchor_start],
                 math.log1p(-float(anchor_prior))),
            ])
        if selective_alpha < 1.0:
            suffix_groups.append((
                k[1:2, anchor_end:joint_seq_len],
                v[1:2, anchor_end:joint_seq_len],
                math.log1p(-selective_alpha)))
        if selective_alpha > 0.0:
            suffix_groups.append((
                k[0:1, anchor_end:seq_len],
                v[0:1, anchor_end:seq_len],
                math.log(selective_alpha)))
        j_suffix_output = grouped_attention(j_suffix_q, suffix_groups)
    else:
        if anchor_prior is None:
            j_suffix_result = flash_attention(
                q=j_suffix_q,
                k=torch.cat([
                    k[1:2, :anchor_start],
                    positioned_joint_anchor_k,
                    k[1:2, anchor_end:joint_seq_len],
                ], dim=1),
                v=torch.cat([
                    v[1:2, :anchor_start], v[1:2, anchor_start:anchor_end],
                    v[1:2, anchor_end:joint_seq_len],
                ], dim=1),
                window_size=window_size,
                dtype=dtype,
                return_lse=use_split_memory)
        else:
            last_prefix_start = anchor_start - tokens_per_frame
            j_suffix_result = _shared_position_prior_attention(
                q=j_suffix_q,
                ordinary_k=torch.cat([
                    k[1:2, :last_prefix_start],
                    k[1:2, anchor_end:joint_seq_len],
                ], dim=1),
                ordinary_v=torch.cat([
                    v[1:2, :last_prefix_start],
                    v[1:2, anchor_end:joint_seq_len],
                ], dim=1),
                anchor_k=positioned_joint_anchor_k,
                anchor_v=v[1:2, anchor_start:anchor_end],
                neighbour_k=k[1:2, last_prefix_start:anchor_start],
                neighbour_v=v[1:2, last_prefix_start:anchor_start],
                anchor_prior=anchor_prior,
                window_size=window_size,
                dtype=dtype,
                return_lse=use_split_memory)

        j_suffix_output = merge_joint_and_m_memory(
            j_suffix_result, j_suffix_q)

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

    direct_joint_update = (
        joint_query_single_memory_alpha is not None
        or selective_suffix_memory_alpha is not None)
    if not direct_joint_update:
        guided_m_prefix_output = inject_joint_residual(
            m_prefix_output, j_prefix_output, anchor_index)
    else:
        guided_m_prefix_output = j_prefix_output
    # Before the J handoff, suffix guidance is exclusively M/J. B is still
    # evaluated so B0 can supply the anchor velocity, but B1: does not enter
    # the layer-wise suffix update until the two-stream M/B handoff route.
    if not direct_joint_update:
        guided_m_suffix_output = inject_joint_residual(
            m_suffix_output, j_suffix_output, frames - anchor_index - 1)
    else:
        # Each split output already lets one J query choose J versus M memory
        # inside one exact normalization. Mixing it with O_M again would
        # reintroduce the completed-output interpolation this ablation avoids.
        guided_m_suffix_output = j_suffix_output
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
