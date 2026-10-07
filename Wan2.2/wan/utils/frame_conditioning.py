from numbers import Integral


def center_crop_resize_geometry(source_height, source_width, target_height,
                                target_width):
    """Return resize dimensions and a deterministic centered target crop.

    Isotropically resize the complete image just enough to cover the target,
    then remove only the minimal centered excess. This preserves the maximum
    possible field of view without letterboxing or aspect-ratio distortion.
    """
    values = (source_height, source_width, target_height, target_width)
    if any(not isinstance(value, Integral) for value in values):
        raise TypeError("crop geometry dimensions must be integers")
    source_height, source_width, target_height, target_width = (
        int(value) for value in values)
    if min(source_height, source_width, target_height, target_width) <= 0:
        raise ValueError("crop geometry dimensions must be positive")

    scale = max(target_height / source_height,
                target_width / source_width)
    resized_height = max(target_height, round(source_height * scale))
    resized_width = max(target_width, round(source_width * scale))

    top = (resized_height - target_height) // 2
    left = (resized_width - target_width) // 2
    return resized_height, resized_width, top, left


def layer_anchor_block_radius(radius, start_layer, layer):
    """Return the active symmetric anchor radius for one transformer layer."""
    values = (radius, start_layer, layer)
    if any(not isinstance(value, Integral) for value in values):
        raise TypeError("anchor radius layer values must be integers")
    radius, start_layer, layer = (int(value) for value in values)
    if radius < 0 or start_layer < 0 or layer < 0:
        raise ValueError("anchor radius layer values must be non-negative")
    return radius if layer >= start_layer else 0


def pixel_frame_range_for_latent(latent_index, temporal_stride=4):
    """Return the pixel-frame range packed into one causal VAE latent.

    Wan's first latent is the special one-frame prefix. Every later latent
    corresponds to ``temporal_stride`` consecutive pixel frames.
    """
    if not isinstance(latent_index, Integral):
        raise TypeError("latent index must be an integer")
    if latent_index < 0:
        raise ValueError("latent index must be non-negative")
    if not isinstance(temporal_stride, Integral) or temporal_stride <= 0:
        raise ValueError("temporal stride must be a positive integer")
    latent_index = int(latent_index)
    temporal_stride = int(temporal_stride)
    if latent_index == 0:
        return 0, 1
    start = 1 + (latent_index - 1) * temporal_stride
    return start, start + temporal_stride


def build_sacrificial_fractional_position_map(num_frames, anchors):
    """Place temporary anchors inside the compact retained-frame timeline.

    Retained frames use the compact timeline produced by deleting all
    temporary anchors. An isolated temporary anchor is placed halfway between
    its retained predecessor and retained successor.
    Consecutive anchors are spaced evenly inside their shared retained-frame
    interval. The same map is applied to queries and keys, preserving a
    consistent relative-position geometry without giving an anchor and an
    ordinary frame identical temporal RoPE.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if any(not isinstance(anchor, Integral) for anchor in anchors):
        raise TypeError("sacrificial anchors must be integers")
    anchors = tuple(sorted(int(anchor) for anchor in anchors))
    if len(anchors) != len(set(anchors)):
        raise ValueError("sacrificial anchors must be unique")
    if any(anchor <= 0 or anchor >= num_frames - 1 for anchor in anchors):
        raise ValueError(
            "every sacrificial anchor must have frames on both sides")

    anchor_set = set(anchors)
    positions = [None] * num_frames
    for frame in range(num_frames):
        if frame not in anchor_set:
            anchors_before_frame = sum(anchor < frame for anchor in anchors)
            positions[frame] = float(frame - anchors_before_frame)

    block_start = 0
    while block_start < len(anchors):
        block_end = block_start + 1
        while (block_end < len(anchors)
               and anchors[block_end] == anchors[block_end - 1] + 1):
            block_end += 1
        block = anchors[block_start:block_end]
        left_position = positions[block[0] - 1]
        if len(block) == 1:
            positions[block[0]] = left_position + 0.5
        else:
            denominator = len(block) + 1
            for offset, anchor in enumerate(block, start=1):
                positions[anchor] = left_position + offset / denominator
        block_start = block_end
    return tuple(positions)


def build_compressed_position_map(num_frames, anchors):
    """Map a sequence onto the timeline left after removing ``anchors``."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if any(not isinstance(anchor, Integral) for anchor in anchors):
        raise TypeError("sacrificial anchors must be integers")
    anchors = tuple(sorted(int(anchor) for anchor in anchors))
    if len(anchors) != len(set(anchors)):
        raise ValueError("sacrificial anchors must be unique")
    if any(anchor <= 0 or anchor >= num_frames for anchor in anchors):
        raise ValueError("sacrificial anchors must be nonzero valid frames")
    return tuple(
        frame - sum(anchor < frame for anchor in anchors)
        for frame in range(num_frames))


def build_anchor_query_positions(num_frames, anchors):
    """Return integer positions used only while an anchor is the query.

    Each temporary anchor is placed one unit before the first retained frame
    to its right. Earlier frames are invisible to that anchor query, so sharing
    their compact coordinate does not create a visible positional collision.
    """
    build_retained_frame_indices(num_frames, anchors)
    anchors = tuple(sorted(int(anchor) for anchor in anchors))
    return {
        anchor: float(anchor - sum(other <= anchor for other in anchors))
        for anchor in anchors
    }


def build_retained_frame_indices(num_frames, anchors):
    """Return expanded-frame indices after removing temporary anchors."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if any(not isinstance(anchor, Integral) for anchor in anchors):
        raise TypeError("temporary anchors must be integers")
    anchors = tuple(sorted(int(anchor) for anchor in anchors))
    if len(anchors) != len(set(anchors)):
        raise ValueError("temporary anchors must be unique")
    if any(anchor <= 0 or anchor >= num_frames for anchor in anchors):
        raise ValueError("temporary anchors must be nonzero valid frames")
    anchor_set = set(anchors)
    return tuple(frame for frame in range(num_frames)
                 if frame not in anchor_set)


def build_local_branch_ranges(num_frames, anchors):
    """Assign each temporary anchor the interval ending at the next anchor."""
    build_retained_frame_indices(num_frames, anchors)
    anchors = tuple(sorted(int(anchor) for anchor in anchors))
    return tuple((anchor,
                  anchors[index + 1]
                  if index + 1 < len(anchors) else num_frames)
                 for index, anchor in enumerate(anchors))


def build_attention_segments(num_frames,
                             conditioned_frame_indices,
                             anchor_block_radius=0):
    """Build anchor-started query intervals with normal temporal RoPE.

    Returns ``(start, end, anchor, blocked_frames)`` tuples.
    Ordinary queries see all ordinary frames, the first image anchor, and the
    image anchor that owns their interval. Queries in the first anchor's
    interval also see the next image anchor, giving the prefix direct endpoint
    guidance. Keeping the first anchor globally visible gives later intervals
    a shared appearance/scene reference. All query intervals use the model's
    normal temporal RoPE orientation.

    ``anchor_block_radius`` is symmetric: the nearest N ordinary queries on
    either side of every non-first image anchor hide that anchor's K/V tokens.

    The first conditioned query retains ordinary global I2V visibility. Every
    later image-anchor query sees itself and every future frame, but no earlier
    frame.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if not isinstance(anchor_block_radius, Integral):
        raise TypeError("anchor block radius must be an integer")
    anchor_block_radius = int(anchor_block_radius)
    if anchor_block_radius < 0:
        raise ValueError("anchor block radius must be non-negative")

    if any(not isinstance(frame, Integral)
           for frame in conditioned_frame_indices):
        raise TypeError("conditioned frame indices must be integers")
    conditioned = sorted(int(frame) for frame in conditioned_frame_indices)
    if len(conditioned) != len(set(conditioned)):
        raise ValueError("conditioned frame indices must be unique")
    if any(frame < 0 or frame >= num_frames for frame in conditioned):
        raise ValueError(
            f"conditioned frame indices must be in [0, {num_frames})")
    segments = []

    def append_segment(start, end, anchor, blocked):
        if start == end:
            return
        blocked = tuple(blocked)
        if (segments and segments[-1][1] == start
                and segments[-1][2:] == (anchor, blocked)):
            previous_start, _, _, _ = segments[-1]
            segments[-1] = (previous_start, end, anchor, blocked)
        else:
            segments.append((start, end, anchor, blocked))

    if not conditioned:
        return [(0, num_frames, None, ())]

    # Frames before the first anchor use it as their sole image condition.
    first_condition = conditioned[0]
    append_segment(
        0,
        first_condition,
        first_condition,
        (frame for frame in conditioned if frame != first_condition),
    )

    for condition_index, condition in enumerate(conditioned):

        if condition_index == 0:
            # Preserve the pretrained first-image query behavior, including
            # endpoint access to the next image anchor.
            future_conditions = conditioned[condition_index + 2:]
            blocked = (*range(condition), *future_conditions)
        else:
            # A non-first image anchor behaves like a local I2V first frame:
            # it sees itself and every future frame, but no past frame.
            blocked = tuple(range(condition))
        append_segment(condition, condition + 1, condition, blocked)

        if condition_index + 1 < len(conditioned):
            next_condition = conditioned[condition_index + 1]
            visible_anchors = {first_condition, condition}
            if condition_index == 0:
                visible_anchors.add(next_condition)
            blocked = tuple(frame for frame in conditioned
                            if frame not in visible_anchors)
            interval_end = next_condition
        else:
            visible_anchors = {first_condition, condition}
            blocked = tuple(frame for frame in conditioned
                            if frame not in visible_anchors)
            next_condition = None
            interval_end = num_frames

        # Build the ordinary interval one frame at a time; append_segment
        # merges adjacent frames with identical visibility. This supports a
        # symmetric exclusion radius even when the left/right windows overlap.
        for query_frame in range(condition + 1, interval_end):
            query_blocked = set(blocked)
            if anchor_block_radius > 0:
                if (condition != first_condition
                        and query_frame <=
                        condition + anchor_block_radius):
                    query_blocked.add(condition)
                if (next_condition is not None
                        and next_condition not in blocked
                        and query_frame >=
                        next_condition - anchor_block_radius):
                    query_blocked.add(next_condition)
            append_segment(query_frame, query_frame + 1, condition,
                           sorted(query_blocked))
    return segments


def blocked_frames_for_query(num_frames,
                             conditioned_frame_indices,
                             query_frame,
                             anchor_block_radius=0):
    """Return the frame indices hidden from one temporal query frame."""
    if not isinstance(query_frame, Integral):
        raise TypeError("query frame index must be an integer")
    query_frame = int(query_frame)
    if query_frame < 0 or query_frame >= num_frames:
        raise ValueError(f"query frame index must be in [0, {num_frames})")

    for start, end, _, blocked in build_attention_segments(
            num_frames, conditioned_frame_indices,
            anchor_block_radius=anchor_block_radius):
        if start <= query_frame < end:
            return blocked
    raise RuntimeError("query frame was not covered by attention segments")
