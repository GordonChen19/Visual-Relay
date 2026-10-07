import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = (Path(__file__).parents[1] / "wan" / "utils" /
               "frame_conditioning.py")
SPEC = importlib.util.spec_from_file_location("frame_conditioning", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_attention_segments = MODULE.build_attention_segments
build_sacrificial_fractional_position_map = (
    MODULE.build_sacrificial_fractional_position_map)
build_compressed_position_map = MODULE.build_compressed_position_map
build_anchor_query_positions = MODULE.build_anchor_query_positions
build_retained_frame_indices = MODULE.build_retained_frame_indices
build_local_branch_ranges = MODULE.build_local_branch_ranges
blocked_frames_for_query = MODULE.blocked_frames_for_query
center_crop_resize_geometry = MODULE.center_crop_resize_geometry
pixel_frame_range_for_latent = MODULE.pixel_frame_range_for_latent
layer_anchor_block_radius = MODULE.layer_anchor_block_radius


class FrameConditioningSegmentsTest(unittest.TestCase):

    def test_larger_image_is_downscaled_before_minimal_center_crop(self):
        self.assertEqual(
            center_crop_resize_geometry(1145, 1374, 720, 1280),
            (1067, 1280, 173, 0))

    def test_landscape_image_keeps_nearly_full_view_at_480p(self):
        self.assertEqual(
            center_crop_resize_geometry(941, 1672, 480, 832),
            (480, 853, 0, 10))

    def test_smaller_image_is_resized_to_cover_before_center_crop(self):
        self.assertEqual(
            center_crop_resize_geometry(720, 1000, 720, 1280),
            (922, 1280, 101, 0))

    def test_anchor_block_radius_activates_at_selected_layer(self):
        self.assertEqual(layer_anchor_block_radius(1, 16, 15), 0)
        self.assertEqual(layer_anchor_block_radius(1, 16, 16), 1)
        self.assertEqual(layer_anchor_block_radius(1, 16, 39), 1)
        self.assertEqual(layer_anchor_block_radius(1, 40, 39), 0)

    def test_first_latent_uses_the_special_one_frame_prefix(self):
        self.assertEqual(pixel_frame_range_for_latent(0), (0, 1))

    def test_later_latent_uses_its_four_frame_group(self):
        self.assertEqual(pixel_frame_range_for_latent(10), (37, 41))

    def test_pixel_frame_groups_are_contiguous(self):
        self.assertEqual(pixel_frame_range_for_latent(1), (1, 5))
        self.assertEqual(pixel_frame_range_for_latent(2), (5, 9))

    def test_single_sacrificial_anchor_uses_midpoint_position(self):
        positions = build_sacrificial_fractional_position_map(22, [10])
        self.assertEqual(positions[9], 9.0)
        self.assertEqual(positions[10], 9.5)
        self.assertEqual(positions[11], 10.0)
        self.assertEqual(positions[21], 20.0)

        # Retained neighbors stay one step apart. The temporary anchor is a
        # half step from each retained neighbor and has a unique position.
        self.assertEqual(positions[11] - positions[9], 1.0)
        self.assertEqual(positions[10] - positions[9], 0.5)
        self.assertEqual(positions[9] - positions[11], -1.0)
        self.assertEqual(positions[10] - positions[11], -0.5)
        self.assertEqual(len(set(positions)), len(positions))

    def test_multiple_sacrificial_anchors_use_cumulative_fractional_positions(self):
        positions = build_sacrificial_fractional_position_map(23, [10, 20])

        self.assertEqual(positions[9], 9.0)
        self.assertEqual(positions[10], 9.5)
        self.assertEqual(positions[11], 10.0)
        self.assertEqual(positions[19], 18.0)
        self.assertEqual(positions[20], 18.5)
        self.assertEqual(positions[21], 19.0)
        self.assertEqual(positions[22], 20.0)
        self.assertEqual(len(set(positions)), len(positions))

    def test_adjacent_sacrificial_anchors_split_the_retained_interval(self):
        positions = build_sacrificial_fractional_position_map(14, [10, 11])
        self.assertAlmostEqual(positions[10], 9.0 + 1.0 / 3.0)
        self.assertAlmostEqual(positions[11], 9.0 + 2.0 / 3.0)
        self.assertEqual(positions[12], 10.0)
        self.assertEqual(len(set(positions)), len(positions))

    def test_global_compression_matches_post_deletion_timeline(self):
        positions = build_compressed_position_map(23, [10, 20])
        self.assertEqual(positions[9], 9)
        self.assertEqual(positions[10], 10)
        self.assertEqual(positions[11], 10)
        self.assertEqual(positions[19], 18)
        self.assertEqual(positions[20], 19)
        self.assertEqual(positions[21], 19)
        self.assertEqual(positions[22], 20)

    def test_anchor_query_uses_integer_forward_offsets(self):
        positions = build_anchor_query_positions(22, [10])
        self.assertEqual(positions, {10: 9.0})
        compressed = build_sacrificial_fractional_position_map(22, [10])
        self.assertEqual(compressed[11] - positions[10], 1.0)

    def test_anchor_query_rope_is_asymmetric_at_frame_twelve(self):
        compressed = build_sacrificial_fractional_position_map(22, [12])
        anchor_query = build_anchor_query_positions(22, [12])

        # Ordinary queries see the anchor halfway between retained neighbors.
        self.assertEqual(compressed[12] - compressed[11], 0.5)
        self.assertEqual(compressed[13] - compressed[12], 0.5)

        # When the anchor itself is the query, its Q and self-K move to the
        # integer coordinate immediately before the future retained timeline.
        self.assertEqual(anchor_query[12], 11.0)
        self.assertEqual(compressed[13] - anchor_query[12], 1.0)
        self.assertEqual(compressed[14] - anchor_query[12], 2.0)

    def test_each_anchor_query_uses_its_compact_left_coordinate(self):
        self.assertEqual(
            build_anchor_query_positions(23, [10, 20]),
            {10: 9.0, 20: 18.0},
        )

    def test_retained_mapping_removes_all_temporary_anchors(self):
        self.assertEqual(
            build_retained_frame_indices(23, [10, 20]),
            (*range(10), *range(11, 20), 21, 22),
        )

    def test_local_branches_end_at_the_next_anchor(self):
        self.assertEqual(
            build_local_branch_ranges(23, [10, 20]),
            ((10, 20), (20, 23)),
        )

    def test_conditions_at_a_and_b(self):
        self.assertEqual(
            build_attention_segments(7, [0, 4]),
            [
                (0, 4, 0, ()),
                (4, 5, 4, tuple(range(4))),
                (5, 7, 4, ()),
            ],
        )

    def test_anchor_exclusion_radius_applies_on_both_sides(self):
        self.assertEqual(
            build_attention_segments(
                22, [0, 10], anchor_block_radius=2),
            [
                (0, 8, 0, ()),
                (8, 10, 0, (10, )),
                (10, 11, 10, tuple(range(10))),
                (11, 13, 10, (10, )),
                (13, 22, 10, ()),
            ],
        )
        self.assertEqual(
            blocked_frames_for_query(
                22, [0, 10], 7, anchor_block_radius=2), ())
        self.assertEqual(
            blocked_frames_for_query(
                22, [0, 10], 8, anchor_block_radius=2), (10, ))
        self.assertEqual(
            blocked_frames_for_query(
                22, [0, 10], 9, anchor_block_radius=2), (10, ))
        self.assertEqual(
            blocked_frames_for_query(
                22, [0, 10], 11, anchor_block_radius=2), (10, ))
        self.assertEqual(
            blocked_frames_for_query(
                22, [0, 10], 12, anchor_block_radius=2), (10, ))
        self.assertEqual(
            blocked_frames_for_query(
                22, [0, 10], 13, anchor_block_radius=2), ())

    def test_queries_before_first_condition_use_first_anchor(self):
        self.assertEqual(
            build_attention_segments(8, [2, 5]),
            [
                (0, 2, 2, (5, )),
                (2, 3, 2, (0, 1)),
                (3, 5, 2, ()),
                (5, 6, 5, tuple(range(5))),
                (6, 8, 5, ()),
            ],
        )

    def test_non_first_conditioned_query_sees_self_and_future_ordinary(self):
        self.assertEqual(
            build_attention_segments(5, [1, 2]),
            [
                (0, 1, 1, (2, )),
                (1, 2, 1, (0,)),
                (2, 3, 2, (0, 1)),
                (3, 5, 2, ()),
            ],
        )

    def test_visibility_switches_at_the_later_anchor(self):
        self.assertEqual(
            build_attention_segments(21, [0, 10]),
            [
                (0, 10, 0, ()),
                (10, 11, 10, tuple(range(10))),
                (11, 21, 10, ()),
            ],
        )

    def test_three_anchors_own_until_the_next_anchor(self):
        self.assertEqual(
            build_attention_segments(25, [0, 10, 20]),
            [
                (0, 10, 0, (20,)),
                (10, 11, 10, tuple(range(10))),
                (11, 20, 10, (20,)),
                (20, 21, 20, tuple(range(20))),
                (21, 25, 20, (10,)),
            ],
        )

    def test_ordinary_queries_see_expected_image_anchors(self):
        conditioned = {0, 10, 20}
        for start, end, owner, blocked in build_attention_segments(
                25, conditioned):
            for query in range(start, end):
                if query in conditioned:
                    continue
                visible_conditions = conditioned.difference(blocked)
                expected = {0, owner}
                if owner == 0 and query < 10:
                    expected.add(10)
                self.assertEqual(visible_conditions, expected)
                self.assertTrue(set(blocked).issubset(conditioned))

    def test_single_first_frame_condition_matches_default_global_attention(self):
        self.assertEqual(
            build_attention_segments(21, [0]),
            [(0, 21, 0, ())],
        )

    def test_probe_uses_the_same_visibility_as_segmented_attention(self):
        self.assertEqual(blocked_frames_for_query(21, [0, 10], 4), ())
        self.assertEqual(blocked_frames_for_query(21, [0, 10], 9), ())
        self.assertEqual(
            blocked_frames_for_query(21, [0, 10], 10),
            tuple(range(10)))
        self.assertEqual(blocked_frames_for_query(21, [0, 10], 11), ())

    def test_rejects_duplicate_or_out_of_range_frames(self):
        with self.assertRaises(ValueError):
            build_attention_segments(8, [2, 2])
        with self.assertRaises(ValueError):
            build_attention_segments(8, [8])
        with self.assertRaises(TypeError):
            build_attention_segments(8, [2.5])


if __name__ == '__main__':
    unittest.main()
