"""Test the paired-anchor CLI without importing the CUDA model stack."""
import argparse
import ast
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class PairedAnchorCliTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parents[1] / 'generate.py'
        tree = ast.parse(path.read_text())
        functions = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ('_parse_args', '_validate_args')
        ]
        cfg = SimpleNamespace(
            sample_steps=40,
            sample_shift=5,
            sample_guide_scale=5,
            frame_num=81,
            vae_stride=(4, 8, 8),
        )
        cls.scope = dict(
            argparse=argparse,
            random=random,
            sys=sys,
            WAN_CONFIGS={'i2v-A14B': cfg},
            SIZE_CONFIGS={'832*480': (832, 480)},
            SUPPORTED_SIZES={'i2v-A14B': ['832*480']},
            EXAMPLE_PROMPT={'i2v-A14B': {'prompt': 'test'}},
            str2bool=lambda value: value.lower() == 'true',
        )
        exec(compile(ast.Module(body=functions, type_ignores=[]),
                     str(path), 'exec'), cls.scope)

    def parse(self, *options):
        argv = [
            'generate.py', '--task', 'i2v-A14B', '--size', '832*480',
            '--ckpt_dir', '/unused', *options
        ]
        with patch.object(sys, 'argv', argv):
            return self.scope['_parse_args']()

    def test_standard_i2v_remains_single_image(self):
        args = self.parse('--image', 'img1.png')
        self.assertEqual(args.image, ['img1.png'])

    def test_joint_img2_guidance_accepts_aligned_images(self):
        args = self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image_prompt', 'full trajectory',
            '--image', 'img2.png', '--image_latent_index', '10',
            '--image_prompt', 'Img2 continuation',
            '--image_condition_joint_img2_guidance',
            '--image_condition_joint_img2_guidance_weight', '0.3')
        self.assertTrue(args.image_condition_joint_img2_guidance)
        self.assertEqual(args.image_condition_joint_img2_guidance_weight, 0.3)

    def test_joint_img2_guidance_accepts_shifted_anchor_view(self):
        args = self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image_prompt', 'full trajectory',
            '--image', 'img2.png', '--image_latent_index', '10',
            '--image_prompt', 'Img2 continuation',
            '--image_condition_joint_img2_guidance',
            '--image_condition_joint_img2_guidance_shifted_anchor')
        self.assertTrue(
            args.image_condition_joint_img2_guidance_shifted_anchor)
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png',
                '--image_condition_joint_img2_guidance_shifted_anchor')

    def test_joint_img2_guidance_rope_compression_is_opt_in_and_scoped(self):
        args = self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image_prompt', 'full trajectory',
            '--image', 'img2.png', '--image_latent_index', '10',
            '--image_prompt', 'Img2 continuation',
            '--image_condition_joint_img2_guidance',
            '--image_condition_joint_img2_guidance_shifted_anchor',
            '--image_condition_joint_img2_guidance_rope_compression')
        self.assertTrue(
            args.image_condition_joint_img2_guidance_rope_compression)
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png', '--image_latent_index', '0',
                '--image_prompt', 'full trajectory',
                '--image', 'img2.png', '--image_latent_index', '10',
                '--image_prompt', 'Img2 continuation',
                '--image_condition_joint_img2_guidance',
                '--image_condition_joint_img2_guidance_rope_compression')

    def test_joint_adaptive_pair_attention_is_opt_in_and_scoped(self):
        common = (
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image_prompt', 'full trajectory',
            '--image', 'img2.png', '--image_latent_index', '10',
            '--image_prompt', 'Img2 continuation',
            '--image_condition_joint_img2_guidance',
            '--image_condition_joint_img2_guidance_shifted_anchor',
            '--image_condition_joint_img2_guidance_rope_compression')
        args = self.parse(
            *common,
            '--image_condition_joint_img2_guidance_adaptive_pair_attention',
            '--image_condition_joint_img2_guidance_adaptive_pair_end_step', '8',
            '--image_condition_joint_img2_guidance_adaptive_pair_margin', '1.05',
            '--image_condition_joint_img2_guidance_adaptive_pair_alpha_max',
            '0.7')
        self.assertTrue(
            args.image_condition_joint_img2_guidance_adaptive_pair_attention)
        self.assertEqual(
            args.image_condition_joint_img2_guidance_adaptive_pair_end_step, 8)
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png', '--image_latent_index', '0',
                '--image_prompt', 'full trajectory',
                '--image', 'img2.png', '--image_latent_index', '10',
                '--image_prompt', 'Img2 continuation',
                '--image_condition_joint_img2_guidance',
                '--image_condition_joint_img2_guidance_shifted_anchor',
                '--image_condition_joint_img2_guidance_adaptive_pair_attention')

    def test_joint_img1_only_handoff_is_opt_in_and_scoped(self):
        common = (
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image_prompt', 'full trajectory',
            '--image', 'img2.png', '--image_latent_index', '10',
            '--image_prompt', 'Img2 continuation',
            '--image_condition_joint_img2_guidance',
            '--image_condition_joint_img2_guidance_shifted_anchor',
            '--image_condition_joint_img2_guidance_rope_compression')
        args = self.parse(
            *common,
            '--image_condition_joint_img2_guidance_adaptive_pair_attention',
            '--image_condition_joint_img2_guidance_adaptive_pair_end_step', '25',
            '--image_condition_joint_img2_guidance_handoff_img1_only')
        self.assertTrue(
            args.image_condition_joint_img2_guidance_handoff_img1_only)
        with self.assertRaises(AssertionError):
            self.parse(
                *common,
                '--image_condition_joint_img2_guidance_handoff_img1_only')

    def test_joint_no_rope_compression_preserves_handoff_option(self):
        common = (
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image_prompt', 'full trajectory',
            '--image', 'img2.png', '--image_latent_index', '10',
            '--image_prompt', 'Img2 continuation',
            '--image_condition_joint_img2_guidance',
            '--image_condition_joint_img2_guidance_shifted_anchor')
        self.assertFalse(self.parse(
            *common).image_condition_joint_img2_guidance_no_rope_compression)
        for compression in ((), (
                '--image_condition_joint_img2_guidance_rope_compression',)):
            with self.subTest(compression=compression):
                args = self.parse(
                    *common, *compression,
                    '--image_condition_joint_img2_guidance_no_rope_compression',
                    '--image_condition_joint_img2_guidance_adaptive_pair_attention',
                    '--image_condition_joint_img2_guidance_adaptive_pair_end_step', '25')
                self.assertTrue(
                    args.image_condition_joint_img2_guidance_no_rope_compression)
                self.assertTrue(
                    args.image_condition_joint_img2_guidance_adaptive_pair_attention)
                self.assertEqual(
                    args.image_condition_joint_img2_guidance_adaptive_pair_end_step, 25)
        with self.assertRaises(AssertionError):
            self.parse('--image', 'img1.png',
                       '--image_condition_joint_img2_guidance_no_rope_compression')

    def test_fractional_anchor_rope_offset_is_configurable_and_scoped(self):
        common = (
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image_prompt', 'full trajectory',
            '--image', 'img2.png', '--image_latent_index', '10',
            '--image_prompt', 'Img2 continuation',
            '--image_condition_joint_img2_guidance',
            '--image_condition_joint_img2_guidance_shifted_anchor',
            '--image_condition_joint_img2_guidance_adaptive_pair_attention',
            '--image_condition_joint_img2_guidance_adaptive_pair_end_step', '25')
        for offset in (0.0, 0.5, 0.8, 1.0):
            with self.subTest(offset=offset):
                args = self.parse(
                    *common,
                    '--image_condition_joint_img2_guidance_anchor_rope_offset',
                    str(offset))
                self.assertEqual(
                    args.image_condition_joint_img2_guidance_anchor_rope_offset,
                    offset)
        for invalid in ('-0.1', '1.1', 'nan'):
            with self.subTest(invalid=invalid), self.assertRaises(AssertionError):
                self.parse(
                    *common,
                    '--image_condition_joint_img2_guidance_anchor_rope_offset',
                    invalid)
        with self.assertRaises(AssertionError):
            self.parse('--image', 'img1.png',
                       '--image_condition_joint_img2_guidance_anchor_rope_offset',
                       '0.8')

    def test_joint_img2_guidance_static_b_prompt_is_scoped(self):
        args = self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image_prompt', 'full trajectory',
            '--image', 'img2.png', '--image_latent_index', '10',
            '--image_prompt', 'Img2 continuation',
            '--image_condition_joint_img2_guidance',
            '--image_condition_joint_img2_guidance_static_b_prompt')
        self.assertTrue(
            args.image_condition_joint_img2_guidance_static_b_prompt)
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png',
                '--image_condition_joint_img2_guidance_static_b_prompt')

    def test_joint_velocity_blend_weights_are_configurable_and_scoped(self):
        base = (
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image_prompt', 'full trajectory',
            '--image', 'img2.png', '--image_latent_index', '14',
            '--image_prompt', 'Img2 continuation')
        args = self.parse(
            *base, '--image_condition_joint_img2_guidance',
            '--image_condition_joint_img2_guidance_velocity_blend_weights',
            '1.0', '0.6', '0.25', '0.1')
        self.assertEqual(
            args.image_condition_joint_img2_guidance_velocity_blend_weights,
            [1.0, 0.6, 0.25, 0.1])
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png',
                '--image_condition_joint_img2_guidance_velocity_blend_weights',
                '1.0')
        for invalid in ('-0.1', '1.1', 'nan'):
            with self.subTest(invalid=invalid), self.assertRaises(AssertionError):
                self.parse(
                    *base, '--image_condition_joint_img2_guidance',
                    '--image_condition_joint_img2_guidance_velocity_blend_weights',
                    invalid)

    def test_joint_single_frame_anchor_is_opt_in_and_scoped(self):
        base = (
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image_prompt', 'full trajectory',
            '--image', 'img2.png', '--image_latent_index', '10',
            '--image_prompt', 'Img2 continuation')
        args = self.parse(*base, '--image_condition_joint_img2_guidance')
        self.assertFalse(args.image_condition_joint_img2_guidance_single_frame_anchor)
        args = self.parse(*base, '--image_condition_joint_img2_guidance',
                          '--image_condition_joint_img2_guidance_single_frame_anchor')
        self.assertTrue(args.image_condition_joint_img2_guidance_single_frame_anchor)
        with self.assertRaises(AssertionError):
            self.parse(*base,
                       '--image_condition_joint_img2_guidance_single_frame_anchor')

    def test_joint_single_frame_four_history_is_separate_and_scoped(self):
        base = (
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image_prompt', 'first',
            '--image', 'img2.png', '--image_latent_index', '10',
            '--image_prompt', 'second',
            '--image_condition_joint_img2_guidance')
        single = '--image_condition_joint_img2_guidance_single_frame_anchor'
        four = '--image_condition_joint_img2_guidance_single_frame_anchor_four_history'
        with self.assertRaises(AssertionError):
            self.parse(*base, four)
        args = self.parse(*base, single, four)
        self.assertTrue(
            args.image_condition_joint_img2_guidance_single_frame_anchor_four_history)

    def test_joint_b1_alignment_requires_shifted_mode_and_defaults_off(self):
        base = (
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image_prompt', 'first',
            '--image', 'img2.png', '--image_latent_index', '14',
            '--image_prompt', 'second',
            '--image_condition_joint_img2_guidance')
        flag = '--image_condition_joint_img2_guidance_align_b1'
        self.assertFalse(self.parse(*base).image_condition_joint_img2_guidance_align_b1)
        with self.assertRaises(AssertionError):
            self.parse(*base, flag)
        args = self.parse(*base, flag,
                          '--image_condition_joint_img2_guidance_shifted_anchor')
        self.assertTrue(args.image_condition_joint_img2_guidance_align_b1)

    def test_backward_pair_rope_is_shifted_and_exclusive(self):
        base = (
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image_prompt', 'first',
            '--image', 'img2.png', '--image_latent_index', '14',
            '--image_prompt', 'second',
            '--image_condition_joint_img2_guidance')
        flag = '--image_condition_joint_img2_guidance_backward_pair_rope_compression'
        with self.assertRaises(AssertionError):
            self.parse(*base, flag)
        args = self.parse(*base, flag,
                          '--image_condition_joint_img2_guidance_shifted_anchor')
        self.assertTrue(
            args.image_condition_joint_img2_guidance_backward_pair_rope_compression)
        with self.assertRaises(AssertionError):
            self.parse(*base, flag,
                       '--image_condition_joint_img2_guidance_shifted_anchor',
                       '--image_condition_joint_img2_guidance_anchor_rope_offset', '0.5')

    def test_suffix_anchor_transfer_requires_routing_and_valid_cap(self):
        base = (
            '--image', 'img1.png', '--image_latent_index', '0', '--image_prompt', 'first',
            '--image', 'img2.png', '--image_latent_index', '14', '--image_prompt', 'second',
            '--image_condition_joint_img2_guidance',
            '--image_condition_joint_img2_guidance_shifted_anchor')
        flag = '--image_condition_joint_img2_guidance_suffix_anchor_transfer'
        self.assertFalse(self.parse(*base).image_condition_joint_img2_guidance_suffix_anchor_transfer)
        with self.assertRaises(AssertionError):
            self.parse(*base, flag)
        routed = (*base, '--image_condition_joint_img2_guidance_anchor_rope_offset', '0.8')
        args = self.parse(*routed, flag)
        self.assertEqual(args.image_condition_joint_img2_guidance_suffix_anchor_transfer_ratio, 1.)
        self.assertEqual(args.image_condition_joint_img2_guidance_suffix_anchor_transfer_max_delta, .05)
        for value in ('-0.1', '1.1', 'nan'):
            with self.assertRaises(AssertionError):
                self.parse(*routed, flag,
                           '--image_condition_joint_img2_guidance_suffix_anchor_transfer_max_delta', value)

    def test_joint_img2_guidance_persistent_b_is_shifted_only(self):
        args = self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image_prompt', 'full trajectory',
            '--image', 'img2.png', '--image_latent_index', '10',
            '--image_prompt', 'Img2 continuation',
            '--image_condition_joint_img2_guidance',
            '--image_condition_joint_img2_guidance_shifted_anchor',
            '--image_condition_joint_img2_guidance_persistent_b')
        self.assertTrue(args.image_condition_joint_img2_guidance_persistent_b)
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png', '--image_latent_index', '0',
                '--image_prompt', 'full trajectory',
                '--image', 'img2.png', '--image_latent_index', '10',
                '--image_prompt', 'Img2 continuation',
                '--image_condition_joint_img2_guidance',
                '--image_condition_joint_img2_guidance_persistent_b')

    def test_joint_img2_guidance_save_b_video_requires_persistent_b(self):
        args = self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image_prompt', 'full trajectory',
            '--image', 'img2.png', '--image_latent_index', '10',
            '--image_prompt', 'Img2 continuation',
            '--image_condition_joint_img2_guidance',
            '--image_condition_joint_img2_guidance_shifted_anchor',
            '--image_condition_joint_img2_guidance_persistent_b',
            '--image_condition_joint_img2_guidance_save_b_video')
        self.assertTrue(
            args.image_condition_joint_img2_guidance_save_b_video)
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png', '--image_latent_index', '0',
                '--image_prompt', 'full trajectory',
                '--image', 'img2.png', '--image_latent_index', '10',
                '--image_prompt', 'Img2 continuation',
                '--image_condition_joint_img2_guidance',
                '--image_condition_joint_img2_guidance_shifted_anchor',
                '--image_condition_joint_img2_guidance_save_b_video')

    def test_joint_img2_guidance_requires_two_prompts(self):
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png', '--image_latent_index', '0',
                '--image', 'img2.png', '--image_latent_index', '10',
                '--image_prompt', 'only one prompt',
                '--image_condition_joint_img2_guidance')

    def test_multiple_images_require_joint_mode(self):
        with self.assertRaises(AssertionError):
            self.parse('--image', 'img1.png', '--image', 'img2.png')

    def test_anchor_must_leave_a_suffix_frame(self):
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png', '--image_latent_index', '0',
                '--image_prompt', 'full trajectory',
                '--image', 'img2.png', '--image_latent_index', '20',
                '--image_prompt', 'Img2 continuation',
                '--image_condition_joint_img2_guidance')


if __name__ == '__main__':
    unittest.main()
