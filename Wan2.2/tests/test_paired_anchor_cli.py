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
        self.assertFalse(args.image_condition_paired_anchor_attention)

    def test_paired_anchor_accepts_two_aligned_images(self):
        args = self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image', 'img2.png', '--image_latent_index', '12',
            '--image_condition_paired_anchor_attention')
        self.assertEqual(args.image, ['img1.png', 'img2.png'])
        self.assertEqual(args.image_latent_index, [0, 12])
        self.assertTrue(args.image_condition_paired_anchor_attention)
        self.assertEqual(args.image_condition_paired_stream_weight, 0.25)
        self.assertIsNone(args.image_condition_paired_fade_start_step)
        self.assertIsNone(args.image_condition_paired_fade_end_step)
        self.assertIsNone(args.image_condition_paired_suffix_fade_start_step)
        self.assertIsNone(args.image_condition_paired_suffix_fade_end_step)
        self.assertIsNone(
            args.image_condition_paired_prefix_route_switch_step)

    def test_paired_stream_weight_is_configurable(self):
        args = self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image', 'img2.png', '--image_latent_index', '12',
            '--image_condition_paired_anchor_attention',
            '--image_condition_paired_stream_weight', '0.4')
        self.assertEqual(args.image_condition_paired_stream_weight, 0.4)

    def test_paired_stream_weight_must_be_a_convex_weight(self):
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png', '--image_latent_index', '0',
                '--image', 'img2.png', '--image_latent_index', '12',
                '--image_condition_paired_anchor_attention',
                '--image_condition_paired_stream_weight', '1.1')

    def test_paired_suffix_weight_can_cosine_fade_to_img1_only(self):
        args = self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image', 'img2.png', '--image_latent_index', '12',
            '--image_condition_paired_anchor_attention',
            '--image_condition_paired_fade_start_step', '25',
            '--image_condition_paired_fade_end_step', '34')
        self.assertEqual(args.image_condition_paired_fade_start_step, 25)
        self.assertEqual(args.image_condition_paired_fade_end_step, 34)

    def test_paired_fade_requires_both_steps(self):
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png', '--image_latent_index', '0',
                '--image', 'img2.png', '--image_latent_index', '12',
                '--image_condition_paired_anchor_attention',
                '--image_condition_paired_fade_start_step', '25')

    def test_suffix_specific_fade_is_configurable(self):
        args = self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image', 'img2.png', '--image_latent_index', '12',
            '--image_condition_paired_anchor_attention',
            '--image_condition_paired_suffix_fade_start_step', '25',
            '--image_condition_paired_suffix_fade_end_step', '30')
        self.assertEqual(
            args.image_condition_paired_suffix_fade_start_step, 25)
        self.assertEqual(args.image_condition_paired_suffix_fade_end_step, 30)

    def test_suffix_fade_requires_matching_end(self):
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png', '--image_latent_index', '0',
                '--image', 'img2.png', '--image_latent_index', '12',
                '--image_condition_paired_anchor_attention',
                '--image_condition_paired_suffix_fade_start_step', '15')

    def test_prefix_route_switch_is_configurable(self):
        args = self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image', 'img2.png', '--image_latent_index', '12',
            '--image_condition_paired_anchor_attention',
            '--image_condition_paired_prefix_route_switch_step', '20')
        self.assertEqual(
            args.image_condition_paired_prefix_route_switch_step, 20)

    def test_paired_fade_end_must_precede_last_sampling_index(self):
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png', '--image_latent_index', '0',
                '--image', 'img2.png', '--image_latent_index', '12',
                '--image_condition_paired_anchor_attention',
                '--image_condition_paired_fade_start_step', '25',
                '--image_condition_paired_fade_end_step', '40')

    def test_multiple_images_require_paired_mode(self):
        with self.assertRaises(AssertionError):
            self.parse('--image', 'img1.png', '--image', 'img2.png')

    def test_anchor_must_leave_a_suffix_frame(self):
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png', '--image_latent_index', '0',
                '--image', 'img2.png', '--image_latent_index', '20',
                '--image_condition_paired_anchor_attention')


if __name__ == '__main__':
    unittest.main()
