"""Test the paired-anchor CLI without importing the CUDA model stack."""
import argparse
import ast
import io
import math
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
        tree = ast.parse(path.read_text(encoding='utf-8'))
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
            math=math,
            random=random,
            sys=sys,
            WAN_CONFIGS={'i2v-A14B': cfg, 't2v-A14B': cfg},
            SIZE_CONFIGS={'832*480': (832, 480)},
            SUPPORTED_SIZES={
                'i2v-A14B': ['832*480'], 't2v-A14B': ['832*480']},
            EXAMPLE_PROMPT={
                'i2v-A14B': {'prompt': 'test'},
                't2v-A14B': {'prompt': 'test'}},
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
        self.assertIsNone(args.image_condition_paired_prefix_reference_budget)
        self.assertEqual(args.image_condition_paired_prefix_reference_curve,
                         'linear')

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
        self.assertIsNone(args.image_condition_paired_prefix_reference_budget)
        self.assertEqual(args.image_condition_paired_prefix_reference_curve,
                         'linear')

    def parse_paired(self, *options):
        return self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image', 'img2.png', '--image_latent_index', '12',
            '--image_condition_paired_anchor_attention', *options)

    def test_prefix_budget_accepts_closed_interval_and_both_curves(self):
        for budget in ('0', '0.4', '1'):
            for curve in ('linear', 'smoothstep'):
                with self.subTest(budget=budget, curve=curve):
                    args = self.parse_paired(
                        '--image_condition_paired_prefix_reference_budget',
                        budget,
                        '--image_condition_paired_prefix_reference_curve',
                        curve)
                    self.assertEqual(
                        args.image_condition_paired_prefix_reference_budget,
                        float(budget))
                    self.assertEqual(
                        args.image_condition_paired_prefix_reference_curve,
                        curve)
                    self.assertEqual(
                        args.image_condition_paired_stream_weight, 0.25)

    def test_prefix_budget_rejects_nonfinite_and_out_of_range_values(self):
        for budget in ('nan', 'inf', '-inf', '-0.1', '1.1'):
            with self.subTest(budget=budget):
                with self.assertRaisesRegex(AssertionError, 'finite'):
                    self.parse_paired(
                        '--image_condition_paired_prefix_reference_budget='
                        + budget)

    def test_prefix_budget_requires_paired_i2v_including_zero(self):
        for budget in ('0', '0.4'):
            for task in ('i2v-A14B', 't2v-A14B'):
                with self.subTest(budget=budget, task=task):
                    with self.assertRaisesRegex(AssertionError, 'paired'):
                        self.parse(
                            '--task', task, '--image', 'img1.png',
                            '--image_condition_paired_prefix_reference_budget',
                            budget)

    def test_nondefault_prefix_curve_requires_budget(self):
        with self.assertRaisesRegex(AssertionError, 'requires'):
            self.parse_paired(
                '--image_condition_paired_prefix_reference_curve',
                'smoothstep')

    def test_prefix_curve_rejects_unknown_values(self):
        with patch.object(sys, 'stderr', io.StringIO()):
            with self.assertRaises(SystemExit):
                self.parse_paired(
                    '--image_condition_paired_prefix_reference_budget', '0.4',
                    '--image_condition_paired_prefix_reference_curve', 'cosine')

    def test_prefix_budget_conflicts_with_anchor_removal_including_zero(self):
        for budget in ('0', '0.4'):
            with self.subTest(budget=budget):
                with self.assertRaisesRegex(AssertionError, 'cannot be combined'):
                    self.parse_paired(
                        '--image_condition_paired_prefix_reference_budget',
                        budget,
                        '--image_condition_paired_prefix_route_switch_step', '20')

    def test_prefix_budget_keeps_suffix_fade_independent(self):
        args = self.parse_paired(
            '--image_condition_paired_prefix_reference_budget', '0.4',
            '--image_condition_paired_suffix_fade_start_step', '25',
            '--image_condition_paired_suffix_fade_end_step', '30')
        self.assertEqual(args.image_condition_paired_prefix_reference_budget, 0.4)
        self.assertEqual(args.image_condition_paired_suffix_fade_start_step, 25)
        self.assertEqual(args.image_condition_paired_suffix_fade_end_step, 30)

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


class _ReachedImagePreprocessing(Exception):
    pass


class PairedPrefixPipelineValidationTest(unittest.TestCase):
    """Execute the real public API's validation before loading any model."""

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parents[1] / 'wan' / 'image2video.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        pipeline = next(node for node in tree.body
                        if isinstance(node, ast.ClassDef)
                        and node.name == 'WanI2V')
        generate = next(node for node in pipeline.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == 'generate')

        def stop_at_preprocessing(_image):
            raise _ReachedImagePreprocessing

        scope = dict(math=math, TF=SimpleNamespace(to_tensor=stop_at_preprocessing))
        exec(compile(ast.Module(body=[generate], type_ignores=[]),
                     str(path), 'exec'), scope)
        cls.generate = staticmethod(scope['generate'])

    def call_generate(self, **options):
        arguments = dict(
            image_condition_paired_anchor_attention=True,
            image_latent_indices=[0, 12],
        )
        arguments.update(options)
        return self.generate(SimpleNamespace(device='cpu'), 'test',
                             [object(), object()], **arguments)

    def test_valid_budgets_and_legacy_reach_image_preprocessing(self):
        for budget in (None, 0.0, 0.4, 1.0):
            for curve in ('linear', 'smoothstep'):
                if budget is None and curve != 'linear':
                    continue
                with self.subTest(budget=budget, curve=curve):
                    with self.assertRaises(_ReachedImagePreprocessing):
                        self.call_generate(
                            image_condition_paired_prefix_reference_budget=budget,
                            image_condition_paired_prefix_reference_curve=curve)

    def test_invalid_budgets_raise_value_error(self):
        for budget in (float('nan'), float('inf'), -float('inf'), -0.1, 1.1):
            with self.subTest(budget=budget):
                with self.assertRaisesRegex(ValueError, 'finite'):
                    self.call_generate(
                        image_condition_paired_prefix_reference_budget=budget)

    def test_budget_requires_paired_mode_including_zero(self):
        for budget in (0.0, 0.4):
            with self.subTest(budget=budget):
                with self.assertRaisesRegex(ValueError, 'requires paired'):
                    self.call_generate(
                        image_condition_paired_anchor_attention=False,
                        image_condition_paired_prefix_reference_budget=budget)

    def test_budget_rejects_route_switch_including_zero(self):
        for budget in (0.0, 0.4):
            with self.subTest(budget=budget):
                with self.assertRaisesRegex(ValueError, 'cannot be combined'):
                    self.call_generate(
                        image_condition_paired_prefix_reference_budget=budget,
                        image_condition_paired_prefix_route_switch_step=20)

    def test_curve_requires_budget_and_known_value(self):
        with self.assertRaisesRegex(ValueError, 'requires'):
            self.call_generate(
                image_condition_paired_prefix_reference_curve='smoothstep')
        with self.assertRaisesRegex(ValueError, 'linear or smoothstep'):
            self.call_generate(
                image_condition_paired_prefix_reference_budget=0.4,
                image_condition_paired_prefix_reference_curve='cosine')


if __name__ == '__main__':
    unittest.main()
