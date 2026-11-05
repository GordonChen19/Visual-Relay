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

    def test_one_way_prompt_guidance_accepts_paired_prompts(self):
        args = self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image_prompt', 'begin from img1',
            '--image', 'img2.png', '--image_latent_index', '10',
            '--image_prompt', 'continue from img2',
            '--image_condition_one_way_prompt_guidance',
            '--image_condition_one_way_prompt_guidance_weight', '0.3')
        self.assertTrue(args.image_condition_one_way_prompt_guidance)
        self.assertEqual(args.image_prompt,
                         ['begin from img1', 'continue from img2'])
        self.assertEqual(
            args.image_condition_one_way_prompt_guidance_weight, 0.3)

    def test_one_way_prefix_m10_only_flag_is_scoped(self):
        args = self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image_prompt', 'begin from img1',
            '--image', 'img2.png', '--image_latent_index', '10',
            '--image_prompt', 'continue from img2',
            '--image_condition_one_way_prompt_guidance',
            '--image_condition_one_way_prompt_guidance_prefix_m10_only')
        self.assertTrue(
            args.image_condition_one_way_prompt_guidance_prefix_m10_only)
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png',
                '--image_condition_one_way_prompt_guidance_prefix_m10_only')

    def test_one_way_auxiliary_anchor_prior_is_configurable(self):
        args = self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image_prompt', 'begin from img1',
            '--image', 'img2.png', '--image_latent_index', '10',
            '--image_prompt', 'continue from img2',
            '--image_condition_one_way_prompt_guidance',
            '--image_condition_one_way_prompt_guidance_aux_anchor_prior',
            '0.7')
        self.assertEqual(
            args.image_condition_one_way_prompt_guidance_aux_anchor_prior,
            0.7)
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png', '--image_latent_index', '0',
                '--image_prompt', 'begin from img1',
                '--image', 'img2.png', '--image_latent_index', '10',
                '--image_prompt', 'continue from img2',
                '--image_condition_one_way_prompt_guidance',
                '--image_condition_one_way_prompt_guidance_aux_anchor_prior',
                '1.0')

    def test_one_way_full_img2_stream_flag_is_scoped(self):
        args = self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image_prompt', 'begin from img1',
            '--image', 'img2.png', '--image_latent_index', '10',
            '--image_prompt', 'continue from img2',
            '--image_condition_one_way_prompt_guidance',
            '--image_condition_one_way_prompt_guidance_full_img2_stream')
        self.assertTrue(
            args.image_condition_one_way_prompt_guidance_full_img2_stream)
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png',
                '--image_condition_one_way_prompt_guidance_full_img2_stream')

    def test_one_way_guided_frames_is_scoped_and_positive(self):
        args = self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image_prompt', 'begin from img1',
            '--image', 'img2.png', '--image_latent_index', '10',
            '--image_prompt', 'continue from img2',
            '--image_condition_one_way_prompt_guidance',
            '--image_condition_one_way_prompt_guidance_guided_frames', '3')
        self.assertEqual(
            args.image_condition_one_way_prompt_guidance_guided_frames, 3)
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png',
                '--image_condition_one_way_prompt_guidance_guided_frames', '3')
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png', '--image_latent_index', '0',
                '--image_prompt', 'begin from img1',
                '--image', 'img2.png', '--image_latent_index', '10',
                '--image_prompt', 'continue from img2',
                '--image_condition_one_way_prompt_guidance',
                '--image_condition_one_way_prompt_guidance_guided_frames', '0')

    def test_one_way_shared_anchor_update_flag_is_scoped(self):
        args = self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image_prompt', 'begin from img1',
            '--image', 'img2.png', '--image_latent_index', '10',
            '--image_prompt', 'continue from img2',
            '--image_condition_one_way_prompt_guidance',
            '--image_condition_one_way_prompt_guidance_shared_anchor_update')
        self.assertTrue(
            args.image_condition_one_way_prompt_guidance_shared_anchor_update)
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png',
                '--image_condition_one_way_prompt_guidance_shared_anchor_update')

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
        self.assertFalse(args.image_condition_self_attention_guidance)

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

    def test_joint_img2_guidance_is_isolated(self):
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png', '--image_latent_index', '0',
                '--image_prompt', 'full trajectory',
                '--image', 'img2.png', '--image_latent_index', '10',
                '--image_prompt', 'Img2 continuation',
                '--image_condition_joint_img2_guidance',
                '--image_condition_self_attention_guidance')

    def test_joint_img2_guidance_requires_two_prompts(self):
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png', '--image_latent_index', '0',
                '--image', 'img2.png', '--image_latent_index', '10',
                '--image_prompt', 'only one prompt',
                '--image_condition_joint_img2_guidance')

    def test_cross_tail_bridge_accepts_prompts_and_controls(self):
        args = self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image_prompt', 'prompt one',
            '--image', 'img2.png', '--image_latent_index', '10',
            '--image_prompt', 'prompt two',
            '--image_condition_cross_tail_bridge',
            '--image_condition_cross_tail_bridge_tail_frames', '3',
            '--image_condition_cross_tail_bridge_memory_prior', '0.15')
        self.assertTrue(args.image_condition_cross_tail_bridge)
        self.assertEqual(args.image_condition_cross_tail_bridge_tail_frames, 3)
        self.assertEqual(
            args.image_condition_cross_tail_bridge_memory_prior, 0.15)

    def test_cross_tail_bridge_is_isolated_and_validated(self):
        common = (
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image_prompt', 'prompt one',
            '--image', 'img2.png', '--image_latent_index', '10',
            '--image_prompt', 'prompt two',
            '--image_condition_cross_tail_bridge')
        with self.assertRaises(AssertionError):
            self.parse(*common, '--image_condition_joint_img2_guidance')
        with self.assertRaises(AssertionError):
            self.parse(
                *common,
                '--image_condition_cross_tail_bridge_tail_frames', '21')
        with self.assertRaises(AssertionError):
            self.parse(
                *common,
                '--image_condition_cross_tail_bridge_memory_prior', '1.1')
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png',
                '--image_condition_cross_tail_bridge_uncontrolled_mass')

    def test_cross_tail_bridge_can_disable_mass_control(self):
        args = self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image_prompt', 'prompt one',
            '--image', 'img2.png', '--image_latent_index', '10',
            '--image_prompt', 'prompt two',
            '--image_condition_cross_tail_bridge',
            '--image_condition_cross_tail_bridge_uncontrolled_mass')
        self.assertTrue(
            args.image_condition_cross_tail_bridge_uncontrolled_mass)

    def test_one_way_prompt_guidance_requires_two_prompts(self):
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png', '--image_latent_index', '0',
                '--image', 'img2.png', '--image_latent_index', '10',
                '--image_prompt', 'only one prompt',
                '--image_condition_one_way_prompt_guidance')

    def test_one_way_prompt_guidance_is_isolated(self):
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png', '--image_latent_index', '0',
                '--image_prompt', 'prompt one',
                '--image', 'img2.png', '--image_latent_index', '10',
                '--image_prompt', 'prompt two',
                '--image_condition_one_way_prompt_guidance',
                '--image_condition_self_attention_guidance')

    def test_prefix_future_eta_range(self):
        args = self.parse('--image', 'img1.png',
                          '--image_condition_paired_prefix_future_eta', '0.2')
        self.assertEqual(args.image_condition_paired_prefix_future_eta, 0.2)
        for value in ('-0.1', '1.1'):
            with self.assertRaises(AssertionError):
                self.parse('--image', 'img1.png',
                           '--image_condition_paired_prefix_future_eta', value)

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

    def test_self_attention_guidance_accepts_aligned_images(self):
        args = self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image', 'img2.png', '--image_latent_index', '12',
            '--image_condition_self_attention_guidance',
            '--image_condition_self_attention_guidance_scale', '0.6')
        self.assertTrue(args.image_condition_self_attention_guidance)
        self.assertEqual(
            args.image_condition_self_attention_guidance_scale, 0.6)
        self.assertEqual(
            args.image_condition_self_attention_guidance_img2_weight, 0.25)
        self.assertFalse(args.image_condition_paired_anchor_attention)

    def test_self_attention_guidance_scale_must_be_nonnegative(self):
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png', '--image_latent_index', '0',
                '--image', 'img2.png', '--image_latent_index', '12',
                '--image_condition_self_attention_guidance',
                '--image_condition_self_attention_guidance_scale', '-0.1')

    def test_self_attention_guidance_phase_weights_are_independent(self):
        args = self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image', 'img2.png', '--image_latent_index', '12',
            '--image_condition_self_attention_guidance',
            '--image_condition_self_attention_guidance_scale', '0.8',
            '--image_condition_self_attention_guidance_img2_weight', '0.3')
        self.assertEqual(args.image_condition_self_attention_guidance_scale,
                         0.8)
        self.assertEqual(
            args.image_condition_self_attention_guidance_img2_weight, 0.3)

    def test_self_attention_guidance_stop_step_is_configurable(self):
        args = self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image', 'img2.png', '--image_latent_index', '12',
            '--image_condition_self_attention_guidance',
            '--image_condition_self_attention_guidance_stop_step', '20')
        self.assertEqual(
            args.image_condition_self_attention_guidance_stop_step, 20)

    def test_post_stop_img2_suffix_guidance_is_configurable(self):
        args = self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image', 'img2.png', '--image_latent_index', '12',
            '--image_condition_self_attention_guidance',
            '--image_condition_self_attention_guidance_stop_step', '20',
            '--image_condition_self_attention_guidance_post_stop_img2_suffix',
            '--image_condition_self_attention_guidance_img2_weight', '0.3')
        self.assertTrue(
            args.image_condition_self_attention_guidance_post_stop_img2_suffix)
        self.assertEqual(
            args.image_condition_self_attention_guidance_img2_weight, 0.3)

    def test_post_stop_img2_suffix_guidance_requires_stop_step(self):
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png', '--image_latent_index', '0',
                '--image', 'img2.png', '--image_latent_index', '12',
                '--image_condition_self_attention_guidance',
                '--image_condition_self_attention_guidance_post_stop_img2_suffix')

    def test_self_attention_guidance_rms_cap_is_configurable(self):
        args = self.parse(
            '--image', 'img1.png', '--image_latent_index', '0',
            '--image', 'img2.png', '--image_latent_index', '12',
            '--image_condition_self_attention_guidance',
            '--image_condition_self_attention_guidance_rms_cap', '0.2')
        self.assertEqual(
            args.image_condition_self_attention_guidance_rms_cap, 0.2)

    def test_self_attention_guidance_stop_requires_guidance_mode(self):
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png',
                '--image_condition_self_attention_guidance_stop_step', '20')

    def test_self_attention_guidance_stop_must_precede_sampling_end(self):
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png', '--image_latent_index', '0',
                '--image', 'img2.png', '--image_latent_index', '12',
                '--image_condition_self_attention_guidance',
                '--image_condition_self_attention_guidance_stop_step', '40')

    def test_two_multi_image_ablations_are_mutually_exclusive(self):
        with self.assertRaises(AssertionError):
            self.parse(
                '--image', 'img1.png', '--image_latent_index', '0',
                '--image', 'img2.png', '--image_latent_index', '12',
                '--image_condition_paired_anchor_attention',
                '--image_condition_self_attention_guidance')

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
