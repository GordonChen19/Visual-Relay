"""Exercise the real paired sampler on CPU without checkpoints or CUDA imports.

Only the external model, text encoder, VAE, and scheduler are doubles. The
production generate method performs conditioning, scheduling, CFG, temporary
anchor replacement, and removal unchanged.
"""

import ast
from contextlib import contextmanager, nullcontext
import gc
import logging
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


class _TextEncoder:

    def __init__(self):
        self.calls = []
        self.embeddings = {}
        self.model = SimpleNamespace(to=lambda _device: None, cpu=lambda: None)

    def __call__(self, prompts, device):
        self.calls.append((list(prompts), str(device)))
        encoded = []
        for prompt in prompts:
            if prompt not in self.embeddings:
                self.embeddings[prompt] = torch.full(
                    (2, 3), float(len(self.embeddings) + 1))
            encoded.append(self.embeddings[prompt].clone())
        return encoded


class _VAE:

    def __init__(self):
        self.encoded_inputs = []
        self.decoded_inputs = []

    def encode(self, videos):
        self.encoded_inputs = [video.clone() for video in videos]
        results = []
        for index, video in enumerate(videos):
            frames = (video.shape[1] - 1) // 4 + 1
            temporal_values = torch.arange(frames, dtype=torch.float32)
            temporal_values += 100 * (index + 1)
            results.append(temporal_values.view(1, frames, 1, 1).expand(
                16, frames, video.shape[2] // 8, video.shape[3] // 8).clone())
        return results

    def decode(self, latents):
        self.decoded_inputs = [latent.clone() for latent in latents]
        return [torch.zeros(
            3, (latent.shape[1] - 1) * 4 + 1,
            latent.shape[2] * 8, latent.shape[3] * 8) for latent in latents]


class _Model:

    def __init__(self):
        self.calls = []

    def __call__(self, latents, t, **kwargs):
        # Different conditional/unconditional predictions make accidental
        # replacement on only one CFG branch detectable at the scheduler.
        conditional = len(self.calls) % 2 == 0
        values = (100.0, 200.0) if conditional else (10.0, 20.0)
        predictions = []
        for latent, value in zip(latents, values):
            frame_values = torch.arange(latent.shape[1], dtype=latent.dtype)
            predictions.append((frame_values + value).view(
                1, -1, 1, 1).expand_as(latent).clone())
        captured = dict(kwargs)
        captured['context'] = [value.clone() for value in kwargs['context']]
        captured['y'] = [value.clone() for value in kwargs['y']]
        captured['latents'] = [value.clone() for value in latents]
        captured['predictions'] = [value.clone() for value in predictions]
        captured['t'] = t.clone()
        self.calls.append(captured)
        return predictions


class _Scheduler:

    def __init__(self, **_kwargs):
        self.calls = []

    def set_timesteps(self, sampling_steps, device, shift):
        self.timesteps = torch.linspace(900, 100, sampling_steps, device=device)

    def step(self, prediction, timestep, sample, **kwargs):
        result = sample - 0.01 * prediction
        self.calls.append(dict(
            prediction=prediction.clone(), sample=sample.clone(),
            result=result.clone(), timestep=timestep.clone(), **kwargs))
        return (result,)


class PairedAnchorSamplingTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.source_path = Path(__file__).parents[1] / 'wan' / 'image2video.py'
        tree = ast.parse(cls.source_path.read_text(encoding='utf-8'))
        pipeline = next(node for node in tree.body
                        if isinstance(node, ast.ClassDef)
                        and node.name == 'WanI2V')
        cls.generate_node = next(node for node in pipeline.body
                                 if isinstance(node, ast.FunctionDef)
                                 and node.name == 'generate')

    def run_sampler(self, *, t5_cpu=True, anchor=12, **options):
        encoder, vae, model = _TextEncoder(), _VAE(), _Model()
        schedulers = []

        def make_scheduler(**kwargs):
            scheduler = _Scheduler(**kwargs)
            schedulers.append(scheduler)
            return scheduler

        scope = dict(
            torch=torch, np=np, math=math, random=random, sys=sys,
            logging=logging, gc=gc, contextmanager=contextmanager,
            TF=SimpleNamespace(to_tensor=lambda image: image.clone()),
            dist=SimpleNamespace(is_initialized=lambda: False),
            tqdm=lambda values: values,
            FlowUniPCMultistepScheduler=make_scheduler,
        )
        exec(compile(ast.Module(body=[self.generate_node], type_ignores=[]),
                     str(self.source_path), 'exec'), scope)
        pipeline = SimpleNamespace(
            device=torch.device('cpu'), vae_stride=(4, 8, 8),
            patch_size=(1, 2, 2), sp_size=1, t5_cpu=t5_cpu,
            text_encoder=encoder, vae=vae, sample_neg_prompt='default negative',
            low_noise_model=model, high_noise_model=model,
            param_dtype=torch.float32, boundary=0.5, num_train_timesteps=1000,
            rank=0, _prepare_model_for_timestep=lambda *args, **kwargs: model,
        )
        arguments = dict(
            max_area=16 * 16, frame_num=81, sampling_steps=5,
            guide_scale=2.0, seed=4827316, offload_model=False,
            image_latent_indices=[0, anchor],
            image_condition_paired_anchor_attention=True,
            image_condition_paired_stream_weight=0.6,
        )
        arguments.update(options)
        # Leave all tensor operations and torch.no_grad intact. There is no GPU
        # work in these doubles, so only the CUDA autocast context is disabled.
        with patch.object(torch.amp, 'autocast',
                          side_effect=lambda *args, **kwargs: nullcontext()):
            video = scope['generate'](
                pipeline, 'main action',
                [torch.full((3, 16, 16), 0.25),
                 torch.full((3, 16, 16), 0.75)], **arguments)
        self.assertEqual(len(schedulers), 1)
        return SimpleNamespace(
            video=video, encoder=encoder, vae=vae, model=model,
            scheduler=schedulers[0])

    def test_distinct_suffix_positive_and_shared_negative_conditioning(self):
        for t5_cpu in (True, False):
            with self.subTest(t5_cpu=t5_cpu):
                result = self.run_sampler(
                    t5_cpu=t5_cpu, n_prompt='shared negative',
                    image_condition_paired_suffix_prompt='continue after anchor')
                self.assertEqual(result.encoder.calls, [
                    (['main action', 'continue after anchor'], 'cpu'),
                    (['shared negative'], 'cpu'),
                ])
                for cond, uncond in zip(
                        result.model.calls[::2], result.model.calls[1::2]):
                    for actual, prompt in zip(cond['context'],
                                              ('main action', 'continue after anchor')):
                        torch.testing.assert_close(
                            actual, result.encoder.embeddings[prompt])
                    for actual in uncond['context']:
                        torch.testing.assert_close(
                            actual, result.encoder.embeddings['shared negative'])

    def test_unset_suffix_preserves_single_positive_encoder_call(self):
        result = self.run_sampler()
        self.assertEqual(result.encoder.calls, [
            (['main action'], 'cpu'), (['default negative'], 'cpu')])
        for cond, uncond in zip(result.model.calls[::2],
                                result.model.calls[1::2]):
            torch.testing.assert_close(cond['context'][0], cond['context'][1])
            torch.testing.assert_close(uncond['context'][0], uncond['context'][1])
            self.assertIsNone(cond['paired_prefix_reference_strength'])
            self.assertFalse(cond['paired_prefix_preserve_first_frame'])

    def test_reference_and_suffix_fades_are_independent_on_both_cfg_branches(self):
        result = self.run_sampler(
            image_condition_paired_prefix_reference_strength=0.8,
            image_condition_paired_prefix_preserve_first_frame=True,
            image_condition_paired_prefix_reference_fade_start_step=0,
            image_condition_paired_prefix_reference_fade_end_step=2,
            image_condition_paired_suffix_fade_start_step=2,
            image_condition_paired_suffix_fade_end_step=4)
        expected_strength = (0.8, 0.4, 0.0, 0.0, 0.0)
        expected_suffix = (0.6, 0.6, 0.6, 0.3, 0.0)
        for step, (strength, suffix) in enumerate(zip(
                expected_strength, expected_suffix)):
            for call in result.model.calls[2 * step:2 * step + 2]:
                self.assertAlmostEqual(
                    call['paired_prefix_reference_strength'], strength)
                self.assertAlmostEqual(call['paired_stream_weight'], suffix)
                self.assertTrue(call['paired_prefix_preserve_first_frame'])
                # Fading redistribution to zero must restore the old formula
                # without removing Img2 from that formula's visible keys.
                self.assertTrue(call['paired_prefix_anchor_active'])
                self.assertIsNone(call['paired_prefix_reference_budget'])

    def test_no_fade_preserves_strength_and_suffix_weight_at_every_step(self):
        for strength in (0.0, 0.4):
            with self.subTest(strength=strength):
                result = self.run_sampler(
                    image_condition_paired_prefix_reference_strength=strength)
                for call in result.model.calls:
                    self.assertEqual(call['paired_prefix_reference_strength'], strength)
                    self.assertEqual(call['paired_stream_weight'], 0.6)
                    self.assertTrue(call['paired_prefix_anchor_active'])
                    self.assertFalse(call['paired_prefix_preserve_first_frame'])

    def test_legacy_route_switch_still_removes_right_route(self):
        result = self.run_sampler(
            image_condition_paired_prefix_route_switch_step=3)
        expected = (True, True, True, False, False)
        for step, active in enumerate(expected):
            for call in result.model.calls[2 * step:2 * step + 2]:
                self.assertEqual(call['paired_prefix_anchor_active'], active)
                self.assertIsNone(call['paired_prefix_reference_strength'])

    def test_b0_prediction_replaces_only_temporary_anchor_before_cfg_and_step(self):
        for anchor in (10, 12):
            with self.subTest(anchor=anchor):
                result = self.run_sampler(anchor=anchor)
                for step, scheduled in enumerate(result.scheduler.calls):
                    cond, uncond = result.model.calls[2 * step:2 * step + 2]
                    expected = (uncond['predictions'][0]
                                + 2.0 * (cond['predictions'][0]
                                         - uncond['predictions'][0]))
                    expected[:, anchor] = (
                        uncond['predictions'][1][:, 0]
                        + 2.0 * (cond['predictions'][1][:, 0]
                                 - uncond['predictions'][1][:, 0]))
                    torch.testing.assert_close(
                        scheduled['prediction'][0], expected)
                    # Ordinary A predictions on both sides must survive.
                    self.assertEqual(scheduled['prediction'].shape[2], 22)
                    self.assertEqual(scheduled['sample'].shape[2], 22)

    def test_temporary_anchor_survives_sampling_and_is_removed_only_for_decode(self):
        result = self.run_sampler(anchor=12)
        self.assertEqual([video.shape[1] for video in result.vae.encoded_inputs],
                         [81, 37, 85])
        for step, call in enumerate(result.model.calls):
            self.assertEqual([latent.shape[1] for latent in call['latents']],
                             [22, 10])
            torch.testing.assert_close(
                call['latents'][1], call['latents'][0][:, 12:])
            if step >= 2:
                torch.testing.assert_close(
                    call['latents'][0],
                    result.scheduler.calls[step // 2 - 1]['result'][0])
        final_working_latents = result.scheduler.calls[-1]['result'][0]
        retained_indices = torch.tensor([*range(12), *range(13, 22)])
        torch.testing.assert_close(
            result.vae.decoded_inputs[0],
            final_working_latents.index_select(1, retained_indices))
        self.assertEqual(result.vae.decoded_inputs[0].shape[1], 21)
        self.assertEqual(tuple(result.video.shape), (3, 81, 16, 16))


if __name__ == '__main__':
    unittest.main()
