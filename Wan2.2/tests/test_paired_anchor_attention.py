import importlib.util
from pathlib import Path
import unittest
from unittest import mock

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    # Load only this module so CPU routing tests do not import model weights,
    # transformers, diffusers or the rest of the GPU-oriented wan package.
    spec = importlib.util.spec_from_file_location(
        'paired_attention_under_test',
        Path(__file__).resolve().parents[1] / 'wan/modules/attention.py')
    attention = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(attention)


class PairedAnchorAttentionRoutingTest(unittest.TestCase):

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_active_prefix_uses_all_ordinary_frames_and_hybrid_anchor(self):
        # Four main frames with anchor 2, plus the aligned two-frame suffix.
        # One token per frame keeps the routing visible in the tensor values.
        q = torch.zeros(2, 4, 1, 2)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        k[0, :, 0, 0] = torch.tensor([10., 11., 12., 13.])
        v[0, :, 0, 0] = torch.tensor([20., 21., 22., 23.])
        k[1, :2, 0, 0] = torch.tensor([30., 31.])
        v[1, :2, 0, 0] = torch.tensor([40., 41.])

        calls = []

        def fake_flash(q, k, v, **kwargs):
            calls.append((q.clone(), k.clone(), v.clone()))
            return torch.zeros(
                q.size(0), q.size(1), q.size(2), v.size(3), dtype=q.dtype)

        rope_shifts = []

        def fake_shift(tensor, shift, freqs):
            rope_shifts.append(shift)
            return tensor.clone() + 100.0 * shift

        with (mock.patch.object(attention, 'flash_attention', fake_flash),
              mock.patch.object(
                  attention, '_shift_temporal_rope',
                  side_effect=fake_shift)):
            attention.paired_anchor_attention(
                q=q,
                k=k,
                v=v,
                seq_lens=torch.tensor([4, 2]),
                grid_sizes=torch.tensor([[4, 1, 1], [2, 1, 1]]),
                freqs=torch.zeros(4, 1, dtype=torch.complex64),
                anchor_index=2,
                dtype=torch.bfloat16)

        # Calls: suffix, one prefix softmax, anchor messenger, main suffix.
        # The active prefix sees A0, A1, one hybrid anchor, and compressed A3.
        # The hybrid key is history-aware A2 while its value is fresh-cache B0.
        _, prefix_key, prefix_value = calls[1]
        self.assertEqual(
            prefix_key[0, :, 0, 0].tolist(), [10.0, 11.0, 12.0, -87.0])
        self.assertEqual(
            prefix_value[0, :, 0, 0].tolist(), [20.0, 21.0, 40.0, 23.0])
        self.assertEqual(rope_shifts, [-1, -1])

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_switched_prefix_uses_all_ordinary_stream_a_frames(self):
        q = torch.zeros(2, 4, 1, 2)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        k[0, :, 0, 0] = torch.tensor([10., 11., 12., 13.])
        v[0, :, 0, 0] = torch.tensor([20., 21., 22., 23.])

        calls = []

        def fake_flash(q, k, v, **kwargs):
            calls.append((q.clone(), k.clone(), v.clone()))
            return torch.zeros(
                q.size(0), q.size(1), q.size(2), v.size(3), dtype=q.dtype)

        def fake_shift(tensor, shift, freqs):
            return tensor.clone() + 100.0 * shift

        with (mock.patch.object(attention, 'flash_attention', fake_flash),
              mock.patch.object(
                  attention, '_shift_temporal_rope',
                  side_effect=fake_shift)):
            attention.paired_anchor_attention(
                q=q,
                k=k,
                v=v,
                seq_lens=torch.tensor([4, 2]),
                grid_sizes=torch.tensor([[4, 1, 1], [2, 1, 1]]),
                freqs=torch.zeros(4, 1, dtype=torch.complex64),
                anchor_index=2,
                prefix_anchor_active=False,
                dtype=torch.bfloat16)

        # Once switched, B0 and A2 are absent; compressed A3 is restored.
        _, prefix_key, prefix_value = calls[1]
        self.assertEqual(
            prefix_key[0, :, 0, 0].tolist(), [10.0, 11.0, -87.0])
        self.assertEqual(
            prefix_value[0, :, 0, 0].tolist(), [20.0, 21.0, 23.0])


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PairedAnchorReferenceBudgetTest(unittest.TestCase):

    def setUp(self):
        self.frames = 7
        self.anchor = 4
        self.height, self.width = 2, 3
        self.tokens = self.height * self.width
        self.main_length = self.frames * self.tokens
        self.suffix_length = (self.frames - self.anchor) * self.tokens
        self.padded_length = self.main_length + 5
        self.heads, self.channels = 3, 8
        generator = torch.Generator().manual_seed(831)
        self.q, self.k, self.v = [
            torch.randn(
                2, self.padded_length, self.heads, self.channels,
                generator=generator)
            for _ in range(3)
        ]
        # Padding is deliberately large so leaking padded tokens is visible.
        for tensor in (self.q, self.k, self.v):
            tensor[0, self.main_length:] = 1000.
            tensor[1, self.suffix_length:] = -1000.
        phases = torch.outer(
            torch.arange(self.frames, dtype=torch.float32),
            torch.tensor([0.7, 0.3, 0.1, 0.05]))
        self.freqs = torch.polar(torch.ones_like(phases), phases)

    @staticmethod
    def sdpa(q, k, v, **kwargs):
        output = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
        if kwargs.get('return_lse', False):
            scores = torch.einsum(
                'bqhd,bkhd->bhqk', q.float(), k.float()) / q.size(-1)**0.5
            return output, scores.logsumexp(-1).transpose(1, 2)
        return output

    @staticmethod
    def dense(q, k, v):
        scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / q.size(-1)**0.5
        return torch.einsum('bhqk,bkhd->bqhd', scores.softmax(-1), v)

    def run_attention(self, **kwargs):
        with mock.patch.object(attention, 'flash_attention', self.sdpa):
            return attention.paired_anchor_attention(
                q=self.q.clone(), k=self.k.clone(), v=self.v.clone(),
                seq_lens=torch.tensor([self.main_length, self.suffix_length]),
                grid_sizes=torch.tensor([
                    [self.frames, self.height, self.width],
                    [self.frames - self.anchor, self.height, self.width],
                ]),
                freqs=self.freqs, anchor_index=self.anchor, **kwargs)

    def shifted_main(self, tensor):
        """Independent real-valued reference for the -1 temporal RoPE shift."""
        result = tensor[0:1, :self.main_length].clone()
        start = (self.anchor + 1) * self.tokens
        temporal_pairs = self.channels // 2 - 2 * (self.channels // 6)
        pairs = result[:, start:, :, :2 * temporal_pairs].reshape(
            1, -1, self.heads, temporal_pairs, 2).clone()
        phase = -torch.angle(self.freqs[1, :temporal_pairs])
        real = pairs[..., 0] * phase.cos() - pairs[..., 1] * phase.sin()
        imag = pairs[..., 0] * phase.sin() + pairs[..., 1] * phase.cos()
        result[:, start:, :, :2 * temporal_pairs] = torch.stack(
            [real, imag], dim=-1).flatten(-2)
        return result

    def prefix_groups(self):
        start, end = self.anchor * self.tokens, (self.anchor + 1) * self.tokens
        k = self.shifted_main(self.k)
        v = self.v[0:1, :self.main_length]
        context_k = torch.cat([k[:, self.tokens:start], k[:, end:]], dim=1)
        context_v = torch.cat([v[:, self.tokens:start], v[:, end:]], dim=1)
        return (
            (context_k, context_v),
            (k[:, :self.tokens], v[:, :self.tokens]),
            (k[:, start:end], self.v[1:2, :self.tokens]),
        )

    def dense_budget_reference(self, rho, curve):
        # A single dense softmax with normalized group logits plus log budgets
        # independently represents the desired three-softmax mixture.
        q = self.q[0:1, :self.anchor * self.tokens]
        w = torch.arange(self.anchor, dtype=q.dtype) / self.anchor
        if curve == 'smoothstep':
            w = w.square() * (3. - 2. * w)
        w = w.repeat_interleave(self.tokens).view(1, 1, -1, 1)
        weights = (torch.full_like(w, 1. - rho), rho * (1. - w), rho * w)
        logits, values = [], []
        for (k, v), weight in zip(self.prefix_groups(), weights):
            scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / q.size(-1)**0.5
            logits.append(scores - scores.logsumexp(-1, keepdim=True)
                          + weight.log())
            values.append(v)
        probabilities = torch.cat(logits, dim=-1).softmax(-1)
        return torch.einsum(
            'bhqk,bkhd->bqhd', probabilities, torch.cat(values, dim=1))

    def test_budget_matches_dense_logit_reference(self):
        for rho in (0., 0.35, 1.):
            for curve in ('linear', 'smoothstep'):
                with self.subTest(rho=rho, curve=curve):
                    actual = self.run_attention(
                        prefix_reference_budget=rho, prefix_reference_curve=curve)
                    expected = self.dense_budget_reference(rho, curve)
                    torch.testing.assert_close(
                        actual[0:1, :self.anchor * self.tokens], expected,
                        rtol=2e-5, atol=2e-6)

    def test_budget_preserves_non_prefix_outputs_and_padding(self):
        original = self.run_attention()
        for rho in (0., 0.35, 1.):
            for curve in ('linear', 'smoothstep'):
                with self.subTest(rho=rho, curve=curve):
                    actual = self.run_attention(
                        prefix_reference_budget=rho, prefix_reference_curve=curve)
                    self.assertEqual(actual.shape, self.q.shape)
                    self.assertEqual(actual.dtype, self.q.dtype)
                    self.assertTrue(torch.equal(
                        actual[0, self.anchor * self.tokens:],
                        original[0, self.anchor * self.tokens:]))
                    self.assertTrue(torch.equal(actual[1], original[1]))
                    self.assertEqual(
                        torch.count_nonzero(actual[0, self.main_length:]).item(), 0)
                    self.assertEqual(
                        torch.count_nonzero(actual[1, self.suffix_length:]).item(), 0)

    def test_none_preserves_legacy_joint_softmax_and_switch(self):
        start, end = self.anchor * self.tokens, (self.anchor + 1) * self.tokens
        main_k = self.shifted_main(self.k)
        main_v = self.v[0:1, :self.main_length]
        for active in (True, False):
            with self.subTest(active=active):
                if active:
                    k = main_k
                    v = torch.cat([
                        main_v[:, :start], self.v[1:2, :self.tokens],
                        main_v[:, end:],
                    ], dim=1)
                else:
                    k = torch.cat([main_k[:, :start], main_k[:, end:]], dim=1)
                    v = torch.cat([main_v[:, :start], main_v[:, end:]], dim=1)
                expected = self.dense(self.q[0:1, :start], k, v)
                implicit = self.run_attention(prefix_anchor_active=active)
                explicit = self.run_attention(
                    prefix_anchor_active=active, prefix_reference_budget=None,
                    prefix_reference_curve='smoothstep')
                torch.testing.assert_close(
                    implicit[0:1, :start], expected, rtol=2e-5, atol=2e-6)
                self.assertTrue(torch.equal(implicit, explicit))

    def test_rho_zero_is_context_only_and_not_legacy(self):
        start = self.anchor * self.tokens
        context_k, context_v = self.prefix_groups()[0]
        expected = self.dense(self.q[0:1, :start], context_k, context_v)
        actual = self.run_attention(prefix_reference_budget=0.)[0:1, :start]
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        self.assertFalse(torch.allclose(actual, self.run_attention()[0:1, :start]))
        # Neither reference's values may leak through the context-only route.
        self.v[0, :self.tokens] += 30.
        self.v[0, start:start + self.tokens] -= 20.
        self.v[1, :self.tokens] += 40.
        changed = self.run_attention(prefix_reference_budget=0.)[0:1, :start]
        self.assertTrue(torch.equal(actual, changed))

    def test_rho_one_time_curve_broadcasts_to_every_spatial_token_and_head(self):
        # Constant group values turn output into a readable time-budget curve.
        # Nonzero context values also detect an accidental context contribution.
        self.v.fill_(100.)
        self.v[0, :self.tokens] = 2.
        self.v[1, :self.tokens] = 10.
        for curve in ('linear', 'smoothstep'):
            with self.subTest(curve=curve):
                actual = self.run_attention(
                    prefix_reference_budget=1., prefix_reference_curve=curve)
                weights = torch.arange(self.anchor) / self.anchor
                if curve == 'smoothstep':
                    weights = weights.square() * (3. - 2. * weights)
                expected = (2. + 8. * weights).view(
                    self.anchor, 1, 1, 1).expand(
                        self.anchor, self.tokens, self.heads, self.channels)
                prefix = actual[0, :self.anchor * self.tokens].reshape_as(expected)
                torch.testing.assert_close(prefix, expected, rtol=1e-5, atol=2e-6)
                torch.testing.assert_close(prefix[0], torch.full_like(prefix[0], 2.))
                torch.testing.assert_close(prefix[2], torch.full_like(prefix[2], 6.))
                self.assertTrue(torch.all(prefix[-1] < 10.))

    def test_hybrid_right_reference_keeps_a_key_and_b_value(self):
        start = self.anchor * self.tokens
        original = self.run_attention(prefix_reference_budget=0.6)[0:1, :start]
        # B0's K belongs to its independent suffix branch, not prefix retrieval.
        self.k[1, :self.tokens] += 13.
        # A's messenger value is also excluded from all three prefix groups.
        self.v[0, start:start + self.tokens] -= 17.
        unchanged = self.run_attention(prefix_reference_budget=0.6)[0:1, :start]
        self.assertTrue(torch.equal(original, unchanged))
        # Alter spatially distinct A-anchor keys to change retrieval within R.
        self.k[0, start, :, 0] += 11.
        key_changed = self.run_attention(prefix_reference_budget=0.6)[0:1, :start]
        self.assertFalse(torch.allclose(original[:, self.tokens:],
                                        key_changed[:, self.tokens:]))
        torch.testing.assert_close(original[:, :self.tokens],
                                   key_changed[:, :self.tokens])
        self.v[1, :self.tokens] += 7.
        value_changed = self.run_attention(prefix_reference_budget=0.6)[0:1, :start]
        self.assertFalse(torch.allclose(key_changed[:, self.tokens:],
                                        value_changed[:, self.tokens:]))
        torch.testing.assert_close(key_changed[:, :self.tokens],
                                   value_changed[:, :self.tokens])

    def test_smallest_anchor_has_nonempty_post_anchor_context(self):
        self.anchor = 1
        self.suffix_length = (self.frames - self.anchor) * self.tokens
        actual = self.run_attention(prefix_reference_budget=0.4)
        expected = self.dense_budget_reference(0.4, 'linear')
        torch.testing.assert_close(
            actual[0:1, :self.tokens], expected, rtol=2e-5, atol=2e-6)

    def test_new_kwargs_preserve_legacy_positional_signature(self):
        with mock.patch.object(attention, 'flash_attention', self.sdpa):
            actual = attention.paired_anchor_attention(
                self.q.clone(), self.k.clone(), self.v.clone(),
                torch.tensor([self.main_length, self.suffix_length]),
                torch.tensor([
                    [self.frames, self.height, self.width],
                    [self.frames - self.anchor, self.height, self.width],
                ]), self.freqs, self.anchor, 0.25, True, (-1, -1), torch.bfloat16)
        self.assertTrue(torch.equal(actual, self.run_attention()))

    def test_rejects_invalid_reference_budgets(self):
        for budget in (-0.01, 1.01, float('nan'), float('inf'), -float('inf')):
            with self.subTest(budget=budget):
                with self.assertRaisesRegex(ValueError, 'prefix reference budget'):
                    self.run_attention(prefix_reference_budget=budget)

    def test_rejects_unknown_curve(self):
        for budget in (None, 0.3):
            with self.subTest(budget=budget):
                with self.assertRaisesRegex(ValueError, 'prefix reference curve'):
                    self.run_attention(
                        prefix_reference_budget=budget, prefix_reference_curve='cubic')

    def test_rejects_budget_with_disabled_anchor_route(self):
        for budget in (0., 0.3, 1.):
            with self.subTest(budget=budget):
                with self.assertRaisesRegex(ValueError, 'active prefix anchor route'):
                    self.run_attention(
                        prefix_reference_budget=budget, prefix_anchor_active=False)

    def dense_strength_reference(self, strength, curve='linear'):
        # Independently redistribute dense per-token probabilities. Context
        # probabilities remain untouched; neither the production residual
        # formula nor its log-partition ratios are used here.
        q = self.q[0:1, :self.anchor * self.tokens].float()
        scores, values = [], []
        for k, v in self.prefix_groups():
            scores.append(torch.einsum(
                'bqhd,bkhd->bhqk', q, k.float()) / q.size(-1)**0.5)
            values.append(v.float())
        native = torch.cat(scores, dim=-1).softmax(-1)
        lengths = [score.size(-1) for score in scores]
        context, left, right = native.split(lengths, dim=-1)
        reference_mass = (left.sum(-1, keepdim=True)
                          + right.sum(-1, keepdim=True))
        w = torch.arange(self.anchor, dtype=torch.float32) / self.anchor
        if curve == 'smoothstep':
            w = w.square() * (3. - 2. * w)
        w = w.repeat_interleave(self.tokens).view(1, 1, -1, 1)
        target = torch.cat([
            context,
            (1. - strength) * left
            + strength * reference_mass * (1. - w) * scores[1].softmax(-1),
            (1. - strength) * right
            + strength * reference_mass * w * scores[2].softmax(-1),
        ], dim=-1)
        output = torch.einsum(
            'bhqk,bkhd->bqhd', target, torch.cat(values, dim=1))
        return output, native, target, lengths

    def test_strength_matches_independent_dense_probability_redistribution(self):
        for strength in (0.1, 0.5, 1.):
            for curve in ('linear', 'smoothstep'):
                with self.subTest(strength=strength, curve=curve):
                    actual = self.run_attention(
                        prefix_reference_strength=strength,
                        prefix_reference_curve=curve)
                    expected = self.dense_strength_reference(strength, curve)[0]
                    torch.testing.assert_close(
                        actual[0:1, :self.anchor * self.tokens], expected,
                        rtol=3e-5, atol=3e-6)

    def test_strength_zero_uses_exact_legacy_fastpath(self):
        original = self.run_attention()
        for preserve in (False, True):
            with self.subTest(preserve=preserve):
                with mock.patch.object(
                        attention, 'flash_attention', wraps=self.sdpa) as backend:
                    actual = attention.paired_anchor_attention(
                        q=self.q.clone(), k=self.k.clone(), v=self.v.clone(),
                        seq_lens=torch.tensor([self.main_length, self.suffix_length]),
                        grid_sizes=torch.tensor([
                            [self.frames, self.height, self.width],
                            [self.frames - self.anchor, self.height, self.width],
                        ]), freqs=self.freqs, anchor_index=self.anchor,
                        prefix_reference_strength=0.,
                        prefix_preserve_first_frame=preserve)
                self.assertTrue(torch.equal(actual, original))
                self.assertEqual(backend.call_count, 4)
                self.assertFalse(any(
                    call.kwargs.get('return_lse', False)
                    for call in backend.call_args_list))

    def test_strength_preserves_native_context_and_total_reference_mass(self):
        # Values act as group indicators so output channels expose the masses
        # actually used by production, independently of ordinary value mixing.
        self.v.zero_()
        start, end = self.anchor * self.tokens, (self.anchor + 1) * self.tokens
        self.v[0, self.tokens:start, :, 0] = 1.
        self.v[0, end:self.main_length, :, 0] = 1.
        self.v[0, :self.tokens, :, 1] = 1.
        self.v[1, :self.tokens, :, 2] = 1.
        baseline = self.run_attention()[0:1, :start]
        for strength in (0.25, 1.):
            with self.subTest(strength=strength):
                actual = self.run_attention(
                    prefix_reference_strength=strength)[0:1, :start]
                torch.testing.assert_close(actual[..., 0], baseline[..., 0])
                torch.testing.assert_close(
                    actual[..., 1:3].sum(-1), baseline[..., 1:3].sum(-1),
                    rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(
                    actual[..., :3].sum(-1), torch.ones_like(actual[..., 0]),
                    rtol=2e-5, atol=2e-6)
        # With full redistribution, direct left and right masses balance at
        # the temporal midpoint, without forcing any particular total mass.
        middle = self.anchor // 2 * self.tokens
        torch.testing.assert_close(
            actual[:, middle:middle + self.tokens, :, 1],
            actual[:, middle:middle + self.tokens, :, 2], rtol=2e-5, atol=2e-6)
        self.assertGreater(baseline[..., 1:3].sum(-1).std().item(), 0.01)

    def test_preserve_first_frame_keeps_native_attention_and_other_queries(self):
        original = self.run_attention()
        changed = self.run_attention(prefix_reference_strength=0.7)
        preserved = self.run_attention(
            prefix_reference_strength=0.7, prefix_preserve_first_frame=True)
        self.assertTrue(torch.equal(
            preserved[0, :self.tokens], original[0, :self.tokens]))
        self.assertTrue(torch.equal(
            preserved[0, self.tokens:], changed[0, self.tokens:]))
        self.assertFalse(torch.allclose(
            preserved[0, :self.tokens], changed[0, :self.tokens]))

    def test_strength_preserves_suffix_messenger_and_padding(self):
        original = self.run_attention()
        for strength in (0.25, 1.):
            with self.subTest(strength=strength):
                actual = self.run_attention(prefix_reference_strength=strength)
                self.assertEqual(actual.shape, self.q.shape)
                self.assertEqual(actual.dtype, self.q.dtype)
                self.assertTrue(torch.equal(
                    actual[0, self.anchor * self.tokens:],
                    original[0, self.anchor * self.tokens:]))
                self.assertTrue(torch.equal(actual[1], original[1]))
                self.assertEqual(
                    torch.count_nonzero(actual[0, self.main_length:]).item(), 0)
                self.assertEqual(
                    torch.count_nonzero(actual[1, self.suffix_length:]).item(), 0)

    def test_strength_keeps_existing_hybrid_reference_sources(self):
        start = self.anchor * self.tokens
        original = self.run_attention(prefix_reference_strength=0.5)[0:1, :start]
        self.k[1, :self.tokens] += 13.
        self.v[0, start:start + self.tokens] -= 17.
        unchanged = self.run_attention(prefix_reference_strength=0.5)[0:1, :start]
        self.assertTrue(torch.equal(original, unchanged))
        self.k[0, start, :, 0] += 11.
        key_changed = self.run_attention(prefix_reference_strength=0.5)[0:1, :start]
        self.assertFalse(torch.allclose(original, key_changed))
        self.v[1, :self.tokens] += 7.
        value_changed = self.run_attention(prefix_reference_strength=0.5)[0:1, :start]
        self.assertFalse(torch.allclose(key_changed, value_changed))

    def test_strength_is_stable_with_extreme_logits(self):
        # These finite logits overflow a direct exp(scores), but stable
        # softmax and differences of log-sum-exp must remain finite.
        self.q *= 50.
        self.k *= 50.
        for strength in (0.25, 1.):
            with self.subTest(strength=strength):
                actual = self.run_attention(prefix_reference_strength=strength)
                expected = self.dense_strength_reference(strength)[0]
                self.assertTrue(torch.isfinite(actual).all().item())
                torch.testing.assert_close(
                    actual[0:1, :self.anchor * self.tokens], expected,
                    rtol=2e-3, atol=2e-3)

    def test_strength_preserves_output_dtype(self):
        self.q, self.k, self.v = [x.to(torch.bfloat16) for x in (self.q, self.k, self.v)]
        actual = self.run_attention(prefix_reference_strength=0.5)
        expected = self.dense_strength_reference(0.5)[0]
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(
            actual[0:1, :self.anchor * self.tokens].float(), expected,
            rtol=0.04, atol=0.015)

    def test_strength_preservation_with_smallest_anchor_is_legacy(self):
        self.anchor = 1
        self.suffix_length = (self.frames - self.anchor) * self.tokens
        original = self.run_attention()
        actual = self.run_attention(
            prefix_reference_strength=1., prefix_preserve_first_frame=True)
        self.assertTrue(torch.equal(actual, original))

    def test_rejects_invalid_strength_configurations(self):
        for strength in (-0.01, 1.01, float('nan'), float('inf'), -float('inf')):
            with self.subTest(strength=strength):
                with self.assertRaisesRegex(ValueError, 'prefix reference strength'):
                    self.run_attention(prefix_reference_strength=strength)
        for strength in (0., 0.5, 1.):
            with self.subTest(strength=strength):
                with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
                    self.run_attention(
                        prefix_reference_strength=strength, prefix_reference_budget=0.)
                with self.assertRaisesRegex(ValueError, 'active prefix anchor route'):
                    self.run_attention(
                        prefix_reference_strength=strength, prefix_anchor_active=False)
        with self.assertRaisesRegex(ValueError, 'preservation requires'):
            self.run_attention(prefix_preserve_first_frame=True)
        with self.assertRaisesRegex(ValueError, 'preservation requires'):
            self.run_attention(
                prefix_preserve_first_frame=True, prefix_reference_budget=0.3)


if __name__ == '__main__':
    unittest.main()
