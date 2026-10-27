import unittest
from unittest import mock

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from wan.modules import attention


class PairedAnchorAttentionRoutingTest(unittest.TestCase):

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_duplicate_position_attention_matches_negative_log2_bias(self):
        generator = torch.Generator().manual_seed(7)
        q = torch.randn(1, 3, 2, 4, generator=generator)
        ordinary_k = torch.randn(1, 5, 2, 4, generator=generator)
        ordinary_v = torch.randn(1, 5, 2, 4, generator=generator)
        duplicate_k = torch.randn(1, 4, 2, 4, generator=generator)
        duplicate_v = torch.randn(1, 4, 2, 4, generator=generator)

        def dense_flash(q, k, v, return_lse=False, **kwargs):
            logits = torch.einsum('bqhd,bkhd->bhqk', q, k) / 2.0
            output = torch.einsum(
                'bhqk,bkhd->bqhd', logits.softmax(-1), v)
            if return_lse:
                return output, logits.logsumexp(-1).transpose(1, 2)
            return output

        with mock.patch.object(attention, 'flash_attention', dense_flash):
            actual = attention._duplicate_position_attention(
                q, ordinary_k, ordinary_v, duplicate_k, duplicate_v,
                dtype=torch.bfloat16)

        ordinary_logits = torch.einsum(
            'bqhd,bkhd->bhqk', q, ordinary_k) / 2.0
        duplicate_logits = (
            torch.einsum('bqhd,bkhd->bhqk', q, duplicate_k) / 2.0
            - torch.log(torch.tensor(2.0)))
        logits = torch.cat([ordinary_logits, duplicate_logits], dim=-1)
        values = torch.cat([ordinary_v, duplicate_v], dim=1)
        expected = torch.einsum(
            'bhqk,bkhd->bqhd', logits.softmax(-1), values)
        torch.testing.assert_close(actual, expected)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_prefix_future_nudge_matches_dense_framewise_reference(self):
        generator = torch.Generator().manual_seed(42)
        q = torch.randn(3, 10, 2, 4, generator=generator)
        k = torch.randn(2, 10, 2, 4, generator=generator)
        v = torch.randn(2, 10, 2, 4, generator=generator)
        # C: five frames, two spatial tokens each; anchor at frame 2.
        # B: three frames. Use two heads to check LSE axes.
        def dense_flash(q, k, v, return_lse=False, **kwargs):
            logits = torch.einsum('bqhd,bkhd->bhqk', q, k) / 2.0
            output = torch.einsum('bhqk,bkhd->bqhd', logits.softmax(-1), v)
            if return_lse:
                return output, logits.logsumexp(-1).transpose(1, 2)
            return output

        for eta in (0.0, 0.2, 1.0):
            with (mock.patch.object(attention, 'flash_attention', dense_flash),
                  mock.patch.object(attention, '_shift_temporal_rope',
                                    side_effect=lambda x, *args: x)):
                output = attention.paired_anchor_attention(
                    q, k, v, torch.tensor([6, 10, 2]),
                    torch.tensor([[3, 1, 2], [5, 1, 2], [1, 1, 2]]),
                    torch.ones(10, 2, dtype=torch.complex64), 2,
                    prefix_anchor_active=False, prefix_future_eta=eta)
            # Prefix memory is C0:C2, H, then C4:C5. C2 is the hidden
            # transport slot and C3 is the regular copy replaced by H.
            keys = torch.cat([k[1:2, :4], k[2:3, :2], k[1:2, 8:10]], dim=1)
            values = torch.cat([v[1:2, :4], v[2:3, :2], v[1:2, 8:10]], dim=1)
            for frame in range(2):
                query = q[1:2, frame * 2:(frame + 1) * 2]
                logits = torch.einsum('bqhd,bkhd->bhqk', query, keys) / 2.0
                probabilities = logits.softmax(-1)
                split = (frame + 1) * 2
                future_mass = probabilities[..., split:].sum(-1, keepdim=True)
                target_mass = future_mass + eta * future_mass * (1 - future_mass)
                adjusted = torch.cat([
                    probabilities[..., :split] * (1 - target_mass) / (1 - future_mass),
                    probabilities[..., split:] * target_mass / future_mass,
                ], dim=-1)
                expected = torch.einsum('bhqk,bkhd->bqhd', adjusted, values)
                torch.testing.assert_close(
                    output[1:2, frame * 2:(frame + 1) * 2], expected)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_guidance_rms_cap_limits_joint_residual_per_frame(self):
        q = torch.zeros(3, 6, 1, 2)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        calls = []

        def fake_flash(q, k, v, **kwargs):
            calls.append((q.clone(), k.clone(), v.clone()))
            result = torch.full_like(q, float(len(calls)))
            if kwargs.get('return_lse', False):
                return result, torch.zeros(
                    q.size(0), q.size(1), q.size(2), dtype=torch.float32)
            return result

        with mock.patch.object(attention, 'flash_attention', fake_flash):
            output = attention.self_attention_guidance_attention(
                q=q,
                k=k,
                v=v,
                seq_lens=torch.tensor([6, 6, 4]),
                grid_sizes=torch.tensor(
                    [[6, 1, 1], [6, 1, 1], [4, 1, 1]]),
                freqs=torch.zeros(5, 1),
                anchor_index=2,
                joint_weight=0.5,
                img2_weight=0.25,
                joint_rms_cap=0.1)

        # M output RMS is 1.0. Both the prefix and suffix injected residuals
        # are capped to 0.1 even though their uncapped deltas are 1 and 2.
        self.assertEqual(
            output[0, :, 0, 0].tolist(),
            [1.1, 1.1, 0.0, 1.1, 1.1, 1.1])

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_guidance_after_stop_keeps_b_suffix_guidance(self):
        q = torch.zeros(2, 6, 1, 2)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        calls = []

        def fake_flash(q, k, v, **kwargs):
            calls.append((q.clone(), k.clone(), v.clone()))
            return torch.full_like(q, float(len(calls)))

        with mock.patch.object(attention, 'flash_attention', fake_flash):
            output = attention.self_attention_guidance_attention(
                q=q,
                k=k,
                v=v,
                seq_lens=torch.tensor([6, 4]),
                grid_sizes=torch.tensor([[6, 1, 1], [4, 1, 1]]),
                freqs=torch.zeros(5, 1),
                anchor_index=2,
                joint_weight=0.0,
                img2_weight=0.25)

        self.assertEqual(len(calls), 2)
        # Prefix remains pure M; suffix receives 25% of aligned B1: output.
        self.assertEqual(
            output[0, :, 0, 0].tolist(),
            [1.0, 1.0, 0.0, 1.25, 1.25, 1.25])
        self.assertEqual(
            output[1, :, 0, 0].tolist(),
            [2.0, 2.0, 2.0, 2.0, 0.0, 0.0])

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_guidance_m_only_handoff_isolates_transport_anchor(self):
        q = torch.zeros(1, 5, 1, 2)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        q[0, :, 0, 0] = torch.arange(5, dtype=torch.float32)
        k[0, :, 0, 0] = torch.arange(10, 15, dtype=torch.float32)
        v[0, :, 0, 0] = torch.arange(20, 25, dtype=torch.float32)
        calls = []

        def fake_flash(q, k, v, **kwargs):
            calls.append((q.clone(), k.clone(), v.clone()))
            return torch.full_like(q, 7.0)

        with mock.patch.object(attention, 'flash_attention', fake_flash):
            output = attention.self_attention_guidance_attention(
                q=q,
                k=k,
                v=v,
                seq_lens=torch.tensor([5]),
                grid_sizes=torch.tensor([[5, 1, 1]]),
                freqs=torch.zeros(5, 1),
                anchor_index=2)

        self.assertEqual(len(calls), 1)
        # The expanded M transport anchor is absent from retained Q/K/V.
        self.assertEqual(
            calls[0][0][0, :, 0, 0].tolist(), [0., 1., 3., 4.])
        self.assertEqual(
            calls[0][1][0, :, 0, 0].tolist(), [10., 11., 13., 14.])
        self.assertEqual(
            calls[0][2][0, :, 0, 0].tolist(), [20., 21., 23., 24.])
        self.assertEqual(
            output[0, :, 0, 0].tolist(), [7., 7., 0., 7., 7.])

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_guidance_aligns_single_joint_anchor_to_retained_m(self):
        q = torch.zeros(3, 6, 1, 2)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        k[0, :, 0, 0] = torch.arange(20, 26, dtype=torch.float32)
        v[0, :, 0, 0] = torch.arange(30, 36, dtype=torch.float32)
        k[1, :, 0, 0] = torch.arange(6, dtype=torch.float32)
        v[1, :, 0, 0] = torch.arange(10, 16, dtype=torch.float32)
        k[2, :4, 0, 0] = torch.tensor([40., 41., 42., 43.])
        v[2, :4, 0, 0] = torch.tensor([50., 51., 52., 53.])
        calls = []

        def fake_flash(q, k, v, **kwargs):
            calls.append((q.clone(), k.clone(), v.clone()))
            result = torch.full(
                (q.size(0), q.size(1), q.size(2), v.size(3)),
                float(len(calls)),
                dtype=q.dtype)
            if kwargs.get('return_lse', False):
                return result, torch.zeros(
                    q.size(0), q.size(1), q.size(2), dtype=torch.float32)
            return result

        def fake_shift(tensor, shift, freqs):
            return tensor + shift * 100

        with (mock.patch.object(attention, 'flash_attention', fake_flash),
              mock.patch.object(
                  attention, '_shift_temporal_rope', side_effect=fake_shift)):
            output = attention.self_attention_guidance_attention(
                q=q,
                k=k,
                v=v,
                seq_lens=torch.tensor([6, 6, 4]),
                grid_sizes=torch.tensor(
                    [[6, 1, 1], [6, 1, 1], [4, 1, 1]]),
                freqs=torch.zeros(5, 1),
                anchor_index=2,
                joint_weight=0.5,
                img2_weight=0.25)

        self.assertEqual(len(calls), 7)
        # M excludes its anchor from retained Q/K/V.
        self.assertEqual(
            calls[0][1][0, :, 0, 0].tolist(), [20., 21., 23., 24., 25.])
        self.assertEqual(
            calls[0][2][0, :, 0, 0].tolist(), [30., 31., 33., 34., 35.])
        # J prefix ordinary group excludes the anchor and first suffix frame;
        # its duplicate group contains exactly those two representations.
        self.assertEqual(
            calls[1][1][0, :, 0, 0].tolist(), [0., 1., 4., 5.])
        self.assertEqual(
            calls[1][2][0, :, 0, 0].tolist(), [10., 11., 14., 15.])
        self.assertEqual(
            calls[2][1][0, :, 0, 0].tolist(),
            [52.0, 3.0])
        self.assertEqual(
            calls[2][2][0, :, 0, 0].tolist(),
            [12.0, 13.0])
        # J suffix ordinary group omits the last prefix and anchor; its
        # duplicate group contains the colocated last-prefix/anchor pair.
        self.assertEqual(
            calls[3][1][0, :, 0, 0].tolist(), [0., 3., 4., 5.])
        self.assertEqual(
            calls[3][2][0, :, 0, 0].tolist(), [10., 13., 14., 15.])
        self.assertEqual(
            calls[4][1][0, :, 0, 0].tolist(), [1., -48.])
        self.assertEqual(
            calls[4][2][0, :, 0, 0].tolist(), [11., 12.])
        # J anchor reads itself and the J suffix; B remains standard.
        self.assertEqual(
            calls[5][0][0, :, 0, 0].tolist(), [-50.])
        self.assertEqual(
            calls[5][1][0, :, 0, 0].tolist(), [-48., 3., 4., 5.])
        self.assertEqual(
            calls[6][1][0, :, 0, 0].tolist(), [40., 41., 42., 43.])
        # Equal group LSEs become a 1/3 duplicate-group weight after -log(2).
        torch.testing.assert_close(
            output[0, :, 0, 0],
            torch.tensor([5 / 3, 5 / 3, 0., 8 / 3, 8 / 3, 8 / 3]))
        torch.testing.assert_close(
            output[1, :, 0, 0],
            torch.tensor([7 / 3, 7 / 3, 6., 13 / 3, 13 / 3, 13 / 3]))
        self.assertEqual(
            output[2, :, 0, 0].tolist(), [7., 7., 7., 7., 0., 0.])

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_guidance_scale_must_be_a_convex_weight(self):
        q = torch.zeros(3, 6, 1, 2)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        calls = []

        def fake_flash(q, k, v, **kwargs):
            calls.append((q.clone(), k.clone(), v.clone(), kwargs))
            return torch.full(
                (q.size(0), q.size(1), q.size(2), v.size(3)),
                float(len(calls)),
                dtype=q.dtype)

        def fake_shift(tensor, shift, freqs):
            return tensor + shift * 100

        with self.assertRaisesRegex(ValueError, "joint attention weight"):
            attention.self_attention_guidance_attention(
                q=q,
                k=k,
                v=v,
                seq_lens=torch.tensor([6, 6, 4]),
                grid_sizes=torch.tensor(
                    [[6, 1, 1], [6, 1, 1], [4, 1, 1]]),
                freqs=torch.zeros(5, 1),
                anchor_index=2,
                joint_weight=1.25,
                img2_weight=0.5)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_temporary_anchor_rope_preserves_aligned_expanded_layout(self):
        # Five frames, one token per frame, with temporary anchor 2.
        values = torch.arange(5, dtype=torch.float32).view(1, 5, 1, 1)

        def fake_shift(tensor, shift, freqs):
            return tensor + shift * 10

        with mock.patch.object(
                attention, '_shift_temporal_rope', side_effect=fake_shift):
            shifted = attention.temporary_anchor_rope(
                values,
                grid_sizes=torch.tensor([[5, 1, 1]]),
                freqs=torch.zeros(5, 1),
                anchor_index=2)

        # Prefix positions stay fixed, the anchor moves to 1.5, and retained
        # post-anchor frames compress left by one full latent position.
        self.assertEqual(
            shifted.flatten().tolist(), [0.0, 1.0, -3.0, -7.0, -6.0])

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_paired_prefix_uses_joint_proxy_but_regular_frame_velocity(self):
        q = torch.zeros(3, 4, 1, 2)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        k[0, :2, 0, 0] = torch.tensor([30., 31.])
        v[0, :2, 0, 0] = torch.tensor([40., 41.])
        k[1, :, 0, 0] = torch.tensor([50., 51., 52., 53.])
        v[1, :, 0, 0] = torch.tensor([60., 61., 62., 63.])
        k[2, 0, 0, 0] = 70.
        v[2, 0, 0, 0] = 80.
        for legacy_anchor_flag in (True, False):
            calls = []
            def fake_flash(q, k, v, **kwargs):
                calls.append((q.clone(), k.clone(), v.clone()))
                return torch.full_like(q, float(len(calls)))
            with (mock.patch.object(attention, 'flash_attention', fake_flash),
                  mock.patch.object(attention, '_shift_temporal_rope',
                                    side_effect=lambda x, shift, f: x + 100 * shift)):
                output = attention.paired_anchor_attention(
                    q, k, v, torch.tensor([2, 4, 1]),
                    torch.tensor([[2, 1, 1], [4, 1, 1], [1, 1, 1]]),
                    torch.ones(4, 1, dtype=torch.complex64), 2,
                    stream_weight=0.25, prefix_anchor_active=legacy_anchor_flag)
            self.assertEqual(len(calls), 3)
            # Prefix reads C0, C1 and H (shifted to C3's compressed
            # coordinate). It reads neither transport C2 nor regular C3.
            self.assertEqual(
                calls[1][1][0, :, 0, 0].tolist(), [50., 51., 270.])
            self.assertEqual(
                calls[1][2][0, :, 0, 0].tolist(), [60., 61., 80.])
            # C suffix uses ordinary C, including regular C3, and excludes C2.
            self.assertEqual(
                calls[2][1][0, :, 0, 0].tolist(), [50., 51., -47.])
            self.assertEqual(
                calls[2][2][0, :, 0, 0].tolist(), [60., 61., 63.])
            self.assertEqual(output.shape, q.shape)
            # B output=1, C prefix=2, hidden slot=B0=1, C suffix=2.5,
            # and H receives aligned B1 output=1. H's prediction is discarded
            # by image2video.py rather than copied into regular C3.
            self.assertEqual(output[1, :, 0, 0].tolist(), [2., 2., 1., 2.5])
            self.assertEqual(output[2, 0, 0, 0].item(), 1.)


if __name__ == '__main__':
    unittest.main()
