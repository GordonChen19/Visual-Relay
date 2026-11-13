import math
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
    def test_joint_img2_guidance_updates_anchor_and_suffix_only(self):
        q = torch.zeros(2, 5, 1, 2)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        calls = []

        def fake_flash(q, k, v, **kwargs):
            calls.append((q.size(1), k.size(1)))
            return torch.full_like(q, float(len(calls)))

        with mock.patch.object(attention, 'flash_attention', fake_flash):
            output = attention.joint_img2_guidance_attention(
                q=q,
                k=k,
                v=v,
                seq_lens=torch.tensor([5, 3]),
                grid_sizes=torch.tensor([[5, 1, 1], [3, 1, 1]]),
                anchor_index=2,
                img2_weight=0.25)

        # Full J attention, suffix-only J-anchor attention, then full B.
        self.assertEqual(calls, [(5, 5), (1, 3), (3, 3)])
        self.assertEqual(output[0, :, 0, 0].tolist(),
                         [1.0, 1.0, 2.25, 1.5, 1.5])
        self.assertEqual(output[1, :3, 0, 0].tolist(), [3.0, 3.0, 3.0])
        self.assertEqual(output[1, 3:, 0, 0].tolist(), [0.0, 0.0])

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_shifted_joint_guidance_keeps_attention_streams_independent(self):
        q = torch.zeros(2, 5, 1, 2)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        calls = []

        def fake_flash(q, k, v, **kwargs):
            calls.append((q.size(1), k.size(1)))
            result = torch.zeros_like(q)
            result[:, :, :, 0] = torch.arange(
                q.size(1), dtype=q.dtype)[None, :, None]
            if len(calls) == 2:
                result[:, :, :, 0] += 10
            return result

        with mock.patch.object(attention, 'flash_attention', fake_flash):
            output = attention.joint_img2_guidance_attention(
                q=q,
                k=k,
                v=v,
                seq_lens=torch.tensor([5, 4]),
                grid_sizes=torch.tensor([[5, 1, 1], [4, 1, 1]]),
                anchor_index=2,
                img2_weight=0.25)

        self.assertEqual(calls, [(5, 5), (4, 4)])
        self.assertEqual(output[0, :, 0, 0].tolist(),
                         [0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(output[1, :4, 0, 0].tolist(),
                         [10.0, 11.0, 12.0, 13.0])

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_aligned_b0_velocity_only_never_injects_layerwise_outputs(self):
        q = torch.zeros(2, 5, 1, 2)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        v[0, :, 0, 0] = torch.arange(5, dtype=torch.float32)
        v[1, :3, 0, 0] = torch.tensor([100., 101., 102.])

        def uniform_attention(q, k, v, **kwargs):
            return v.mean(dim=1, keepdim=True).expand_as(q).clone()

        for compressed in (False, True):
            with self.subTest(compressed=compressed), \
                 mock.patch.object(attention, 'flash_attention', uniform_attention), \
                 mock.patch.object(attention, '_shift_temporal_rope',
                                   lambda x, delta, freqs: x):
                output = attention.joint_img2_guidance_attention(
                    q=q, k=k, v=v, seq_lens=torch.tensor([5, 3]),
                    grid_sizes=torch.tensor([[5, 1, 1], [3, 1, 1]]),
                    anchor_index=2, img2_weight=1.0,
                    rope_compression=compressed, velocity_only=True,
                    freqs=torch.ones(2, 1))
                expected = [2., 2., 3., 2., 2.] if compressed else [2.] * 5
                self.assertEqual(output[0, :, 0, 0].tolist(), expected)
                self.assertEqual(output[1, :, 0, 0].tolist(),
                                 [101., 101., 101., 0., 0.])

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_joint_attention_diagnostics_match_dense_probabilities(self):
        torch.manual_seed(17)
        q = torch.randn(1, 10, 2, 4)
        k = torch.randn_like(q)
        suffix_q = torch.randn(1, 4, 2, 4)
        suffix_k = torch.randn_like(k)
        with attention.collect_joint_attention_shares(True) as samples:
            attention._record_joint_attention_shares(
                q, k, 2, 2, suffix_q, suffix_k)
        expected = []
        for query, memory, targets in (
                (q[:, 2:4], k, (2, 3)),
                (suffix_q[:, :2], suffix_k, (2, 1))):
            probabilities = torch.softmax(
                torch.einsum('bqhd,bkhd->bhqk', query, memory) / 2, dim=-1)
            expected.extend([
                probabilities[..., frame * 2:(frame + 1) * 2].sum()
                for frame in targets])
        torch.testing.assert_close(samples[0][:4], torch.stack(expected))
        self.assertEqual(samples[0][4].item(), 4)
        attention._record_joint_attention_shares(q, k, 2, 2)
        self.assertEqual(len(samples), 1)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_shifted_joint_rope_compression_is_query_dependent(self):
        q = torch.zeros(2, 5, 1, 2)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        calls = []
        shifts = []

        def fake_flash(q, k, v, **kwargs):
            calls.append((q.size(1), k.size(1)))
            return torch.full_like(q, float(len(calls)))

        def fake_shift(x, delta, freqs):
            shifts.append((x.size(1), delta))
            return x

        with mock.patch.object(attention, 'flash_attention', fake_flash), \
             mock.patch.object(attention, '_shift_temporal_rope', fake_shift):
            output = attention.joint_img2_guidance_attention(
                q=q,
                k=k,
                v=v,
                seq_lens=torch.tensor([5, 4]),
                grid_sizes=torch.tensor([[5, 1, 1], [4, 1, 1]]),
                anchor_index=2,
                rope_compression=True,
                freqs=torch.ones(2, 1))

        # Prefix shifts J3:J4, while suffix attention shifts Q3:J4 and
        # K2:J4. With anchor=2 this produces J2/J3@2 for prefix queries and
        # J1/J2@1 for suffix queries without disturbing suffix self-offsets.
        self.assertEqual(shifts, [(2, -1), (2, -1), (3, -1)])
        # Prefix reads all J memory, anchor reads itself/future only, suffix
        # reads all J memory, and B retains one ordinary attention call.
        self.assertEqual(calls, [(2, 5), (1, 3), (2, 5), (4, 4)])
        self.assertEqual(output[0, :, 0, 0].tolist(),
                         [1.0, 1.0, 2.0, 3.0, 3.0])

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_handoff_mode_compresses_both_routes_with_visible_anchor(self):
        q = torch.zeros(2, 5, 1, 2)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        calls = []
        shifts = []

        def fake_flash(q, k, v, **kwargs):
            calls.append((q.size(1), k.size(1),
                          kwargs.get('return_lse', False)))
            output = torch.zeros_like(q)
            if kwargs.get('return_lse', False):
                return output, torch.full(
                    (q.size(0), q.size(1), q.size(2)),
                    math.log(k.size(1)), dtype=torch.float32)
            return output

        def fake_shift(x, delta, freqs):
            shifts.append((x.size(1), delta))
            return x

        with mock.patch.object(attention, 'flash_attention', fake_flash), \
             mock.patch.object(attention, '_shift_temporal_rope', fake_shift):
            attention.joint_img2_guidance_attention(
                q=q,
                k=k,
                v=v,
                seq_lens=torch.tensor([5, 4]),
                grid_sizes=torch.tensor([[5, 1, 1], [4, 1, 1]]),
                anchor_index=2,
                rope_compression=True,
                adaptive_pair_attention=True,
                adaptive_pair_margin=1.05,
                adaptive_pair_alpha_max=0.7,
                freqs=torch.ones(2, 1))

        # Prefix memory J3:J4 shifts left and retains J2. Suffix Q3:Q4
        # and K2:K4 shift left together. Every route uses unpartitioned softmax.
        self.assertEqual(shifts, [(2, -1), (2, -1.0), (3, -1.0)])
        self.assertEqual(calls, [
            (2, 5, False), (1, 3, False), (2, 5, False), (4, 4, False),
        ])


if __name__ == '__main__':
    unittest.main()
