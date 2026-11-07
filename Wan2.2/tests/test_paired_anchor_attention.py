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
    def test_cross_tail_bridge_uses_safe_distance_routes(self):
        q = torch.zeros(3, 6, 1, 2)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        calls = []

        def fake_flash(q, k, v, **kwargs):
            calls.append((q.size(1), k.size(1)))
            result = torch.full_like(q, float(len(calls)))
            if kwargs.get('return_lse', False):
                # Equal token logits have log-sum-exp log(memory length).
                lse = torch.full(
                    (q.size(0), q.size(1), q.size(2)),
                    math.log(k.size(1)), dtype=torch.float32)
                return result, lse
            return result

        with mock.patch.object(attention, 'flash_attention', fake_flash):
            output = attention.cross_tail_bridge_attention(
                q=q,
                k=k,
                v=v,
                seq_lens=torch.tensor([5, 6, 6]),
                grid_sizes=torch.tensor(
                    [[5, 1, 1], [6, 1, 1], [6, 1, 1]]),
                tail_frames=2,
                cross_memory_prior=0.2)

        self.assertEqual(calls, [
            (2, 5), (2, 6),  # M0:1: all M + B0:5
            (1, 5), (1, 5),  # M2: all M + B1:5
            (2, 5), (2, 4),  # M3:4: all M + B2:5
            (2, 6),          # B0:1 overlap: native B only
            (4, 6), (4, 3),  # B2:5 tail: all B + pre-anchor M0:2
            (6, 6),          # D0:5: independent native D only
        ])
        self.assertTrue(torch.allclose(
            output[0, :2, 0, 0], torch.full((2,), 1.2)))
        self.assertTrue(torch.allclose(
            output[0, 2:3, 0, 0], torch.full((1,), 3.2)))
        self.assertTrue(torch.allclose(
            output[0, 3:5, 0, 0], torch.full((2,), 5.2)))
        self.assertEqual(output[0, 5, 0, 0].item(), 0.0)
        self.assertTrue(torch.allclose(
            output[1, :2, 0, 0], torch.full((2,), 7.0)))
        self.assertTrue(torch.allclose(
            output[1, 2:, 0, 0], torch.full((4,), 8.2)))
        self.assertTrue(torch.allclose(
            output[2, :, 0, 0], torch.full((6,), 10.0)))

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_cross_tail_bridge_can_disable_mass_control(self):
        q = torch.zeros(3, 6, 1, 2)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        calls = []

        def fake_flash(q, k, v, **kwargs):
            calls.append((q.size(1), k.size(1),
                          kwargs.get('return_lse', False)))
            return torch.full_like(q, float(len(calls)))

        with mock.patch.object(attention, 'flash_attention', fake_flash):
            output = attention.cross_tail_bridge_attention(
                q=q,
                k=k,
                v=v,
                seq_lens=torch.tensor([5, 6, 6]),
                grid_sizes=torch.tensor(
                    [[5, 1, 1], [6, 1, 1], [6, 1, 1]]),
                tail_frames=2,
                cross_memory_prior=0.2,
                control_memory_mass=False)

        self.assertEqual(calls, [
            (2, 11, False),  # M0:1: all M + B0:5
            (1, 10, False),  # M2: all M + B1:5
            (2, 9, False),   # M3:4: all M + B2:5
            (2, 6, False),  # B0:1 overlap: native B only
            (4, 9, False),  # B2:5 tail: all B + pre-anchor M0:2
            (6, 6, False),  # D0:5: independent native D only
        ])
        self.assertEqual(output[0, :, 0, 0].tolist(),
                         [1.0, 1.0, 2.0, 3.0, 3.0, 0.0])
        self.assertEqual(output[1, :, 0, 0].tolist(),
                         [4.0, 4.0, 5.0, 5.0, 5.0, 5.0])
        self.assertEqual(output[2, :, 0, 0].tolist(), [6.0] * 6)

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

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_one_way_img2_guidance_updates_only_aligned_main_outputs(self):
        q = torch.zeros(2, 6, 1, 2)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        v[0, :, 0, 0] = torch.tensor([10., 11., 12., 13., 14., 15.])
        v[1, :3, 0, 0] = torch.tensor([20., 21., 22.])
        calls = []
        value_calls = []

        def fake_flash(q, k, v, **kwargs):
            calls.append((q.size(1), k.size(1)))
            value_calls.append(v.clone())
            result = torch.full_like(q, float(len(calls)))
            if kwargs.get('return_lse', False):
                return result, torch.zeros(
                    q.size(0), q.size(1), q.size(2), dtype=torch.float32)
            return result

        with mock.patch.object(attention, 'flash_attention', fake_flash):
            output = attention.one_way_img2_attention(
                q=q,
                k=k,
                v=v,
                seq_lens=torch.tensor([6, 3]),
                grid_sizes=torch.tensor([[6, 1, 1], [3, 1, 1]]),
                anchor_index=2,
                img2_weight=0.25)

        self.assertEqual(
            calls,
            [(2, 4), (2, 1), (2, 1), (3, 5), (1, 3), (3, 3)])
        self.assertEqual(value_calls[0][0, :, 0, 0].tolist(),
                         [10.0, 11.0, 14.0, 15.0])
        self.assertEqual(value_calls[1][0, :, 0, 0].tolist(), [12.0])
        self.assertEqual(value_calls[2][0, :, 0, 0].tolist(), [13.0])
        self.assertEqual(output[0, :, 0, 0].tolist(),
                         [1.75, 1.75, 5.0, 4.0, 4.5, 4.5])
        self.assertEqual(output[1, :3, 0, 0].tolist(), [6.0, 6.0, 6.0])
        self.assertEqual(output[1, 3:, 0, 0].tolist(), [0.0, 0.0, 0.0])

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_one_way_prefix_uses_configurable_auxiliary_anchor_prior(self):
        q = torch.zeros(2, 6, 1, 2)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        calls = []

        def fake_flash(q, k, v, **kwargs):
            calls.append((q.size(1), k.size(1)))
            result = torch.full_like(q, float(len(calls)))
            if kwargs.get('return_lse', False):
                return result, torch.zeros(
                    q.size(0), q.size(1), q.size(2), dtype=torch.float32)
            return result

        with mock.patch.object(attention, 'flash_attention', fake_flash):
            output = attention.one_way_img2_attention(
                q=q,
                k=k,
                v=v,
                seq_lens=torch.tensor([6, 3]),
                grid_sizes=torch.tensor([[6, 1, 1], [3, 1, 1]]),
                anchor_index=2,
                img2_weight=0.25,
                aux_anchor_prior=0.75)

        # Equal group log-normalizers give 0.5 ordinary mass, then split the
        # remaining 0.5 as 0.75 A and 0.25 M: 0.5*1 + .375*2 + .125*3.
        self.assertEqual(output[0, :2, 0, 0].tolist(), [1.625, 1.625])

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_one_way_prefix_can_exclude_auxiliary_anchor(self):
        q = torch.zeros(2, 6, 1, 2)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        calls = []

        def fake_flash(q, k, v, **kwargs):
            calls.append((q.size(1), k.size(1)))
            return torch.full_like(q, float(len(calls)))

        with mock.patch.object(attention, 'flash_attention', fake_flash):
            output = attention.one_way_img2_attention(
                q=q,
                k=k,
                v=v,
                seq_lens=torch.tensor([6, 3]),
                grid_sizes=torch.tensor([[6, 1, 1], [3, 1, 1]]),
                anchor_index=2,
                img2_weight=0.25,
                prefix_reads_aux_anchor=False)

        self.assertEqual(calls, [(2, 5), (3, 5), (1, 3), (3, 3)])
        self.assertEqual(output[0, :, 0, 0].tolist(),
                         [1.0, 1.0, 3.0, 2.0, 2.5, 2.5])

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_one_way_full_b_uses_only_overlapping_outputs(self):
        q = torch.zeros(2, 7, 1, 2)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        calls = []

        def fake_flash(q, k, v, **kwargs):
            calls.append((q.size(1), k.size(1)))
            result = torch.full_like(q, float(len(calls)))
            if kwargs.get('return_lse', False):
                return result, torch.zeros(
                    q.size(0), q.size(1), q.size(2), dtype=torch.float32)
            return result

        with mock.patch.object(attention, 'flash_attention', fake_flash):
            output = attention.one_way_img2_attention(
                q=q,
                k=k,
                v=v,
                seq_lens=torch.tensor([6, 7]),
                grid_sizes=torch.tensor([[6, 1, 1], [7, 1, 1]]),
                anchor_index=2,
                img2_weight=0.25,
                prefix_reads_aux_anchor=False)

        self.assertEqual(calls, [(2, 5), (3, 5), (1, 3), (7, 7)])
        # Only B1:B2 overlap the two retained M suffix frames. B3:B6 remain
        # internal B context and cannot add output positions to M.
        self.assertEqual(output[0, :, 0, 0].tolist(),
                         [1.0, 1.0, 3.0, 2.0, 2.5, 2.5, 0.0])
        self.assertEqual(output[1, :, 0, 0].tolist(), [4.0] * 7)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_one_way_guided_frames_localizes_anchor_suffix_updates(self):
        q = torch.zeros(2, 7, 1, 2)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        calls = []

        def fake_flash(q, k, v, **kwargs):
            calls.append((q.size(1), k.size(1)))
            return torch.full_like(q, float(len(calls)))

        with mock.patch.object(attention, 'flash_attention', fake_flash):
            output = attention.one_way_img2_attention(
                q=q,
                k=k,
                v=v,
                seq_lens=torch.tensor([7, 7]),
                grid_sizes=torch.tensor([[7, 1, 1], [7, 1, 1]]),
                anchor_index=2,
                img2_weight=0.25,
                prefix_reads_aux_anchor=False,
                guided_frames=3)

        self.assertEqual(calls, [(2, 6), (4, 6), (1, 3), (7, 7)])
        # B0:B2 guide the retained anchor and next two frames. The final
        # retained M frame keeps its own output from call 2.
        self.assertEqual(output[0, :, 0, 0].tolist(),
                         [1.0, 1.0, 3.0, 2.5, 2.5, 2.5, 2.0])
        self.assertEqual(output[1, :, 0, 0].tolist(), [4.0] * 7)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_one_way_shared_anchor_uses_guided_retained_anchor_output(self):
        q = torch.zeros(2, 7, 1, 2)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        calls = []

        def fake_flash(q, k, v, **kwargs):
            calls.append((q.size(1), k.size(1)))
            return torch.full_like(q, float(len(calls)))

        with mock.patch.object(attention, 'flash_attention', fake_flash):
            output = attention.one_way_img2_attention(
                q=q,
                k=k,
                v=v,
                seq_lens=torch.tensor([7, 7]),
                grid_sizes=torch.tensor([[7, 1, 1], [7, 1, 1]]),
                anchor_index=2,
                img2_weight=0.25,
                prefix_reads_aux_anchor=False,
                guided_frames=3,
                shared_anchor_update=True)

        self.assertEqual(calls, [(2, 6), (4, 6), (1, 3), (7, 7)])
        # A2 and retained M2 both receive the same interpolated anchor output:
        # 0.75 * call-2 M output + 0.25 * call-4 B0 output = 2.5.
        self.assertEqual(output[0, :, 0, 0].tolist(),
                         [1.0, 1.0, 2.5, 2.5, 2.5, 2.5, 2.0])
        self.assertEqual(output[1, :, 0, 0].tolist(), [4.0] * 7)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_shared_position_prior_matches_one_biased_softmax(self):
        generator = torch.Generator().manual_seed(17)
        q = torch.randn(1, 3, 2, 4, generator=generator)
        ordinary_k = torch.randn(1, 5, 2, 4, generator=generator)
        ordinary_v = torch.randn(1, 5, 2, 4, generator=generator)
        anchor_k = torch.randn(1, 2, 2, 4, generator=generator)
        anchor_v = torch.randn(1, 2, 2, 4, generator=generator)
        neighbour_k = torch.randn(1, 2, 2, 4, generator=generator)
        neighbour_v = torch.randn(1, 2, 2, 4, generator=generator)
        anchor_prior = 0.8

        def dense_flash(q, k, v, return_lse=False, **kwargs):
            logits = torch.einsum('bqhd,bkhd->bhqk', q, k) / 2.0
            output = torch.einsum(
                'bhqk,bkhd->bqhd', logits.softmax(-1), v)
            if return_lse:
                return output, logits.logsumexp(-1).transpose(1, 2)
            return output

        with mock.patch.object(attention, 'flash_attention', dense_flash):
            actual = attention._shared_position_prior_attention(
                q, ordinary_k, ordinary_v, anchor_k, anchor_v,
                neighbour_k, neighbour_v, anchor_prior,
                dtype=torch.bfloat16)

        ordinary_logits = torch.einsum(
            'bqhd,bkhd->bhqk', q, ordinary_k) / 2.0
        anchor_logits = (
            torch.einsum('bqhd,bkhd->bhqk', q, anchor_k) / 2.0
            + torch.log(torch.tensor(anchor_prior)))
        neighbour_logits = (
            torch.einsum('bqhd,bkhd->bhqk', q, neighbour_k) / 2.0
            + torch.log(torch.tensor(1.0 - anchor_prior)))
        logits = torch.cat(
            [ordinary_logits, anchor_logits, neighbour_logits], dim=-1)
        values = torch.cat(
            [ordinary_v, anchor_v, neighbour_v], dim=1)
        expected = torch.einsum(
            'bhqk,bkhd->bqhd', logits.softmax(-1), values)
        torch.testing.assert_close(actual, expected)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_paired_memory_alpha_matches_dense_biased_softmax(self):
        generator = torch.Generator().manual_seed(42)
        q = torch.randn(2, 5, 2, 4, generator=generator)
        k = torch.randn(2, 5, 2, 4, generator=generator)
        v = torch.randn(2, 5, 2, 4, generator=generator)
        alpha = 0.25

        def dense_flash(q, k, v, return_lse=False, **kwargs):
            logits = torch.einsum('bqhd,bkhd->bhqk', q, k) / 2.0
            output = torch.einsum('bhqk,bkhd->bqhd', logits.softmax(-1), v)
            if return_lse:
                return output, logits.logsumexp(-1).transpose(1, 2)
            return output

        with (mock.patch.object(attention, 'flash_attention', dense_flash),
              mock.patch.object(attention, '_shift_temporal_rope',
                                side_effect=lambda x, *args: x)):
            output = attention.paired_anchor_attention(
                q, k, v, torch.tensor([3, 5]),
                torch.tensor([[3, 1, 1], [5, 1, 1]]),
                torch.ones(5, 1, dtype=torch.complex64), 2,
                stream_weight=alpha)

        retained_q = torch.cat([q[1:2, :2], q[1:2, 3:5]], dim=1)
        prefix_logits = torch.einsum(
            'bqhd,bkhd->bhqk', retained_q, k[1:2, :2]) / 2.0
        c_suffix_logits = (
            torch.einsum('bqhd,bkhd->bhqk', retained_q, k[1:2, 3:5])
            / 2.0 + math.log(1.0 - alpha))
        b_suffix_logits = (
            torch.einsum('bqhd,bkhd->bhqk', retained_q, k[0:1, 1:3])
            / 2.0 + math.log(alpha))
        logits = torch.cat(
            [prefix_logits, c_suffix_logits, b_suffix_logits], dim=-1)
        values = torch.cat(
            [v[1:2, :2], v[1:2, 3:5], v[0:1, 1:3]], dim=1)
        expected = torch.einsum(
            'bhqk,bkhd->bqhd', logits.softmax(-1), values)
        actual = torch.cat([output[1:2, :2], output[1:2, 3:5]], dim=1)
        torch.testing.assert_close(actual, expected)

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
    def test_joint_prefix_and_suffix_split_memory_match_dense_softmax(self):
        generator = torch.Generator().manual_seed(91)
        q = torch.randn(3, 6, 1, 2, generator=generator)
        k = torch.randn(3, 6, 1, 2, generator=generator)
        v = torch.randn(3, 6, 1, 2, generator=generator)
        alpha = 0.25

        def dense_flash(q, k, v, return_lse=False, **kwargs):
            logits = torch.einsum(
                'bqhd,bkhd->bhqk', q, k) / math.sqrt(q.size(-1))
            output = torch.einsum(
                'bhqk,bkhd->bqhd', logits.softmax(-1), v)
            if return_lse:
                return output, logits.logsumexp(-1).transpose(1, 2)
            return output

        with (mock.patch.object(attention, 'flash_attention', dense_flash),
              mock.patch.object(attention, '_shift_temporal_rope',
                                side_effect=lambda x, *args: x)):
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
                joint_query_single_memory_alpha=alpha)

        m_k = torch.cat([k[0:1, :2], k[0:1, 3:6]], dim=1)
        m_v = torch.cat([v[0:1, :2], v[0:1, 3:6]], dim=1)

        j_prefix_q = q[1:2, :2]
        j_prefix_logits = (
            torch.einsum('bqhd,bkhd->bhqk', j_prefix_q, k[1:2, :6])
            / math.sqrt(q.size(-1)) + math.log(1.0 - alpha))
        m_prefix_logits = (
            torch.einsum('bqhd,bkhd->bhqk', j_prefix_q, m_k)
            / math.sqrt(q.size(-1)) + math.log(alpha))
        prefix_logits = torch.cat(
            [j_prefix_logits, m_prefix_logits], dim=-1)
        prefix_values = torch.cat([v[1:2, :6], m_v], dim=1)
        expected_prefix = torch.einsum(
            'bhqk,bkhd->bqhd', prefix_logits.softmax(-1), prefix_values)
        torch.testing.assert_close(output[1:2, :2], expected_prefix)
        torch.testing.assert_close(output[0:1, :2], expected_prefix)

        j_suffix_q = q[1:2, 3:6]
        j_logits = (
            torch.einsum('bqhd,bkhd->bhqk', j_suffix_q, k[1:2, :6])
            / math.sqrt(q.size(-1)) + math.log(1.0 - alpha))
        m_logits = (
            torch.einsum('bqhd,bkhd->bhqk', j_suffix_q, m_k)
            / math.sqrt(q.size(-1)) + math.log(alpha))
        logits = torch.cat([j_logits, m_logits], dim=-1)
        values = torch.cat([v[1:2, :6], m_v], dim=1)
        expected = torch.einsum(
            'bhqk,bkhd->bqhd', logits.softmax(-1), values)
        torch.testing.assert_close(output[1:2, 3:6], expected)
        # The optional split-memory mode directly uses the joint-query result
        # for M's suffix instead of interpolating it with completed O_M.
        torch.testing.assert_close(output[0:1, 3:6], expected)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_selective_suffix_memory_matches_dense_biased_softmax(self):
        generator = torch.Generator().manual_seed(123)
        q = torch.randn(3, 6, 1, 2, generator=generator)
        k = torch.randn(3, 6, 1, 2, generator=generator)
        v = torch.randn(3, 6, 1, 2, generator=generator)
        alpha = 0.25
        anchor_prior = 0.8

        def dense_flash(q, k, v, return_lse=False, **kwargs):
            logits = torch.einsum(
                'bqhd,bkhd->bhqk', q, k) / math.sqrt(q.size(-1))
            output = torch.einsum(
                'bhqk,bkhd->bqhd', logits.softmax(-1), v)
            if return_lse:
                return output, logits.logsumexp(-1).transpose(1, 2)
            return output

        with (mock.patch.object(attention, 'flash_attention', dense_flash),
              mock.patch.object(attention, '_shift_temporal_rope',
                                side_effect=lambda x, *args: x)):
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
                anchor_prior=anchor_prior,
                selective_suffix_memory_alpha=alpha)

        scale = math.sqrt(q.size(-1))
        j_prefix_q = q[1:2, :2]
        prefix_logits = torch.cat([
            torch.einsum(
                'bqhd,bkhd->bhqk', j_prefix_q, k[1:2, :2]) / scale,
            torch.einsum(
                'bqhd,bkhd->bhqk', j_prefix_q, k[1:2, 2:3]) / scale
            + math.log(anchor_prior),
            torch.einsum(
                'bqhd,bkhd->bhqk', j_prefix_q, k[1:2, 3:4]) / scale
            + math.log(1.0 - anchor_prior) + math.log(1.0 - alpha),
            torch.einsum(
                'bqhd,bkhd->bhqk', j_prefix_q, k[1:2, 4:6]) / scale
            + math.log(1.0 - alpha),
            torch.einsum(
                'bqhd,bkhd->bhqk', j_prefix_q, k[0:1, 3:4]) / scale
            + math.log(1.0 - anchor_prior) + math.log(alpha),
            torch.einsum(
                'bqhd,bkhd->bhqk', j_prefix_q, k[0:1, 4:6]) / scale
            + math.log(alpha),
        ], dim=-1)
        prefix_values = torch.cat([
            v[1:2, :2], v[1:2, 2:3], v[1:2, 3:4],
            v[1:2, 4:6], v[0:1, 3:4], v[0:1, 4:6]
        ], dim=1)
        expected_prefix = torch.einsum(
            'bhqk,bkhd->bqhd', prefix_logits.softmax(-1), prefix_values)

        j_suffix_q = q[1:2, 3:6]
        suffix_logits = torch.cat([
            torch.einsum(
                'bqhd,bkhd->bhqk', j_suffix_q, k[1:2, :1]) / scale,
            torch.einsum(
                'bqhd,bkhd->bhqk', j_suffix_q, k[1:2, 2:3]) / scale
            + math.log(anchor_prior),
            torch.einsum(
                'bqhd,bkhd->bhqk', j_suffix_q, k[1:2, 1:2]) / scale
            + math.log(1.0 - anchor_prior),
            torch.einsum(
                'bqhd,bkhd->bhqk', j_suffix_q, k[1:2, 3:6]) / scale
            + math.log(1.0 - alpha),
            torch.einsum(
                'bqhd,bkhd->bhqk', j_suffix_q, k[0:1, 3:6]) / scale
            + math.log(alpha),
        ], dim=-1)
        suffix_values = torch.cat([
            v[1:2, :1], v[1:2, 2:3], v[1:2, 1:2],
            v[1:2, 3:6], v[0:1, 3:6]
        ], dim=1)
        expected_suffix = torch.einsum(
            'bhqk,bkhd->bqhd', suffix_logits.softmax(-1), suffix_values)

        torch.testing.assert_close(output[1:2, :2], expected_prefix)
        torch.testing.assert_close(output[0:1, :2], expected_prefix)
        torch.testing.assert_close(output[1:2, 3:6], expected_suffix)
        torch.testing.assert_close(output[0:1, 3:6], expected_suffix)

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

        self.assertEqual(len(calls), 5)
        # M excludes its anchor from retained Q/K/V.
        self.assertEqual(
            calls[0][1][0, :, 0, 0].tolist(), [20., 21., 23., 24., 25.])
        self.assertEqual(
            calls[0][2][0, :, 0, 0].tolist(), [30., 31., 33., 34., 35.])
        # J prefix uses one softmax over prefix, shifted anchor, and suffix.
        self.assertEqual(
            calls[1][1][0, :, 0, 0].tolist(),
            [0., 1., 52., 3., 4., 5.])
        self.assertEqual(
            calls[1][2][0, :, 0, 0].tolist(),
            [10., 11., 12., 13., 14., 15.])
        # J suffix likewise uses one softmax with its shifted anchor key.
        self.assertEqual(
            calls[2][1][0, :, 0, 0].tolist(),
            [0., 1., -48., 3., 4., 5.])
        self.assertEqual(
            calls[2][2][0, :, 0, 0].tolist(),
            [10., 11., 12., 13., 14., 15.])
        # J anchor reads itself and the J suffix; B remains standard.
        self.assertEqual(
            calls[3][0][0, :, 0, 0].tolist(), [-50.])
        self.assertEqual(
            calls[3][1][0, :, 0, 0].tolist(), [-48., 3., 4., 5.])
        self.assertEqual(
            calls[4][1][0, :, 0, 0].tolist(), [40., 41., 42., 43.])
        torch.testing.assert_close(
            output[0, :, 0, 0],
            torch.tensor([1.5, 1.5, 0., 2., 2., 2.]))
        torch.testing.assert_close(
            output[1, :, 0, 0],
            torch.tensor([2., 2., 4., 3., 3., 3.]))
        self.assertEqual(
            output[2, :, 0, 0].tolist(), [5., 5., 5., 5., 0., 0.])

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
    def test_duplicate_anchor_rope_preserves_native_retained_positions(self):
        values = torch.arange(6, dtype=torch.float32).view(1, 6, 1, 1)

        def fake_shift(tensor, shift, freqs):
            return tensor + shift * 10

        with mock.patch.object(
                attention, '_shift_temporal_rope', side_effect=fake_shift):
            shifted = attention.duplicate_anchor_rope(
                values,
                grid_sizes=torch.tensor([[6, 1, 1]]),
                freqs=torch.zeros(6, 1),
                anchor_index=2)

        # Auxiliary frame 2 stays at position 2. The retained frame that was
        # expanded to slot 3 shifts back to position 2, and its suffix retains
        # the native timeline positions 3 and 4.
        self.assertEqual(
            shifted.flatten().tolist(), [0.0, 1.0, 2.0, -7.0, -6.0, -5.0])

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_paired_queries_use_mass_normalized_c_b_suffix_memory(self):
        q = torch.zeros(2, 4, 1, 2)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        # B0,B1 and expanded C0,C1,hidden-C2,C3.
        k[0, :2, 0, 0] = torch.tensor([30., 31.])
        v[0, :2, 0, 0] = torch.tensor([40., 41.])
        k[1, :, 0, 0] = torch.tensor([50., 51., 52., 53.])
        v[1, :, 0, 0] = torch.tensor([60., 61., 62., 63.])
        calls = []

        def fake_flash(q, k, v, **kwargs):
            calls.append((q.clone(), k.clone(), v.clone(), kwargs))
            result = torch.full_like(q, float(len(calls)))
            if kwargs.get('return_lse', False):
                return result, torch.zeros(
                    q.size(0), q.size(1), q.size(2), dtype=torch.float32)
            return result

        with (mock.patch.object(attention, 'flash_attention', fake_flash),
              mock.patch.object(attention, '_shift_temporal_rope',
                                side_effect=lambda x, shift, f: x + 100 * shift)):
            output = attention.paired_anchor_attention(
                q, k, v, torch.tensor([2, 4]),
                torch.tensor([[2, 1, 1], [4, 1, 1]]),
                torch.ones(4, 1, dtype=torch.complex64), 2,
                stream_weight=0.25)

        self.assertEqual(len(calls), 4)
        # Call 0 is standard B self-attention. The remaining calls use the
        # same retained C queries for unique prefix, C suffix and B1 memory.
        self.assertEqual(calls[1][1][0, :, 0, 0].tolist(), [50., 51.])
        self.assertEqual(calls[2][1][0, :, 0, 0].tolist(), [-47.])
        self.assertEqual(calls[3][1][0, :, 0, 0].tolist(), [131.])
        self.assertEqual(calls[2][2][0, :, 0, 0].tolist(), [63.])
        self.assertEqual(calls[3][2][0, :, 0, 0].tolist(), [41.])
        self.assertEqual(output.shape, q.shape)
        # Prefix group LSE=0, C suffix gets log(.75), B suffix gets log(.25).
        # Fake outputs 2,3,4 therefore merge to 2*.5 + 3*.375 + 4*.125=2.625.
        torch.testing.assert_close(
            output[1, :, 0, 0], torch.tensor([2.625, 2.625, 1., 2.625]))


if __name__ == '__main__':
    unittest.main()
