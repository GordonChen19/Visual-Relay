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
    def test_active_prefix_uses_stream_a_anchor_key_and_stream_b_value(self):
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
        # The active prefix sees A0, A1, and one hybrid anchor only: its key is
        # history-aware A2 while its value is fresh-cache B0. A3 is still
        # RoPE-compressed for the main suffix routes, but is absent here.
        _, prefix_key, prefix_value = calls[1]
        self.assertEqual(
            prefix_key[0, :, 0, 0].tolist(), [10.0, 11.0, 12.0])
        self.assertEqual(
            prefix_value[0, :, 0, 0].tolist(), [20.0, 21.0, 40.0])
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


if __name__ == '__main__':
    unittest.main()
