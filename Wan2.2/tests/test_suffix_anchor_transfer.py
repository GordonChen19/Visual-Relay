"""Verify group probability redistribution against a dense biased softmax."""
import ast
import math
from pathlib import Path
import types
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(torch is None, 'requires PyTorch')
class SuffixAnchorTransferTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((ROOT / 'wan/modules/attention.py').read_text())
        names = {'_joint_suffix_transfer_weights', '_joint_first_suffix_attention',
                 'joint_img2_guidance_attention', '_record_joint_attention_shares'}
        functions = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        self.samples = []
        self.scope = dict(
            torch=torch, math=math, flash_attention=self.flash,
            _shift_temporal_rope=lambda x, delta, freqs: x,
            _joint_attention_samples=types.SimpleNamespace(get=lambda: None))
        exec(compile(ast.Module(body=functions, type_ignores=[]),
                     '<actual transfer attention>', 'exec'), self.scope)

    @staticmethod
    def flash(q, k, v, return_lse=False, **kwargs):
        logits = torch.einsum('bqhd,bkhd->bqhk', q.float(), k.float()) / math.sqrt(q.size(-1))
        output = torch.einsum('bqhk,bkhd->bqhd', logits.softmax(-1), v.float())
        return (output, logits.logsumexp(-1)) if return_lse else output

    def test_transfer_cap_target_and_unchanged_suffix_mass(self):
        original = torch.tensor([[.30, .10, .60], [.10, .30, .60], [.80, .10, .10]])
        before, after = self.scope['_joint_suffix_transfer_weights'](
            original.log(), 1.0, .05)
        torch.testing.assert_close(before, original)
        torch.testing.assert_close(after, torch.tensor([
            [.25, .15, .60], [.10, .30, .60], [.75, .15, .10]]))
        _, uncapped = self.scope['_joint_suffix_transfer_weights'](original.log(), 1., 1.)
        torch.testing.assert_close(uncapped[0], torch.tensor([.20, .20, .60]))
        torch.testing.assert_close(after.sum(-1), torch.ones(3))

    def test_group_attention_equals_one_dense_biased_softmax(self):
        torch.manual_seed(71)
        q, k, v = torch.randn(1, 4, 2, 8), torch.randn(1, 15, 2, 8), torch.randn(1, 15, 2, 8)
        logits = torch.einsum('bqhd,bkhd->bqhk', q, k) / math.sqrt(8)
        evidence = torch.stack([logits[..., :6].logsumexp(-1),
                                logits[..., 6:9].logsumexp(-1),
                                logits[..., 9:].logsumexp(-1)], dim=-1)
        before, after = self.scope['_joint_suffix_transfer_weights'](evidence, 1., .05)
        bias = (after / before).log()
        biased = logits.clone()
        for group, (start, end) in enumerate(((0, 6), (6, 9), (9, 15))):
            biased[..., start:end] += bias[..., group:group + 1]
        expected_probabilities = biased.softmax(-1)
        torch.testing.assert_close(expected_probabilities[..., 9:], logits.softmax(-1)[..., 9:])
        expected = torch.einsum('bqhk,bkhd->bqhd', expected_probabilities, v)
        actual = self.scope['_joint_first_suffix_attention'](
            q, k, v, 6, 9, (1., .05), (-1, -1), torch.float32)
        torch.testing.assert_close(actual, expected)
        zero = self.scope['_joint_first_suffix_attention'](
            q, k, v, 6, 9, (1., 0.), (-1, -1), torch.float32)
        torch.testing.assert_close(zero, self.flash(q, k, v))

    def test_only_first_suffix_queries_change_including_b1_mapping(self):
        torch.manual_seed(39)
        q, k, v = (torch.randn(2, 21, 2, 8) for _ in range(3))
        args = dict(q=q, k=k, v=v, seq_lens=torch.tensor([21, 8]),
                    grid_sizes=torch.tensor([[21, 1, 1], [8, 1, 1]]),
                    anchor_index=14, rope_compression=True,
                    anchor_rope_offset=.8, velocity_only=True, freqs=object(),
                    dtype=torch.float32)
        function = self.scope['joint_img2_guidance_attention']
        base = function(**args)
        corrected = function(**args, suffix_anchor_transfer=(10., .05))
        torch.testing.assert_close(base[0, :15], corrected[0, :15])
        torch.testing.assert_close(base[0, 16:], corrected[0, 16:])
        torch.testing.assert_close(base[1], corrected[1])
        self.assertGreater((base[0, 15] - corrected[0, 15]).abs().max().item(), 1e-5)

    def test_diagnostics_report_corrected_probabilities(self):
        torch.manual_seed(19)
        q, k = torch.randn(1, 6, 2, 4), torch.randn(1, 6, 2, 4)
        self.scope['_joint_attention_samples'] = types.SimpleNamespace(get=lambda: self.samples)
        transfer = (2., .05)
        self.scope['_record_joint_attention_shares'](
            q, k, 2, 1, q[:, 3:], k, True, transfer)
        logits = torch.einsum('bqhd,bkhd->bqhk', q[:, 3:4], k) / 2
        evidence = torch.stack([logits[..., :2].logsumexp(-1),
                                logits[..., 2:3].logsumexp(-1),
                                logits[..., 3:].logsumexp(-1)], dim=-1)
        before, after = self.scope['_joint_suffix_transfer_weights'](evidence, *transfer)
        expected_previous = (logits.softmax(-1)[..., 1] * after[..., 0] / before[..., 0]).sum()
        torch.testing.assert_close(self.samples[0][2], after[..., 1].sum())
        torch.testing.assert_close(self.samples[0][3], expected_previous)


class TransferForwardingTests(unittest.TestCase):
    def test_config_reaches_all_standard_and_sequence_parallel_calls(self):
        name = 'joint_img2_guidance_suffix_anchor_transfer'
        for relative in ('wan/modules/model.py', 'wan/distributed/sequence_parallel.py',
                         'wan/distributed/ulysses.py'):
            tree = ast.parse((ROOT / relative).read_text())
            for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
                keywords = {kw.arg for kw in call.keywords}
                if 'joint_img2_guidance_anchor_rope_offset' in keywords:
                    self.assertIn(name, keywords, relative)
                if 'anchor_rope_offset' in keywords:
                    self.assertIn('suffix_anchor_transfer', keywords, relative)


if __name__ == '__main__':
    unittest.main()
