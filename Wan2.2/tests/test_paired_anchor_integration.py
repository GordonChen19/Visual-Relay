"""Execute tiny real DiT blocks on CPU, with inference kernels replaced.

The model definitions are loaded without the optional diffusers checkpoint
mixins. Distributed collectives are single-rank identity operations here;
these tests do not claim to exercise CUDA, FlashAttention, or NCCL.
"""
import ast
from functools import partial
import importlib.util
import math
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
import warnings
from unittest import mock

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None


ROOT = Path(__file__).parents[1]


def definitions(path, scope):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    body = [node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), scope)
    return scope


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class PairedAnchorModelIntegrationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            '_integration_attention', ROOT / 'wan/modules/attention.py')
        cls.attention = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.attention)

    @staticmethod
    def cpu_attention(q, k, v, **kwargs):
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).float(), k.transpose(1, 2).float(),
            v.transpose(1, 2).float()).transpose(1, 2).to(q.dtype)

    def build_model(self, distributed=False):
        class ConfigMixin:
            pass

        scope = definitions(ROOT / 'wan/modules/model.py', dict(
            torch=torch, nn=nn, math=math,
            ModelMixin=nn.Module, ConfigMixin=ConfigMixin,
            register_to_config=lambda method: method,
            flash_attention=self.cpu_attention,
            paired_anchor_attention=self.core_spy,
        ))
        torch.manual_seed(51)
        model = scope['WanModel'](
            model_type='i2v', patch_size=(1, 1, 1), text_len=5,
            in_dim=4, dim=24, ffn_dim=32, freq_dim=8, text_dim=8,
            out_dim=2, num_heads=2, num_layers=2)
        # Wan initializes this projection to zero; use a nonzero tiny head so
        # a missing parameter forwarding path cannot hide behind zero outputs.
        with torch.no_grad():
            model.head.head.weight.normal_(std=0.1)

        if distributed:
            ulysses = definitions(ROOT / 'wan/distributed/ulysses.py', dict(
                torch=torch,
                dist=SimpleNamespace(is_initialized=lambda: True),
                all_to_all=lambda value, **kwargs: value,
                flash_attention=self.cpu_attention,
                paired_anchor_attention=self.core_spy,
            ))
            parallel = definitions(
                ROOT / 'wan/distributed/sequence_parallel.py', dict(
                    torch=torch, amp=torch.cuda.amp,
                    sinusoidal_embedding_1d=scope['sinusoidal_embedding_1d'],
                    distributed_attention=ulysses['distributed_attention'],
                    get_rank=lambda: 0, get_world_size=lambda: 1,
                    gather_forward=lambda value, **kwargs: value,
                ))
            model.forward = MethodType(parallel['sp_dit_forward'], model)
            for block in model.blocks:
                block.self_attn.forward = MethodType(
                    partial(parallel['sp_attn_forward'], dtype=torch.float32),
                    block.self_attn)
        return model.eval()

    def inputs(self):
        torch.manual_seed(74)
        return dict(
            x=[torch.randn(2, 7, 1, 2), torch.randn(2, 3, 1, 2)],
            y=[torch.randn(2, 7, 1, 2), torch.randn(2, 3, 1, 2)],
            t=torch.tensor([500., 500.]),
            context=[torch.randn(3, 8), torch.randn(3, 8)],
            seq_len=16, paired_anchor_index=4,
        )

    def check_forward(self, distributed):
        with (mock.patch.object(self.attention, 'flash_attention',
                                self.cpu_attention),
              warnings.catch_warnings(), torch.no_grad()):
            warnings.filterwarnings('ignore', message=".*CUDA is not available.*")
            self.core_spy = mock.Mock(
                wraps=self.attention.paired_anchor_attention)
            model = self.build_model(distributed=distributed)
            inputs = self.inputs()
            original = model(**inputs)
            self.assertEqual(self.core_spy.call_count, 2)
            self.assertTrue(all(
                call.kwargs['prefix_reference_budget'] is None
                for call in self.core_spy.call_args_list))

            self.core_spy.reset_mock()
            modified = model(
                **inputs, paired_prefix_reference_budget=0.4,
                paired_prefix_reference_curve='smoothstep')
            self.assertEqual(self.core_spy.call_count, 2)
            for call in self.core_spy.call_args_list:
                self.assertEqual(call.kwargs['prefix_reference_budget'], 0.4)
                self.assertEqual(call.kwargs['prefix_reference_curve'], 'smoothstep')
            self.assertEqual([tuple(value.shape) for value in modified],
                             [(2, 7, 1, 2), (2, 3, 1, 2)])
            self.assertTrue(all(torch.isfinite(value).all() for value in modified))
            self.assertGreater(float((original[0] - modified[0]).abs().max()), 1e-6)
            # The independently evolved suffix stream is unchanged within this
            # model call; the shared latent may transmit effects on later steps.
            torch.testing.assert_close(original[1], modified[1])

    def test_budget_reaches_each_standard_dit_block(self):
        self.check_forward(distributed=False)

    def test_budget_reaches_each_sequence_parallel_dit_block(self):
        self.check_forward(distributed=True)


if __name__ == '__main__':
    unittest.main()
