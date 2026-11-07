"""CPU tests for the optional contextual single-frame VAE anchor.

Chunk and resampler tests run without Torch. Full convolution/shape tests use
a tiny randomly initialized encoder when Torch and einops are available;
neither the production checkpoint nor a CUDA device is required.
"""
import ast
import importlib.util
from pathlib import Path
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'wan/modules/vae2_1.py'
TREE = ast.parse(SOURCE.read_text())
scope = {}
exec(compile(ast.Module(body=[n for n in TREE.body
                             if isinstance(n, ast.FunctionDef)
                             and n.name == '_encoding_chunk_plan'],
                        type_ignores=[]), str(SOURCE), 'exec'), scope)
chunk_plan = scope['_encoding_chunk_plan']


class Pixels:
    """Temporal-only tensor stub for the actual Resample.forward method."""
    def __init__(self, values):
        self.values = list(values)

    def size(self, dim=None):
        shape = (1, 1, len(self.values), 1, 1)
        return shape if dim is None else shape[dim]

    @property
    def shape(self):
        return self.size()

    def clone(self):
        return Pixels(self.values)

    def __getitem__(self, index):
        return Pixels(self.values[index[2]])


resample_class = next(n for n in TREE.body
                      if isinstance(n, ast.ClassDef) and n.name == 'Resample')
resample_forward = next(n for n in resample_class.body
                        if isinstance(n, ast.FunctionDef) and n.name == 'forward')
resample_scope = dict(
    rearrange=lambda value, pattern, **kwargs: value,
    torch=types.SimpleNamespace(cat=lambda parts, dim: Pixels(
        [value for part in parts for value in part.values])))
exec(compile(ast.Module(body=[resample_forward], type_ignores=[]),
             str(SOURCE), 'exec'), resample_scope)


class ChunkAndCacheTests(unittest.TestCase):
    def test_default_chunks(self):
        plan = chunk_plan(81)
        self.assertEqual(len(plan), 21)
        self.assertEqual(plan[:3], [(0, 1, False), (1, 5, False), (5, 9, False)])
        self.assertEqual(plan[-1], (77, 81, False))

    def test_anchor_and_suffix_chunk_alignment(self):
        for anchor in (1, 10, 20):
            with self.subTest(anchor=anchor):
                plan = chunk_plan(78, (anchor,))
                self.assertEqual(len(plan), 21)
                self.assertEqual(plan[anchor],
                                 (1 + 4 * (anchor - 1),
                                  2 + 4 * (anchor - 1), True))
                self.assertEqual(plan[-1][1], 78)
                self.assertEqual(sum(end - start for start, end, _ in plan), 78)
                for left, right in zip(plan, plan[1:]):
                    self.assertEqual(left[1], right[0])

    def test_multiple_boundaries(self):
        plan = chunk_plan(75, (5, 10))
        self.assertEqual(len(plan), 21)
        self.assertEqual([i for i, (_, _, single) in enumerate(plan) if single],
                         [5, 10])

    def test_invalid_chunks_are_rejected(self):
        for length, indices in ((81, (10,)), (78, (0,)), (78, (21,)),
                                (78, (10, 10)), (78, (1.5,)), (0, ())):
            with self.subTest(length=length, indices=indices):
                with self.assertRaises(ValueError):
                    chunk_plan(length, indices)

    def test_boundary_bypasses_only_resampler_and_seeds_continuation(self):
        calls = []

        def temporal_conv(value):
            calls.append(value.values)
            return Pixels([sum(value.values[i:i + 3])
                           for i in range(0, len(value.values) - 2, 2)])

        layer = types.SimpleNamespace(mode='downsample3d',
                                      resample=lambda x: x, time_conv=temporal_conv)
        forward = resample_scope['forward']
        context = object()
        other = object()
        cache = [context, Pixels([999]), other]
        result = forward(layer, Pixels([7]), cache, [1],
                         single_frame_boundary=True)
        self.assertEqual(result.values, [7])
        self.assertEqual(calls, [])
        self.assertEqual(cache[1].values, [7])
        self.assertIs(cache[0], context)
        self.assertIs(cache[2], other)
        # Four-frame continuation consumes the newly installed anchor cache.
        result = forward(layer, Pixels([1, 2, 3, 4]), cache, [1])
        self.assertEqual(calls, [[7, 1, 2, 3, 4]])
        self.assertEqual(result.values, [10, 9])


try:
    import torch
    import einops
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch/einops not installed')
class TinyEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('isolated_wan_vae', SOURCE)
        cls.vae = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.vae)

    def test_real_encoder_shapes_prefix_history_and_bypasses(self):
        torch.manual_seed(27)
        model = self.vae.WanVAE_(
            dim=4, z_dim=2, dim_mult=[1, 2, 2, 2], num_res_blocks=1,
            temperal_downsample=[True, True, False]).eval()
        ordinary = torch.randn(1, 3, 17, 16, 16)
        # Five latents, with latent 2 consuming one pixel instead of four.
        compact = torch.cat([ordinary[:, :, :6], ordinary[:, :, 9:]], dim=2)
        with torch.no_grad():
            baseline = model.encode(ordinary, (0.0, 1.0))
            downsamplers = [layer for layer in model.encoder.downsamples
                            if isinstance(layer, self.vae.Resample)
                            and layer.mode == 'downsample3d']
            calls = [[] for _ in downsamplers]
            handles = [layer.time_conv.register_forward_hook(
                lambda module, inputs, output, i=i: calls[i].append(output.shape[2]))
                for i, layer in enumerate(downsamplers)]
            result = model.encode(compact, (0.0, 1.0), single_frame_indices=(2,))
            for handle in handles:
                handle.remove()
            self.assertEqual(result.shape, baseline.shape)
            self.assertEqual(result.shape[2], 5)
            torch.testing.assert_close(result[:, :, :2], baseline[:, :, :2])
            # First frame and interior anchor both bypass each downsampler.
            self.assertEqual([len(values) for values in calls], [3, 3])
            changed = compact.clone()
            changed[:, :, :5] += 1.0
            different_history = model.encode(
                changed, (0.0, 1.0), single_frame_indices=(2,))
            self.assertGreater(float((result[:, :, 2] - different_history[:, :, 2])
                                     .abs().max()), 1e-6)
            self.assertTrue(all(value is None for value in model._enc_feat_map))
            # Boundary processing leaves no persistent mode on the encoder.
            repeat = model.encode(ordinary, (0.0, 1.0))
            torch.testing.assert_close(repeat, baseline)


if __name__ == '__main__':
    unittest.main()
