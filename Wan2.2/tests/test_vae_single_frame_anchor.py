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
    torch=types.SimpleNamespace(
        cat=lambda parts, dim: Pixels(
            [value for part in parts for value in part.values]),
        zeros_like=lambda value: Pixels([0] * value.size(2))))
exec(compile(ast.Module(body=[resample_forward], type_ignores=[]),
             str(SOURCE), 'exec'), resample_scope)


class ChunkAndCacheTests(unittest.TestCase):
    def test_joint_guidance_builder_and_vae_receive_matching_frame_counts(self):
        """Regression: the flag must reach J's builder, not another mode's."""
        pipeline_path = ROOT / 'wan/image2video.py'
        pipeline = ast.parse(pipeline_path.read_text())
        assignments = [node for node in ast.walk(pipeline)
                       if isinstance(node, ast.Assign)
                       and ast.unparse(node.targets[0]) == 'joint_encode_inputs']
        joint_branches = [node for node in ast.walk(pipeline)
                          if isinstance(node, ast.If)
                          and ast.unparse(node.test) == 'image_condition_joint_img2_guidance'
                          and any(isinstance(child, ast.Assign)
                                  and ast.unparse(child.targets[0]) ==
                                  'suffix_latent_frame_num' for child in node.body)]
        encoding = [node for node in ast.walk(pipeline)
                    if isinstance(node, ast.If)
                    and ast.unparse(node.test) ==
                    'image_condition_joint_img2_guidance_single_frame_anchor'
                    and 'self.vae.encode' in ast.unparse(node)]
        self.assertEqual(len(assignments), 1)
        self.assertEqual(len(encoding), 1)
        self.assertEqual(len(joint_branches), 1)
        lengths = [node for node in joint_branches[0].body
                   if isinstance(node, ast.Assign)
                   and ast.unparse(node.targets[0]) in
                   ('suffix_latent_frame_num', 'suffix_frame_num')]
        code = compile(ast.Module(body=lengths + [assignments[0], encoding[0]],
                                  type_ignores=[]), str(pipeline_path), 'exec')

        class Video:
            def __init__(self, frames):
                self.frames = frames

            def size(self, dimension):
                assert dimension == 1
                return self.frames

        for enabled, history, four_history, three, b_index in (
                (False, False, False, False, 0),
                (True, False, False, False, 0),
                (True, True, False, False, 0),
                (True, False, True, False, 0),
                (False, False, False, True, 0),
                (True, False, True, False, 1)):
            with self.subTest(single_frame_anchor=enabled, history=history,
                              four_history=four_history,
                              three_frame_anchor=three):
                calls = []

                def encode(videos, single_frame_indices=None,
                           single_frame_history=False, three_frame_indices=None,
                           single_frame_four_history=False):
                    calls.append(([v.frames for v in videos], single_frame_indices,
                                  single_frame_history, three_frame_indices,
                                  single_frame_four_history))
                    return [Video(len(chunk_plan(v.frames, single_frame_indices,
                                                 three_frame_indices)))
                            for v in videos]

                def joint_builder(first, anchor, frames, index,
                                  single_frame_anchor=False,
                                  three_frame_anchor=False):
                    return Video(frames - (3 if single_frame_anchor else
                                           1 if three_frame_anchor else 0))

                env = dict(
                    self=types.SimpleNamespace(
                        vae=types.SimpleNamespace(encode=encode), vae_stride=(4, 8, 8)),
                    joint_anchor_conditioning_video=joint_builder,
                    conditioning_video=lambda image, frames: Video(frames),
                    imgs=['Img1', 'Img2'], F=81, anchor_index=10,
                    suffix_frame_num=41, latent_frame_num=21,
                    joint_b_anchor_index=b_index,
                    image_condition_joint_img2_guidance_single_frame_anchor=enabled,
                    image_condition_joint_img2_guidance_single_frame_anchor_history=history,
                    image_condition_joint_img2_guidance_single_frame_anchor_four_history=(
                        four_history),
                    image_condition_joint_img2_guidance_three_frame_anchor=three,
                    logging=types.SimpleNamespace(info=lambda *args: None))
                exec(code, env)
                b_pixels = 41 + 4 * b_index
                self.assertEqual(env['suffix_latent_frame_num'], 11 + b_index)
                self.assertEqual(env['suffix_frame_num'], b_pixels)
                self.assertEqual([v.frames for v in env['encoded']], [21, 11 + b_index])
                if enabled:
                    self.assertEqual(
                        calls, [([78], (10,), history, None, four_history),
                                ([b_pixels], None, False, None, False)])
                elif three:
                    self.assertEqual(calls, [([80], None, False, (10,), False),
                                             ([41], None, False, None, False)])
                else:
                    self.assertEqual(
                        calls, [([81, 41], None, False, None, False)])

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

    def test_three_frame_anchor_and_suffix_chunk_alignment(self):
        plan = chunk_plan(80, three_frame_indices=(10,))
        self.assertEqual(len(plan), 21)
        self.assertEqual(plan[10], (37, 40, 3))
        self.assertEqual(plan[11], (40, 44, False))
        self.assertEqual(plan[-1][1], 80)

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

    def history_layer(self):
        calls = []

        def temporal_conv(value):
            calls.append(value.values)
            return Pixels([sum(value.values[i:i + 3])
                           for i in range(0, len(value.values) - 2, 2)])

        return types.SimpleNamespace(mode='downsample3d',
                                     resample=lambda x: x, time_conv=temporal_conv), calls

    def test_history_anchor_runs_kernel_and_continuation_uses_anchor(self):
        layer, calls = self.history_layer()
        forward = resample_scope['forward']
        unrelated_cache = object()
        cache = [unrelated_cache, Pixels([10, 20])]
        result = forward(layer, Pixels([7]), cache, [1],
                         single_frame_boundary=True, single_frame_history=True)
        self.assertEqual(calls, [[10, 20, 7]])
        self.assertEqual(result.values, [37])
        self.assertEqual(cache[1].values, [20, 7])
        self.assertIs(cache[0], unrelated_cache)
        result = forward(layer, Pixels([1, 2, 3, 4]), cache, [1],
                         single_frame_history=True)
        self.assertEqual(calls[-1], [7, 1, 2, 3, 4])
        self.assertEqual(result.values, [10, 9])
        self.assertEqual(cache[1].values, [3, 4])

    def test_two_cached_slices_do_not_change_ordinary_chunk_filtering(self):
        layer, calls = self.history_layer()
        forward = resample_scope['forward']
        cache = [Pixels([99, 100])]
        result = forward(layer, Pixels([1, 2, 3, 4]), cache, [0],
                         single_frame_history=True)
        self.assertEqual(calls, [[100, 1, 2, 3, 4]])
        self.assertEqual(result.values, [103, 9])
        self.assertEqual(cache[0].values, [3, 4])

    def test_history_at_both_temporal_scales_produces_one_anchor(self):
        first, first_calls = self.history_layer()
        second, second_calls = self.history_layer()
        cache = [Pixels([10, 20]), Pixels([100, 200])]
        cursor = [0]
        out = resample_scope['forward'](
            first, Pixels([7]), cache, cursor,
            single_frame_boundary=True, single_frame_history=True)
        out = resample_scope['forward'](
            second, out, cache, cursor,
            single_frame_boundary=True, single_frame_history=True)
        self.assertEqual(first_calls, [[10, 20, 7]])
        self.assertEqual(second_calls, [[100, 200, 37]])
        self.assertEqual(out.size(2), 1)
        self.assertEqual(out.values, [337])
        self.assertEqual(cursor, [2])

    def test_three_frame_anchor_uses_two_then_one_cached_features(self):
        first, first_calls = self.history_layer()
        second, second_calls = self.history_layer()
        cache = [Pixels([10, 20]), Pixels([100])]
        cursor = [0]
        out = resample_scope['forward'](
            first, Pixels([1, 2, 3]), cache, cursor,
            three_frame_boundary=True, three_frame_history=True,
            temporal_stage=0)
        out = resample_scope['forward'](
            second, out, cache, cursor,
            three_frame_boundary=True, three_frame_history=True,
            temporal_stage=1)
        self.assertEqual(first_calls, [[10, 20, 1, 2, 3]])
        self.assertEqual(second_calls, [[100, 31, 6]])
        self.assertEqual(out.size(2), 1)
        self.assertEqual(cache[0].values, [2, 3])
        self.assertEqual(cache[1].values, [6])
        self.assertEqual(cursor, [2])

    def test_single_frame_anchor_uses_four_then_one_cached_features(self):
        first, first_calls = self.history_layer()
        second, second_calls = self.history_layer()
        cache = [Pixels([10, 20, 30, 40]), Pixels([100])]
        cursor = [0]
        out = resample_scope['forward'](
            first, Pixels([7]), cache, cursor,
            single_frame_boundary=True, single_frame_four_history=True,
            temporal_stage=0)
        out = resample_scope['forward'](
            second, out, cache, cursor,
            single_frame_boundary=True, single_frame_four_history=True,
            temporal_stage=1)
        self.assertEqual(first_calls, [[10, 20, 30, 40, 7]])
        self.assertEqual(second_calls, [[100, 60, 77]])
        self.assertEqual(out.values, [237])
        self.assertEqual(cache[0].values, [20, 30, 40, 7])
        self.assertEqual(cache[1].values, [77])
        self.assertEqual(cursor, [2])

    def test_near_start_anchor_pads_only_missing_older_feature(self):
        layer, calls = self.history_layer()
        cache = [Pixels([6])]
        result = resample_scope['forward'](
            layer, Pixels([7]), cache, [0],
            single_frame_boundary=True, single_frame_history=True)
        self.assertEqual(calls, [[0, 6, 7]])
        self.assertEqual(result.values, [13])
        self.assertEqual(cache[0].values, [6, 7])

    def test_actual_first_frame_still_bypasses_even_in_history_mode(self):
        layer, calls = self.history_layer()
        cache = [None]
        result = resample_scope['forward'](
            layer, Pixels([7]), cache, [0], single_frame_history=True)
        self.assertEqual(calls, [])
        self.assertEqual(result.values, [7])
        self.assertEqual(cache[0].values, [7])


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

    def test_real_history_filters_shapes_and_prefix_at_different_boundaries(self):
        torch.manual_seed(28)
        model = self.vae.WanVAE_(
            dim=4, z_dim=2, dim_mult=[1, 2, 2, 2], num_res_blocks=1,
            temperal_downsample=[True, True, False]).eval()
        ordinary = torch.randn(1, 3, 17, 16, 16)
        with torch.no_grad():
            baseline = model.encode(ordinary, (0.0, 1.0))
            for anchor in (1, 2, 4):
                with self.subTest(anchor=anchor):
                    start = 1 + 4 * (anchor - 1)
                    compact = torch.cat([
                        ordinary[:, :, :start + 1], ordinary[:, :, start + 4:]
                    ], dim=2)
                    layers = [layer for layer in model.encoder.downsamples
                              if isinstance(layer, self.vae.Resample)
                              and layer.mode == 'downsample3d']
                    calls = [[] for _ in layers]
                    handles = [layer.time_conv.register_forward_hook(
                        lambda module, inputs, output, i=i:
                        calls[i].append((inputs[0].shape[2], output.shape[2])))
                        for i, layer in enumerate(layers)]
                    try:
                        result = model.encode(
                            compact, (0.0, 1.0), single_frame_indices=(anchor,),
                            single_frame_history=True)
                    finally:
                        for handle in handles:
                            handle.remove()
                    self.assertEqual(result.shape, baseline.shape)
                    torch.testing.assert_close(result[:, :, :anchor],
                                               baseline[:, :, :anchor])
                    self.assertEqual([len(values) for values in calls], [4, 4])
                    # The first true video frame bypasses; the interior anchor
                    # is call anchor-1 at both scales and uses one 3-slice window.
                    self.assertEqual(calls[0][anchor - 1], (3, 1))
                    self.assertEqual(calls[1][anchor - 1], (3, 1))
                    self.assertTrue(torch.isfinite(result).all())
                    self.assertTrue(all(x is None for x in model._enc_feat_map))
            repeat = model.encode(ordinary, (0.0, 1.0))
            torch.testing.assert_close(repeat, baseline)

    def test_real_three_frame_anchor_uses_two_first_stage_outputs(self):
        torch.manual_seed(29)
        model = self.vae.WanVAE_(
            dim=4, z_dim=2, dim_mult=[1, 2, 2, 2], num_res_blocks=1,
            temperal_downsample=[True, True, False]).eval()
        ordinary = torch.randn(1, 3, 17, 16, 16)
        anchor = 2
        compact = torch.cat([ordinary[:, :, :8], ordinary[:, :, 9:]], dim=2)
        with torch.no_grad():
            baseline = model.encode(ordinary, (0.0, 1.0))
        layers = [layer for layer in model.encoder.downsamples
                  if isinstance(layer, self.vae.Resample)
                  and layer.mode == 'downsample3d']
        calls = [[] for _ in layers]
        handles = [layer.time_conv.register_forward_hook(
            lambda module, inputs, output, i=i:
            calls[i].append((inputs[0].shape[2], output.shape[2])))
            for i, layer in enumerate(layers)]
        try:
            with torch.no_grad():
                result = model.encode(
                    compact, (0.0, 1.0), three_frame_indices=(anchor,))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(result.shape, baseline.shape)
        torch.testing.assert_close(result[:, :, :anchor], baseline[:, :, :anchor])
        self.assertEqual(calls[0][anchor - 1], (5, 2))
        self.assertEqual(calls[1][anchor - 1], (3, 1))
        self.assertTrue(torch.isfinite(result).all())

    def test_real_single_frame_four_history_uses_two_first_stage_outputs(self):
        torch.manual_seed(30)
        model = self.vae.WanVAE_(
            dim=4, z_dim=2, dim_mult=[1, 2, 2, 2], num_res_blocks=1,
            temperal_downsample=[True, True, False]).eval()
        ordinary = torch.randn(1, 3, 17, 16, 16)
        anchor = 2
        start = 1 + 4 * (anchor - 1)
        compact = torch.cat([
            ordinary[:, :, :start + 1], ordinary[:, :, start + 4:]
        ], dim=2)
        with torch.no_grad():
            baseline = model.encode(ordinary, (0.0, 1.0))
        layers = [layer for layer in model.encoder.downsamples
                  if isinstance(layer, self.vae.Resample)
                  and layer.mode == 'downsample3d']
        calls = [[] for _ in layers]
        handles = [layer.time_conv.register_forward_hook(
            lambda module, inputs, output, i=i:
            calls[i].append((inputs[0].shape[2], output.shape[2])))
            for i, layer in enumerate(layers)]
        try:
            with torch.no_grad():
                result = model.encode(
                    compact, (0.0, 1.0), single_frame_indices=(anchor,),
                    single_frame_four_history=True)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(result.shape, baseline.shape)
        torch.testing.assert_close(result[:, :, :anchor], baseline[:, :, :anchor])
        self.assertEqual(calls[0][anchor - 1], (5, 2))
        self.assertEqual(calls[1][anchor - 1], (3, 1))
        self.assertTrue(torch.isfinite(result).all())

    def test_history_option_requires_a_boundary(self):
        model = self.vae.WanVAE_(
            dim=4, z_dim=2, dim_mult=[1, 2, 2, 2], num_res_blocks=1,
            temperal_downsample=[True, True, False]).eval()
        with self.assertRaises(ValueError):
            model.encode(torch.zeros(1, 3, 17, 16, 16), (0.0, 1.0),
                         single_frame_history=True)


if __name__ == '__main__':
    unittest.main()
