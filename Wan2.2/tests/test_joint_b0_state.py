"""Exercise state-aligned B0 guidance without importing the CUDA stack."""
import ast
import math
from pathlib import Path
import types
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


class Frames:
    """Small shared-storage temporal tensor stub for sampling-block tests."""
    def __init__(self, values, storage=None, indices=None):
        self.storage = list(values) if storage is None else storage
        self.indices = list(range(len(self.storage))) if indices is None else indices

    @property
    def values(self):
        return [self.storage[i] for i in self.indices]

    def __getitem__(self, index):
        return Frames([], self.storage, self.indices[index[1]])

    def clone(self):
        return Frames(self.values)

    def size(self, dim):
        if dim != 1:
            raise ValueError('stub only models the temporal dimension')
        return len(self.indices)

    def copy_(self, source):
        if len(self.indices) != len(source.values):
            raise ValueError('frame count mismatch')
        for i, value in zip(self.indices, source.values):
            self.storage[i] = value
        return self


class B0StateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.tree = ast.parse((cls.root / 'wan/image2video.py').read_text())
        helpers = [node for node in cls.tree.body
                   if isinstance(node, ast.FunctionDef)
                   and node.name == '_replace_joint_anchor_velocity']
        cls.scope = {}
        exec(compile(ast.Module(body=helpers, type_ignores=[]),
                     '<actual anchor replacement>', 'exec'), cls.scope)

    def guided_block(self):
        nodes = [node for node in ast.walk(self.tree)
                 if isinstance(node, ast.If)
                 and ast.unparse(node.test) == 'joint_guidance_active']
        self.assertEqual(len(nodes), 1)
        return compile(ast.Module(body=nodes, type_ignores=[]),
                       '<actual guided input>', 'exec')

    def environment(self):
        return dict(
            torch=types.SimpleNamespace(cat=lambda parts, dim: Frames(
                [value for part in parts for value in part.values])),
            anchor_index=2, current_latent=Frames([0, 1, 2, 3, 4]), t=999,
            joint_guidance_active=True,
            joint_handoff_arg_c={'conditioning': 'retained joint'},
            joint_handoff_arg_null={'conditioning': 'retained joint negative'},
            sampling_step=15,
            image_condition_joint_img2_guidance_adaptive_pair_end_step=15,
            logging=types.SimpleNamespace(info=lambda *args: None))

    def test_b0_is_a_view_of_the_current_j_anchor(self):
        env = self.environment()
        exec(self.guided_block(), env)
        main, b = env['latent_model_input']
        self.assertIs(main, env['current_latent'])
        self.assertEqual(b.values, [2, 3, 4])
        main[:, 2:3].copy_(Frames([77]))
        self.assertEqual(b.values, [77, 3, 4])
        env['current_latent'] = Frames([10, 11, 12, 13, 14])
        exec(self.guided_block(), env)
        self.assertEqual(env['latent_model_input'][1].values, [12, 13, 14])

    def test_b0_velocity_replaces_only_anchor_in_both_cfg_branches(self):
        assignments = [node for node in ast.walk(self.tree)
                       if isinstance(node, ast.Assign)
                       and isinstance(node.value, ast.Call)
                       and ast.unparse(node.value.func) == '_replace_joint_anchor_velocity']
        self.assertEqual(len(assignments), 2)
        self.assertEqual({ast.unparse(n.targets[0]) for n in assignments},
                         {'noise_pred_cond', 'noise_pred_uncond'})
        env = dict(self.scope, anchor_index=2,
                   noise_pred_cond=Frames([10, 11, 12, 13, 14]),
                   noise_pred_uncond=Frames([1, 2, 3, 4, 5]),
                   predictions_cond=[None, Frames([100, 200, 300])],
                   predictions_uncond=[None, Frames([7, 8, 9])])
        original_c = env['noise_pred_cond']
        exec(compile(ast.Module(body=assignments, type_ignores=[]),
                     '<actual CFG branch replacements>', 'exec'), env)
        self.assertEqual(env['noise_pred_cond'].values, [10, 11, 100, 13, 14])
        self.assertEqual(env['noise_pred_uncond'].values, [1, 2, 7, 4, 5])
        self.assertEqual(original_c.values, [10, 11, 12, 13, 14])
        scale = 3.5
        guided = [u + scale * (c - u) for c, u in zip(
            env['noise_pred_cond'].values, env['noise_pred_uncond'].values)]
        self.assertEqual(guided[2], 7 + scale * (100 - 7))

    def test_joint_b_has_no_independent_state_or_scheduler(self):
        identifiers = {node.id for node in ast.walk(self.tree)
                       if isinstance(node, ast.Name)}
        self.assertNotIn('joint_guidance_b_scheduler', identifiers)
        self.assertNotIn('joint_guidance_b_latent', identifiers)

    def test_ordinary_rope_reaches_both_cfg_branches_without_changing_handoff(self):
        assignments = [node for node in ast.walk(self.tree)
                       if isinstance(node, ast.Assign)
                       and isinstance(node.value, ast.Dict)
                       and any(isinstance(key, ast.Constant)
                               and key.value == 'joint_img2_guidance_compression_ratio'
                               for key in node.value.keys)]
        self.assertEqual(len(assignments), 2)
        for ordinary, compressed in ((False, True), (True, True), (True, False)):
            for node in assignments:
                names = {n.id for n in ast.walk(node.value)
                         if isinstance(n, ast.Name)}
                env = dict.fromkeys(names, False)
                env.update(
                    context=['J', 'B'], context_null=['negative'],
                    image_condition_joint_img2_guidance_no_rope_compression=ordinary,
                    image_condition_joint_img2_guidance_anchor_rope_offset=None,
                    image_condition_joint_img2_guidance_rope_compression=compressed,
                    image_condition_joint_img2_guidance_shifted_anchor=True,
                    image_condition_joint_img2_guidance_adaptive_pair_attention=True)
                exec(compile(ast.Module(body=[node], type_ignores=[]),
                             '<actual joint CFG arguments>', 'exec'), env)
                args = env[node.targets[0].id]
                self.assertTrue(args['joint_img2_guidance_rope_compression'])
                self.assertEqual(args['joint_img2_guidance_compression_ratio'],
                                 0.0 if ordinary else 1.0)
                self.assertTrue(args['joint_img2_guidance_velocity_only'])
                self.assertTrue(args['joint_img2_guidance_adaptive_pair_attention'])

    def test_handoff_removes_anchor_and_stops_b_input(self):
        env = self.environment()
        env['joint_guidance_active'] = False
        exec(self.guided_block(), env)
        self.assertEqual(len(env['latent_model_input']), 1)
        self.assertEqual(env['latent_model_input'][0].values, [0, 1, 3, 4])
        self.assertIs(env['current_arg_c'], env['joint_handoff_arg_c'])
        self.assertIs(env['current_arg_null'], env['joint_handoff_arg_null'])

    def test_ordinary_rope_still_discards_anchor_before_decoding(self):
        branch = next(node for node in ast.walk(self.tree)
                      if isinstance(node, ast.If)
                      and any(isinstance(child, ast.Assign)
                              and ast.unparse(child.targets[0]) == 'retained_x0'
                              for child in node.body))
        for ordinary, compressed in ((True, False), (True, True), (False, True)):
            env = self.environment()
            env.update(
                temporary_anchor_mode=False,
                image_condition_joint_img2_guidance=True,
                image_condition_joint_img2_guidance_no_rope_compression=ordinary,
                image_condition_joint_img2_guidance_rope_compression=compressed,
                latent_frame_num=21, anchor_index=10,
                x0=[Frames(list(range(21)))],
                self=types.SimpleNamespace(vae=types.SimpleNamespace(
                    decode=lambda frames: [frame.values for frame in frames])))
            self.assertTrue(eval(compile(ast.Expression(branch.test),
                                         '<actual decode guard>', 'eval'), env))
            exec(compile(ast.Module(body=branch.body, type_ignores=[]),
                         '<actual retained-frame decoding>', 'exec'), env)
            self.assertEqual(env['videos'], [list(range(10)) + list(range(11, 21))])

    def test_handoff_preserves_joint_features_and_masks_in_both_cfg_calls(self):
        def assignment(name):
            matches = [node for node in ast.walk(self.tree)
                       if isinstance(node, ast.Assign)
                       and ast.unparse(node.targets[0]) == name
                       and isinstance(node.value, (ast.Call, ast.Dict))]
            self.assertEqual(len(matches), 1)
            return compile(ast.Module(body=matches, type_ignores=[]),
                           '<actual conditioning handoff>', 'exec')

        for anchor in (1, 10, 19):
            with self.subTest(anchor=anchor):
                values = [(i == 0 or i == anchor, 100 + i, 200 + i)
                          for i in range(21)]
                env = self.environment()
                env.update(anchor_index=anchor, joint_y=Frames(values),
                           context=['J', 'B', 'global'],
                           context_null=['negative'] * 3, max_seq_len=21)
                exec(assignment('joint_handoff_y'), env)
                exec(assignment('joint_handoff_arg_c'), env)
                exec(assignment('joint_handoff_arg_null'), env)
                expected = values[:anchor] + values[anchor + 1:]
                self.assertEqual(env['joint_handoff_y'].values, expected)
                for name in ('joint_handoff_arg_c', 'joint_handoff_arg_null'):
                    self.assertIs(env[name]['y'][0], env['joint_handoff_y'])
                    self.assertEqual(env[name]['y'][0].values, expected)
                self.assertEqual(env['joint_y'].values, values)


class TemporalRows:
    """Track temporal coordinates through the actual attention routing code."""
    def __init__(self, rows):
        self.rows = rows

    def size(self, dim):
        return len(self.rows) if dim == 0 else len(self.rows[0])

    def __getitem__(self, index):
        if isinstance(index, int):
            return types.SimpleNamespace(tolist=lambda: self.rows[index])
        return TemporalRows([row[index[1]] for row in self.rows[index[0]]])


class OrdinaryJointRopeTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_fractional_anchor_has_real_temporal_rope_phase(self):
        path = Path(__file__).resolve().parents[1] / 'wan/modules/attention.py'
        tree = ast.parse(path.read_text())
        functions = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef)
                     and node.name in ('_shift_temporal_rope',
                                       'joint_img2_guidance_attention')]
        calls = []

        def flash(q, k, v, **kwargs):
            calls.append((q.clone(), k.clone()))
            return torch.zeros_like(q)

        scope = dict(
            torch=torch, flash_attention=flash,
            _record_joint_attention_shares=lambda *args: None)
        exec(compile(ast.Module(body=functions, type_ignores=[]),
                     str(path), 'exec'), scope)
        freqs = torch.polar(
            torch.ones(6, 6),
            torch.outer(torch.arange(6, dtype=torch.float32),
                        torch.tensor([0.5, 0.3, 0.2, 0.1, 0.1, 0.1])))
        q = torch.zeros(2, 5, 1, 12)
        for frame in range(5):
            q[:, frame, 0, 0] = math.cos(0.5 * frame)
            q[:, frame, 0, 1] = math.sin(0.5 * frame)
        scope['joint_img2_guidance_attention'](
            q=q, k=q, v=q,
            seq_lens=torch.tensor([5, 3]),
            grid_sizes=torch.tensor([[5, 1, 1], [3, 1, 1]]),
            anchor_index=2, rope_compression=True,
            anchor_rope_offset=0.8, velocity_only=True, freqs=freqs)

        def positions(x):
            return torch.atan2(x[0, :, 0, 1], x[0, :, 0, 0]) / 0.5

        self.assertEqual(len(calls), 4)
        expected = torch.tensor([0., 1., 1.8, 2., 3.])
        torch.testing.assert_close(positions(calls[0][1]), expected,
                                   atol=1e-5, rtol=0)
        torch.testing.assert_close(positions(calls[1][0]), torch.tensor([2.]),
                                   atol=1e-5, rtol=0)
        torch.testing.assert_close(positions(calls[1][1]),
                                   torch.tensor([2., 3., 4.]),
                                   atol=1e-5, rtol=0)
        torch.testing.assert_close(positions(calls[2][0]),
                                   torch.tensor([2., 3.]),
                                   atol=1e-5, rtol=0)
        torch.testing.assert_close(positions(calls[2][1]), expected,
                                   atol=1e-5, rtol=0)

    def test_fractional_offset_is_forwarded_through_each_model_path(self):
        root = Path(__file__).resolve().parents[1]
        for relative in ('wan/modules/model.py',
                         'wan/distributed/sequence_parallel.py',
                         'wan/distributed/ulysses.py'):
            with self.subTest(path=relative):
                tree = ast.parse((root / relative).read_text())
                calls = [node for node in ast.walk(tree)
                         if isinstance(node, ast.Call)]
                carriers = [call for call in calls
                            if any(keyword.arg ==
                                   'joint_img2_guidance_compression_ratio'
                                   for keyword in call.keywords)]
                for call in carriers:
                    self.assertIn('joint_img2_guidance_anchor_rope_offset',
                                  {keyword.arg for keyword in call.keywords})
                if relative != 'wan/distributed/ulysses.py':
                    self.assertGreater(len(carriers), 0)
                attention_calls = [call for call in calls
                                   if isinstance(call.func, ast.Name)
                                   and call.func.id ==
                                   'joint_img2_guidance_attention']
                if relative != 'wan/distributed/sequence_parallel.py':
                    self.assertEqual(len(attention_calls), 1)
                    self.assertIn('anchor_rope_offset',
                                  {keyword.arg for keyword in
                                   attention_calls[0].keywords})

    def test_real_attention_routes_keep_native_offsets_and_anchor_mask(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / 'wan/modules/attention.py').read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == 'joint_img2_guidance_attention')

        def cat(parts, dim):
            if dim == 0:
                return TemporalRows([row for part in parts for row in part.rows])
            return TemporalRows([sum((part.rows[i] for part in parts), [])
                                 for i in range(parts[0].size(0))])

        calls, shifts = [], []

        def flash(q, k, v, **kwargs):
            calls.append((q.rows[0], k.rows[0]))
            return q

        def shift(x, delta, freqs):
            shifts.append(delta)
            return TemporalRows([[value + delta for value in row] for row in x.rows])

        env = dict(
            torch=types.SimpleNamespace(
                bfloat16=None, cat=cat,
                zeros_like=lambda x: TemporalRows([[0] * len(row) for row in x.rows])),
            flash_attention=flash, _shift_temporal_rope=shift,
            _record_joint_attention_shares=lambda *args: None)
        exec(compile(ast.Module(body=[function], type_ignores=[]),
                     '<actual joint attention>', 'exec'), env)
        for ratio, anchor_offset in (
                (0.0, None), (1.0, None),
                (1.0, 0.0), (1.0, 0.5), (1.0, 0.8), (1.0, 1.0)):
            calls.clear()
            shifts.clear()
            coordinates = TemporalRows([list(range(21)), list(range(21))])
            output = env[function.name](
                q=coordinates, k=coordinates, v=coordinates,
                seq_lens=[21, 11], grid_sizes=TemporalRows([[21, 1, 1], [11, 1, 1]]),
                anchor_index=10, rope_compression=True,
                adaptive_pair_attention=True, compression_ratio=ratio,
                anchor_rope_offset=anchor_offset,
                velocity_only=True, freqs=object())
            self.assertEqual(len(calls), 4)  # J prefix, anchor, suffix, native B
            self.assertEqual(calls[0][0], list(range(10)))
            self.assertEqual(calls[1], ([10], list(range(10, 21))))
            self.assertEqual(calls[3], (list(range(11)), list(range(11))))
            self.assertEqual(output.size(0), 2)
            if anchor_offset is not None:
                self.assertEqual(shifts, [anchor_offset - 1.0, -1.0, -1.0])
                self.assertEqual(calls[0][1][9], 9)
                self.assertAlmostEqual(calls[0][1][10], 9 + anchor_offset)
                self.assertEqual(calls[0][1][11], 10)
                self.assertEqual(calls[2][0][0], 10)
                self.assertEqual(calls[2][1], calls[0][1])
                self.assertAlmostEqual(
                    calls[2][0][0] - calls[2][1][10], 1 - anchor_offset)
            elif ratio == 0:
                self.assertEqual(shifts, [])
                self.assertEqual(calls[0][1], list(range(21)))
                self.assertEqual(calls[2], (list(range(11, 21)), list(range(21))))
                self.assertEqual(calls[0][1][11] - calls[0][0][9], 2)
                self.assertEqual(calls[2][0][0] - calls[2][1][9], 2)
            else:
                self.assertEqual(shifts, [-1.0] * 3)
                self.assertEqual(calls[0][1][10:12], [10, 10])
                self.assertEqual(calls[2][1][9:11], [9, 9])
                self.assertEqual(calls[2][0][0], 10)

if __name__ == '__main__':
    unittest.main()
