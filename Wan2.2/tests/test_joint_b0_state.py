"""Exercise state-aligned B0 guidance without importing the CUDA stack."""
import ast
from pathlib import Path
import types
import unittest


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

    def test_handoff_removes_anchor_and_stops_b_input(self):
        env = self.environment()
        env['joint_guidance_active'] = False
        exec(self.guided_block(), env)
        self.assertEqual(len(env['latent_model_input']), 1)
        self.assertEqual(env['latent_model_input'][0].values, [0, 1, 3, 4])
        self.assertIs(env['current_arg_c'], env['joint_handoff_arg_c'])
        self.assertIs(env['current_arg_null'], env['joint_handoff_arg_null'])

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


if __name__ == '__main__':
    unittest.main()
