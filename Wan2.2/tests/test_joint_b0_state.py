"""Exercise the actual B view/scheduler blocks without importing CUDA models."""
import ast
from pathlib import Path
import types
import unittest


class Frames:
    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, index):
        return Frames(self.values[index[1]])

    def to(self, device):
        return self

    def unsqueeze(self, dim):
        return self

    def squeeze(self, dim):
        return self


class Scheduler:
    def step(self, prediction, timestep, state, **kwargs):
        self.prediction = prediction.values
        self.state = state.values
        return (Frames([a + b for a, b in zip(
            state.values, prediction.values)]),)


class B0StateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / 'wan/image2video.py'
        cls.tree = ast.parse(source.read_text())

    def block(self, first_assignment):
        matches = [node for node in ast.walk(self.tree)
                   if isinstance(node, ast.If) and node.body
                   and isinstance(node.body[0], ast.Assign)
                   and ast.unparse(node.body[0].targets[0]) == first_assignment
                   and 'joint_guidance_b_latent' in ast.unparse(node.test)]
        self.assertEqual(len(matches), 1)
        return compile(ast.Module(body=[matches[0]], type_ignores=[]),
                       '<actual sampling block>', 'exec')

    def environment(self):
        return dict(
            torch=types.SimpleNamespace(cat=lambda parts, dim: Frames(
                [value for part in parts for value in part.values])),
            self=types.SimpleNamespace(device='cpu'), anchor_index=2,
            joint_guidance_b_latent=Frames([100, -1, -2, -3]),
            current_latent=Frames([0, 1, 2, 3, 4]))

    def test_input_uses_own_b0_and_canonical_j_suffix(self):
        env = self.environment()
        exec(self.block('joint_guidance_b_latent'), env)
        self.assertEqual(env['joint_guidance_b_latent'].values, [100, 2, 3, 4])
        # A later J9-equivalent change must never replace B0.
        env['current_latent'] = Frames([10, 999, 20, 30, 40])
        exec(self.block('joint_guidance_b_latent'), env)
        self.assertEqual(env['joint_guidance_b_latent'].values,
                         [100, 20, 30, 40])

    def test_only_b0_is_scheduled_and_j_is_not_overwritten(self):
        env = self.environment()
        scheduler = Scheduler()
        env.update(joint_guidance_active=True,
                   joint_guidance_b_scheduler=scheduler,
                   joint_guidance_b_noise_pred=Frames([7]),
                   joint_guidance_b_pred_cond=Frames([8]),
                   joint_guidance_b_pred_uncond=Frames([1]),
                   latent=Frames([10, 11, 12, 13, 14]), t=999, seed_g=None)
        exec(self.block('b_temp_x0'), env)
        self.assertEqual(scheduler.state, [100])
        self.assertEqual(scheduler.prediction, [7])
        self.assertEqual(env['joint_guidance_b_latent'].values,
                         [107, 12, 13, 14])
        self.assertEqual(env['latent'].values, [10, 11, 12, 13, 14])

    def test_b0_stops_at_handoff(self):
        env = self.environment()
        env['joint_guidance_active'] = False
        exec(self.block('b_temp_x0'), env)
        self.assertEqual(env['joint_guidance_b_latent'].values,
                         [100, -1, -2, -3])


if __name__ == '__main__':
    unittest.main()
