"""CPU-only checks for controlled experiment commands and safe resumability."""
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'paired_ablation_runner', REPO / 'experiments' / 'run_ablation.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class AblationCommandsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads(runner.PLAN_PATH.read_text(encoding='utf-8'))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='ablation-test-', dir=REPO.parent)
        self.addCleanup(self.temporary.cleanup)
        self.options = SimpleNamespace(python='python', gpu_count=1, ckpt_dir=None,
                                       output_dir=self.temporary.name)

    def job(self, variant, case='warriors', seed=4827316, **kwargs):
        return runner.make_job(self.plan, case, variant, seed, self.options,
                               {'revision': 'test', 'dirty_status': ''}, **kwargs)

    @staticmethod
    def flags(job):
        argv = job['argv']
        result = {}
        for index, value in enumerate(argv):
            if value.startswith('--') and value != '--save_file':
                result[value] = (argv[index + 1] if index + 1 < len(argv)
                                 and not argv[index + 1].startswith('--') else True)
        return result

    def test_primary_plan_has_two_cases_and_five_variants(self):
        jobs = runner.selected_jobs(self.plan, 'primary')
        self.assertEqual(len(jobs), 10)
        self.assertEqual({seed for _, _, seed in jobs}, {4827316})
        self.assertEqual(len(set(jobs)), 10)

    def test_baselines_omit_all_reference_interventions(self):
        for name in ('legacy_open', 'legacy_switch'):
            flags = self.flags(self.job(name))
            self.assertNotIn('--image_condition_paired_prefix_reference_budget', flags)
            self.assertNotIn('--image_condition_paired_prefix_reference_strength', flags)
            self.assertNotIn('--image_condition_paired_prefix_reference_curve', flags)
            self.assertNotIn('--image_condition_paired_prefix_preserve_first_frame', flags)
        open_flags = self.flags(self.job('legacy_open'))
        switch_flags = self.flags(self.job('legacy_switch'))
        self.assertEqual(set(switch_flags) - set(open_flags),
                         {'--image_condition_paired_prefix_route_switch_step'})
        self.assertEqual(switch_flags['--image_condition_paired_prefix_route_switch_step'], '25')

    def test_first_frame_protection_is_one_toggle(self):
        base = self.flags(self.job('adaptive'))
        protected = self.flags(self.job('adaptive_first'))
        self.assertEqual(set(protected) - set(base),
                         {'--image_condition_paired_prefix_preserve_first_frame'})
        protected.pop('--image_condition_paired_prefix_preserve_first_frame')
        self.assertEqual(base, protected)

    def test_diagnostics_change_only_named_interventions(self):
        base = self.flags(self.job('adaptive_first'))
        expected = {
            'adaptive_fade': {'--image_condition_paired_prefix_reference_fade_start_step',
                              '--image_condition_paired_prefix_reference_fade_end_step'},
            'suffix_off': {'--image_condition_paired_stream_weight'},
            'continuation': {'--image_condition_paired_suffix_prompt'},
            'timing_prompt': {'--prompt', '--image_condition_paired_suffix_prompt'}}
        for variant, changed in expected.items():
            with self.subTest(variant=variant):
                actual = self.flags(self.job(variant))
                difference = {key for key in base.keys() | actual.keys()
                              if base.get(key) != actual.get(key)}
                self.assertEqual(difference, changed)
                self.assertEqual(actual['--image_condition_paired_suffix_fade_start_step'], '25')
                self.assertEqual(actual['--image_condition_paired_suffix_fade_end_step'], '30')

    def test_original_images_indices_prompts_and_sampling_are_preserved(self):
        for case, anchor, weight in [('warriors', 12, 0.5), ('falling', 10, 1.0)]:
            job = self.job('adaptive_first', case)
            flags = self.flags(job)
            self.assertEqual(flags['--image_latent_index'], str(anchor))
            self.assertEqual(flags['--image_condition_paired_stream_weight'], str(weight))
            self.assertEqual(flags['--frame_num'], '81')
            self.assertEqual(flags['--sample_steps'], '40')
            self.assertEqual(flags['--sample_shift'], '3')
            self.assertEqual(flags['--base_seed'], '4827316')
            self.assertEqual(flags['--size'], '832*480')
            self.assertIsNone(job['config']['suffix_prompt'])
            self.assertEqual(job['config']['prompt']['text'],
                             runner.read_prompt(f'prompts/{case}.txt')['text'])

    def test_two_gpu_launch_and_single_gpu_omissions(self):
        single = self.job('adaptive')['argv']
        self.assertNotIn('--dit_fsdp', single)
        self.assertNotIn('--ulysses_size', single)
        self.options.gpu_count = 2
        paired = self.job('adaptive')['argv']
        self.assertEqual(paired[:5], ['python', '-m', 'torch.distributed.run',
                                     '--standalone', '--nproc_per_node=2'])
        self.assertIn('--dit_fsdp', paired)

    def test_output_directory_inside_repository_is_rejected(self):
        for target in [REPO, REPO / 'outputs', REPO / 'experiments' / '..' / 'runs']:
            with self.subTest(path=str(target)), self.assertRaises(ValueError):
                runner.external_directory(target)
        self.assertEqual(runner.external_directory(self.temporary.name),
                         Path(self.temporary.name).resolve())

    def test_confirmation_requires_explicit_winner_and_adds_paired_baselines(self):
        with self.assertRaises(ValueError):
            runner.selected_jobs(self.plan, 'confirmation')
        jobs = runner.selected_jobs(self.plan, 'confirmation', winner='adaptive_first')
        self.assertEqual(len(jobs), 8)
        self.assertEqual({seed for _, _, seed in jobs}, {4827317, 4827318})
        for case in self.plan['cases']:
            for seed in (4827317, 4827318):
                self.assertIn((case, 'legacy_open', seed), jobs)
                self.assertIn((case, 'adaptive_first', seed), jobs)

    def test_default_is_dry_run_and_checkpoint_only_required_to_execute(self):
        self.assertFalse(runner.parse_args([]).execute)
        self.assertFalse(runner.parse_args(['--dry-run']).execute)
        with mock.patch('sys.stderr'), self.assertRaises(SystemExit):
            runner.parse_args(['--execute'])

    def test_identity_changes_invalidate_resume_signature(self):
        base = self.job('adaptive')
        different = [self.job('adaptive_first'), self.job('adaptive', seed=4827317),
                     self.job('adaptive', runtime={'torch': 'different'})]
        self.assertTrue(all(item['signature'] != base['signature'] for item in different))
        self.assertEqual(self.job('adaptive')['signature'], base['signature'])

    def test_resume_requires_completed_full_decode_metadata_and_matching_bytes(self):
        job = self.job('adaptive')
        path = Path(job['directory'])
        path.mkdir()
        video = path / 'video.mp4'
        video.write_bytes(b'test video bytes')
        self.assertFalse(runner.completed_matches(job))
        result = {'signature': job['signature'], 'validation': {
            'frames': 81, 'fps': 16, 'bytes': video.stat().st_size,
            'sha256': runner.file_hash(video), 'validator': 'imageio-full-decode'}}
        runner.write_json(path / 'completed.json', result)
        self.assertTrue(runner.completed_matches(job))
        result['validation']['validator'] = 'metadata-only'
        runner.write_json(path / 'completed.json', result)
        self.assertFalse(runner.completed_matches(job))
        result['validation']['validator'] = 'imageio-full-decode'
        runner.write_json(path / 'completed.json', result)
        video.write_bytes(b'corrupted bytes')
        self.assertFalse(runner.completed_matches(job))

    def test_queue_lock_rejects_duplicate_queue_and_releases(self):
        path = Path(self.temporary.name)
        with runner.queue_lock(path):
            with self.assertRaises(RuntimeError):
                with runner.queue_lock(path):
                    pass
        with runner.queue_lock(path):
            pass

    def test_filter_rejects_cross_stage_and_negative_seed(self):
        with self.assertRaises(ValueError):
            runner.selected_jobs(self.plan, 'primary', variants=['suffix_off'])
        with self.assertRaises(ValueError):
            runner.selected_jobs(self.plan, 'primary', seed=-1)

    def test_dry_run_creates_no_files_and_does_not_probe_gpu_runtime(self):
        with mock.patch.object(runner, 'source_identity', return_value={}), \
                mock.patch.object(runner, 'runtime_identity') as runtime, \
                mock.patch('sys.stdout', new_callable=io.StringIO) as output:
            runner.main(['--dry-run', '--cases', 'falling', '--variants', 'legacy_open',
                         '--output-dir', self.temporary.name])
        self.assertEqual(json.loads(output.getvalue())['mode'], 'dry-run')
        runtime.assert_not_called()
        self.assertEqual(list(Path(self.temporary.name).iterdir()), [])

    def test_failed_generation_leaves_failure_record_and_no_completion(self):
        job = self.job('adaptive')
        process = mock.Mock()
        process.wait.return_value = 1
        process.poll.return_value = 1
        with mock.patch.object(runner.subprocess, 'Popen', return_value=process), \
                mock.patch.object(runner, 'validate_video') as validate, \
                mock.patch('sys.stdout', new_callable=io.StringIO), \
                self.assertRaises(runner.subprocess.CalledProcessError):
            runner.run_job(job, 'python')
        validate.assert_not_called()
        directory = Path(job['directory'])
        self.assertTrue((directory / 'run.json').exists())
        self.assertTrue((directory / 'failure.json').exists())
        self.assertFalse((directory / 'completed.json').exists())

    def test_partial_artifact_is_archived_before_restart(self):
        job = self.job('adaptive')
        directory = Path(job['directory'])
        directory.mkdir()
        (directory / 'video.mp4').write_bytes(b'interrupted old artifact')
        process = mock.Mock()
        process.wait.return_value = 0
        def launch(*args, **kwargs):
            (directory / 'video.mp4').write_bytes(b'new artifact')
            return process
        validation = {'frames': 81, 'fps': 16, 'validator': 'imageio-full-decode'}
        with mock.patch.object(runner.subprocess, 'Popen', side_effect=launch), \
                mock.patch.object(runner, 'validate_video', return_value=validation), \
                mock.patch('sys.stdout', new_callable=io.StringIO):
            runner.run_job(job, 'python')
        self.assertEqual((directory / 'attempt-1' / 'video.mp4').read_bytes(),
                         b'interrupted old artifact')
        self.assertEqual((directory / 'video.mp4').read_bytes(), b'new artifact')
        self.assertTrue((directory / 'completed.json').exists())

    def test_execute_requires_passing_preflight_before_any_generation(self):
        events = []
        command = ['--execute', '--ckpt-dir', self.temporary.name, '--cases', 'warriors',
                   '--variants', 'legacy_open', '--output-dir', self.temporary.name]
        with mock.patch.object(runner, 'source_identity', return_value={}), \
                mock.patch.object(runner, 'runtime_identity', return_value={}), \
                mock.patch.object(runner, 'require_video_validator'), \
                mock.patch.object(runner, 'run_cuda_preflight',
                                  side_effect=lambda *a, **k: events.append('preflight') or
                                  {'status': 'passed'}) as preflight, \
                mock.patch.object(runner, 'run_job',
                                  side_effect=lambda *a, **k: events.append('generation')) as run:
            runner.main(command)
        self.assertEqual(events, ['preflight', 'generation'])
        self.assertEqual(run.call_args.args[0]['cuda_preflight']['status'], 'passed')
        self.assertEqual(preflight.call_args.args[0], runner.sys.executable)
        self.assertEqual(run.call_args.kwargs['lock_fd'], preflight.call_args.kwargs['lock_fd'])

    def test_failed_preflight_stops_queue(self):
        command = ['--execute', '--ckpt-dir', self.temporary.name,
                   '--output-dir', self.temporary.name]
        with mock.patch.object(runner, 'source_identity', return_value={}), \
                mock.patch.object(runner, 'runtime_identity', return_value={}), \
                mock.patch.object(runner, 'require_video_validator'), \
                mock.patch.object(runner, 'run_cuda_preflight',
                                  side_effect=runner.subprocess.CalledProcessError(
                                      1, ['selected-python', 'check_attention_cuda.py'])), \
                mock.patch.object(runner, 'run_job') as run, \
                self.assertRaises(runner.subprocess.CalledProcessError):
            runner.main(command)
        run.assert_not_called()

    def test_preflight_uses_selected_python_and_unique_external_output(self):
        def check(command, **kwargs):
            self.assertEqual(command[0], 'selected-python')
            self.assertEqual(command[1], str(runner.PREFLIGHT_PATH))
            path = Path(command[3])
            self.assertEqual(path.parent, Path(self.temporary.name).resolve())
            self.assertFalse(path.exists())
            path.write_text('{"status":"passed"}', encoding='utf-8')
            return runner.subprocess.CompletedProcess(command, 0)
        with mock.patch.object(runner.subprocess, 'run', side_effect=check), \
                mock.patch('sys.stdout', new_callable=io.StringIO):
            first = runner.run_cuda_preflight('selected-python', self.temporary.name, 1)
            second = runner.run_cuda_preflight('selected-python', self.temporary.name, 1)
        self.assertNotEqual(first['path'], second['path'])
        self.assertEqual(first['sha256'], runner.file_hash(first['path']))

    def test_source_or_input_change_stops_before_next_job(self):
        command = ['--execute', '--ckpt-dir', self.temporary.name, '--cases', 'warriors',
                   '--variants', 'legacy_open', 'adaptive',
                   '--output-dir', self.temporary.name]
        original = {'input_and_prompt_sha256': {'Img1.png': 'original'}}
        changed = {'input_and_prompt_sha256': {'Img1.png': 'changed'}}
        with mock.patch.object(runner, 'source_identity',
                               side_effect=[original, original, changed]), \
                mock.patch.object(runner, 'runtime_identity', return_value={}), \
                mock.patch.object(runner, 'require_video_validator'), \
                mock.patch.object(runner, 'run_cuda_preflight', return_value={'status': 'passed'}), \
                mock.patch.object(runner, 'run_job') as run, \
                self.assertRaisesRegex(RuntimeError, 'files changed'):
            runner.main(command)
        self.assertEqual(run.call_count, 1)

    def test_termination_handler_restores_previous_handler(self):
        previous = signal.getsignal(signal.SIGTERM)
        with self.assertRaisesRegex(RuntimeError, 'interrupted by signal'):
            with runner.termination_handlers():
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    def test_stop_signal_is_deferred_until_child_handle_is_assigned(self):
        events = []
        with self.assertRaisesRegex(RuntimeError, 'interrupted by signal'):
            with runner.termination_handlers():
                with runner.defer_termination_during_spawn():
                    signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                    events.append('child handle assigned')
                events.append('must not reach generation wait')
        self.assertEqual(events, ['child handle assigned'])

    @unittest.skipIf(os.name == 'nt', 'POSIX process groups and inherited flock required')
    def test_signal_stops_generation_group_records_failure_and_passes_lock(self):
        job = self.job('adaptive')
        process = mock.Mock(pid=12345)
        def interrupted_wait(*args, **kwargs):
            if kwargs.get('timeout') is None:
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return 0
        process.wait.side_effect = interrupted_wait
        process.poll.return_value = None
        with runner.queue_lock(Path(self.temporary.name)) as lock_fd, \
                runner.termination_handlers(), \
                mock.patch.object(runner.subprocess, 'Popen', return_value=process) as launch, \
                mock.patch.object(runner.os, 'killpg') as kill, \
                mock.patch('sys.stdout', new_callable=io.StringIO), \
                self.assertRaisesRegex(RuntimeError, 'interrupted by signal'):
            runner.run_job(job, 'python', lock_fd=lock_fd)
        kill.assert_called_once_with(12345, signal.SIGTERM)
        self.assertEqual(launch.call_args.kwargs['pass_fds'], (lock_fd,))
        self.assertTrue((Path(job['directory']) / 'failure.json').exists())
        self.assertFalse((Path(job['directory']) / 'completed.json').exists())

    @unittest.skipIf(os.name == 'nt', 'POSIX inherited flock behavior')
    def test_child_retains_queue_lock_after_parent_sigkill(self):
        # No GPU or model imports: the child only sleeps while holding the FD.
        script = (
            'import importlib.util, pathlib, os, signal, subprocess, sys\n'
            f's=importlib.util.spec_from_file_location("runner", {str(runner.__file__)!r})\n'
            'r=importlib.util.module_from_spec(s); s.loader.exec_module(r)\n'
            f'with r.queue_lock(pathlib.Path({self.temporary.name!r})) as fd:\n'
            ' child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"], '
            'pass_fds=(fd,), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, '
            'stderr=subprocess.DEVNULL, start_new_session=True)\n'
            ' print(child.pid, flush=True)\n'
            ' os.kill(os.getpid(), signal.SIGKILL)\n')
        result = subprocess.run([runner.sys.executable, '-c', script], capture_output=True,
                                text=True, timeout=10)
        self.assertEqual(result.returncode, -signal.SIGKILL, result.stderr)
        child_pid = int(result.stdout.strip())
        self.assertGreater(child_pid, 0)
        try:
            with self.assertRaises(RuntimeError):
                with runner.queue_lock(Path(self.temporary.name)):
                    pass
        finally:
            try:
                os.kill(child_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for _ in range(100):
            try:
                with runner.queue_lock(Path(self.temporary.name)):
                    return
            except RuntimeError:
                time.sleep(0.05)
        self.fail('Child exit did not release the inherited queue lock.')


if __name__ == '__main__':
    unittest.main()
