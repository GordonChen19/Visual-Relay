#!/usr/bin/env python3
"""Print controlled experiments, or run them sequentially with --execute.

Only standard-library modules are imported by the runner. Video validation uses
FFmpeg/ffprobe when available, otherwise imageio in the selected Python runtime.
All generated artifacts and run records must stay outside the repository.
"""
import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import uuid


REPO = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO / 'experiments' / 'ablation_plan.json'
PREFLIGHT_PATH = REPO / 'experiments' / 'check_attention_cuda.py'


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def signature(value):
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                   ensure_ascii=True).encode('utf-8'))


def external_directory(path, repo=REPO):
    target, repo = Path(path).expanduser().resolve(), Path(repo).resolve()
    if target == repo or repo in target.parents:
        raise ValueError('Output directory must be outside the repository.')
    return target


def read_prompt(relative):
    path = (REPO / 'experiments' / relative).resolve()
    text = path.read_text(encoding='utf-8').strip()
    if not text:
        raise ValueError(f'Empty prompt: {path}')
    return {'path': str(path), 'text': text,
            'sha256': digest_bytes(text.encode('utf-8')),
            'file_sha256': file_hash(path)}


def default_negative_prompt():
    path = REPO / 'Wan2.2' / 'wan' / 'configs' / 'shared_config.py'
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Attribute)
                        and target.attr == 'sample_neg_prompt'):
                    text = ast.literal_eval(node.value)
                    return {'text': text, 'sha256': digest_bytes(text.encode())}
    raise ValueError('Could not read the default negative prompt without imports.')


def source_identity():
    def git(*args):
        try:
            result = subprocess.run(
                ['git', '-c', f'safe.directory={REPO.as_posix()}', '-C', str(REPO),
                 *args], capture_output=True, text=True, check=False)
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    paths = [REPO / 'Wan2.2' / 'generate.py', Path(__file__).resolve(), PLAN_PATH,
             PREFLIGHT_PATH]
    paths += sorted((REPO / 'Wan2.2' / 'wan').rglob('*.py'))
    paths += sorted((REPO / 'Wan2.2').glob('requirements*.txt'))
    plan = json.loads(PLAN_PATH.read_text(encoding='utf-8'))
    input_paths = {REPO / image for case in plan['cases'].values()
                   for image in case['images']}
    input_paths.update((REPO / 'experiments' / 'prompts').glob('*.txt'))
    return {'revision': git('rev-parse', 'HEAD'),
            'dirty_status': git('status', '--porcelain', '--untracked-files=normal'),
            'runtime_source_sha256': {
                str(p.relative_to(REPO).as_posix()): file_hash(p) for p in paths},
            'input_and_prompt_sha256': {
                str(p.relative_to(REPO).as_posix()): file_hash(p)
                for p in sorted(input_paths)}}


def selected_jobs(plan, stage, cases=None, variants=None, winner=None, seed=None):
    cases = cases or list(plan['cases'])
    for name in cases:
        if name not in plan['cases']:
            raise ValueError(f'Unknown case: {name}')
    if stage == 'confirmation':
        if variants or not winner or winner not in plan['variants']:
            raise ValueError('Confirmation requires --variant with one known candidate.')
        if winner in ('legacy_switch', 'legacy_open', 'fixed_budget'):
            raise ValueError('Choose an improved candidate for confirmation.')
        chosen = ['legacy_open', winner]
        seeds = [seed] if seed is not None else plan['defaults']['confirmation_seeds']
    else:
        if winner:
            raise ValueError('--variant is only used for confirmation.')
        chosen = variants or [name for name, value in plan['variants'].items()
                              if value['stage'] == stage]
        for name in chosen:
            if name not in plan['variants'] or plan['variants'][name]['stage'] != stage:
                raise ValueError(f'Variant {name!r} is not in stage {stage!r}.')
        seeds = [plan['defaults']['seed'] if seed is None else seed]
    if any(value < 0 for value in seeds):
        raise ValueError('Seeds must be nonnegative for reproducibility.')
    return [(case, variant, value) for case in dict.fromkeys(cases)
            for value in seeds for variant in dict.fromkeys(chosen)]


def selected_environment(gpu_count):
    return {
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES',
                                              '0,1' if gpu_count == 2 else '0'),
        'PYTORCH_CUDA_ALLOC_CONF': os.environ.get('PYTORCH_CUDA_ALLOC_CONF',
                                                 'expandable_segments:True')}


def make_job(plan, case_name, variant_name, seed, options, identity, runtime=None):
    case, variant = plan['cases'][case_name], plan['variants'][variant_name]
    defaults = plan['defaults']
    prompt = read_prompt(case['timing_prompt'] if variant.get('use_timing_prompt')
                         else case['prompt'])
    suffix_prompt = (read_prompt(case['suffix_prompt'])
                     if variant.get('use_suffix_prompt') else None)
    images = [{'path': str((REPO / name).resolve()), 'sha256': file_hash(REPO / name)}
              for name in case['images']]
    launch = [options.python]
    if options.gpu_count == 2:
        launch += ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2']
    command = launch + ['generate.py', '--task', 'i2v-A14B', '--size', defaults['size'],
                       '--frame_num', str(defaults['frame_num']), '--ckpt_dir',
                       str(Path(options.ckpt_dir).expanduser().resolve())
                       if options.ckpt_dir else '<CKPT_DIR>']
    for image, index in zip(images, [0, case['anchor_index']]):
        command += ['--image', image['path'], '--image_latent_index', str(index)]
    command += ['--image_condition_paired_anchor_attention',
                '--image_condition_paired_stream_weight',
                str(variant.get('suffix_weight', case['suffix_weight'])),
                '--image_condition_paired_suffix_fade_start_step',
                str(defaults['suffix_fade_start_step']),
                '--image_condition_paired_suffix_fade_end_step',
                str(defaults['suffix_fade_end_step'])]
    for key, flag in (
            ('prefix_switch_step', '--image_condition_paired_prefix_route_switch_step'),
            ('budget', '--image_condition_paired_prefix_reference_budget'),
            ('strength', '--image_condition_paired_prefix_reference_strength')):
        if key in variant:
            command += [flag, str(variant[key])]
    if 'budget' in variant or 'strength' in variant:
        command += ['--image_condition_paired_prefix_reference_curve', 'linear']
    if variant.get('preserve_first'):
        command += ['--image_condition_paired_prefix_preserve_first_frame']
    if 'prefix_fade' in variant:
        command += ['--image_condition_paired_prefix_reference_fade_start_step',
                    str(variant['prefix_fade'][0]),
                    '--image_condition_paired_prefix_reference_fade_end_step',
                    str(variant['prefix_fade'][1])]
    if suffix_prompt:
        command += ['--image_condition_paired_suffix_prompt', suffix_prompt['text']]
    command += ['--sample_steps', str(defaults['sample_steps']), '--sample_shift',
                str(defaults['sample_shift']), '--base_seed', str(seed),
                '--prompt', prompt['text']]
    if options.gpu_count == 2:
        command += ['--ulysses_size', '2', '--dit_fsdp']
    command += ['--t5_cpu', '--offload_model', 'True', '--convert_model_dtype']
    environment = selected_environment(options.gpu_count)
    config = {'schema_version': 1, 'case': case_name, 'variant': variant_name,
              'seed': seed, 'gpu_count': options.gpu_count, 'argv_without_output': command,
              'cwd': str(REPO / 'Wan2.2'), 'environment': environment,
              'inputs': images, 'prompt': prompt, 'suffix_prompt': suffix_prompt,
              'negative_prompt_default': default_negative_prompt(), 'source': identity,
              'runtime': runtime, 'expected_frames': defaults['frame_num'],
              'expected_fps': defaults['fps']}
    job_signature = signature(config)
    directory = external_directory(options.output_dir) / (
        f'{case_name}-{variant_name}-seed{seed}-{job_signature[:16]}')
    command = command + ['--save_file', str(directory / 'video.mp4')]
    return {'signature': job_signature, 'config': config, 'argv': command,
            'directory': str(directory)}


def runtime_identity(python, checkpoint):
    script = ('import json,sys,platform,importlib.metadata as m; '
              'print(json.dumps({"python":sys.version,"executable":sys.executable,'
              '"platform":platform.platform(),'
              '"packages":sorted((d.metadata.get("Name", ""),d.version) '
              'for d in m.distributions())}))')
    result = subprocess.run([python, '-c', script], capture_output=True, text=True, check=True)
    identity = json.loads(result.stdout)
    identity['cuda_preflight_source_sha256'] = file_hash(PREFLIGHT_PATH)
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                          '--format=csv,noheader'], capture_output=True, text=True, check=True)
    identity['gpus'] = gpu.stdout.strip().splitlines()
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        raise ValueError(f'Checkpoint directory does not exist: {checkpoint}')
    files = sorted(p for p in checkpoint.rglob('*') if p.is_file() and '.cache' not in p.parts)
    identity['checkpoint_inventory'] = [
        {'path': str(p.relative_to(checkpoint)), 'bytes': p.stat().st_size,
         'mtime_ns': p.stat().st_mtime_ns} for p in files]
    identity['checkpoint_inventory_note'] = (
        'Size/mtime inventory only; independently verify weight checksums before experiments.')
    return identity


def validate_video(path, expected_frames, expected_fps, python):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError('Generation did not produce a nonempty video.')
    if shutil.which('ffmpeg') and shutil.which('ffprobe'):
        # Decode all frames with errors fatal, then independently count decoded frames.
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path),
                        '-map', '0:v:0', '-f', 'null', '-'], check=True,
                       capture_output=True, text=True)
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
            '-show_entries', 'stream=nb_read_frames,width,height,avg_frame_rate',
            '-of', 'json', str(path)], check=True, capture_output=True, text=True)
        stream = json.loads(result.stdout)['streams'][0]
        numerator, denominator = stream['avg_frame_rate'].split('/')
        data = {'frames': int(stream['nb_read_frames']), 'width': stream['width'],
                'height': stream['height'], 'fps': float(numerator) / float(denominator),
                'validator': 'ffmpeg-full-decode-and-ffprobe'}
    else:
        script = ('import imageio.v2 as i,json,sys; r=i.get_reader(sys.argv[1]); '
                  'm=r.get_meta_data(); n=0; shape=None\n'
                  'for f in r:\n n+=1; shape=f.shape\n'
                  'r.close(); print(json.dumps({"frames":n,"width":shape[1],'
                  '"height":shape[0],"fps":m["fps"],'
                  '"validator":"imageio-full-decode"}))')
        result = subprocess.run([python, '-c', script, str(path)], check=True,
                                capture_output=True, text=True)
        data = json.loads(result.stdout)
    if (data['frames'] != expected_frames or abs(data['fps'] - expected_fps) > 0.01
            or data['width'] <= 0 or data['height'] <= 0):
        raise ValueError(f'Unexpected video metadata: {data}')
    data.update({'sha256': file_hash(path), 'bytes': path.stat().st_size,
                 'validated_at': datetime.now(timezone.utc).isoformat()})
    return data


def require_video_validator(python):
    if shutil.which('ffmpeg') and shutil.which('ffprobe'):
        for executable in ('ffmpeg', 'ffprobe'):
            subprocess.run([executable, '-version'], check=True, capture_output=True)
        return
    subprocess.run([python, '-c',
                    'import imageio.v2, imageio_ffmpeg, subprocess; '
                    'subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-version"], '
                    'check=True, capture_output=True)'],
                   check=True, capture_output=True, text=True)


def run_cuda_preflight(python, output_dir, gpu_count, lock_fd=None):
    directory = external_directory(output_dir)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = directory / f'cuda-preflight-{stamp}-{uuid.uuid4().hex}.json'
    log_path = output.with_suffix('.log')
    command = [python, str(PREFLIGHT_PATH), '--output', str(output)]
    environment = os.environ.copy()
    environment.update(selected_environment(gpu_count))
    print(f'PREFLIGHT {output.name}', flush=True)
    with log_path.open('x', encoding='utf-8') as log:
        subprocess.run(command, cwd=str(REPO), env=environment,
                       stdout=log, stderr=subprocess.STDOUT, check=True,
                       **inherited_lock_options(lock_fd))
    report = json.loads(output.read_text(encoding='utf-8'))
    if report.get('status') != 'passed':
        raise RuntimeError(f'CUDA preflight did not pass: {output}')
    return {'path': str(output), 'sha256': file_hash(output), 'log': str(log_path),
            'argv': command, 'status': 'passed'}


def completed_matches(job):
    directory = Path(job['directory'])
    try:
        result = json.loads((directory / 'completed.json').read_text(encoding='utf-8'))
        video = directory / 'video.mp4'
        validation = result['validation']
        return (result['signature'] == job['signature']
                and validation['frames'] == job['config']['expected_frames']
                and abs(validation['fps'] - job['config']['expected_fps']) <= 0.01
                and validation['validator'] in ('imageio-full-decode',
                                                'ffmpeg-full-decode-and-ffprobe')
                and video.stat().st_size == validation['bytes']
                and file_hash(video) == validation['sha256'])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
    temporary.replace(path)


@contextmanager
def queue_lock(directory):
    """Yield a lock FD; POSIX children retain it if the queue parent dies abruptly."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.queue.lock').open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f'Another queue is using {directory}.') from error
        try:
            yield stream.fileno()
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def inherited_lock_options(lock_fd):
    # flock belongs to the shared open-file description. A surviving child keeps
    # the lock after SIGKILL closes the parent's descriptors. Do not explicitly
    # unlock until normal child cleanup has completed.
    return {'pass_fds': (lock_fd,)} if os.name != 'nt' and lock_fd is not None else {}


@contextmanager
def termination_handlers():
    """Turn ordinary service termination into the same cleanup path as an error."""
    signals = [signal.SIGTERM]
    if hasattr(signal, 'SIGHUP'):
        signals.append(signal.SIGHUP)
    previous = {number: signal.getsignal(number) for number in signals}

    def terminate(number, frame):
        # Repeated termination must not interrupt the bounded child cleanup.
        for item in signals:
            signal.signal(item, signal.SIG_IGN)
        raise RuntimeError(f'Queue interrupted by signal {number}.')

    try:
        for number in signals:
            signal.signal(number, terminate)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


@contextmanager
def defer_termination_during_spawn():
    """Assign the child handle before a stop signal unwinds the queue parent.

    Defer Python handlers instead of blocking OS signals: a blocked mask would
    be inherited by the generator and prevent its normal SIGTERM cleanup.
    """
    numbers = {signal.SIGTERM, signal.SIGINT}
    if hasattr(signal, 'SIGHUP'):
        numbers.add(signal.SIGHUP)
    previous = {number: signal.getsignal(number) for number in numbers}
    pending = []
    try:
        for number in numbers:
            signal.signal(number, lambda signum, frame: pending.append(signum))
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    if pending:
        number = pending[0]
        handler = previous[number]
        if callable(handler):
            handler(number, None)
        elif handler != signal.SIG_IGN:
            signal.raise_signal(number)


def run_job(job, python, lock_fd=None):
    directory = external_directory(job['directory'])
    directory.mkdir(parents=True, exist_ok=True)
    if completed_matches(job):
        print(f'SKIP validated completed job: {directory.name}', flush=True)
        return
    # Preserve failed/interrupted attempts. Diffusion starts from the initial noise.
    attempt = 1
    while (directory / f'attempt-{attempt}').exists():
        attempt += 1
    previous = [directory / name for name in ('video.mp4', 'run.log', 'completed.json',
                                              'failure.json', 'run.json')]
    if any(path.exists() for path in previous):
        archive = directory / f'attempt-{attempt}'
        archive.mkdir()
        for path in previous:
            if path.exists():
                path.replace(archive / path.name)
    write_json(directory / 'run.json', job)
    print(f'RUN {directory.name}', flush=True)
    environment = os.environ.copy()
    environment.update(job['config']['environment'])
    try:
        with (directory / 'run.log').open('w', encoding='utf-8') as log:
            process_options = ({'start_new_session': True} if os.name != 'nt' else
                               {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP})
            process_options.update(inherited_lock_options(lock_fd))
            process = None
            try:
                with defer_termination_during_spawn():
                    process = subprocess.Popen(
                        job['argv'], cwd=job['config']['cwd'], env=environment,
                        stdout=log, stderr=subprocess.STDOUT, **process_options)
                return_code = process.wait()
                if return_code:
                    raise subprocess.CalledProcessError(return_code, job['argv'])
            except BaseException:
                if process is not None and process.poll() is None:
                    if os.name == 'nt':
                        try:
                            process.send_signal(signal.CTRL_BREAK_EVENT)
                        except OSError:
                            process.terminate()
                    else:
                        os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        if os.name == 'nt':
                            process.kill()
                        else:
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
        validation = validate_video(directory / 'video.mp4', job['config']['expected_frames'],
                                    job['config']['expected_fps'], python)
        write_json(directory / 'completed.json', {'signature': job['signature'],
                                                  'validation': validation})
        print(f'COMPLETE {directory.name}', flush=True)
    except BaseException as error:
        write_json(directory / 'failure.json', {'signature': job['signature'],
                                               'error': repr(error),
                                               'stderr': getattr(error, 'stderr', None),
                                               'time': datetime.now(timezone.utc).isoformat()})
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('primary', 'diagnostic', 'confirmation'),
                        default='primary')
    parser.add_argument('--cases', nargs='+', help='Case names; default: both cases.')
    parser.add_argument('--variants', nargs='+', help='Filter variants in primary/diagnostic.')
    parser.add_argument('--variant', help='Explicit selected candidate for confirmation.')
    parser.add_argument('--seed', type=int, help='Override the stage seed(s).')
    parser.add_argument('--ckpt-dir')
    parser.add_argument('--output-dir', default=str(REPO.parent / 'visual-relay-ablation-runs'))
    parser.add_argument('--python', default=sys.executable, help='Generation/validation Python.')
    parser.add_argument('--gpu-count', type=int, choices=(1, 2), default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true', help='Actually run sequential experiments.')
    mode.add_argument('--dry-run', action='store_true', help='Print JSON commands only (default).')
    options = parser.parse_args(argv)
    if options.execute and not options.ckpt_dir:
        parser.error('--ckpt-dir is required with --execute')
    return options


def main(argv=None):
    options = parse_args(argv)
    options.output_dir = str(external_directory(options.output_dir))
    plan = json.loads(PLAN_PATH.read_text(encoding='utf-8'))
    selections = selected_jobs(plan, options.stage, options.cases, options.variants,
                               options.variant, options.seed)
    identity = source_identity()
    if not options.execute:
        jobs = [make_job(plan, *selection, options, identity) for selection in selections]
        print(json.dumps({'mode': 'dry-run', 'jobs': jobs}, indent=2, ensure_ascii=True))
        return 0
    # Hold the lock during runtime inspection as well as all jobs and validation.
    with queue_lock(Path(options.output_dir)) as lock_fd, termination_handlers():
        runtime = runtime_identity(options.python, options.ckpt_dir)
        require_video_validator(options.python)
        preflight = run_cuda_preflight(options.python, options.output_dir, options.gpu_count,
                                       lock_fd=lock_fd)
        for selection in selections:
            if source_identity() != identity:
                raise RuntimeError(
                    'Source, input, or prompt files changed after the queue started. '
                    'The queue stopped; review the change before restarting.')
            job = make_job(plan, *selection, options, identity, runtime)
            # A unique preflight report belongs to the invocation, not the stable
            # generation signature, so rechecking CUDA does not invalidate resume.
            job['cuda_preflight'] = preflight
            run_job(job, options.python, lock_fd=lock_fd)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(1)
