#!/usr/bin/env python3
"""Validate paired attention on the target GPU without downloading weights.

Run before generating videos. Missing CUDA or FlashAttention is a failure, not
a skipped check. JSON goes to stdout; --output may save it outside this checkout.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
from unittest import mock


REPO_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ATOL = 0.02
LSE_ATOL = 0.001
HEAD_DIM = 128  # Wan2.2-I2V-A14B: hidden dimension 5120 / 40 attention heads.


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output', type=Path,
        help='Optional new JSON file outside the repository; existing files are refused.')
    args = parser.parse_args()
    if args.output is not None:
        args.output = args.output.expanduser().resolve()
        if args.output.is_relative_to(REPO_DIR):
            parser.error('--output must be outside the repository')
        if args.output.exists():
            parser.error('--output already exists; choose a new file')
    return args


def load_attention():
    spec = importlib.util.spec_from_file_location(
        'attention_cuda_preflight', REPO_DIR / 'Wan2.2/wan/modules/attention.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def max_error(torch, actual, expected, tolerance, label):
    if actual.shape != expected.shape:
        raise AssertionError(f'{label}: shape {actual.shape} != {expected.shape}')
    if not torch.isfinite(actual).all().item():
        raise AssertionError(f'{label}: output contains non-finite values')
    error = (actual.float() - expected.float()).abs().max().item()
    if error > tolerance:
        raise AssertionError(f'{label}: max error {error:.8g} exceeds {tolerance}')
    return error


def dense_scores(torch, q, k):
    # The model supplies fp32 RoPE Q/K and bf16 V; FlashAttention converts Q/K
    # to V's bf16 dtype before its dot product. Mirror that input quantization,
    # then independently compute dense scores and probabilities in fp32.
    q = q.to(torch.bfloat16).float()
    k = k.to(torch.bfloat16).float()
    return torch.einsum('bqhd,bkhd->bhqk', q, k) / q.size(-1)**0.5


def dense_attention(torch, q, k, v):
    scores = dense_scores(torch, q, k)
    output = torch.einsum('bhqk,bkhd->bqhd', scores.softmax(-1), v.float())
    return output, scores.logsumexp(-1).transpose(1, 2)


def check_lse(torch, module, backend, generator):
    q = torch.randn(2, 9, 3, HEAD_DIM, generator=generator).cuda()
    k = torch.randn(2, 13, 3, HEAD_DIM, generator=generator).cuda()
    v = torch.randn(2, 13, 3, HEAD_DIM, generator=generator).cuda().bfloat16()
    output, lse = module.flash_attention(q, k, v, version=backend, return_lse=True)
    expected, expected_lse = dense_attention(torch, q, k, v)
    if lse.shape != (2, 9, 3) or lse.dtype != torch.float32:
        raise AssertionError(f'FA{backend}: unexpected normalized LSE shape/dtype')
    result = {
        'backend': f'FA{backend}',
        'output_max_abs_error': max_error(
            torch, output, expected, OUTPUT_ATOL, f'FA{backend} wrapper output'),
        'lse_max_abs_error': max_error(
            torch, lse, expected_lse, LSE_ATOL, f'FA{backend} LSE'),
        'lse_shape': list(lse.shape),
    }
    if backend == 2:
        # Verify the installed FA2 build, not just the documented version:
        # requesting the public LSE tuple at zero dropout must return an empty
        # third tensor, rather than a quadratic attention-probability buffer.
        native = module.flash_attn.flash_attn_varlen_func(
            q=q.bfloat16().flatten(0, 1),
            k=k.bfloat16().flatten(0, 1),
            v=v.flatten(0, 1),
            cu_seqlens_q=torch.tensor([0, 9, 18], device='cuda', dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 13, 26], device='cuda', dtype=torch.int32),
            max_seqlen_q=9, max_seqlen_k=13,
            dropout_p=0., return_attn_probs=True)
        if not isinstance(native, tuple) or len(native) != 3:
            raise AssertionError('FA2 did not return the expected output/LSE/probability tuple')
        if not isinstance(native[2], torch.Tensor) or native[2].numel() != 0:
            raise AssertionError('FA2 allocated attention probabilities at zero dropout')
        result['zero_dropout_probability_elements'] = native[2].numel()
    return result


def make_fixture(torch, generator):
    frames, anchor, height, width, heads, channels = 7, 4, 2, 3, 3, HEAD_DIM
    tokens = height * width
    main_len, suffix_len = frames * tokens, (frames - anchor) * tokens
    padded_len = main_len + 5
    q, k = [torch.randn(2, padded_len, heads, channels, generator=generator).cuda()
            for _ in range(2)]
    v = torch.randn(2, padded_len, heads, channels, generator=generator).cuda().bfloat16()
    for tensor in (q, k, v):
        tensor[0, main_len:] = 1000.
        tensor[1, suffix_len:] = -1000.
    phase = torch.outer(
        torch.arange(frames, device='cuda', dtype=torch.float32),
        torch.linspace(0.7, 0.01, channels // 2, device='cuda'))
    return {
        'q': q, 'k': k, 'v': v, 'anchor_index': anchor,
        'seq_lens': torch.tensor([main_len, suffix_len], device='cuda'),
        'grid_sizes': torch.tensor([[frames, height, width],
                                    [frames - anchor, height, width]], device='cuda'),
        'freqs': torch.polar(torch.ones_like(phase), phase),
        'stream_weight': 0.5,
    }


def independent_shift(torch, tensor, freqs, first_shifted_token):
    """Real-valued rotation independent of production's complex RoPE helper."""
    result = tensor.clone()
    channels = tensor.size(-1)
    temporal_pairs = channels // 2 - 2 * (channels // 6)
    part = result[:, first_shifted_token:, :, :2 * temporal_pairs]
    pairs = part.reshape(*part.shape[:-1], temporal_pairs, 2).clone()
    angle = -torch.angle(freqs[1, :temporal_pairs])
    rotated = torch.stack([
        pairs[..., 0] * angle.cos() - pairs[..., 1] * angle.sin(),
        pairs[..., 0] * angle.sin() + pairs[..., 1] * angle.cos(),
    ], dim=-1).flatten(-2)
    result[:, first_shifted_token:, :, :2 * temporal_pairs] = rotated
    return result


def dense_prefix(torch, fixture, strength, curve, preserve):
    tokens = int(fixture['grid_sizes'][0, 1:].prod().item())
    main_len = int(fixture['seq_lens'][0])
    start = fixture['anchor_index'] * tokens
    end = start + tokens
    q = fixture['q'][0:1, :start]
    k = independent_shift(torch, fixture['k'][0:1, :main_len], fixture['freqs'], end)
    v = fixture['v'][0:1, :main_len]
    groups = [
        (torch.cat([k[:, tokens:start], k[:, end:]], dim=1),
         torch.cat([v[:, tokens:start], v[:, end:]], dim=1)),
        (k[:, :tokens], v[:, :tokens]),
        (k[:, start:end], fixture['v'][1:2, :tokens]),
    ]
    scores = [dense_scores(torch, q, group_k) for group_k, _ in groups]
    probabilities = torch.cat(scores, dim=-1).softmax(-1)
    context, left, right = probabilities.split([x.size(-1) for x in scores], dim=-1)
    mass = left.sum(-1, keepdim=True) + right.sum(-1, keepdim=True)
    w = torch.arange(fixture['anchor_index'], device=q.device).float()
    w = w / fixture['anchor_index']
    if curve == 'smoothstep':
        w = w.square() * (3. - 2. * w)
    w = w.repeat_interleave(tokens).view(1, 1, -1, 1)
    adjusted = torch.cat([
        context,
        (1. - strength) * left + strength * mass * (1. - w) * scores[1].softmax(-1),
        (1. - strength) * right + strength * mass * w * scores[2].softmax(-1),
    ], dim=-1)
    if preserve:
        adjusted[:, :, :tokens] = probabilities[:, :, :tokens]
    return torch.einsum(
        'bhqk,bkhd->bqhd', adjusted,
        torch.cat([group_v.float() for _, group_v in groups], dim=1))


def check_paired(torch, module, backend, fixture):
    original_flash = module.flash_attention

    def selected_flash(*args, **kwargs):
        return original_flash(*args, version=backend, **kwargs)

    def run(**options):
        # Paired attention shifts Q/K RoPE in place. Never reuse tensors across
        # the baseline, a candidate, or another backend without cloning.
        inputs = {key: value.clone() if isinstance(value, torch.Tensor) else value
                  for key, value in fixture.items()}
        with mock.patch.object(module, 'flash_attention', selected_flash):
            return module.paired_anchor_attention(**inputs, **options)

    baseline = run()
    anchor = fixture['anchor_index']
    tokens = int(fixture['grid_sizes'][0, 1:].prod().item())
    start = anchor * tokens
    results = []
    for strength in (0., 0.25, 0.5, 1.):
        for curve in ('linear', 'smoothstep'):
            for preserve in (False, True):
                output = run(
                    prefix_reference_strength=strength,
                    prefix_reference_curve=curve,
                    prefix_preserve_first_frame=preserve)
                if output.dtype != fixture['q'].dtype or output.shape != fixture['q'].shape:
                    raise AssertionError('Paired attention changed the output shape or dtype')
                if not torch.isfinite(output).all().item():
                    raise AssertionError('Paired attention produced non-finite values')
                if strength == 0. and not torch.equal(output, baseline):
                    raise AssertionError('Strength zero differs from the exact legacy fastpath')
                if not torch.equal(output[0, start:], baseline[0, start:]):
                    raise AssertionError('The messenger, main suffix, or padding changed')
                if not torch.equal(output[1], baseline[1]):
                    raise AssertionError('The independent suffix stream or padding changed')
                for sample, length in enumerate(fixture['seq_lens'].tolist()):
                    if torch.count_nonzero(output[sample, length:]).item() != 0:
                        raise AssertionError('Nonzero output padding')
                expected = dense_prefix(torch, fixture, strength, curve, preserve)
                case = {
                    'backend': f'FA{backend}', 'strength': strength,
                    'curve': curve, 'preserve_first_frame': preserve,
                    'max_abs_error': max_error(
                        torch, output[0:1, :start], expected, OUTPUT_ATOL,
                        f'FA{backend} strength={strength} curve={curve} preserve={preserve}'),
                }
                if preserve:
                    # Allow at most one bf16 unit at the largest baseline
                    # magnitude, while also reporting bitwise equality.
                    first_tolerance = (torch.finfo(torch.bfloat16).eps
                                       * baseline[0, :tokens].abs().max().item()
                                       + 1e-6)
                    case['first_frame_max_abs_error'] = max_error(
                        torch, output[0, :tokens], baseline[0, :tokens], first_tolerance,
                        f'FA{backend} preserved first frame')
                    case['first_frame_abs_tolerance'] = first_tolerance
                    case['first_frame_bitwise_equal'] = torch.equal(
                        output[0, :tokens], baseline[0, :tokens])
                results.append(case)
    return results


def run_checks(report):
    import torch

    report['torch_version'] = torch.__version__
    report['torch_cuda_version'] = torch.version.cuda
    report['attention_source_sha256'] = hashlib.sha256(
        (REPO_DIR / 'Wan2.2/wan/modules/attention.py').read_bytes()).hexdigest()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; this mandatory preflight cannot be skipped')
    module = load_attention()
    backends = []
    if module.FLASH_ATTN_2_AVAILABLE:
        backends.append(2)
        report['flash_attention_2_version'] = importlib.metadata.version('flash-attn')
    if module.FLASH_ATTN_3_AVAILABLE:
        backends.append(3)
    if not backends:
        raise RuntimeError('FlashAttention is unavailable; this mandatory preflight cannot be skipped')
    report['device'] = torch.cuda.get_device_name()
    report['device_capability'] = list(torch.cuda.get_device_capability())
    report['default_backend'] = 'FA3' if 3 in backends else 'FA2'
    report['output_abs_tolerance'] = OUTPUT_ATOL
    report['lse_abs_tolerance'] = LSE_ATOL
    report['head_dimension'] = HEAD_DIM
    report['lse_checks'], report['paired_checks'] = [], []
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    generator = torch.Generator().manual_seed(4827316)
    with torch.inference_mode():
        fixture = make_fixture(torch, generator)
        for backend in backends:
            report['lse_checks'].append(check_lse(torch, module, backend, generator))
            report['paired_checks'].extend(check_paired(torch, module, backend, fixture))
        torch.cuda.synchronize()


def main():
    args = parse_args()
    report = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'failed'}
    try:
        run_checks(report)
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    serialized = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as handle:
            handle.write(serialized + '\n')
    print(serialized)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
