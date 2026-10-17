# Controlled paired-reference experiments

These experiments compare the original paired route, the fixed reference budget, and native-mass redistribution on the supplied warrior and falling-object image pairs. They use the same official Wan2.2-I2V-A14B weights. No retraining is involved.

The executable definition is [ablation_plan.json](ablation_plan.json). The runner prints commands by default and starts generation only with `--execute`. Generated videos, logs, validation records, and reports must stay outside the repository and must not be pushed to GitHub.

## Primary comparisons

Run each row for both image pairs with seed `4827316`: ten videos in total. These are screening experiments, not proof of robustness or input-order symmetry.

| Variant | Prefix attention | Purpose |
| --- | --- | --- |
| `legacy_switch` | Original joint softmax; remove right reference at step 25 | Same-code reference for the originally supplied command |
| `legacy_open` | Original joint softmax; keep right reference throughout | Baseline isolating the new attention method |
| `fixed_budget` | Fixed `rho=0.3`, linear curve | Earlier grouped-budget method |
| `adaptive` | Preserve native reference total, `strength=0.5`, linear curve | Isolate reference-mass redistribution |
| `adaptive_first` | `adaptive` plus original attention for first-latent queries | Isolate first-frame attention protection |

The comparison `legacy_switch` versus `legacy_open` isolates route scheduling. Compare `legacy_open`, `fixed_budget`, and `adaptive` for the reference allocation mechanism. Compare `adaptive` versus `adaptive_first` for first-frame protection. Fixed `rho=0` is **not** the original baseline. Native-mass `strength=0` uses the exact original prefix path; numerical preflight checks this property.

Both primary cases preserve the original positive prompts, this revision's English negative prompt, 81 frames, size budget `832*480`, 40 sampling steps, shift 3, suffix fade steps 25 to 30, and initial seed. Prompt expansion is disabled.

| Case | Input images | Img2 latent index | Suffix weight | Original prompt |
| --- | --- | --- | --- | --- |
| `warriors` | `Img1.png`, `Img2.png` | 12 | 0.5 | [warriors.txt](prompts/warriors.txt) |
| `falling` | `fnr1.png`, `fnr2.png` | 10 | 1.0 | [falling.txt](prompts/falling.txt) |

Some earlier code revisions used Chinese negative prompts. Do not use those videos as a controlled comparison with this branch's English defaults. Rerun comparison variants on the same server environment; GPU counts and kernel implementations can change outputs even with identical seeds.

## Running the plan

From the repository root, inspect the exact commands without loading the model or creating run artifacts:

```bash
python experiments/run_ablation.py --stage primary
```

Once the server environment and checkpoint checksums are verified:

```bash
python experiments/run_ablation.py \
  --stage primary --execute \
  --ckpt-dir /path/to/Wan2.2-I2V-A14B \
  --python /path/to/venv/bin/python \
  --output-dir /path/outside/the/repository/visual-relay-runs
```

The execution path first runs [check_attention_cuda.py](check_attention_cuda.py) in the selected environment. It fails if CUDA or FlashAttention is unavailable or if the installed backend fails LSE and dense-reference comparisons. Passing CPU tests is insufficient for this gate. The standalone preflight can also be run before downloading weights:

```bash
python experiments/check_attention_cuda.py
```

The runner defaults to one GPU and to `repo.parent/visual-relay-ablation-runs` for output. `--gpu-count 2` enables the original two-process Ulysses/FSDP configuration. Keep the same GPU configuration throughout a comparison. `CUDA_VISIBLE_DEVICES` may select the devices explicitly. The runner does not silently change precision, resolution, frame count, steps, or offloading after a failure.

To run or print a subset, add `--cases warriors` and/or `--variants legacy_open adaptive_first`. Jobs run sequentially. A queue lock prevents two runner processes from using the same output root; do not launch independent queues on the same GPUs.

On Linux, generation and CUDA-preflight children inherit the lock, so a surviving child keeps the queue locked after abrupt parent termination. SIGTERM/SIGHUP use bounded process-group cleanup. Windows supports graceful cleanup but does not provide the same hard-parent-termination guarantee. Source, input, and prompt hashes are checked again before each job; changes stop the queue.

Each job records exact arguments, revision and dirty status, source/input/prompt hashes, negative prompt text, runtime versions, GPU identity, and checkpoint inventory. The inventory records sizes and timestamps, **not a substitute for model-file checksum verification**. A video is marked complete only after full decoding, frame/FPS checks, and SHA-256 calculation.

Rerunning the same command skips only completed jobs with matching configurations and unchanged validated video hashes. Failed or interrupted attempts are preserved before retry. An interrupted diffusion generation restarts from the seed; the runner does not save or resume denoising checkpoints. Model download resumption is a separate setup concern.

## Diagnostic comparisons

Review the primary videos before selecting these runs. The reference is `adaptive_first`; the variants below each change one intervention except the explicitly separate timing test.

| Variant | Additional change | Question |
| --- | --- | --- |
| `adaptive_fade` | Fade redistribution strength during steps 25 to 30; keep the right route | Does restoring original late-step attention help detail formation? |
| `suffix_off` | Set suffix fusion weight to zero; retain the B0 right-reference value | Does independently queried suffix motion contribute to event restart? |
| `continuation` | Change only B's text to continue from Img2 | Does repeating the full-event prompt conflict with the suffix state? |
| `timing_prompt` | Use an explicit internal milestone in the main prompt and a continuation in B | Can timing instructions help reach the intended intermediate state? This is a conditioning experiment, not an isolated attention comparison. |

```bash
python experiments/run_ablation.py \
  --stage diagnostic --variants suffix_off continuation
```

Add `--execute`, checkpoint/runtime paths, and the same output directory when running on the server. Do not infer that step 25 is a universal expert boundary: the sampler logs actual timestep, high/low-noise expert, reference strength, suffix weight, and right-route status. The fade is a testable schedule, not a proven optimum.

Continuation and timing prompts are stored in [prompts/](prompts/). Changing B's prompt also changes its evolving B0 value read by prefix queries; its effects are not confined to frames after the anchor. The main full-sequence prompt is unchanged in `continuation`.

## Evaluation and confirmation

Watch each full video at normal speed and frame by frame. Inspect every frame in the first second and dense windows around the internal reference: frames 35 to 60 for warriors and 25 to 50 for falling objects, using zero-based indices.

- **Opening:** compare first-frame fidelity, subject detail, noise, blur, and flicker. A sharpness score alone can reward noise and does not establish quality.
- **Warriors:** require continuous intermediate viewpoints, background parallax, coherent occlusion, and persistent identities while both characters retain world positions. Swapping characters or morphing directly to the second composition does not count as an orbit.
- **Falling objects:** track object height relative to the same ground reference and record first contact. Img2 depicts the rock grounded and the feather still airborne. Landing before that point and jumping back up is a failure; later flutter alone is not a failure.
- **Boundary:** temporary Img2 removal nominally makes the first retained suffix block frames 45 to 48 for warriors and 37 to 40 for falling. Temporal VAE dependencies can spread changes beyond those blocks. A coincident discontinuity is evidence to investigate, not proof of its cause.
- **Control:** assess consistency near both reference times, motion and identity together. A clearer static video or equal direct attention coefficients does not establish balanced visual control.

After explicitly selecting a candidate, compare it with `legacy_open` at two additional seeds:

```bash
python experiments/run_ablation.py \
  --stage confirmation --variant adaptive_first
```

This produces eight jobs across both cases and seeds `4827317` and `4827318`. If the candidate changes prompts, report that conditioning difference explicitly; use the matching primary/diagnostic pair to attribute individual changes. No winner is selected automatically. Input-order symmetry requires separate, carefully matched reversed-order/noise/motion comparisons; reversing a falling event is not the same physical task.

## Research rationale and limits

Native-mass preservation is an algebraic design choice, not an experimentally established quality improvement. Small, scheduled attention interventions in [Anchoring and Rescaling Attention for Semantically Coherent Inbetweening](https://arxiv.org/html/2603.17651v1) motivate a conservative approach, but that work uses Wan2.1 FLF2V and different attention mechanisms. Its results do not validate this implementation. The [official Wan2.2 architecture description](https://github.com/Wan-Video/Wan2.2#introduction-of-wan22) distinguishes high-noise layout and low-noise detail experts, motivating an independent schedule ablation.

The main conditioning, mixed right K/V sources, residuals, and temporal VAE remain asymmetric. Img2 is still a temporary reference removed before decoding. No explicit camera-pose, object-trajectory, ground-contact, or strict endpoint constraint is added. If conservative attention and phase conditioning preserve quality but still fail the requested motion, the next research step is an explicit intermediate-state or motion constraint, rather than declaring another scalar weight sufficient.

## Earlier fixed-budget command

The original helper remains available for reproducing fixed-budget examples:

```bash
CKPT_DIR=/path/to/Wan2.2-I2V-A14B \
OUTPUT_DIR=/path/outside/the/repository/fixed-budget-runs \
bash experiments/run_reference_budget.sh warriors
```

Use the controlled Python runner for new comparisons and its safer external output default. No experiment results or reports belong in this README.
