# Visual-Relay: Paired Reference Attention Experiments

This branch adds **time-dependent left/right reference allocation** to the existing `i2v-A14B` paired-anchor mechanism. It supports the earlier fixed reference budget and a native-mass redistribution mode that preserves the model's current reference/context attention totals. Optional first-frame attention protection, redistribution fading, and a separate suffix continuation prompt can be tested independently.

Branch: `codex/paired-reference-budget`. Original baseline: [`2affd871`](https://github.com/GordonChen19/Visual-Relay/commit/2affd871f23b01026fb5118bed7b43f91e332ae7). The feature is disabled by default and uses the existing Wan2.2-I2V-A14B checkpoint without additional training weights. See the [Wan2.2 documentation](Wan2.2/README.md) for model downloads, dependencies, and hardware requirements.

This is an inference experiment. It constrains the direct reference coefficients; **it does not establish equal control over the generated video**.

## Shared reference sources

The existing A stream covers the full video; the B stream covers the suffix starting at Img2. Both receive slices of the same noisy latent at each sampling step, but their hidden representations evolve differently inside the model. Let `a` be the latent index of the temporary Img2 anchor. Prefix keys and values have these sources:

| Group | Key source | Value source |
| --- | --- | --- |
| Left reference `L` | First position of A | First position of A |
| Right reference `R` | Temporary anchor `a` in A | First position of B |
| Video context `V` | Remaining ordinary positions in A | Remaining ordinary positions in A |

Video context includes ordinary positions on both sides of the anchor, excluding the two separately read reference positions to avoid directly counting them twice. Existing RoPE alignment for ordinary suffix positions is preserved. **The right reference remains `K_A(anchor) + V_B(0)`; it is not changed to B/B.**

## Fixed-budget mode

After separately normalizing the three groups to obtain `O_L`, `O_R`, and `O_V`, the outputs are combined as follows:

```text
u = j / a                         # prefix latent index j = 0, ..., a-1
w(u) = u                          # linear
# or w(u) = 3*u*u - 2*u*u*u       # smoothstep

O(j) = (1-rho) * O_V(j)
     + rho * ((1-w(u)) * O_L(j) + w(u) * O_R(j))
```

- `rho` is the fixed total image-reference budget; `1-rho` is reserved for ordinary video context.
- `w(u)` controls the handoff between references and satisfies `w(1-u)=1-w(u)`. The direct coefficients are equal at the midpoint when it falls on the latent grid.
- `j` is a **video latent time position**, not a denoising step. All spatial tokens at a given time position share the same handoff coefficient.
- The temporary anchor is not a prefix query, so no query in this calculation has `u=1`. Temporal VAE compression and removal of the temporary position also mean this is not an exact constraint on individual decoded frames or pixels.

Only prefix attention is replaced. The temporary anchor update, main-stream residuals, suffix blending, sampler, and decoder retain their existing mechanisms. The suffix parameter `image_condition_paired_stream_weight` still defaults to `0.25` and applies to a different region from `rho`.

The new path uses three grouped prefix attention calls instead of one, adding two prefix attention calls per invocation. It uses the existing attention kernel without materializing the full attention score matrix. GPU memory and runtime costs still need measurement.

## Native-mass redistribution

The original joint softmax already chooses a reference total for each query and head. A fixed budget replaces that total, potentially amplifying weak references or suppressing strong self/reference attention. The alternative mode preserves that total and moves only the left/right split toward the video-time curve:

```text
p_L = exp(LSE_left - LSE_native)
p_R = exp(LSE_right - LSE_native)
m = p_L + p_R
delta = strength * (m*w - p_R)
O_new = O_native + delta * (O_R - O_L)
```

`O_native` uses the original paired joint-softmax route, including the existing hybrid right reference. Its context mass stays `1-m`. The new left and right masses are `(1-strength)*p_L + strength*m*(1-w)` and `(1-strength)*p_R + strength*m*w`. This identity holds at the current hidden states; later layers and sampling steps can still diverge from the baseline.

- `strength=0` executes the original attention path directly, avoiding multi-kernel reconstruction drift. It differs fundamentally from fixed `rho=0`, which excludes both references.
- `strength=1` enforces the temporal split within the available reference mass. Intermediate strengths retain part of the original split; they do not enforce equal midpoint coefficients.
- Optional first-frame protection copies the native output for A0 queries. B0 continues its original computation. This protects a direct attention route, not the final first-frame pixels or end-to-end symmetry.
- Optional cosine fading reduces only redistribution strength over **sampling steps**. At zero, original joint attention resumes with the right reference still present. This differs from the old route switch, which removes that reference.
- An optional suffix prompt gives B continuation semantics from Img2. It changes text conditioning, including B0's evolved reference value, and must be evaluated separately from attention changes. Both branches retain the same negative prompt. Unset preserves the original single-positive-prompt encoder call.

Native-mass mode uses the native joint attention plus the two reference reads and their log-sum-exp values. Corrections accumulate in FP32. FlashAttention 2.8.3 returns LSE without a quadratic probability buffer when dropout is zero; a mandatory CUDA preflight checks the installed backend before ablation generation. See the [upstream Python interface](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/flash_attn/flash_attn_interface.py#L896-L928).

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--image_condition_paired_prefix_reference_budget` | Unset; feature disabled | Finite value in `[0,1]`, corresponding to `rho` |
| `--image_condition_paired_prefix_reference_curve` | `linear` | `linear` or `smoothstep` |
| `--image_condition_paired_prefix_reference_strength` | Unset | Native-mass redistribution strength in `[0,1]`; mutually exclusive with fixed budget |
| `--image_condition_paired_prefix_preserve_first_frame` | Off | Keep A0's direct attention on the original route in native-mass mode |
| `--image_condition_paired_prefix_reference_fade_start_step` / `end_step` | Unset | Cosine fade of redistribution strength; require `0 <= start < end < sample_steps` |
| `--image_condition_paired_suffix_prompt` | Unset | Separate continuation prompt for B; otherwise reuse main prompt |

Usage notes:

- Enable `--image_condition_paired_anchor_attention` with exactly two images and latent indices `[0,a]`. Img2 must be internal, with a video segment remaining to its right.
- **Omit both budget and strength for the original attention baseline. `rho=0` is context-only; `strength=0` is the original route.** At `rho=1`, the context group is not read directly, but residuals and other network paths remain active.
- Both reference modes are mutually exclusive with `--image_condition_paired_prefix_route_switch_step`, which disables the right reference route during denoising.
- Keep suffix weights, suffix fades, prompts and sampling settings identical within each method comparison. The supplied ablation plan retains the original suffix fade and treats changes as separate variants.
- No K/V source switch, explicit camera trajectory, object track, or hard endpoint constraint is added. Appearance/reference allocation alone cannot guarantee a camera orbit or prevent an object from restarting its motion.

## Example

For the supplied warrior-orbit and falling-object experiments, see the [reproducible commands and exact prompts](experiments/README.md).

After installing dependencies and obtaining the checkpoint, enter `Wan2.2` from the repository root. The root images `Img1.png` and `Img2.png` can be used directly; `fnr1.png` and `fnr2.png` provide another pair. Replace the checkpoint path and supply a prompt that describes the intended motion for your pair. Without an explicit prompt, the existing I2V example falls back to a white-cat beach description, which is unsuitable for these pairs. This is a Bash command; in PowerShell, put the Python command on one line.

```bash
cd Wan2.2
python generate.py \
  --task i2v-A14B \
  --size '832*480' \
  --frame_num 81 \
  --ckpt_dir /path/to/Wan2.2-I2V-A14B \
  --image ../Img1.png --image_latent_index 0 \
  --image ../Img2.png --image_latent_index 12 \
  --image_condition_paired_anchor_attention \
  --image_condition_paired_prefix_reference_strength 0.5 \
  --image_condition_paired_prefix_preserve_first_frame \
  --image_condition_paired_prefix_reference_curve linear \
  --base_seed 123 \
  --sample_steps 40 \
  --prompt "A continuous shot with natural motion and a smooth transition." \
  --save_file /path/outside/the/repository/paired-native-0.5-linear.mp4
```

`0.5` is an initial redistribution strength, not a validated optimum. For an original-route baseline, keep images, indices, prompts, seed, and other sampling settings fixed; omit strength, first-frame protection, and any reference fade/curve options. Keep suffix conditioning unchanged. The ablation runner defines separate comparisons for each intervention.

## Validation and experiments

CPU unit tests use small tensors and a substitute attention kernel to check routing, the formula, and boundary arguments. They do not load model weights or validate CUDA kernels or full video quality. Run from `Wan2.2`:

```bash
python -X utf8 -m unittest discover -s tests -p 'test_paired_anchor_*.py' -v
```

CPU tests cover dense attention oracles, native-mass conservation, exact zero-strength fallback, first-frame routing, CLI validation, a small real DiT through standard and simulated sequence-parallel paths, and the real sampling method with stubbed external models. They also check experiment isolation and queue resume validation. Distributed collectives are replaced with identity operations; this does not validate real multi-GPU or NCCL execution. Test pass counts belong to local run records.

See [tests/README.md](Wan2.2/tests/README.md) for details. CPU attention tests require PyTorch; a skip caused by missing PyTorch is not a successful validation.

Compare the following, using multiple fixed seeds for multiple image pairs:

| Comparison | Change | What to evaluate |
| --- | --- | --- |
| Original attention baseline | Omit the budget | Existing control and motion |
| Fixed budget | Vary `rho` with the curve fixed | Reference strength versus motion continuity |
| Native mass | Preserve total reference mass and vary strength | Quality and temporal allocation without a fixed context fraction |
| First-frame protection | Toggle only the first-frame exception | Opening fidelity versus intermediate behavior |
| Suffix conditioning | Change only the suffix prompt | Continuation versus repeated event descriptions |
| Handoff curve | Compare linear and smoothstep at fixed `rho` | Effect of temporal reference allocation |
| Future independent experiment; not implemented | Compare A/B with B/B after correct RoPE alignment | Budget effects versus reference representation effects |

In addition to recording reference coefficients, perturb Img1 and Img2 separately and inspect how each change propagates over video time. When swapping the endpoints for a time-reversed comparison, account for motion descriptions, canvas geometry, and noise correspondence: using the same seed alone does not ensure time-reversed noise. Inspect consistency near both references, intermediate motion, identity preservation, ghosting, freezing, and abrupt switches. Equal coefficients are an implementation property, not evidence of equal visual control.

## Limitations

- Reference features are evolving network hidden states. Group budgets are not causal or pixel-level contributions from the original images.
- Img1 conditioning, main-stream queries and residuals, anchor histories, and the causal VAE remain asymmetric. This change does not eliminate end-to-end differences between the input roles.
- Ordinary context can already contain information from both images. Grouping only avoids directly counting the reference positions twice.
- Img2 remains a temporary reference latent removed before decoding. No strict endpoint constraint has been added.
- Stronger or more balanced direct references may improve control or impair natural motion. Full-model video generation has not established a quality improvement; the experiments above are still required.

The [ablation runner and plan](experiments/README.md) keep generated videos, logs, measurements, and reports outside this repository. Only source code, method documentation, input prompts, and experiment definitions belong on GitHub.

Implementation: [attention.py](Wan2.2/wan/modules/attention.py), [model.py](Wan2.2/wan/modules/model.py), [image2video.py](Wan2.2/wan/image2video.py), and [generate.py](Wan2.2/generate.py).
