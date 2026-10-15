# Visual-Relay: Paired Reference Budget Experiment

This branch adds **time-dependent left/right reference budgets** to the existing `i2v-A14B` paired-anchor mechanism. For the segment between Img1 and Img2, the left reference coefficient decreases while the right reference coefficient increases, with a separate share reserved for ordinary video context.

Branch: `codex/paired-reference-budget`. Original baseline: [`2affd871`](https://github.com/GordonChen19/Visual-Relay/commit/2affd871f23b01026fb5118bed7b43f91e332ae7). The feature is disabled by default and uses the existing Wan2.2-I2V-A14B checkpoint without additional training weights. See the [Wan2.2 documentation](Wan2.2/README.md) for model downloads, dependencies, and hardware requirements.

This is an inference experiment. It constrains the direct reference coefficients; **it does not establish equal control over the generated video**.

## Method

The existing A stream covers the full video; the B stream covers the suffix starting at Img2. Let `a` be the latent index of the temporary Img2 anchor. For queries before that anchor, attention is computed and normalized separately over three groups:

| Group | Key source | Value source |
| --- | --- | --- |
| Left reference `L` | First position of A | First position of A |
| Right reference `R` | Temporary anchor `a` in A | First position of B |
| Video context `V` | Remaining ordinary positions in A | Remaining ordinary positions in A |

Video context includes ordinary positions on both sides of the anchor, excluding the two separately read reference positions to avoid directly counting them twice. Existing RoPE alignment for ordinary suffix positions is preserved. **The right reference remains `K_A(anchor) + V_B(0)`; it is not changed to B/B.**

After computing `O_L`, `O_R`, and `O_V`, the outputs are combined as follows:

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

Only prefix attention is replaced. The temporary anchor update, B stream, main-stream residuals, suffix blending, sampler, and decoder retain their existing mechanisms. The suffix parameter `image_condition_paired_stream_weight` still defaults to `0.25` and applies to a different region from the new `rho`.

The new path uses three grouped prefix attention calls instead of one, adding two prefix attention calls per invocation. It uses the existing attention kernel without materializing the full attention score matrix. GPU memory and runtime costs still need measurement.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--image_condition_paired_prefix_reference_budget` | Unset; feature disabled | Finite value in `[0,1]`, corresponding to `rho` |
| `--image_condition_paired_prefix_reference_curve` | `linear` | `linear` or `smoothstep` |

Usage notes:

- Enable `--image_condition_paired_anchor_attention` with exactly two images and latent indices `[0,a]`. Img2 must be internal, with a video segment remaining to its right.
- **Omit the budget argument to obtain the original attention baseline. `rho=0` uses only ordinary context with both anchors excluded; it does not disable the new mechanism.** At `rho=1`, the ordinary context group is not read directly, but residuals and other network paths remain active.
- The budget is mutually exclusive with `--image_condition_paired_prefix_route_switch_step`, which would otherwise disable the right reference route during denoising and invalidate the explicit budget interpretation.
- For initial comparisons, omit all `paired_*fade*` parameters, keep the suffix weight fixed, and disable prompt expansion.
- This branch does not add adaptive `rho`, a fallback coefficient `eta`, a denoising-step budget schedule, or a K/V source switch.

## Example

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
  --image_condition_paired_prefix_reference_budget 0.3 \
  --image_condition_paired_prefix_reference_curve linear \
  --base_seed 123 \
  --sample_steps 40 \
  --prompt "A continuous shot with natural motion and a smooth transition." \
  --save_file paired-budget-0.3-linear.mp4
```

`0.3` is an illustrative value, not a validated optimum. To produce a baseline, keep the images, order, indices, prompt, seed, and other sampling settings fixed; remove both `paired_prefix_reference_*` arguments and use a different output filename.

## English text and defaults

Documentation, comments, built-in prompts, negative prompts, and prompt-expansion instructions are written in English. Prompt expansion defaults to English in both the CLI and Python API; explicitly selecting the existing `zh` option still requests Chinese output for compatibility. Its instructions are also written in English.

The TTS example uses an English transcript and a matching synthesized English reference recording at `Wan2.2/examples/zero_shot_prompt.wav`. The reference says: "I hope that one day you will do even better than I have." It was synthesized with Microsoft Zira Desktop as 24 kHz, 16-bit mono PCM. The generated-speech example text is also English. This sample is a functional input example, not a benchmark of TTS quality.

**Translating the negative prompts changes model conditioning.** Results from the earlier Chinese-default revision may differ even when the attention budget is disabled. Run the original-attention baseline and budget experiments on the **same revision**, with identical positive and negative prompts, to isolate the attention change. Prompt-expansion results may also change after translation; keep expansion disabled for controlled comparisons.

## Validation and experiments

CPU unit tests use small tensors and a substitute attention kernel to check routing, the formula, and boundary arguments. They do not load model weights or validate CUDA kernels or full video quality. Run from `Wan2.2`:

```bash
python -X utf8 -m unittest discover -s tests -p 'test_paired_anchor_*.py' -v
```

Local CPU validation completed with **39 tests passing and no skips**. This includes actual forward passes through a small, randomly initialized DiT, checking that the standard path and a single-process simulation of sequence parallelism deliver the arguments to every layer. Distributed collectives are replaced with identity operations; this does not validate real multi-GPU or NCCL execution.

See [tests/README.md](Wan2.2/tests/README.md) for details. CPU attention tests require PyTorch; a skip caused by missing PyTorch is not a successful validation.

Compare the following, using multiple fixed seeds for multiple image pairs:

| Comparison | Change | What to evaluate |
| --- | --- | --- |
| Original attention baseline | Omit the budget | Existing control and motion |
| Fixed budget | Vary `rho` with the curve fixed | Reference strength versus motion continuity |
| Handoff curve | Compare linear and smoothstep at fixed `rho` | Effect of temporal reference allocation |
| Future independent experiment; not implemented | Compare A/B with B/B after correct RoPE alignment | Budget effects versus reference representation effects |

In addition to recording reference coefficients, perturb Img1 and Img2 separately and inspect how each change propagates over video time. When swapping the endpoints for a time-reversed comparison, account for motion descriptions, canvas geometry, and noise correspondence: using the same seed alone does not ensure time-reversed noise. Inspect consistency near both references, intermediate motion, identity preservation, ghosting, freezing, and abrupt switches. Equal coefficients are an implementation property, not evidence of equal visual control.

## Limitations

- Reference features are evolving network hidden states. Group budgets are not causal or pixel-level contributions from the original images.
- Img1 conditioning, main-stream queries and residuals, anchor histories, and the causal VAE remain asymmetric. This change does not eliminate end-to-end differences between the input roles.
- Ordinary context can already contain information from both images. Grouping only avoids directly counting the reference positions twice.
- Img2 remains a temporary reference latent removed before decoding. No strict endpoint constraint has been added.
- Stronger or more balanced direct references may improve control or impair natural motion. Full-model video generation has not established a quality improvement; the experiments above are still required.

Implementation: [attention.py](Wan2.2/wan/modules/attention.py), [model.py](Wan2.2/wan/modules/model.py), [image2video.py](Wan2.2/wan/image2video.py), and [generate.py](Wan2.2/generate.py).
