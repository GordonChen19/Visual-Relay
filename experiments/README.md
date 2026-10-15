# Reference budget experiments

These commands adapt the two supplied original-repository experiments to the grouped prefix reference budget. They use the same Wan2.2-I2V-A14B weights; no separate trained checkpoint is required.

## Changes from the supplied commands

- Add `--image_condition_paired_prefix_reference_budget 0.3` and `--image_condition_paired_prefix_reference_curve linear`.
- Remove `--image_condition_paired_prefix_route_switch_step 25`, which conflicts with the new budget. The right reference route remains enabled throughout denoising.
- Preserve the supplied prompts, image indices, seed, suffix weights, suffix fade steps, sampling shift, resolution, and frame count.
- Set `--sample_steps 40` explicitly, matching the I2V task default.
- Correct `--convert_model_dtyp` to `--convert_model_dtype` in the falling-objects command.
- Default to a single GPU. On a server with two GPUs, `GPU_COUNT=2` restores the supplied Ulysses and FSDP settings.

`rho=0.3` and the linear curve are initial experimental settings, not validated optima. The suffix fade is retained to match the supplied setup; it is independent of the prefix budget and can still indirectly affect the intermediate segment through ordinary context.

| Experiment | Images | Img2 latent index | Suffix weight |
| --- | --- | --- | --- |
| `warriors` | `Img1.png`, `Img2.png` | 12 | 0.5 |
| `falling` | `fnr1.png`, `fnr2.png` | 10 | 1.0 |

The exact English prompts are stored in [prompts/warriors.txt](prompts/warriors.txt) and [prompts/falling.txt](prompts/falling.txt).

## Run on one GPU

Activate the installed Python environment, enter the repository root, and set the checkpoint directory:

```bash
export CKPT_DIR=/home/gchen/Wan2.2-I2V-A14B
export CUDA_VISIBLE_DEVICES=0

bash experiments/run_reference_budget.sh warriors
bash experiments/run_reference_budget.sh falling
```

Run the commands sequentially. The script uses the repository images by default and saves timestamped videos under `outputs/`. Override `IMAGE_DIR` or `OUTPUT_DIR` if needed. A server-side managed queue may already be running these jobs; do not start duplicates in that case.

## Run on two GPUs

```bash
export CKPT_DIR=/home/gchen/Wan2.2-I2V-A14B
export CUDA_VISIBLE_DEVICES=0,1
export GPU_COUNT=2

bash experiments/run_reference_budget.sh warriors
bash experiments/run_reference_budget.sh falling
```

Two-GPU mode requires two visible GPUs. Single-GPU mode omits Ulysses and FSDP and uses `--convert_model_dtype`. Different hardware and execution paths need not produce bitwise identical outputs.

To explore the budget or curve, set `RHO` and `CURVE` explicitly, for example `RHO=0.4 CURVE=smoothstep bash experiments/run_reference_budget.sh warriors`.

## Interpretation and controlled comparisons

The original supplied commands and these new commands differ in both the reference budget and the prefix route switch. To isolate the budget's effect, also run a same-revision baseline with **both** the budget and the prefix route switch omitted; keep every other setting fixed. `rho=0` is context-only attention and is not the original-attention baseline.

Use the same positive and negative prompts for comparisons. This branch uses English negative prompts, so results from an older Chinese-default revision introduce another conditioning difference.

Img2 remains an internal temporary anchor removed before decoding. Neither the budget nor the prompt imposes a strict final-frame match to Img2. Equal direct reference coefficients do not establish equal visual control. These two runs are initial examples, not sufficient evidence of symmetry across image pairs and seeds.
