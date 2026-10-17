
# Tests

## Paired-anchor CPU regression tests

From the `Wan2.2` directory:

```bash
python -X utf8 -m unittest discover -s tests -p 'test_paired_anchor_*.py' -v
```

These tests check CLI/API validation, original paired routing, fixed-budget attention, native-mass redistribution, first-frame protection, and experiment configuration/resume behavior. Attention tests use small tensors and a substituted kernel. They do not load model weights or import the full CUDA model stack. PyTorch is required; a skipped test is not a successful verification of the attention implementation.

The integration tests execute tiny, randomly initialized DiT blocks through standard and sequence-parallel entry points. They load actual definitions without optional checkpoint-loading mixins, substitute CPU attention, and model collectives as single-rank identity operations. Sampling tests execute the actual generation method with stubbed external models to check separate positive/shared negative prompts, independent schedules on both CFG passes, B0 prediction replacement, and removal of only the temporary latent before decoding. These verify wiring and numerical execution, not real multi-GPU communication or generated-video quality.

Before server generation, run the mandatory no-weight GPU check from the repository root:

```bash
python experiments/check_attention_cuda.py
```

This checks installed FlashAttention backends against a dense oracle, including LSE, zero-strength legacy equivalence, first-frame protection, and untouched suffix/padding. Missing CUDA/FlashAttention fails instead of skipping. The ablation runner invokes this check before execution. Keep generated validation JSON outside the repository.

The tests do not establish FlashAttention GPU compatibility, video quality, or equal causal influence of the input images. See the [experiment README](../../README.md) for the method, generation command, and visual validation plan.

## Full model smoke tests

Put all your models (Wan2.2-T2V-A14B, Wan2.2-I2V-A14B, Wan2.2-TI2V-5B) in a folder and specify the max GPU number you want to use.

```bash
bash ./tests/test.sh <local model dir> <gpu number>
```
