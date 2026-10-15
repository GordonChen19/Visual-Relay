
# Tests

## Paired-anchor CPU regression tests

From the `Wan2.2` directory:

```bash
python -X utf8 -m unittest discover -s tests -p 'test_paired_anchor_*.py' -v
```

These tests check the CLI validation, original paired routing, and optional fixed-budget prefix attention using small tensors and a substituted attention kernel. They do not load model weights or import the full CUDA model stack. PyTorch is required for the attention tests; those tests are skipped if it is unavailable. A skipped test is not a successful verification of the attention implementation.

The integration tests execute tiny, randomly initialized DiT blocks through the standard and sequence-parallel entry points. They load the actual model definitions without optional checkpoint-loading mixins, substitute CPU attention, and model collectives as single-rank identity operations. This verifies parameter forwarding and numerical execution, not real multi-GPU communication. The local validation run completed 39 tests with no failures or skips.

The tests do not establish FlashAttention GPU compatibility, video quality, or equal causal influence of the input images. See the [experiment README](../../README.md) for the method, generation command, and visual validation plan.

## Full model smoke tests

Put all your models (Wan2.2-T2V-A14B, Wan2.2-I2V-A14B, Wan2.2-TI2V-5B) in a folder and specify the max GPU number you want to use.

```bash
bash ./tests/test.sh <local model dir> <gpu number>
```
