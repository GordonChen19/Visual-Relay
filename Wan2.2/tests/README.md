
Put all your models (Wan2.2-T2V-A14B, Wan2.2-I2V-A14B, Wan2.2-TI2V-5B) in a folder and specify the max GPU number you want to use.

For the paired-anchor main experiment, opt into framewise future attention with
`--image_condition_paired_prefix_future_eta 0.2`. For each prefix query token and
head, the ordinary softmax future mass `w` becomes `w + eta*w*(1-w)`; probabilities
within each group stay proportional to their original scores. Own/earlier and future keys all come from ordinary C frames.
The temporary transport slot is always excluded, including at eta=0. There are
only two streams: Img1-only C and Img2-first B, with no joint VAE encoding.
The legacy prefix anchor route-switch has no effect. Eta must be in [0, 1];
the default 0 uses ordinary softmax over the retained C timeline. Two FlashAttention calls per prefix frame
return outputs and LSE values for the merge, so this can add runtime overhead.
The C/B suffix fusion and anchor update/velocity replacement are unchanged.

Run the CPU dense-reference attention tests with PyTorch installed:
`python3 -m unittest discover -s tests -p 'test_paired_anchor*.py'`.

```bash
bash ./tests/test.sh <local model dir> <gpu number>
```
