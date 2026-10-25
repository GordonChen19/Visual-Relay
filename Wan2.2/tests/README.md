
Put all your models (Wan2.2-T2V-A14B, Wan2.2-I2V-A14B, Wan2.2-TI2V-5B) in a folder and specify the max GPU number you want to use.

For the paired-anchor main experiment, opt into framewise future attention with
`--image_condition_paired_prefix_future_eta 0.2`. For each prefix query token and
head, the ordinary softmax future mass `w` becomes `w + eta*w*(1-w)`; probabilities
within each group stay proportional to their original scores. Own/earlier keys
come from C prefix frames; future keys include later C prefix frames, the
one-frame joint-conditioned H proxy, and ordinary C suffix frames after the
proxied frame. The DiT inputs are Img2-first B, Img1-only C, and H. H is a
second representation of C's first retained suffix frame and has the same
compressed temporal RoPE position. Prefix attention substitutes H for regular
C at that one frame, so the two copies never compete in the same softmax.
Suffix queries use regular C and exclude C's hidden transport slot. Eta must be
in [0, 1]; the default 0 uses ordinary softmax over this substituted timeline.
Two FlashAttention calls per prefix frame
return outputs and LSE values for the merge, so this can add runtime overhead.
H receives B1's attention update per layer, but its head prediction is always
discarded. The retained first-suffix velocity comes from regular C; B0 updates
and predicts only the hidden transport slot that is discarded before decode.

Run the CPU dense-reference attention tests with PyTorch installed:
`python3 -m unittest discover -s tests -p 'test_paired_anchor*.py'`.

```bash
bash ./tests/test.sh <local model dir> <gpu number>
```
