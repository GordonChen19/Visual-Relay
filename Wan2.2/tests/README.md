
Put all your models (Wan2.2-T2V-A14B, Wan2.2-I2V-A14B, Wan2.2-TI2V-5B) in a folder and specify the max GPU number you want to use.

Run the CPU dense-reference joint-guidance attention tests with PyTorch
installed: `python3 -m unittest discover -s tests -p 'test_*.py'`.

```bash
bash ./tests/test.sh <local model dir> <gpu number>
```
