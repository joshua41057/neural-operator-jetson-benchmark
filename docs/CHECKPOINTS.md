# Checkpoint release

Trained checkpoints are attached to the repository release, not tracked in git.
Every checkpoint here is deployed **as trained**: no retraining, no quantization-aware
fine-tuning, no architectural modification. Retraining will not reproduce the error
and dominance verdicts of Table 5.

| Asset | Family | Files | Size |
|---|---|---|---|
| `checkpoints-deeponet.tar` | deeponet | 16 | 591M |
| `checkpoints-fno.tar` | fno | 27 | 1.6G |
| `checkpoints-sp2gno.tar` | sp2gno | 14 | 17M |
| `checkpoints-wno.tar` | wno | 14 | 16M |

Heat exchanger checkpoints are **not** included: they are contributed by the companion
study and are available from <https://github.com/joshua41057/virso-jetson-inference>.

## Verify

```bash
sha256sum -c SHA256SUMS.txt
```

## Checksums

```
04f626076504c3a93eec1035314624ee51be57949b4e7eb047463bd0baf63eb3  checkpoints-deeponet.tar
b995fd8e80eeea18098a6cf745e185bc7611ee4c4d84a08856755464d1ec3c47  checkpoints-fno.tar
9012d83692c2696875da845480929b80ab0707d578dce1699b9861538b0c3b60  checkpoints-sp2gno.tar
20d0353e2ce10f07eb8e15e190c05bfebb24d16c3d6cfa56911325281a0a33a9  checkpoints-wno.tar
```
