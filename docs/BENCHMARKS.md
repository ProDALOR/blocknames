# Reference measurements

Measured on an RTX 5090 32 GB, Ryzen 9 9900X, Windows, release profile, NVIDIA driver
621.01, and NVRTC 13.0.88 on October 31, 2026. Measurements are not performance
guarantees for other systems or future versions.

| Recovery / address engine | Batch | Duration | Valid wallets/s |
| --- | ---: | ---: | ---: |
| BIP39-12 / CPU addresses | 65,536 | 20.03 s | 687,008 |
| BIP39-12 / CUDA addresses | 65,536 | 20.03 s | 1,321,869 |
| BIP39-12 / CUDA addresses | 131,072 | 30.07 s | 1,665,104 |
| TON24 / CUDA addresses | 4,194,304 | 15.16 s | 17,287 |

Rates measure complete candidate-to-address work, including rejected legacy TON
phrases, device transfers/clearing, pipeline fill, and producer shutdown. Startup
compilation/loading and mandatory cross-checks are reported separately. A public
checksum reduction forces all GPU address work to execute during benchmarks.

The 12-word scheme was about 96 times faster in this comparison. This changes the
standard mnemonic scheme as well as the implementation; it does not shorten either
scheme's required KDF. At the same 65,536 batch, moving addresses to CUDA improved
throughput by about 92%.

A 12-word `_ai` search checked 393,216 candidates in 0.248 s, plus 4.936 s setup and
checks (5.184 s measured total). Individual searches are random; an average-work
estimate is not a deadline. Batch rounding also affects short searches.

To measure your own machine:

```text
blocknames bench --mode bip39-12 --seconds 30 --report benchmark.json
blocknames bench --mode bip39-12 --address-engine cpu --seconds 30
blocknames bench --mode ton24 --seconds 30
blocknames bench --mode bip39-12 --backend cpu --seconds 10
```

Phase times overlap and should not be summed to estimate wall time. Workspace
bytes exclude driver/JIT allocations and public tables. Raw local benchmark logs
are deliberately excluded from the repository.
