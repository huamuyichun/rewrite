# Reference Environment

Environment fingerprint captured on 2026-07-16:

| Component | Version |
| --- | --- |
| Python | 3.12.13 (conda-forge) |
| PyTorch | 2.10.0+cu129 |
| CUDA runtime | 12.9 |
| NVIDIA driver | 575.57.08 |
| Primary GPU | NVIDIA A100 80GB PCIe |
| Conda prefix | `/pub/data/hjwz/miniconda3/envs/rewrite_miniexp` |

Experiments record the live fingerprint again in each session's
`environment.json`. The table above is documentation, not a substitute for
that manifest.

Run without installing the package:

```bash
PYTHONPATH=src /pub/data/hjwz/miniconda3/envs/rewrite_miniexp/bin/python \
  scripts/run_phase1_audit.py --help
```

Inductor caches and generated artifacts must remain under `/pub/data/hjwz`.
