# Part 2 follow-up evidence — 2026-11-09

CPU host only transmitted code/files; GPU work ran in Colab on A100 40GB.

- `benchmark-summary.json`: final run, including expert-group counts and 90 timed trials per shape/method.
- `benchmark-first-pass-summary.json`: earlier run retained for audit.
- `router-header-summary.json`: frozen source-router sensitivity on 512 explored-validation samples.
- `readiness.json`: seven existing test suites and pinned model commit verification.
- `restore-verification.json`: archive and per-file SHA verification.
- `*-source.py`, config and requirements: exact run sources.
- `latency.png`, `router-headers.png`: exportable figures.
- `manifest.json`: SHA256 of all other files.

Do not interpret timing repeats as training seeds, source recognition as best-expert selection, or synthetic input timing as task accuracy.
