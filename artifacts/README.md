# Experiment results

[11 November complete activation/routing evidence](activation-strengthening-2026-11-11/README.md): independently audited predictions, controls, figures and repeat comparison; [final report/materials](../report-20261109/README.md). [Verified partial recovery](recovered-partial-2026-11-09/README.md) preserves the incomplete older archive and explicitly lists missing records.

[Independent protocol v3, three training seeds](independent-v3-2026-11-09/README.md) contains audited results, teacher manifests, fixed choices, exact sources and three figures; [report](../PART2-RESULTS-V3-2026-11-09.zh-CN.md), [checkpoint restore verification](independent-v3-restore-2026-11-09.json). Evidence files remain covered by their original manifest.

[2026-11-09 A100 follow-up](part2-followup-2026-11-09/README.md) contains raw deployment timings, router header diagnostics, figures, exact execution sources and checksums. See the [report](../PART2-FOLLOWUP-2026-11-09.zh-CN.md) for scope and limitations; the diagnostic uses previously explored validation data.

This folder contains recorded result JSON, experiment plans and generated figures. See the root README for the baseline, coefficient, routing and transfer reports.

Runtime package versions help reproduce the original runs. On the team branch, team-launcher identifies the embedded Colab code snapshot. Checkpoint files are shared separately through the assets manifest.

For a new experiment, a separate output name plus its config, seed and useful context is usually enough to get started. Keep prior results so comparisons remain understandable. Personal tooling preferences and local migration/check logs live outside this repository.

Latest follow-up / 最新补强：[results and Q&A](../report-20261109/ROBUSTNESS-RESULTS.zh-CN.md)；三 seed、λ敏感性、匹配 shuffled、保守 gate、2,580条去重 validation。真实响应未超过 shuffled，gate 均回退原专家；不声称盲测或部署增益。
