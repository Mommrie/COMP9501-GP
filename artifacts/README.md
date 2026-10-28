# Recorded experiment evidence

[中文使用指南](README.zh-CN.md)

`team-launcher-2026-10-28.json` identifies the embedded source snapshot and file hashes. Checkpoint distribution metadata lives in [assets](../assets/README.md); the payload remains outside Git.

`team-workflow-validation-2026-10-28.json` records local execution of the generated bootstrap, restore of all 12 adapters, 19 offline tests and export-manifest verification. It does not claim browser authorization or live Colab GPU execution was tested.

`chinese-guides-validation-2026-10-28.json` records the 17 section guides, bilingual notebook code equivalence, unchanged embedded runtime snapshot and documentation checks. It is not a new model experiment.

JSON summaries and generated figures cover the original bank, merge study, three-seed replication, local functional matching, coefficient capacity controls, feature routing, selection-label control and fixed transfer checks. [Reports](../README.md#recorded-part-2-findings) explain their metric and information boundaries.

`*-backup-*.json` records checksums for private local archives. Those ZIP payloads are not distributed in Git. Paths in backup metadata may be redacted for sharing. Numerical result values remain unchanged.

`source-provenance-2026-10-28.json` lists the archived input and shared output hashes. LF normalization and Markdown cleanup are sharing adaptations. Recorded training/evaluation code text matches the original after LF normalization. The runtime path strings in effective-runtime are historical Colab provenance, not required checkout paths.
