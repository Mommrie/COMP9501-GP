# COMP9501-GP

[中文使用指南](README.zh-CN.md)

Codebase for Group Project of COMP9501

Shared code and experiment notes for COMP9501 Group 13.

Part 1 studies **Concurrent Subspaces Analysis**. Part 2 studies **Activation-Aware Merging of LoRA Adapters for Multi-Task Learning**. Implementation and recorded experiments are available. The activation signal's unique benefit and robust transfer remain unestablished.

## Start here

- [Proposal-aligned course report and presentation package](report-20261109/README.md): 11-page compiled report, Chinese result explanations, slide-by-slide narration, Q&A, and complete audited L4 controls.
- [Completed three-seed v3 results and presentation materials](PART2-RESULTS-V3-2026-11-09.zh-CN.md): 12 fresh teachers, matched CE/KL, fixed weight baselines, audited metrics, figures and verified checkpoint restoration; [raw evidence](artifacts/independent-v3-2026-11-09/README.md).
- [Independent protocol v3](INDEPENDENT-PROTOCOL-V3.zh-CN.md): fresh teachers, five content-grouped partitions and matched-capacity CE/KL; [related-work boundaries](PART2-RELATED-WORK.zh-CN.md).
- [Part 2 implementation walkthrough and 26 Q&A (Chinese/English)](PART2-IMPLEMENTATION-QA.zh-CN.md).
- [2026-11-09 A100 follow-up report (Chinese)](PART2-FOLLOWUP-2026-11-09.zh-CN.md): measured deployment costs, frozen-router header diagnostic and presentation Q&A; [raw evidence and figures](artifacts/part2-followup-2026-11-09/README.md).
- [Proposal, progress, timeline and TODO](PROPOSAL.md): original Part 2 text, Chinese overview and evidence-backed status.
- [Team collaboration and Colab](COLLABORATION.md): embedded-source launcher, separate GPU accounts, checkpoint restore and run export.
- [Team Colab launcher](notebooks/team_colab.ipynb): no runtime GitHub token or git clone. Private-notebook access still needs your own repository authorization.

- [Build and reproduction](BUILD.md): environment, checks, pipeline order and output paths.
- [Part 2 code](code/README.md) and [frozen configurations](configs/README.md).
- [Part 1 implementation and results](part1-reproduction-20261025/README.md), [presentation outline](part1-reproduction-20261025/PRESENTATION-OUTLINE.md).
- [Part 1 code and presentation evidence pack](part1-reproduction-20261025/presentation-20261028/README.md): formula-to-code walkthrough, ten PNG/SVG figures, storage-budget experiments, original-paper evidence and Q&A.
- [Colab starter](notebooks/quickstart.ipynb): use an authenticated private checkout or an uploaded repository ZIP.
- [Contribution conventions](CONTRIBUTING.md) and [repository checks](scripts/README.md).

## Historical protocol-v2 findings

Qwen2.5-0.5B, q/v LoRA, four source tasks: SST-2, MRPC, RTE and CoLA. Three adapter seeds: 42/43/44. Every method uses the original, known task-specific classification head.

| Configuration | Four-task primary score, mean +/- sample SD |
|---|---:|
| Individual adapters | 0.7684 +/- 0.0027 |
| Uniform mixture, gain 2 | 0.6365 +/- 0.0365 |
| Layer coefficients, CE fit and label-based selection | 0.6997 +/- 0.0124 |
| Layer coefficients, teacher-KL fit and teacher-KL selection | 0.6996 +/- 0.0126 |

The score averages SST-2 accuracy, MRPC (accuracy+F1)/2, RTE accuracy and CoLA MCC on native scales. It is not the official full GLUE score. Source validation has been examined repeatedly during exploration. Teacher training and original checkpoint selection used labels, including the original calibration pool.

- [Three-seed baselines and pair diagnostics](REPLICATION-2026-10-25.md).
- [Static fitting, capacity controls and routing](PROTOTYPES-2026-10-25.md).
- [Selection-label control](SELECTION-CONTROL-2026-10-25.md).
- [Fixed IMDb/PAWS transfer checks](TRANSFER-CHECK-2026-10-25.md): no stable layer-KL advantage. PAWS results fall below the subset majority baseline, including individual adapters. IMDb is heavily truncated at the original 128-token limit.

## Repository contents

Source, configurations, derived result JSON and generated plots are included. Raw datasets, trained weights, runtime caches, local backups, course lecture files and original member documents stay outside Git. Download data through the original publishers and recreate weights using the build instructions, or arrange a separate authorized transfer.
