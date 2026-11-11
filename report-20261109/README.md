# Part 2 report and presentation materials

The verified English [report PDF](report.pdf) contains 15 pages; [LaTeX source](report.tex) and generated tables are included. This is a research-material package, not a course submission receipt. Model work, plotting, compilation and rendering ran in Colab.

Start with [latest results and ten additional Q&A](ROBUSTNESS-RESULTS.zh-CN.md), then [slide-by-slide Chinese/English material](PRESENTATION-MATERIALS.zh-CN.md) and [implementation reading guide](READING-GUIDE.zh-CN.md). The [frozen follow-up protocol](ROBUSTNESS-PROTOCOL.zh-CN.md) explains commands, selection budgets, loss weights, routing gates and data filtering.

The follow-up reuses 12 teachers from three training seeds. It adds 12 coefficient fits, matched aligned/shuffled weight grids, confidence and benefit gates, and 2,580 filtered official-validation examples per seed. Native task means are CE 0.7050, selected aligned 0.7106, matched shuffled 0.7107 and individual 0.7618. This does not establish a unique activation benefit. Both gate families select always-own for all seeds; some Ridge candidates tie and lose by the conservative tie-break. Historical validation exploration prevents a pristine blind-test claim.

- [Latest immutable evidence and four figures](../artifacts/robustness-v2-2026-11-11/README.md): 52 checksummed files, saved predictions, selection locks and filtered sample IDs.
- [Original activation-stage results](ACTUAL-RESULTS.zh-CN.md), [mechanism Q&A](EXTENSION-QA.zh-CN.md), and [proposal/claim map](EXPERIMENT-AND-CLAIM-MAP.zh-CN.md) preserve earlier findings and limitations.
- [Original v3 results](../PART2-RESULTS-V3-2026-11-09.zh-CN.md) and [26 implementation Q&A](../PART2-IMPLEMENTATION-QA.zh-CN.md) explain the teacher training and merge implementation.
- [Follow-up summarizer](../code/summarize_robustness.py), [cloud finalizer](finalize_robustness_cloud.py), [code validation](code-validation.json), [PDF build verification](build-verification.json) and [visual review](visual-review.json) record the audit. All 28 cloud code tests passed. The four scientific manifests cover 192 files; run `python scripts/verify_part2_artifacts.py` from the repository root to check them and the final PDF.

For slides, select the key rows and figures rather than copying every backup table. Native task score, task-macro accuracy and sample-weighted switching rate are different metrics. The actual PPTX, team rehearsal and course submission remain team tasks.
