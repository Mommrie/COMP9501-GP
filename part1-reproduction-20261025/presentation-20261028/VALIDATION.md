# Validation record — 2026-10-28

入口：[材料包](README.md) · [结果 JSON](figures/results.json)

This records executed checks, not planned work or a claim of original-table reproduction.

## Scientific and implementation checks

- **9 mathematical/API tests passed**: the original two tests plus explicit-index
  mode products/unfolded covariance, 2D parity with the frozen implementation,
  1D PCA equivalence, 3D held-out exact reconstruction/Kronecker equivalence,
  random-tensor energy/PCA bound, full-rank/constant input, and invalid/unfitted inputs.
- **21 grid configurations passed**: 3 training-subset seeds x 7 ranks. All
  converged within 100 sweeps, with orthonormal factors, nondecreasing coordinate
  energy, the reconstruction-energy identity and the matched-latent PCA training bound.
- **63 CSV rows** agree with the JSON configuration count; mean/sample-SD
  aggregates were recomputed independently from the per-subset values.
- **6 archival overlaps passed**: sample IDs identical, largest accuracy/MSE
  discrepancy below 1.4e-17 against 2026-10-25. Old source, plan and raw results remain unchanged.
- All integer budget dimensions satisfy `PCA(d) <= CSA budget < PCA(d+1)`,
  including the explicitly described d=0 mean-only case.
- Nine initialization fits and the independent noiseless 3D held-out reconstruction
  ran successfully; outcomes and limitations are in [EXPERIMENTS](EXPERIMENTS.zh-CN.md).

## Reproducibility and integration

- Data SHA, plan SHA, Python-source SHAs and generated-output SHAs checked.
  Generated text uses LF so Git's line-ending normalization preserves these hashes.
  CSV includes all methods and seeds; SVG timestamp metadata is removed and its
  generated IDs use a fixed salt.
- The CPU runner was executed from a fresh output directory; a new figure/table
  generation run completed after the final source changes. No GPU or live Colab was used.
- Both team launchers rebuilt together; executable cells remain synchronized.
  The new Part 1 code, independent requirements and plan are present in the
  SHA-verified embedded source. The 9 Part 1 tests also passed from that extracted copy.
- The **4 packaging tests passed** for source materialization, traversal/corruption
  rejection, checkpoint validation and result export. The current notebook source
  snapshot check passed.
- Repository syntax/JSON/notebook/link/privacy checks and `git diff --check`
  completed before commit. Original course PDFs, licensed paper screenshots,
  dataset archive, model arrays, caches and private notes are excluded from the commit.

## Visual and source review

- All 10 PNG figures visually inspected for titles, axes, legends and labels;
  detailed reconstruction, budget and original-paper panels inspected at useful resolution.
  Each also has an SVG counterpart. Complete source numbers remain available in JSON/CSV.
- Original Procedure 1, compression formula and Table IV visually checked on
  PDF pages 4, 7 and 9. The older assertion of a reversed stopping inequality was
  corrected in the presentation outline. Original Table IV values are redrawn
  with attribution and kept separate from our measurements.
- Reused test-set exploration, shared test IDs, sample-SD interpretation,
  different comparison budgets and non-reproduction of Gabor/face/LDA results
  are disclosed in the figure guide and experiment report.

## Commands for another member

```bash
python -m pip install -r part1-reproduction-20261025/requirements.txt
python -m unittest discover -s part1-reproduction-20261025/code -p "test_csa*.py" -v
python part1-reproduction-20261025/code/prepare_presentation.py --download --output part1-reproduction-20261025/outputs/new-check
python scripts/build_team_notebook.py --check
python scripts/check_repository.py
```

Use a previously nonexistent output directory. Data downloading needs network
access; local rendering/linear algebra require CPU only. No new external GPU,
new teammate account or live Colab integration has been tested this round.
