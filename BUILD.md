# Build and reproduction

[中文版](BUILD.zh-CN.md)

For the browser-first team workflow, use [COLLABORATION.md](COLLABORATION.md) and the [embedded source launcher](notebooks/team_colab.ipynb). It initializes the runtime without a separate authenticated clone. Source changes require `python scripts/build_team_notebook.py` before committing the regenerated launcher.

Run commands from the repository root. Python 3.11 or newer is required for the training dependencies. Recorded GPU experiments used Python 3.13.16 and Colab's torch 2.11.0+cu130 on an A100 40 GB. Keep the runtime's torch/CUDA build. [Effective imported versions](artifacts/effective-runtime-2026-10-25.json) and [pinned dependencies](requirements.txt) document the run.

```bash
python -m pip install -r requirements.txt
```

Set `USE_TF=0`, `USE_FLAX=0` and `TOKENIZERS_PARALLELISM=false` before launching Python. The [Colab notebook](notebooks/quickstart.ipynb) sets these. Model and dataset downloads require network access. No private access token belongs in a notebook or committed file.

## Checks

```bash
python scripts/check_repository.py
python -m unittest discover -s code -p test_math.py
python -m unittest discover -s code -p test_protocol.py
python -m unittest discover -s code/functional_merge -p test_maths.py
python -m unittest discover -s code/coefficient_merge -p test_core.py
```

Tiny-model integration checks additionally need the pinned transformers/peft stack. They run with random miniature weights and do not replace real training/evaluation. The static repository check uses only the standard library and imports no model.

## Recreate the source adapter banks

All run directories must be new. Scripts refuse to overwrite an existing experiment.

```bash
python code/run_bank.py --config configs/protocol-v2-a100.json --seed 42 --output outputs/bank-v2-a100-s42
python code/run_bank.py --config configs/protocol-v2-a100.json --seed 43 --output outputs/replication-s43-s44/bank-s43
python code/run_bank.py --config configs/protocol-v2-a100.json --seed 44 --output outputs/replication-s43-s44/bank-s44
```

The bank runner trains all four task adapters, evaluates individual/average backbones, collects probe features and writes status/manifests. Training and calibration are separated within original GLUE train. Validation is a separate source split, but repeated research use makes the reported study exploratory.

## Follow-up pipeline

The following runners consume the recorded bank paths above. They resolve the checkout root from their source location, so the directory need not be named after the original Colab session.

```bash
python code/functional_merge/run_all.py
python code/coefficient_merge/run_all.py
python code/coefficient_merge/run_layer_ce_all.py
python code/router_v1/run.py --config configs/protocol-v2-a100.json --plan configs/router-v1.json --output outputs/router-v1
python code/selection_control/run.py
python code/transfer_check/run.py
```

`selection_control` needs coefficient history. `transfer_check` needs the CE control and selection-control outputs. The original source-baseline study also has an explicit CLI; use `python code/merge_study/run.py --help` and the frozen merge-study configuration. [Code](code/README.md) and [reports](README.md#recorded-part-2-findings) explain protocols and limitations.

Archived result JSON under `artifacts/` stays as evidence for the recorded run. New experiments belong under fresh `outputs/` paths. Do not replace archived results with a new run without changing the provenance and report date. Plot/aggregate helpers may expect the archived artifact filenames.

## Part 1

```bash
python -m unittest discover -s part1-reproduction-20261025/code -p test_csa.py
python part1-reproduction-20261025/code/run.py --data /path/to/mnist.npz --output part1-reproduction-20261025/outputs/mnist-v1
```

The script verifies the dataset checksum. This is a 2D grayscale MNIST illustration, not an exact reproduction of the original face/Gabor-MNIST tables. See [results and limits](part1-reproduction-20261025/RESULTS.md).
