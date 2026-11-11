# Independent protocol v3 evidence — 2026-11-09

Fresh teachers on A100; seeds 42/43/44; one content-grouped partition of the original GLUE train split. This is internally isolated evaluation on previously studied datasets, not a pristine external blind test.

- summary.json, per-task.csv, per-seed.csv: audited aggregate and task metrics.
- main-comparison.png, paired-ce-kl.png, task-accuracy.png: exportable figures.
- partitions.json and locked-config.json: source revision, IDs, content hashes and configuration lock.
- bank-s* run/manifests: teacher checkpoints, data permissions and environment.
- coefficients-s* summaries/selection/split IDs: all checkpoint histories and locked choices.
- weight-baselines*: fixed prior grid, selection and final results.
- source__*: exact execution sources, checked against the initial study manifest. Additional baseline/export sources carry their own result/manifest hashes.
- manifest.json: SHA256 of every other evidence file.

Weights and optimizer state are not committed here. A separate checkpoint ZIP backs up adapter weights and runtime logs; shared base/tokenizer remain pinned Hub downloads.
