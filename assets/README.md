# Shared checkpoint asset

[中文使用指南](README.zh-CN.md)

`checkpoints-2026-10-25.json` describes a separately distributed ZIP containing the original 12 task adapters, bank metadata and saved coefficient/selection histories. It excludes raw data, unrelated courses, runtime credentials and base model weights.

The ZIP stays outside Git. The owner can place it in a private group Drive folder or another agreed private asset store. Collaborators download it, upload it to Colab, or provide its path from their own authorized Drive mount. The team notebook validates the archive and every member SHA before restoring under outputs/. It will not replace a differing existing file.

The manifest is available in the embedded source snapshot. File storage permissions are separate from GitHub collaborator permissions. A cloud download location has not yet been configured.
