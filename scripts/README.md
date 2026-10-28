# Repository utilities

- `build_team_notebook.py` builds/checks the deterministic embedded source launcher.
- `colab_bundle.py` validates source/checkpoint ZIPs, protects existing files and exports runs with metadata.
- `test_colab_bundle.py` checks idempotence, conflicting workspaces, corruption/path traversal and export metadata.

`check_repository.py` validates Python syntax, JSON and notebook structure, relative Markdown file links, accidental credentials and local absolute-path leakage. It rejects redistributed course attachments and model/data files outside ignored output directories. It imports no training library and writes no result files.

Run from the repository root: `python scripts/check_repository.py`.

The original import manifest stays frozen. Later legitimate edits appear in `changed_since_archive_import`; they are not rejected merely for changing an imported source. Use `--strict-import` when checking an unchanged import. Launcher freshness remains a required check for current runtime source/configuration.
