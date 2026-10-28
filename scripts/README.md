# Repository utilities

`check_repository.py` validates Python syntax, JSON and notebook structure, relative Markdown file links, accidental credentials and local absolute-path leakage. It rejects redistributed course attachments and model/data files outside ignored output directories. It imports no training library and writes no result files.

Run from the repository root: `python scripts/check_repository.py`.
