# Small project utilities

check_repository.py is an optional syntax/link/privacy check. It does not impose a contribution or documentation format.

build_team_notebook.py refreshes both language launchers after runtime code/config changes; --check checks freshness. colab_bundle.py restores checkpoint packages and exports results. test_colab_bundle.py checks those packaging operations.

verify_part2_artifacts.py checks the five final Part 2 evidence manifests (222 files) and the compiled/visually reviewed PDF with the standard library. Small derived prediction arrays are included in the evidence; model weights remain separate.

`test_team_notebook.py` guards against platform-specific ZIP creator metadata changing the embedded source SHA. The final launcher was checked on both Windows and Linux after this fix.
