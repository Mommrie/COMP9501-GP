# Colab entry point

`team_colab.ipynb` is the preferred group entry point: source snapshot embedded and SHA-checked, pinned dependency installation, offline checks, optional checkpoint restore and training, and a result export cell. No runtime GitHub token or MCP client is required. Each user still authorizes access to the private notebook and uses their own Colab runtime. [Group instructions](../COLLABORATION.md).

Rebuild after source/config changes: `python scripts/build_team_notebook.py`. Verify without writing: `python scripts/build_team_notebook.py --check`.

`quickstart.ipynb` installs the pinned training stack, sets runtime environment flags and shows a small check/training entry point. Start with an authenticated private clone or a locally downloaded repository ZIP unpacked to `/content/COMP9501-GP`. No access token is embedded.
