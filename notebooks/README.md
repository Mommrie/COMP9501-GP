# Colab entry point

[中文使用指南](README.zh-CN.md)

`team_colab.zh-CN.ipynb` has Chinese instructions and identical executable cells. `team_notebook.zh-CN.json` stores translations and the reviewed English-markdown digest; the builder rejects stale translations. See the Chinese guide for the update procedure.

`team_colab.ipynb` is the preferred group entry point: source snapshot embedded and SHA-checked, pinned dependency installation, offline checks, optional checkpoint restore and training, and a result export cell. No runtime GitHub token or MCP client is required. Each user still authorizes access to the private notebook and uses their own Colab runtime. [Group instructions](../COLLABORATION.md).

Rebuild after source/config changes: `python scripts/build_team_notebook.py`. Verify without writing: `python scripts/build_team_notebook.py --check`.

The embedded snapshot includes the Part 1 tensor implementation, its independent
requirements and presentation experiment plan. The Part 1 offline check uses
`test_csa*.py`; see the [CPU presentation runner](../part1-reproduction-20261025/presentation-20261028/README.md)
to generate figures. Figure assets and prose are viewed in GitHub, not embedded in the runtime snapshot.

`quickstart.ipynb` installs the pinned training stack, sets runtime environment flags and shows a small check/training entry point. Start with an authenticated private clone or a locally downloaded repository ZIP unpacked to `/content/COMP9501-GP`. No access token is embedded.
