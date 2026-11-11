# Colab entry point

[中文使用指南](README.zh-CN.md)

`team_colab.zh-CN.ipynb` has Chinese instructions and identical executable cells. `team_notebook.zh-CN.json` stores translations and the reviewed English-markdown digest; the builder rejects stale translations. See the Chinese guide for the update procedure.

`team_colab.ipynb` is the preferred group entry point: source snapshot embedded and SHA-checked, pinned dependency installation, offline checks, optional checkpoint restore and training, and a result export cell. No runtime GitHub token or MCP client is required. Each user still authorizes access to the private notebook and uses their own Colab runtime. [Group instructions](../COLLABORATION.md).

Rebuild after source/config changes: `python scripts/build_team_notebook.py`. Verify without writing: `python scripts/build_team_notebook.py --check`.

The final Part 2 cells optionally restore the v3 twelve-teacher ZIP or train a fresh independent study, then reproduce the five FP32 activation objectives and source/expert routing controls. All switches default to disabled. The extension needs at least 20 GiB CUDA VRAM; the recorded recovery used L4. Repeating the same three seeds does not create additional independent replications.

The embedded snapshot includes the Part 1 tensor implementation, its independent
requirements and presentation experiment plan. The Part 1 offline check uses
`test_csa*.py`; see the [CPU presentation runner](../part1-reproduction-20261025/presentation-20261028/README.md)
to generate figures. Figure assets and prose are viewed in GitHub, not embedded in the runtime snapshot.

`quickstart.ipynb` installs the pinned training stack, sets runtime environment flags and shows a small check/training entry point. Start with an authenticated private clone or a locally downloaded repository ZIP unpacked to `/content/COMP9501-GP`. No access token is embedded.

Direct CSA bridge / CSA直接迁移：[protocol and commands](../report-20261109/CSA-BRIDGE-PROTOCOL.zh-CN.md)、[results and 12 Q&A](../report-20261109/CSA-BRIDGE-RESULTS.zh-CN.md)。`code/csa_bridge.py`直接导入Part1 TensorCSA；`configs/csa-bridge-v1.json`固定网格；新源码/数学检查已纳入双语notebook。

Regularized discriminant follow-up / CSA判别强化：[protocol](../report-20261109/CSA-DISCRIMINANT-PROTOCOL.zh-CN.md)、[results and QA](../report-20261109/CSA-DISCRIMINANT-RESULTS.zh-CN.md)。新增`csa_discriminant.py`、五项测试、独立候选重载审计；所有表示相同网格，中英notebook已同步。
