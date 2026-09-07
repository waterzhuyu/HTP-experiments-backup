# 隔离实验建立记录

- 日期：2026-09-07；状态：spec-ready，未执行新实验。
- Remote：`3090s:/workspace/guozuyu/HTP-hierarchical-rqvae`
- Branch：`exp/hierarchical-rqvae-gps-grid-road`
- Fork：`da3b80e3d9dffe525c67a7722e99387cdb345347`
- Spec commit：`1b29f5949dcda54e1f2d1a93a53dc96f0d718959`
- Remote spec：`docs/plans/hierarchical_rqvae/spec.md`
- Remote config：`stage_1/configs/hierarchical_rqvae/plan.json`
- [Notion planned 页](https://app.notion.com/p/3d4c24a261be8142899becea7c0bb24f)

## 已执行命令

```bash
cd /workspace/guozuyu/HTP
git status --short
git worktree add -b exp/hierarchical-rqvae-gps-grid-road /workspace/guozuyu/HTP-hierarchical-rqvae da3b80e3d9dffe525c67a7722e99387cdb345347
cd /workspace/guozuyu/HTP-hierarchical-rqvae
uv sync --frozen
uv run python --version
uv run pytest -q
uv run python -m json.tool stage_1/configs/hierarchical_rqvae/plan.json
git diff --check
git add docs/plans/hierarchical_rqvae/spec.md stage_1/configs/hierarchical_rqvae/plan.json
git commit -m 'docs: specify hierarchical view RQ-VAE experiment'
git status --short
```

环境同步成功，Python 3.12.3。pytest：255 passed，23 warnings，22.87s；warnings 为继承 Transformer nested-tensor 提示。JSON 解析与 diff check 通过；提交后新 worktree clean。测试仅验证分叉基线，不证明新模型实现正确。pyproject.toml/uv.lock 未改动。

初次 SSH 受 sandbox 网络限制，以 require_escalated 重试成功；不是 VPN 断开，未操作 EasyConnect。远端无 rg，改用定向 grep/sed/find。uv 跨文件系统 hardlink 回退为 copy，安装成功。原工作树已有未跟踪数据/日志，本次 worktree 从已提交的集成 revision 创建。

实际实验 command/config解析结果/split hashes/model checkpoints/metrics：N/A（尚未实现与执行）。下一步为模块实现与数据/指标 preflight，然后 smoke。训练配置 execution_enabled=false。
