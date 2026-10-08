# 3090s 实验代码本地备份

备份时间：2026-10-08 13:32（北京时间）。来源：`3090s:/workspace/guozuyu/HTP` 及其全部已注册工作树。备份完成后总占用约 47 MB。

## 已保存内容

- `htp-all.bundle`：完整 Git 分支、标签、远程跟踪引用及所有活动工作树 HEAD 的提交历史；共 22 个引用。
- `repository.git/`：从 bundle 还原的本地镜像仓库，保留原引用。
- `worktrees/`：21 个工作树对应的已提交源码，目录名与服务器相同；每个目录为对应完整 revision 的 detached HEAD，可直接浏览。
- `manifest.json`：远端路径、分支、完整 revision、当前状态、未提交源码 SHA256 和排除文件清单。
- `uncommitted-source/HTP/`：主工作树中未提交的 `.gitignore`、`stage_1/src/preprocessing/data_audit.py`、`stage_1/tests/test_data_audit.py`。单独保存，未叠加到已提交源码目录。
- `states/`：每个工作树的状态、相对于 HEAD 的 patch 和 staged patch。
- `SHA256SUMS`、`capture_receipt.json`、`verification.json`：传输与恢复校验记录。
- `capture_remote.py`、`restore_verify.py`：本次捕获和恢复方法。

未跟踪的原始 GPS 数据、checkpoint、大型运行产物、日志、报告及被 Git 忽略的文件不在本次备份范围内；已被 Git 提交的文件随历史完整保留。未执行训练、环境同步或依赖安装，未修改服务器实验分支、提交或工作树。

## 验证结果

下载归档 SHA256 与远端一致；内部文件校验和全部通过；`git bundle verify` 与 `git fsck --full` 通过；22 个引用逐一匹配；21 个工作树 HEAD 均与远端清单一致，已提交源码 checkout 均 clean；3 个补充源码/配置文件 SHA256 全部匹配。具体记录见 `verification.json`。

## 使用与恢复

直接打开 `worktrees/HTP-independent-latent-sid/` 等目录即可查看对应版本。复现某次实验时，应先查该次运行清单里的完整 revision；工作树当前 HEAD 可能包含训练后的分析提交。

`htp-all.bundle` 是可独立复制和恢复的历史备份。在本备份目录运行以下命令可恢复到一个新的目录：

```bash
git clone -b main htp-all.bundle /目标路径/HTP-restored
cd /目标路径/HTP-restored
git switch exp/independent-latent-sid
```

恢复指定实验提交可使用 `git switch --detach <完整revision>`。无分支的 recovery 工作树 HEAD 已记录在 `manifest.json`，其对象可从 bundle 恢复。

当前 `repository.git/` 与 `worktrees/` 使用本地绝对路径关联。跨位置恢复优先使用 bundle 新建仓库；不要把单个 worktree 目录当作独立 Git 仓库复制。
