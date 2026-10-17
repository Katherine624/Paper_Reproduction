# 下载与恢复完整备份

这份备份保存了本机实际存在的文件，包含数据、官方模型、训练输出、视频、虚拟环境和缓存。论文目录供浏览；大文件在同一仓库的 Release 附件里。

## 1. 下载入口

[Cosmos Policy 完整备份](https://github.com/Katherine624/Paper_Reproduction/releases/tag/cosmos-policy-backup-20261017)

先下载 `backup-manifest.json`、`restore_backup.py`、`RESTORE.md`。`SHA256SUMS` 可供手动核对每个分卷。`source-inventory.jsonl` 记录原始项目文件、大小、时间和所属分组；硬链接会计入逻辑大小，实际磁盘占用可能更小。

## 2. 自动恢复（推荐）

在 **WSL/Linux** 使用 Python 3.10 或更新版本运行；只依赖 Python 标准库和 GNU tar。目标必须是一个新的空目录。完整恢复建议预留至少 75 GB 空间。脚本边下载边校验、解压，每次只暂存约 128 MiB，不需要额外保存整份压缩包。

```bash
python3 restore_backup.py \
  --manifest backup-manifest.json \
  --output /mnt/d/cosmos-restored
```

断线后用同样的命令追加 `--resume`。已经完整恢复的分组会跳过；被中断的分组会重新下载、解压。只要实验文件可加 `--groups project`；要官方推理通常还需要 `models environment linux-runtime`。

恢复后目录如下：

```text
cosmos-restored/
  project/      # 原始 cosmos 工作区，包含 scripts、upstream、数据和权重等
  linux-root/   # 原项目外的 /home/xiao/... 解释器、LIBERO 资源和配置
```

`linux-root` 只是另存的目录树，脚本不会自动覆盖当前 Linux 用户目录。六组归档分别是 `project`、`models`、`datasets`、`environment`、`caches`、`linux-runtime`；最后一组以 Linux 根目录为相对路径起点，其余以原项目目录为起点。

## 3. 原路径与换机器时的区别

原项目的 Windows 路径为 `D:\xiao\Documents\ChatGPT\cosmos`，WSL 路径为 `/mnt/d/xiao/Documents/ChatGPT/cosmos`。虚拟环境的激活脚本、editable 安装路径、Linux Python 符号链接及 LIBERO 配置含绝对路径。

- 在原机器、原用户下，将恢复出的 `project/` 内容放回上述原项目位置，保留原来的 `/home/xiao/.local/share/uv/python` 和 `/home/xiao/.cache/libero/assets`，最容易恢复运行。覆盖旧目录前先核对其中是否有新的实验。
- 如果这些项目外目录已经没有了，可从 `linux-root/home/xiao/` 恢复对应的 Python 3.10 目录、Python 别名符号链接、LIBERO 资源和 `.libero` 配置。不要覆盖无关版本的解释器或其他项目配置。
- 在不同用户、不同路径或不同 Linux 系统上，不能直接认为 `.venv` 可移植。应按源码 `SETUP.md`、`pyproject.toml`、`uv.lock` 重建环境，或修正激活脚本、入口脚本 shebang、`pyvenv.cfg`、`_editable_impl_cosmos_policy.pth` 和 LIBERO 配置里的绝对路径，再验证导入。Linux 二进制扩展也受驱动、CUDA 和系统库版本影响。
- 这不是整机镜像；Windows、WSL 发行版、NVIDIA 驱动和 apt 系统库没有打入归档。实际评测版本在 `project/eval-results/official_spatial_10x10_seed195/environment.json`，系统软件包记录另见 `system-packages.txt`（如提供）。
- 登录令牌和 SSH 私钥没有备份。日后需要在线下载时自行登录；本次评测脚本已使用本地缓存及离线模式。

## 4. 手动恢复

同一组的分卷必须齐全，按四位序号拼接。Linux/WSL 中可以直接流式解压，不必先合并成大文件：

```bash
sha256sum -c SHA256SUMS --ignore-missing
mkdir -p restored-project
cat project.tar.gz.part* | tar -xzf - -C restored-project
```

`--ignore-missing` 只核对已下载文件，不能证明分卷齐全。必须同时按 `backup-manifest.json` 核对该组的完整序号；不齐全就不要解压。`linux-runtime` 解压到独立目录，不要直接在 `/` 下解压。

## 5. 备份的边界

- 正式官方评测为 Spatial 10×10、seed 195、99/100 成功；视频、逐回合结果、日志、配置、源码快照均保留。
- 短程训练保存的 LoRA 导出 `lora_iter_000001900.pt` 在 `project` 分组中；官方模型在 `models` 分组中，二者用途不同。
- 本机云端结果目录本来就缺少完整模型与优化器分片。本次备份完整保存了现有文件，但不会补出当时未回传的分片；不能据此承诺精确恢复云端优化器状态。
- 模型许可证及来源声明包含在每组归档的 `backup-licenses/`。第三方包、模型和数据保留原许可证。
