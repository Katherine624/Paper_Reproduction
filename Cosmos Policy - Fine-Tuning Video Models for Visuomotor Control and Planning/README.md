# Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning

Built on NVIDIA Cosmos

[论文](https://arxiv.org/abs/2601.16163) · [作者代码](https://github.com/NVlabs/cosmos-policy) · [完整备份附件](https://github.com/Katherine624/Paper_Reproduction/releases/tag/cosmos-policy-backup-20261017)

## 已完成的实验

| 部分 | 实际完成内容 |
| --- | --- |
| 云端训练 | RTX PRO 6000 96 GB，约 3 小时，2B 基座的 LoRA 短程微调；日志至 step 1960，导出 iter 1900 适配器 |
| 官方模型评测 | 作者完整 LIBERO 策略权重，LIBERO-Spatial 10 个任务 × 10 回合，seed=195；99/100 成功，99.0% |
| 本机推理 | RTX 5060 Laptop 8 GB；CPU 初始化并转 BF16 后放入 GPU，未量化；100 回合耗时 67.7 分钟 |
| 教学模型 | 单独的小模型训练和试跑，用于学习流程，不等同于 Cosmos 论文模型 |

**99.0% 是作者官方权重的成绩，没有加载我们训练的 LoRA。** 训练日志没有 `Done with training` 完成标记；这是预算内的短程训练。这里只评测 Spatial 的一个种子、每任务 10 回合，不是论文全部套件与多种子评测。

云端回传保留了 `lora_iter_000001900.pt`（560 个张量、45,875,200 个参数）以及配置、日志。`checkpoints/iter_000001900/` 缺少完整模型和优化器分片，不能用它声称可精确恢复全部训练状态。LoRA 导出文件完整保留在 Release 备份中。

## 目录说明

| 路径 | 内容 |
| --- | --- |
| `scripts/` | 教学训练、云端训练、官方模型评测启动和结果汇总脚本 |
| `upstream/cosmos-policy/` | 实际使用并修改过的作者源码，包含原许可证和依赖锁文件；浏览区不放虚拟环境 |
| `eval-results/official_spatial_10x10_seed195/` | 正式缩小评测：结果表、逐回合记录、日志、命令、环境、源码快照及视频 |
| `eval-results/official_spatial_pilot/` | 评测前的单回合试跑 |
| `eval-results/video-explanation/` | 用来解释视频的截图及慢放片段 |
| `cloud-results/spatial_rtxpro6000_01/` | 云端短程训练的配置和日志；大权重、原始分片在 Release 中 |
| `runs/` | 教学模型评测记录；教学权重在 Release 中 |
| `cloud/` | 当时准备云端的补丁及脚本包；其中 README 是历史方案，以本页实际执行记录为准 |
| `reference/` | 论文副本 |
| `backup-licenses/` | 模型许可证副本与来源声明 |

### 先看结果

- [结果说明](eval-results/official_spatial_10x10_seed195/RESULTS.txt)
- [每个任务的结果](eval-results/official_spatial_10x10_seed195/per_task_results.csv)
- [100 个回合记录](eval-results/official_spatial_10x10_seed195/episodes.csv)
- [成功视频](eval-results/official_spatial_10x10_seed195/examples/success.mp4) / [失败视频](eval-results/official_spatial_10x10_seed195/examples/failure.mp4)

正式评测的 `videos/` 中有 200 个视频：100 个普通回放与 100 个带预测画面对照的视频；同一个回合会有两份，并不是 200 个回合。task 5 为 9/10，其余任务均为 10/10。

## 完整数据、权重和环境备份

大文件分卷保存在本仓库的 Releases。**上传期间该 Release 为草稿；只有发布后，且带有 `backup-manifest.json` 和 `SHA256SUMS`，才表示整套备份完成。** 不能只下载某个分卷就解压。

备份覆盖本机现有项目文件，包括数据集、官方权重、LoRA/教学权重、Python 虚拟环境及下载缓存；也补充 Linux Python 3.10 解释器、LIBERO 仿真资源和配置。登录令牌、SSH 密钥和项目 `.git` 不在备份中。

| 分卷前缀 | 内容 |
| --- | --- |
| `project.tar.gz.part` | 原始源码、所有实验输出、视频、日志、教学/LoRA 权重、论文、历史脚本包 |
| `models.tar.gz.part` | 两个官方模型目录和 Hugging Face 缓存；保留硬链接关系 |
| `datasets.tar.gz.part` | 本机 LIBERO-Cosmos-Policy 数据集及其下载元数据 |
| `environment.tar.gz.part` | 原始 Linux `.venv` 虚拟环境 |
| `caches.tar.gz.part` | uv、Xet 缓存及其他项目环境配置 |
| `linux-runtime.tar.gz.part` | 项目外的 Python 3.10 解释器、LIBERO 资源和配置 |

恢复步骤及各分卷的字节数、SHA-256 校验值见发布时补充的 `RESTORE.md`、`backup-manifest.json`、`SHA256SUMS`。这是文件和软件包备份，不包含 Windows/WSL 系统镜像、显卡驱动或系统 apt 依赖；迁移到其他路径或机器时，需要按恢复说明调整绝对路径。

## 再次运行评测

恢复环境、数据和权重后，在 WSL/Linux 的项目根目录执行：

```bash
bash scripts/eval_official_libero.sh libero_spatial 10 -1 NEW_RUN_NAME
```

`10` 是每个任务 10 回合，`-1` 是运行该套件全部任务。先试一个回合可用 `libero_spatial 1 0 NEW_PILOT_NAME`。使用新运行名，以免覆盖历史结果。详细参数与环境见 `PREPARATION.md` 和结果目录里的 `command.txt`、`environment.json`。

## 来源与版本

- 作者源码基线：`18a2accadf4e7a3531e56754102af5a24d2316da`，本目录保留实际本地修改。
- LIBERO-Cosmos-Policy 数据：`b463cbd5d114909e4daea65eb03bed9787290042`。
- 官方 LIBERO 策略权重：`cb689ec0e3347c13667d70a78a3447388f5c3bb8`。
- Cosmos-Predict2 基座和 tokenizer：`f50c09f5d8ab133a90cac3f4886a6471e9ba3f18`。

原作者代码按其 Apache-2.0 许可证使用；模型按 NVIDIA Open Model License 使用。第三方数据、模型、包和资源保留其原许可证，本仓库不对它们重新授权。
