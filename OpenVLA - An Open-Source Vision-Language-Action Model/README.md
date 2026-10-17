# OpenVLA: An Open-Source Vision-Language-Action Model

Kim 等，2024。实验于 **2026-10-13** 完成，2026-10-17 归档到 GitHub。

## 复现结果

| 项目 | 本次实际完成 |
| --- | --- |
| 推理评测 | 官方 `openvla/openvla-7b-finetuned-libero-spatial`，LIBERO Spatial 10 个任务 × 10 条轨迹，**84/100 成功（84%）** |
| 训练验证 | 官方 `openvla/openvla-7b` 基础模型 + 真实 RLDS 数据，LoRA 训练 100 个 optimizer steps |
| 断点恢复 | 新进程从第 100 步恢复 adapter、优化器及随机数状态，继续到第 105 步 |
| checkpoint | 第 50、100、105 步；每份含 adapter、优化器状态及加载配置 |
| 视频 | 全部 **103 条**：100 条正式评测 + 3 条预检 |
| 环境 | Ubuntu 22.04，单卡 NVIDIA L40S；PyTorch 2.2.0+cu121，Transformers 4.40.1 |

**84% 来自官方 Spatial checkpoint，不能归因于本次短训的 LoRA adapter。** 100 步训练仅验证训练流程，不代表模型收敛；本次 100 条轨迹也不等同于论文多随机种子的完整统计复现。恢复时 RLDS 迭代器重新开始，不保证精确延续原样本顺序。

详细指标、评测设置与限制见 [复现报告](artifacts/experiment_report/复现报告.md)；机器可读结果见 [verified_summary.json](artifacts/results/verified_summary.json)。

## 完整备份下载

**[打开本论文的 GitHub Release](https://github.com/Katherine624/Paper_Reproduction/releases/tag/openvla-2026-10-13-reproduction)**。

完整原始备份共 **2,162,839,133 字节**，由于单附件大小限制分成两卷，需同时下载并按顺序拼接：

1. [openvla_full_backup.tar.gz.part01](https://github.com/Katherine624/Paper_Reproduction/releases/download/openvla-2026-10-13-reproduction/openvla_full_backup.tar.gz.part01)
2. [openvla_full_backup.tar.gz.part02](https://github.com/Katherine624/Paper_Reproduction/releases/download/openvla-2026-10-13-reproduction/openvla_full_backup.tar.gz.part02)

两卷包含原始导出的全部 209 个文件及内置校验清单，覆盖三个 checkpoint、103 个视频、原始日志、结果、脚本、版本来源和实验报告。完整压缩包 SHA256：

```text
fe4a3105cb3d83fe9d67fc39697df3b617898861862bf6e6b4662a5a997546e5
```

分卷大小与 SHA256 见 [release_manifest.json](release_manifest.json) 和 [SHA256SUMS](SHA256SUMS)。

### 一条命令恢复

在本目录运行，使用 Python 3.10 或更新版本，无需第三方依赖：

```bash
python restore_backup.py --output /your/path/restored_backup
```

脚本依次下载两个分卷，检查每卷及完整压缩包 SHA256，安全解压后逐文件核对原始清单。需要约 5 GB 空余空间。也可只下载校验、不在本地保留文件：

```bash
python restore_backup.py --verify-only
```

恢复后的 `openvla_full_backup/results/train-smoke/checkpoint-105` 是最新训练 checkpoint。这些是 **LoRA adapter**，仍需基础模型 [openvla/openvla-7b，固定 revision](https://huggingface.co/openvla/openvla-7b/tree/47a0ec7fc4ec123775a391911046cf33cf9ed83f)。官方基础模型、官方 Spatial 模型与公开训练数据未重复上传；精确来源、revision 和校验值在 [source_manifest.json](artifacts/source_manifest.json)。

## 目录说明

- `artifacts/scripts/`：服务器实际执行的训练、推理补丁、检查和结果汇总脚本。
- `artifacts/results/`：原始 JSON/JSONL 指标、配置、环境信息与评测补丁；checkpoint 目录中的小型配置可直接查看，大型权重和优化器文件在 Release 完整备份中。
- `artifacts/logs/`：本次安装、训练、评测、下载等日志，包含排错时的失败尝试。
- `artifacts/openvla/rollouts/2026_10_13/`：全部 103 个原始视频，亦包含在 Release 备份中。
- `artifacts/experiment_report/`：报告、重跑命令与视频清单。
- `local_tools/`：本机编写的环境准备、下载、导出、回传及校验工具。
- `source_reference/`：当时检查及修改过的少量上游源码快照；原始版权归上游项目。
- `examples/`：成功与失败示例视频。
- `reproduction_plan.md`：当时采用的复现计划。

原始档案保留历史执行路径和交付时说明。报告中早期“仍在服务器”的描述属于首次交付状态；截至本次 GitHub 归档，三个 checkpoint 和全部视频均已备份。迁移运行时请按实际位置调整模型、数据、源码及输出路径。

## 重新执行

本项目是对官方 OpenVLA 的复现脚本与实验记录，不包含完整上游代码。先按固定提交准备 OpenVLA 和 LIBERO、下载 `source_manifest.json` 对应的模型与数据，再参考：

- [环境准备](artifacts/scripts/bootstrap.sh)
- [训练与评测实际参数](artifacts/scripts/run_pipeline.sh)
- [重跑命令](artifacts/experiment_report/REPRODUCE.txt)
- [继续训练示例](artifacts/回传说明.md)
- [评测代码补丁](artifacts/results/evaluation.patch)

上游版本：OpenVLA `c8f03f48af692657d3060c19588038c7220e9af9`；LIBERO `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`；dlimp `040105d256bd28866cc6620621a3d5f7b6b91b46`。

论文与项目：[OpenVLA 官方代码](https://github.com/openvla/openvla)、[论文](https://arxiv.org/abs/2406.09246)。本目录不改变上游代码或模型的许可条件。
