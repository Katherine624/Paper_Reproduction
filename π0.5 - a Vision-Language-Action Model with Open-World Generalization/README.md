# π0.5: a Vision-Language-Action Model with Open-World Generalization

本目录保存 **2026-10-13** 完成的简化复现实验，于 **2026-10-17** 归档。
范围是“真实示范数据短训练验证 + 官方权重的小规模仿真评测”，不代表原论文完整复现。

| 项目 | 本次结果 |
| --- | --- |
| 训练 | 从官方 `pi05_base` 开始，8 条示范、2197 帧，LoRA 配置，batch 2，100 次更新 |
| 保存与恢复 | 参数抽样更新检查通过；新进程恢复步数为 100，23 组参数样本一致 |
| 评测权重 | 官方 `pi05_libero`，不是上述短训练产物 |
| 评测范围 | LIBERO Spatial，10 个任务 × 每任务 10 次，seed 7 |
| 评测结果 | 97/100 成功，100 段正式评测回放齐全 |

## 从哪里开始看

1. [结果与操作记录](output/π0.5复现结果.txt)
2. [逐任务评测结果](output/评测结果.csv) · [100 步训练指标](output/训练日志.csv)
3. [正式评测的全部视频](output/reproduction-evidence/results/eval-libero-spatial-seed7-trials10/videos)
4. [完整文件导览](00_先看这里_文件导览.txt)
5. [服务器实际运行的脚本](output/reproduction-evidence/scripts) · [本地辅助代码](src)

`output/reproduction-evidence/` 是最终证据；`training-evidence` 和 `environment-evidence` 是较早阶段的备份。
历史报告中的“未训练”“实例未关闭”等内容仅代表当时状态，不是实时状态。
原始文件保持原样，原有相对路径可继续使用。

## 下载完整备份

[打开本论文的 Release：pi05-reproduction-2026-10-17](https://github.com/Katherine624/Paper_Reproduction/releases/tag/pi05-reproduction-2026-10-17)

其中 `pi05-reproduction-2026-10-17.tar.gz` 包含本目录的结果、代码、配置、日志、视频，
以及实际使用的 **8 条训练示范 Parquet 数据**。根目录是论文标题文件夹。
数据的原始版本、选择编号和 SHA256 在 `data/libero-subset/subset-manifest.json` 中。
训练数据的大型二进制文件只放在 Release 中；Git 文件树保留元信息和下载脚本。

归档完整性清单：[archive-manifest.json](backup/archive-manifest.json)。
原始本地文件清单：[original-files.json](backup/original-files.json)。
解压后可用 `python backup/verify_archive.py` 检查归档内每个文件的 SHA256。

## 代码、数据和权重来源

- [论文](https://www.pi.website/download/pi05.pdf)
- [官方 OpenPI](https://github.com/Physical-Intelligence/openpi)，版本 `215abfb217dbac7d5f1273282331b9b1866c0479`
- LIBERO 子模块版本 `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`
- [官方 LIBERO 数据](https://huggingface.co/datasets/physical-intelligence/libero/tree/a4336d589d589045d1c56423ffdf3b88a0e19b1f)，只选 episode 0–7
- 基础权重：`gs://openpi-assets/checkpoints/pi05_base`
- 评测权重：`gs://openpi-assets/checkpoints/pi05_libero`

官方大型权重、未完成下载分块和本地辅助库缓存不重复上传。
其清单保存在 `work/remote-setup/`，下载工具在 `src/` 和最终证据的 `scripts/` 中。
`src/download_checkpoints.py` 是早期下载尝试，只支持 MD5；最终服务器流程使用
`output/reproduction-evidence/scripts/download-server-ranges.py`，支持 MD5 / CRC32C。
第三方代码、模型和数据继续遵循各自上游许可，本归档不改变其许可。

## 训练权重的保存位置与复现边界

本次生成的完整训练 checkpoint 仅在原实验服务器上：
`/root/gpufree-data/pi05-repro/checkpoints/training/pi05_libero_lora_smoke/real8_steps100_seed42/99`

该大型 checkpoint 原本未下载到本地，因此本 GitHub 归档也不包含它。
保存、重载和参数检查的证据均已归档。`99` 是从 0 开始计数的第 100 次更新。
100 步实验不用于声称模型收敛，97% 只对应官方权重的本次 Spatial 评测。

训练、推理脚本依赖 Linux GPU 服务器上的官方 OpenPI / LIBERO 环境。
Windows 克隆时建议使用 `git -c core.longpaths=true clone ...`，以支持完整的视频文件名。
重建环境可参考 `output/environment-evidence-2026-10-13/scripts/` 和保存的依赖列表。
重跑训练或评测应使用新的实验与结果目录，避免覆盖旧结果。
