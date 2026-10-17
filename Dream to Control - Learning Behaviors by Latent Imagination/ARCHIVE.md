# Dream to Control: Learning Behaviors by Latent Imagination — 归档与恢复

本目录对应 **Dreamer v1（ICLR 2020）**。本次结果是 Walker Walk 单任务、单种子 20,000 环境步的短程训练与 10 回合独立评测，平均回报 **30.93 ± 8.83**；不代表论文最终性能。

## 下载完整复现包

- [完整归档（代码、结果、全部视频、权重、replay、原始证据包和 TensorFlow 安装包）](https://github.com/Katherine624/Paper_Reproduction/releases/download/dreamer-v1-reproduction-2026-10-17/dreamer-v1-complete-2026-10-17.tar.gz)
- [Release 页面](https://github.com/Katherine624/Paper_Reproduction/releases/tag/dreamer-v1-reproduction-2026-10-17)
- [逐文件 SHA256 清单](ARCHIVE_MANIFEST.json)
- [归档 SHA256](SHA256SUMS)
- [上传前重新核验结果](verification-before-upload.json)

仓库目录保存可直接阅读的源代码、补丁、依赖、配置、日志、报告、曲线及两段视频。较大的 checkpoint、replay、TensorFlow wheel 和历史证据包保存在 Release 完整归档中。所有原始复现文件逐字节保留；不包含可重建的 Git 对象缓存、SSH 客户端依赖缓存和 Python 字节码。

## 恢复

将完整归档与本目录的 `SHA256SUMS` 下载到同一位置。Linux/macOS 下执行：

```bash
sha256sum -c SHA256SUMS
mkdir dreamer-v1-restored
tar -xzf dreamer-v1-complete-2026-10-17.tar.gz -C dreamer-v1-restored
cd dreamer-v1-restored
```

Windows 可用 `Get-FileHash -Algorithm SHA256` 对照清单，并使用 `tar -xzf` 解压。归档解压后按照原始 [README](README.md) 配置 Linux/GPU 环境；`work/` 内保留本次使用的 TensorFlow wheel。该 wheel 为 Python 3.10、Linux x86_64、sm_120 的社区构建，不适用于本地 Windows 直接安装。

无需 TensorFlow 即可重新检查原始回报统计和 checkpoint：

```bash
python scripts/verify_results.py --training runs/walker_walk_seed0 --evaluation output/walker_walk_eval10 --output output/verification-restored.json
```

历史 README 提到的 `output/dreamer_v1_evidence.tar.gz` 包含在完整归档内部。原论文和作者 MIT 许可证随归档保留。
