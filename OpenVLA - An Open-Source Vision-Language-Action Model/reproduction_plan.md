**OpenVLA 当晚复现范围**

2026年10月13日，用户授权开始执行。使用2024版OpenVLA官方模型与代码，云实例为单张L40S、Ubuntu22.04。无实体机械臂，评测采用LIBERO-Spatial。

1. 固定OpenVLA、LIBERO和dlimp提交，建立独立环境；模型和数据固定Hugging Face revision并验证SHA256。
2. 官方LIBERO-Spatial checkpoint完成第0号任务的3次预检。
3. 从官方OpenVLA-7B基础权重出发，使用公开libero_spatial_no_noops训练数据，LoRA rank32、batch size2、学习率5e-4训练100个optimizer steps。记录loss、梯度和参数更新；在新进程恢复第100步checkpoint，继续至第105步。
4. 使用未经本轮短训练改动的官方LIBERO-Spatial checkpoint评测全部10个任务，每任务10次，seed7、初始状态索引0至9，共100次。保留官方center crop、动作归一化、夹爪变换和220步任务上限。
5. 输出环境清单、代码修改、执行命令、机器可读训练/评测结果、成功与失败视频及简短报告。

此范围是训练管线验证与官方权重的缩减评测，不是完整预训练或论文全部实验复现。100步不承诺模型收敛；单种子100次结果不等于官方三种子的完整统计结果。若依赖或硬件兼容性需要改动，记录实际差异。

服务端项目目录：`/root/gpufree-data/openvla-repro`。全部模型、依赖缓存和数据放在数据盘。密码不写入文件。

来源：[官方仓库](https://github.com/openvla/openvla)、[基础权重](https://huggingface.co/openvla/openvla-7b)、[Spatial评测权重](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial)、[公开训练数据](https://huggingface.co/datasets/openvla/modified_libero_rlds)。精确revision和原文件校验值保存在同目录的openvla_source_manifest.json。
