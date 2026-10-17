from pathlib import Path
import h5py

data_dir = Path(
    "/mnt/d/xiao/Documents/ChatGPT/cosmos/datasets/"
    "LIBERO-Cosmos-Policy/success_only/libero_spatial_regen"
)
file_path = next(data_dir.glob("*_demo.hdf5"))

with h5py.File(file_path, "r") as file:
    demo_name = next(iter(file["data"]))
    demo = file["data"][demo_name]

    print("任务文件：", file_path.name)
    print("示范编号：", demo_name)
    print("动作形状：", demo["actions"].shape)
    print("机器人状态形状：", demo["robot_states"].shape)
    print("相机数据：", list(demo["obs"].keys()))