import h5py
import cv2
import imageio
import numpy as np

# 替换为你的 HDF5 文件路径，例如 "source_demos.hdf5" 或 "pick_n_place_258.hdf5"
hdf5_path = "/mnt/afs/dongyifei/DreamFlyWheel/Gen2Act/demo_data/source_demos.hdf5"

def visualize_hdf5_video(file_path, demo_index=0, camera_type="table_cam"):
    # 1. 打开 HDF5 文件
    with h5py.File(file_path, "r") as f:
        # 获取所有的 demo 列表（例如 demo_0, demo_1, ...）
        demos = list(f["data"].keys())
        demo_key = demos[demo_index]
        print(f"正在读取: {demo_key}")
        
        # 2. 提取指定相机的视频数据
        # 路径结构为： data/demo_X/obs/table_cam
        frames_path = f"data/{demo_key}/obs/{camera_type}"
        if frames_path not in f:
            print(f"找不到路径: {frames_path}")
            return
            
        video_frames = f[frames_path][:] # 形状 (T, H, W, 3), RGB 格式
        
    print(f"成功提取 {video_frames.shape[0]} 帧画面，分辨率为 {video_frames.shape[2]}x{video_frames.shape[1]}")

    # ==========================================
    # 方法 A: 导出保存为 MP4 视频文件
    # ==========================================
    output_filename = f"{demo_key}_{camera_type}.mp4"
    print(f"正在保存视频至: {output_filename}")
    # Isaac Lab 通常是 60Hz 的仿真，但如果是观察数据可能做了降采样。这里以 30 FPS 保存
    imageio.mimwrite(output_filename, video_frames, fps=30)
    print("视频保存完成！")

if __name__ == "__main__":
    # 可视化第 0 个 demo 的顶视相机
    visualize_hdf5_video(hdf5_path, demo_index=0, camera_type="table_cam")
    
    # 如果想看机械臂腕部的相机，可以将 camera_type 改为 "wrist_cam"
    # visualize_hdf5_video(hdf5_path, demo_index=0, camera_type="wrist_cam")