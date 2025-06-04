import cv2
import os

def extract_frames(video_path, output_dir, frame_rate=1):
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    # 提取帧
    count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频读取完

        # 根据帧率决定是否保存当前帧
        if count % frame_rate == 0:
            # 保存帧
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            print(f"Saved: {frame_filename}")

        count += 1

    # 释放视频对象
    cap.release()
    print("Done extracting frames.")

# 设置参数
video_path = '/home/zsj/ljz/ViewCrafter-main/output/20241027_2059_DSC05587/diffusion0.mp4'  # 替换为你的输入视频文件路径
output_dir = 'frames'           # 输出帧的目录
frame_rate = 30                 # 每隔多少帧保存一次（例如每30帧保存1帧）

# 提取视频帧
extract_frames(video_path, output_dir, frame_rate)