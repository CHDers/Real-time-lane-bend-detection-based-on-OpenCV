import cv2
from tqdm import tqdm

# 输入视频文件路径
input_video_path = '8月14日.mp4'
# 输出视频文件路径
output_video_path = '8月14日_720p.mp4'

# 打开视频文件
cap = cv2.VideoCapture(input_video_path)

# 获取视频的原始宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 设置新的宽度和高度为720p
new_width = 1280
new_height = 720

# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码
out = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height))

# 获取视频总帧数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 使用 tqdm 显示进度条
with tqdm(total=total_frames, desc="Processing frames", unit="frame", colour="green") as pbar:
    # 逐帧读取并处理视频
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 将每一帧调整为720p大小
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # 写入调整大小后的帧
        out.write(resized_frame)

        # 更新进度条
        pbar.update(1)

# 释放视频捕获和写入对象
cap.release()
out.release()

print("视频转换完成！")

