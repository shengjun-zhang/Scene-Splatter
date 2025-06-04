from modelscope.pipelines import pipeline	
from modelscope.outputs import OutputKeys
import cv2
import os

def video_enhance(VID_PATH, TEXT):
    pipe = pipeline(task="video-to-video", model='damo/Video-to-Video', model_revision='v1.1.0', device='cuda:2')	
    p_input = {	
                'video_path': VID_PATH,	
                'text': TEXT	
            }	
        
    output_video_path = pipe(p_input, output_video='new_diffusion.mp4')[OutputKeys.OUTPUT_VIDEO]
    return output_video_path

def video2frame(video_path, frame_path):
    video= cv2.VideoCapture(video_path)
    success, image = video.read()
    count = 0
    while success:
        cv2.imwrite(f"{frame_path}/{count}.jpg", image)  #可以是png或jpg
        success, image = video.read()
        count += 1
    print("over")    
    video.release()

def frames_to_video(frame_path, output_video_path, fps=10):
    # 获取帧目录中的所有PNG文件（按编号排序）
    images = [img for img in sorted(os.listdir(frame_path))[15:25] if img.endswith(".png")]

    if not images:
        print("No PNG images found in the specified directory.")
        return

    # 获取第一帧以确定视频的尺寸
    first_frame = cv2.imread(os.path.join(frame_path, images[0]))
    height, width, layers = first_frame.shape
    
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 格式
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 逐帧写入视频
    for image in images:
        img_path = os.path.join(frame_path, image)
        frame = cv2.imread(img_path)
        video.write(frame)
    
    video.release()
    print(f"Video saved successfully as '{output_video_path}'")


if __name__ == "__main__":
    # frame_path="../ljz/gaussian-splatting/news"

    # output_video_path = 'new.mp4'
    # frames_to_video(frame_path, output_video_path)              
    #上述用来将帧生成视频

    #======================================================


    video_path="./new.mp4"
    text=""
    enhanced_video = video_enhance(video_path,text)
    
    #上述用来做视频增强
    #======================================================

    # new_frame='render_boy_image'
    # video2frame('render0.mp4',new_frame)
    #把增强后的视频逐帧提取出来












