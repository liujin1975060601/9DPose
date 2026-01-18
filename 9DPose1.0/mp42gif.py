import os
from moviepy.editor import VideoFileClip

def videos_to_gifs(
    video_dir,
    output_dir=None,
    fps=10,
    resize_width=480,
    max_duration=None
):
    """
    将文件夹下所有 mp4 / avi 视频转为 gif
    
    参数:
        video_dir: 视频文件夹路径
        output_dir: gif 输出路径（默认与视频同目录）
        fps: gif 帧率（网页推荐 8~12）
        resize_width: gif 宽度（保持比例），None 表示不缩放
        max_duration: 最大截取秒数（None 表示完整视频）
    """

    if output_dir is None:
        output_dir = video_dir
    os.makedirs(output_dir, exist_ok=True)

    video_exts = ('.mp4', '.avi')

    for file in os.listdir(video_dir):
        if file.lower().endswith(video_exts):
            video_path = os.path.join(video_dir, file)
            gif_name = os.path.splitext(file)[0] + '.gif'
            gif_path = os.path.join(output_dir, gif_name)

            print(f'Converting: {file} -> {gif_name}')

            clip = VideoFileClip(video_path)

            if max_duration is not None:
                clip = clip.subclip(0, min(max_duration, clip.duration))

            if resize_width is not None:
                clip = clip.resize(width=resize_width)

            clip.write_gif(
                gif_path,
                fps=fps,
                program='ffmpeg'
            )

            clip.close()

    print('✅ 所有视频转换完成')

if __name__ == '__main__':
    video_path = r'D:\Articles\+9DPose\github\9DPose\demos\videos'
    videos_to_gifs(
        video_dir=video_path,  # 替换为你的视频文件夹路径
        output_dir=video_path,   # 替换为你想要保存 gif 的文件夹路径
        fps=10,
        resize_width=512,
        max_duration=10
    )