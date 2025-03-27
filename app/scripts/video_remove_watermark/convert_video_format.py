import os
import subprocess
from typing import List, Tuple
import cv2
from tqdm import tqdm

# 文件夹配置
INPUT_DIR = r"E:\workspace\ai-ground\removed-video-mark-clear"  # 输入视频文件夹
OUTPUT_DIR = r"E:\workspace\ai-ground\removed-video-mark-clear-format"  # 输出视频文件夹

def get_video_info(video_path: str) -> Tuple[int, int, float]:
    """
    获取视频的基本信息
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        Tuple[int, int, float]: (宽度, 高度, 帧率)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    return width, height, fps

def convert_video(input_path: str, output_path: str) -> bool:
    """
    转换单个视频为通用格式
    
    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        
    Returns:
        bool: 转换是否成功
    """
    try:
        # 获取视频信息
        width, height, fps = get_video_info(input_path)
        
        # 构建ffmpeg命令
        ffmpeg_cmd = [
            'ffmpeg', '-y',  # 覆盖已存在的文件
            '-i', input_path,  # 输入文件
            '-c:v', 'libx264',  # 使用H.264编码
            '-preset', 'medium',  # 编码速度预设
            '-crf', '23',  # 质量参数(0-51)，23为默认值
            '-vf', f'scale={width}:{height}',  # 保持原始分辨率
            '-r', str(fps),  # 保持原始帧率
            '-movflags', '+faststart',  # 支持流式播放
            '-c:a', 'aac',  # 音频使用AAC编码
            '-b:a', '128k',  # 音频比特率
            output_path  # 输出文件
        ]
        
        # 执行转换
        print(f"开始转换视频: {os.path.basename(input_path)}")
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # 等待处理完成
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            print(f"视频转换成功: {os.path.basename(output_path)}")
            return True
        else:
            print(f"视频转换失败: {stderr}")
            return False
            
    except Exception as e:
        print(f"处理视频时出错: {str(e)}")
        return False

def convert_directory(input_dir: str, output_dir: str) -> None:
    """
    批量转换目录中的视频
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
    """
    # 支持的视频格式
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有视频文件
    video_files = [f for f in os.listdir(input_dir) 
                  if os.path.isfile(os.path.join(input_dir, f)) 
                  and f.lower().endswith(video_extensions)]
    
    if not video_files:
        print(f"在目录 {input_dir} 中未找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 处理统计
    successful = 0
    failed = 0
    
    # 使用tqdm创建进度条
    with tqdm(total=len(video_files), desc="转换进度") as pbar:
        # 处理每个视频
        for video_file in video_files:
            input_path = os.path.join(input_dir, video_file)
            # 修改输出文件名，添加标记并确保扩展名为.mp4
            output_filename = f"converted_{os.path.splitext(video_file)[0]}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            if convert_video(input_path, output_path):
                successful += 1
            else:
                failed += 1
                
            pbar.update(1)
    
    # 打印处理统计
    print("\n转换完成!")
    print(f"总计: {len(video_files)} 个视频")
    print(f"成功: {successful} 个")
    print(f"失败: {failed} 个")

def check_ffmpeg() -> bool:
    """
    检查系统是否安装了ffmpeg
    
    Returns:
        bool: 是否安装了ffmpeg
    """
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

if __name__ == "__main__":
    import argparse
    
    # 检查ffmpeg是否已安装
    if not check_ffmpeg():
        print("错误: 未检测到ffmpeg，请先安装ffmpeg")
        print("Windows: 下载ffmpeg并添加到系统环境变量")
        print("Linux: sudo apt-get install ffmpeg")
        print("macOS: brew install ffmpeg")
        exit(1)
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='视频格式转换工具')
    parser.add_argument('--input', '-i', help='输入视频文件或目录的路径')
    parser.add_argument('--output', '-o', help='输出视频文件或目录的路径')
    
    args = parser.parse_args()
    
    if args.input:
        # 如果提供了命令行参数,使用命令行参数
        input_path = os.path.abspath(args.input)
        if os.path.isfile(input_path):
            # 处理单个文件
            output_path = args.output if args.output else f"converted_{os.path.basename(input_path)}"
            if not output_path.lower().endswith('.mp4'):
                output_path = os.path.splitext(output_path)[0] + '.mp4'
            convert_video(input_path, output_path)
        else:
            # 处理目录
            output_dir = args.output if args.output else OUTPUT_DIR
            convert_directory(input_path, output_dir)
    else:
        # 没有命令行参数时,使用配置的文件夹
        print(f"使用配置的输入文件夹: {INPUT_DIR}")
        print(f"输出文件夹: {OUTPUT_DIR}")
        
        # 检查输入文件夹是否存在
        if not os.path.exists(INPUT_DIR):
            print(f"错误: 输入文件夹 {INPUT_DIR} 不存在")
            os.makedirs(INPUT_DIR)
            print(f"已创建输入文件夹: {INPUT_DIR}")
            
        # 检查输出文件夹是否存在
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"已创建输出文件夹: {OUTPUT_DIR}")
            
        convert_directory(INPUT_DIR, OUTPUT_DIR) 