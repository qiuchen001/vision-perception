# 文件夹配置
INPUT_DIR = r"E:\workspace\ai-ground\videos-before"  # 输入视频文件夹
OUTPUT_DIR = r"E:\workspace\ai-ground\removed-video-mark"  # 输出视频文件夹
TRIM_SECONDS = 3  # 需要截取的结尾秒数

from openai import OpenAI
import os
from dotenv import load_dotenv
from PIL import Image
import json
import numpy as np
import cv2
import tempfile
from PIL import Image
import dashscope


load_dotenv()
import base64


def encode_image(image_path):
    """base64编码图片"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def detect_watermark(image_path):
    """使用Qwen-VL检测水印位置"""
    # 获取图片原始尺寸
    original_image = Image.open(image_path)
    orig_width = original_image.width
    orig_height = original_image.height


    response = dashscope.MultiModalConversation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model='qwen-vl-max-latest',
        messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "image": image_path,
                    },
                    {"type": "text", "text": "输出关于360记录仪的检测框"},
                ],
            }
        ],
        response_format={"type": "json_object"},
        vl_high_resolution_images=True
    )

    json_str = response.output.choices[0].message.content
    bbox = json.loads(json_str[0]["text"])[0]["bbox_2d"]

    return bbox

def create_mask(frame_shape, bbox):
    """创建水印区域的掩码"""
    x1, y1, x2, y2 = bbox
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    
    # 扩大去除区域，水平和垂直方向使用不同的扩展范围
    padding_x = 10  # 水平方向扩展10像素
    padding_y = 5   # 垂直方向扩展5像素
    
    # 确保不超出图片边界
    x1 = max(0, x1 - padding_x)
    y1 = max(0, y1 - padding_y)
    x2 = min(frame_shape[1], x2 + padding_x)
    y2 = min(frame_shape[0], y2 + padding_y)
    
    mask[y1:y2, x1:x2] = 255
    return mask


def remove_watermark_frame(frame, bbox):
    """去除单帧图像中的水印"""
    # 创建水印区域的掩码
    mask = create_mask(frame.shape, bbox)

    # 使用OpenCV的图像修复算法
    frame_repaired = cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return frame_repaired


def remove_video_watermark(video_path, output_path):
    """去除视频中的水印"""
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")

    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 读取第一帧并保存为临时图片
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("无法读取视频第一帧")

    # 创建临时文件保存首帧
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        cv2.imwrite(temp_file.name, first_frame)
        # 检测水印位置
        bbox = detect_watermark(temp_file.name)

        # 保存原始首帧
        first_frame_path = 'first_frame.png'
        cv2.imwrite(first_frame_path, first_frame)
        print(f"已保存原始首帧到: {first_frame_path}")

        # 在原图上画出检测框并保存
        debug_image = first_frame.copy()
        x1, y1, x2, y2 = bbox
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # BGR格式，(0, 255, 0)是绿色
        bbox_path = first_frame_path.replace('.png', '_bbox.png')
        cv2.imwrite(bbox_path, debug_image)
        print(f"已保存检测框图片到: {bbox_path}")

        # 保存处理后的首帧
        processed_first_frame = remove_watermark_frame(first_frame, bbox)
        processed_path = first_frame_path.replace('.png', '_processed.png')
        cv2.imwrite(processed_path, processed_first_frame)
        print(f"已保存处理后的首帧到: {processed_path}")

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 重置视频到开始
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 处理每一帧
    processed_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 去除水印
        processed_frame = remove_watermark_frame(frame, bbox)
        out.write(processed_frame)

        # 更新进度
        processed_frames += 1
        if processed_frames % 100 == 0:
            print(f"已处理 {processed_frames}/{total_frames} 帧 "
                  f"({processed_frames / total_frames * 100:.1f}%)")

    # 释放资源
    cap.release()
    out.release()
    print(f"视频去水印完成，已保存到: {output_path}")


def trim_video(video_path, output_path, trim_seconds):
    """截取视频，去掉末尾指定秒数"""
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")

    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算需要保留的帧数
    keep_frames = total_frames - (fps * trim_seconds)
    if keep_frames <= 0:
        raise ValueError(f"视频时长小于{trim_seconds}秒,无法处理")

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 处理每一帧
    processed_frames = 0
    while processed_frames < keep_frames:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        processed_frames += 1
        
        if processed_frames % 100 == 0:
            print(f"已处理 {processed_frames}/{keep_frames} 帧 "
                  f"({processed_frames / keep_frames * 100:.1f}%)")

    # 释放资源
    cap.release()
    out.release()
    print(f"视频截取完成，已保存到: {output_path}")
    print(f"已去除最后 {trim_seconds} 秒，处理了 {processed_frames} 帧")


def process_video(input_video, output_video, trim_seconds=3):
    """处理单个视频：去水印并截取"""
    try:
        # 创建临时文件路径
        temp_output = os.path.join(os.path.dirname(output_video), f"temp_{os.path.basename(output_video)}")
        
        try:
            # 第一步：去除水印
            print(f"\n开始处理视频: {os.path.basename(input_video)}")
            print("1. 去除水印...")
            remove_video_watermark(input_video, temp_output)
            
            # 第二步：截取视频
            print("2. 截取视频...")
            trim_video(temp_output, output_video, trim_seconds)
            
            return True
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_output):
                os.remove(temp_output)
                
    except Exception as e:
        print(f"处理视频 {input_video} 时出错: {str(e)}")
        return False


def process_directory(input_dir, output_dir, trim_seconds=3):
    """批量处理文件夹中的视频"""
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
    
    # 处理每个视频
    for i, video_file in enumerate(video_files, 1):
        input_path = os.path.join(input_dir, video_file)
        output_path = os.path.join(output_dir, f"processed_{video_file}")
        
        print(f"\n[{i}/{len(video_files)}] 正在处理: {video_file}")
        
        if process_video(input_path, output_path, trim_seconds):
            successful += 1
        else:
            failed += 1
    
    # 打印处理统计
    print("\n处理完成!")
    print(f"总计: {len(video_files)} 个视频")
    print(f"成功: {successful} 个")
    print(f"失败: {failed} 个")


if __name__ == "__main__":
    # remove_video_watermark("8b1de0729c40bedd7c28936f894b6625.mp4", "out.mp4")


    import argparse

    # 命令行参数解析
    parser = argparse.ArgumentParser(description='视频批量去水印和截取处理工具')
    parser.add_argument('--input', '-i', help='输入视频文件或目录的路径')
    parser.add_argument('--output', '-o', help='输出视频文件或目录的路径')
    parser.add_argument('--trim', '-t', type=int, default=TRIM_SECONDS, help=f'需要截取的结尾秒数（默认{TRIM_SECONDS}秒）')

    args = parser.parse_args()

    if args.input:
        # 如果提供了命令行参数,使用命令行参数
        input_path = os.path.abspath(args.input)
        if os.path.isfile(input_path):
            output_path = args.output if args.output else "processed_" + os.path.basename(input_path)
            process_video(input_path, output_path, args.trim)
        else:
            output_dir = args.output if args.output else OUTPUT_DIR
            process_directory(input_path, output_dir, args.trim)
    else:
        # 没有命令行参数时,使用配置的文件夹
        print(f"使用配置的输入文件夹: {INPUT_DIR}")
        print(f"输出文件夹: {OUTPUT_DIR}")
        print(f"截取结尾: {TRIM_SECONDS} 秒")

        # 检查输入文件夹是否存在
        if not os.path.exists(INPUT_DIR):
            print(f"错误: 输入文件夹 {INPUT_DIR} 不存在")
            os.makedirs(INPUT_DIR)
            print(f"已创建输入文件夹: {INPUT_DIR}")

        # 检查输出文件夹是否存在
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"已创建输出文件夹: {OUTPUT_DIR}")

        process_directory(INPUT_DIR, OUTPUT_DIR, TRIM_SECONDS)