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
    # 扩大一点去除区域，确保完全覆盖水印
    x1 = max(0, x1 - 2)
    y1 = max(0, y1 - 2)
    x2 = min(frame_shape[1], x2 + 2)
    y2 = min(frame_shape[0], y2 + 2)
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
        cv2.imwrite('first_frame.png', first_frame)

        # 保存处理后的首帧
        processed_first_frame = remove_watermark_frame(first_frame, bbox)
        cv2.imwrite('first_frame_processed.png', processed_first_frame)

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


if __name__ == "__main__":
    input_video = "4cf55cb9b1a4313ebac796d83482f9ef.mp4"
    # 创建临时文件路径用于存储中间结果
    temp_output = "temp_no_watermark.mp4"
    final_output = "output_video.mp4"
    
    try:
        # 第一步：去除水印
        print("开始去除水印...")
        remove_video_watermark(input_video, temp_output)
        
        # 第二步：截取视频
        print("\n开始截取视频...")
        trim_video(temp_output, final_output, trim_seconds=3)
        
        print("\n所有处理完成!")
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_output):
            os.remove(temp_output)
            print("已清理临时文件")