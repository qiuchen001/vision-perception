from openai import OpenAI
import os
from dotenv import load_dotenv
from PIL import Image
import json
import numpy as np
import cv2
import tempfile
load_dotenv()
import base64
import math
from PIL import Image
import dashscope




def smart_resize(
        image_path, factor=28, vl_high_resolution_images=False):
    """
    对图像进行预处理。

    参数:
        image_path：图像的路径
        factor：图像转换为Token的最小单位
        vl_high_resolution_images：是否提高模型的单图Token上限

    """
    # 打开指定的PNG图片文件
    image = Image.open(image_path)

    # 获取图片的原始尺寸
    height = image.height
    width = image.width
    # 将高度调整为28的整数倍
    h_bar = round(height / factor) * factor
    # 将宽度调整为28的整数倍
    w_bar = round(width / factor) * factor

    # 图像的Token下限：4个Token
    min_pixels = 28 * 28 * 4

    # 根据vl_high_resolution_images参数确定图像的Token上限
    if not vl_high_resolution_images:
        max_pixels = 1280 * 28 * 28
    else:
        max_pixels = 16384 * 28 * 28

    # 对图像进行缩放处理，调整像素的总数在范围[min_pixels,max_pixels]内
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


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
    
    # 获取模型处理时的缩放尺寸
    h_bar, w_bar = smart_resize(image_path)
    
    # 计算缩放比例
    scale_x = orig_width / w_bar
    scale_y = orig_height / h_bar
    
    base64_image = encode_image(image_path)
    client = OpenAI(
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    # completion = client.chat.completions.create(
    #     model="qwen-vl-max",
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": [{"type":"text","text": "You are a helpful assistant."}]
    #         },
    #         {
    #             "role": "user",
    #             "content": [
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {"url": f"data:image/png;base64,{base64_image}"},
    #                 },
    #                 {"type": "text", "text": "输出关于360记录仪的检测框"},
    #             ],
    #         }
    #     ],
    #     response_format={"type": "json_object"},
    #
    # )

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

    # 将检测到的bbox坐标根据缩放比例还原到原始图片尺寸
    bbox_original = [
        int(bbox[0] * scale_x),
        int(bbox[1] * scale_y),
        int(bbox[2] * scale_x),
        int(bbox[3] * scale_y)
    ]
    
    # return bbox_original
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

def remove_image_watermark(input_path, output_path):
    """去除图片中的水印"""
    # 读取图片
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"无法读取图片: {input_path}")
    
    # 检测水印位置
    bbox = detect_watermark(input_path)
    # bbox = [30, 671, 108, 724]
    print(f"检测到水印位置: {bbox}")
    
    # 在原图上画出检测框并保存
    debug_image = image.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # BGR格式，(0, 255, 0)是绿色
    cv2.imwrite(output_path.replace('.png', '_bbox.png'), debug_image)
    print(f"已保存检测框图片到: {output_path.replace('.png', '_bbox.png')}")
    
    # 去除水印
    processed_image = remove_watermark_frame(image, bbox)
    
    # 保存处理后的图片
    cv2.imwrite(output_path, processed_image)
    print(f"已保存处理后的图片到: {output_path}")

if __name__ == "__main__":
    # 测试代码
    input_image = "qiyewx_20250327155329.png"  # 输入图片路径
    # input_image = "360_watermark.png"  # 输入图片路径
    output_image = "qiyewx_20250327155329_removed_01.png"  # 输出图片路径
    
    try:
        remove_image_watermark(input_image, output_image)
    except Exception as e:
        print(f"处理失败: {str(e)}")