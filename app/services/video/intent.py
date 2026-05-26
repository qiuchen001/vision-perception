import os
import json
import base64
from typing import List, Dict, Any, Union
from openai import OpenAI
from app.utils.logger import logger
from app.prompt.intent import system_instruction

class IntentService:
    """意图识别服务类"""

    def __init__(self):
        """初始化意图识别服务"""
        external_api_key = os.getenv("API_KEY")
        external_base_url = os.getenv("BASE_URL")
        if external_api_key and external_base_url:
            api_key = external_api_key
            base_url = external_base_url
            model_name = os.getenv("VISION_MODEL_NAME")
        else:
            api_key = os.getenv("SCENE_MINING_API_KEY") or "EMPTY"
            base_url = os.getenv("SCENE_MINING_API_BASE_URL")
            model_name = os.getenv("SCENE_MINING_API_MODEL_NAME")
        if not base_url or not model_name:
            raise ValueError("意图识别服务未配置 BASE_URL/VISION_MODEL_NAME 或本地 SCENE_MINING_API_BASE_URL/SCENE_MINING_API_MODEL_NAME")

        self.model_name = model_name
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def _call_api(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        调用API。

        Args:
            messages: 消息列表

        Returns:
            Dict[str, Any]: API响应
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            return response

        except Exception as e:
            logger.error(f"API调用异常: {str(e)}")
            return None

    @staticmethod
    def _normalize_intent_result(result: Any, log_prefix: str = "意图识别") -> List[Dict[str, Any]]:
        if isinstance(result, dict):
            if isinstance(result.get("intents"), list):
                result = result["intents"]
            elif isinstance(result.get("output"), list):
                result = result["output"]
            elif isinstance(result.get("output"), str):
                try:
                    result = json.loads(result["output"])
                except json.JSONDecodeError:
                    pass
            elif "type" in result and "list" in result:
                result = [result]
            elif isinstance(result.get("输出"), str):
                try:
                    result = json.loads(result["输出"])
                except json.JSONDecodeError:
                    pass

        if not isinstance(result, list):
            logger.error(f"{log_prefix}结果格式错误: {result}")
            return []

        for item in result:
            if not isinstance(item, dict) or 'type' not in item or 'list' not in item:
                logger.error(f"{log_prefix}结果项格式错误: {item}")
                return []

            if item['type'] not in ['tag', 'text']:
                logger.error(f"{log_prefix}结果类型错误: {item['type']}")
                return []

            if not isinstance(item['list'], list):
                logger.error(f"{log_prefix}结果列表格式错误: {item['list']}")
                return []

        return result

    def recognize_intent(self, text: str) -> List[Dict[str, Any]]:
        """
        识别用户输入的意图。

        Args:
            text: 用户输入的文本

        Returns:
            List[Dict[str, Any]]: 意图识别结果
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": system_instruction
                },
                {
                    "role": "user",
                    "content": text
                }
            ]

            response = self._call_api(messages)
            if not response:
                return []

            try:
                # 解析JSON响应
                result = json.loads(response.choices[0].message.content)
                
                return self._normalize_intent_result(result, "意图识别")

            except json.JSONDecodeError as e:
                logger.error(f"意图识别结果JSON解析失败: {str(e)}")
                return []

        except Exception as e:
            logger.error(f"意图识别失败: {str(e)}")
            return []

    def recognize_intent_with_image(self, text: str, image_path: str) -> List[Dict[str, Any]]:
        """
        基于文本和图像进行意图识别。

        Args:
            text: 用户输入的文本
            image_path: 图像文件路径

        Returns:
            List[Dict[str, Any]]: 意图识别结果
        """
        try:
            # 读取并编码图像
            with open(image_path, 'rb') as f:
                image_data = f.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')

            messages = [
                {
                    "role": "system",
                    "content": system_instruction
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ]

            response = self._call_api(messages)
            if not response:
                return []

            try:
                # 解析JSON响应
                result = json.loads(response.choices[0].message.content)
                
                return self._normalize_intent_result(result, "多模态意图识别")

            except json.JSONDecodeError as e:
                logger.error(f"多模态意图识别结果JSON解析失败: {str(e)}")
                return []

        except Exception as e:
            logger.error(f"多模态意图识别失败: {str(e)}")
            return []


# 使用示例
if __name__ == "__main__":
    # 设置环境变量
    os.environ["API_KEY"] = "your-api-key"
    os.environ["BASE_URL"] = "your-base-url"
    os.environ["VISION_MODEL_NAME"] = "your-model-name"
    
    intent_service = IntentService()

    # 测试文本意图识别
    text = "我要搜索带有急刹车的数据"
    result = intent_service.recognize_intent(text)
    print(f"文本意图识别结果: {result}")

    # # 测试多模态意图识别
    # text = "这个视频里有没有急刹车的情况？"
    # image_path = "path/to/your/image.jpg"
    # result = intent_service.recognize_intent_with_image(text, image_path)
    # print(f"多模态意图识别结果: {result}")
