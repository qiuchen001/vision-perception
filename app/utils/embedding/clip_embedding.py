import os
from typing import Optional, List, Tuple, Dict
import torch
# import cn_clip.clip as clip
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from PIL import Image
from config.config import Config
from app.utils.embedding.embedding_base import EmbeddingBase


# # 从视频中提取帧，并跳过指定数量的帧。
# def extract_frames(video_path, N):
#     video_frames = []
#     capture = cv2.VideoCapture(video_path)
#     fps = capture.get(cv2.CAP_PROP_FPS)
#     current_frame = 0
#
#     while capture.isOpened():
#         ret, frame = capture.read()
#         if ret:
#             video_frames.append(Image.fromarray(frame[:, :, ::-1]))
#         else:
#             break
#         current_frame += N
#         capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
#
#     capture.release()
#     return video_frames, fps
#
# def embedding_video(video_path, N):
#     print(f"Processing video: {video_path}")
#
#     idxs, embeddings, paths, at_seconds = [], [], [], []
#
#     total_count = 0
#
#     try:
#         video_frames, fps = extract_frames(video_path, N)
#         for frame_idx, frame in enumerate(video_frames):
#             frame_embedding = clip_embeding.embeding_image(frame)
#
#             idxs.append(total_count)
#             embeddings.append(frame_embedding[0].detach().cpu().numpy().tolist())
#             paths.append(video_path)
#             # Calculate the timestamp in seconds for each frame
#             timestamp = int((frame_idx * N) / fps)
#             at_seconds.append(np.int32(timestamp))
#             total_count += 1
#
#             if total_count % 50 == 0:
#                 data = [idxs, embeddings, paths, at_seconds]
#                 print(f'Successfully inserted {operator.coll_name} items: {len(idxs)}')
#                 idxs, embeddings, paths, at_seconds = [], [], [], []
#
#     except Exception as e:
#         print(f"Error processing video {video_path}: {e}")


class ClipEmbedding(EmbeddingBase):
    """CN-CLIP模型实现"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.processor = clip.load_from_name(
            name=Config.CN_CLIP_MODEL_PATH,
            device=self.device,
            vision_model_name="ViT-L-14-336",
            text_model_name="RoBERTa-wwm-ext-base-chinese",
            input_resolution=336)
        self.model.eval()
        self.tokenizer = clip.tokenize

    def embedding_image(self, image: Image.Image) -> List[float]:
        """生成图片的embedding向量"""
        process_image = self.processor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(process_image)
            # 添加L2归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features[0].detach().cpu().numpy().tolist()

    def embedding_text(self, text: str) -> List[float]:
        """生成文本的embedding向量"""
        text = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            # 添加L2归一化
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features[0].detach().cpu().numpy().tolist()

    def embedding(self, image: Image.Image, text: str) -> Tuple[List[float], List[float]]:
        img_emb = self.embedding_image(image)
        txt_emb = self.embedding_text(text)
        return img_emb, txt_emb

    def probs(self, image: Image):
        process_image = self.processor(image).unsqueeze(0).to(self.device)
        text = self.tokenizer(["a diagram", "a dog", "a cat"]).to(self.device)

        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(process_image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        print("Label probs:", probs)

    def match(self, image: Image, texts: List[str]) -> Dict[str, float]:
        """计算图像和文本的匹配相似度

        Args:
            image: PIL Image对象
            texts: 文本描述列表

        Returns:
            Dict[str, float]: 文本及其对应的相似度分数(0-1之间)
        """
        # 预处理图像和文本
        processed_image = self.processor(image).unsqueeze(0).to(self.device)
        text = self.tokenizer(texts).to(self.device)

        with torch.no_grad():
            # 提取特征并归一化
            image_features = self.model.encode_image(processed_image)
            text_features = self.model.encode_text(text)

            # 特征归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 计算余弦相似度
            similarity = image_features @ text_features.t()
            
            # 将相似度转换到0-1范围
            similarity = (similarity + 1) / 2

            # 转换为概率分布
            similarity = similarity.softmax(dim=-1)

            # 返回文本-相似度字典
            return {text: float(score) for text, score in zip(texts, similarity[0].cpu().numpy())}


clip_embedding = ClipEmbedding()

if __name__ == "__main__":
    # image_path = r"E:\playground\ai\datasets\bdd100k\bdd100k\images\10k\train\00a7ef03-00000000.jpg"
    # image_path = r"pokemon.jpeg"
    image_path = r"first_frame.png"

    pil_image = Image.open(image_path)
    # clip_embedding.probs(pil_image)

    # texts = ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘", "白天汽车行驶在道路上"]
    # match = clip_embedding.match(pil_image, texts)
    # print(match)



    # texts = ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]
    # for text in texts:
    #     score = clip_embedding.match(pil_image, text)
    #     print(f"'{text}' 的匹配概率: {score:.4f}")



    image_embeddings = clip_embedding.embedding_image(pil_image)
    print(len(image_embeddings))

    # res = image_embeddings[0].detach().numpy().tolist()
    #
    # print(type(res))
    #
    # print(res)

    # embedding = clip_embedding.embedding_text("a cat")
    # print(len(embedding[0]))
