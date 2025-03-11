import cv2
import subprocess
import json
import os
from typing import Dict, Any, Optional


class VideoFormatAnalyzer:
    """视频格式分析器"""

    def __init__(self, video_path: str):
        """
        初始化分析器

        Args:
            video_path: 视频文件路径
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        self.video_path = video_path

    def get_opencv_info(self) -> Dict[str, Any]:
        """使用OpenCV获取基本视频信息"""
        cap = cv2.VideoCapture(self.video_path)
        try:
            info = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fourcc": self._get_codec_name(int(cap.get(cv2.CAP_PROP_FOURCC))),
                "backend": cap.getBackendName()
            }
            return info
        finally:
            cap.release()

    def get_ffmpeg_info(self) -> Optional[Dict[str, Any]]:
        """使用ffmpeg获取详细的视频信息"""
        try:
            cmd = [
                "ffmpeg",
                "-i", self.video_path,
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"ffmpeg分析失败: {result.stderr}")
                return None

            return json.loads(result.stdout)
        except Exception as e:
            print(f"获取ffmpeg信息失败: {str(e)}")
            return None

    def analyze(self) -> Dict[str, Any]:
        """分析视频并返回完整信息"""
        opencv_info = self.get_opencv_info()
        ffmpeg_info = self.get_ffmpeg_info()

        analysis = {
            "基本信息": {
                "分辨率": f"{opencv_info['width']}x{opencv_info['height']}",
                "帧率": f"{opencv_info['fps']:.2f} fps",
                "总帧数": opencv_info['frame_count'],
                "时长": f"{opencv_info['frame_count'] / opencv_info['fps']:.2f} 秒",
                "编码": opencv_info['fourcc'],
                "OpenCV后端": opencv_info['backend']
            }
        }

        if ffmpeg_info:
            # 提取视频流信息
            video_stream = next(
                (s for s in ffmpeg_info["streams"] if s["codec_type"] == "video"),
                None
            )

            if video_stream:
                analysis["FFmpeg信息"] = {
                    "容器格式": ffmpeg_info["format"]["format_name"],
                    "编码器": video_stream["codec_name"],
                    "像素格式": video_stream.get("pix_fmt", "未知"),
                    "比特率": self._format_bitrate(
                        int(ffmpeg_info["format"].get("bit_rate", 0))
                    ),
                    "文件大小": self._format_size(
                        int(ffmpeg_info["format"].get("size", 0))
                    )
                }

        # 添加建议
        analysis["建议"] = self._generate_recommendations(opencv_info, ffmpeg_info)

        return analysis

    def _get_codec_name(self, fourcc: int) -> str:
        """将fourcc码转换为可读的编码器名称"""
        return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    def _format_bitrate(self, bitrate: int) -> str:
        """格式化比特率"""
        if bitrate > 1000000:
            return f"{bitrate / 1000000:.2f} Mbps"
        elif bitrate > 1000:
            return f"{bitrate / 1000:.2f} Kbps"
        return f"{bitrate} bps"

    def _format_size(self, size: int) -> str:
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} TB"

    def _generate_recommendations(
            self,
            opencv_info: Dict[str, Any],
            ffmpeg_info: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """生成视频格式建议"""
        recommendations = {}

        # 检查编码格式
        if ffmpeg_info:
            video_stream = next(
                (s for s in ffmpeg_info["streams"] if s["codec_type"] == "video"),
                None
            )
            if video_stream:
                codec = video_stream["codec_name"]
                if codec not in ["h264", "avc1"]:
                    recommendations["编码建议"] = (
                        "建议使用H.264编码,这是目前浏览器支持最广泛的视频编码格式。"
                        "可以使用以下ffmpeg命令转换:\n"
                        f"ffmpeg -i {self.video_path} -c:v libx264 -preset medium "
                        "-crf 23 output.mp4"
                    )

        # 检查容器格式
        if ffmpeg_info and ffmpeg_info["format"]["format_name"] != "mp4":
            recommendations["容器格式建议"] = (
                "建议使用MP4容器格式,这是目前浏览器支持最广泛的视频容器格式。"
                "可以使用以下ffmpeg命令转换:\n"
                f"ffmpeg -i {self.video_path} -c copy output.mp4"
            )

        # 检查分辨率
        width, height = opencv_info["width"], opencv_info["height"]
        if width % 2 != 0 or height % 2 != 0:
            recommendations["分辨率建议"] = (
                "视频分辨率的宽度和高度应该是偶数,否则可能导致兼容性问题。"
                "建议调整到最接近的偶数分辨率。"
            )

        return recommendations


def main():
    """主函数"""
    import sys
    if len(sys.argv) != 2:
        print("使用方法: python video_format_analyzer.py <视频文件路径>")
        sys.exit(1)

    video_path = sys.argv[1]
    analyzer = VideoFormatAnalyzer(video_path)

    try:
        analysis = analyzer.analyze()
        print("\n=== 视频格式分析报告 ===\n")

        for section, info in analysis.items():
            print(f"\n{section}:")
            for key, value in info.items():
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"分析失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()