import base64
import hashlib
import hmac
import mimetypes
import os
import time
from pathlib import Path
from urllib.parse import unquote, urlencode, urlparse


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def media_secret() -> str:
    secret = (
        os.getenv("SCENE_MINING_MEDIA_SIGN_SECRET")
        or os.getenv("DIRECT_MINING_SIGN_SECRET")
        or os.getenv("SECRET_KEY")
    )
    if not secret:
        raise RuntimeError("SCENE_MINING_MEDIA_SIGN_SECRET 未配置")
    return secret


def _sign(path: str, expires: int) -> str:
    payload = f"{path}\n{expires}".encode("utf-8")
    return hmac.new(media_secret().encode("utf-8"), payload, hashlib.sha256).hexdigest()


def encode_media_path(path: str) -> str:
    return base64.urlsafe_b64encode(path.encode("utf-8")).decode("ascii").rstrip("=")


def decode_media_path(encoded_path: str) -> str:
    padding = "=" * (-len(encoded_path) % 4)
    return base64.urlsafe_b64decode(f"{encoded_path}{padding}".encode("ascii")).decode("utf-8")


def build_local_media_url(local_path: str, ttl_seconds: int | None = None) -> str:
    resolved_path = str(Path(local_path).resolve())
    ttl = int(ttl_seconds or os.getenv("SCENE_MINING_MEDIA_URL_TTL_SECONDS", "7200"))
    expires = int(time.time()) + max(60, ttl)
    query = urlencode({"expires": str(expires), "signature": _sign(resolved_path, expires)})
    base_url = os.getenv("SCENE_MINING_MEDIA_BASE_URL") or os.getenv(
        "APP_INTERNAL_BASE_URL",
        "http://127.0.0.1:30012",
    )
    return f"{base_url.strip().rstrip('/')}/api/scene-mining/media/{encode_media_path(resolved_path)}?{query}"


def verify_media_url_signature(encoded_path: str, expires: str, signature: str) -> str:
    try:
        expires_value = int(str(expires or ""))
    except (TypeError, ValueError) as exc:
        raise PermissionError("media expires 参数无效") from exc
    if expires_value < int(time.time()):
        raise PermissionError("media URL 已过期")

    try:
        resolved_path = str(Path(decode_media_path(encoded_path)).resolve())
    except Exception as exc:
        raise PermissionError("media path 参数无效") from exc
    expected = _sign(resolved_path, expires_value)
    if not hmac.compare_digest(expected, str(signature or "")):
        raise PermissionError("media signature 参数无效")
    return resolved_path


def allowed_media_roots() -> list[Path]:
    root = project_root()
    output_dir = Path(os.getenv("SCENE_MINING_OUTPUT_DIR", root / "outputs" / "scene_mining")).resolve()
    roots = [
        Path(os.getenv("SCENE_MINING_VIDEO_CACHE_DIR", root / "data" / "scene_mining_videos")).resolve(),
        output_dir,
        Path(os.getenv("SCENE_MINING_TOOL_CLIP_DIR", output_dir / "_tool_clips")).resolve(),
    ]
    video_url_prefix = os.getenv("SCENE_MINING_VIDEO_URL_PREFIX", "").strip()
    parsed_prefix = urlparse(video_url_prefix)
    if parsed_prefix.scheme == "file" and parsed_prefix.path:
        roots.append(Path(unquote(parsed_prefix.path)).resolve())
    return roots


def is_allowed_media_path(local_path: str) -> bool:
    path = Path(local_path).resolve()
    for root in allowed_media_roots():
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def guess_media_type(local_path: str) -> str:
    return mimetypes.guess_type(local_path)[0] or "application/octet-stream"
