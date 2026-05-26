from flask import Blueprint, send_file, abort, current_app, Response, request
import requests
from urllib.parse import unquote, urlparse
import os

bp = Blueprint('video_proxy', __name__)

@bp.route('/proxy/<path:video_path>')
def video_proxy(video_path):
    try:
        if os.getenv('ENABLE_LEGACY_VIDEO_PROXY', 'false').lower() != 'true':
            return abort(410)

        # 解码URL
        video_url = unquote(video_path)
        if not video_url.startswith('http'):
            video_url = f'http://{video_url}'

        allowed_hosts = {
            host.strip()
            for host in os.getenv('LEGACY_VIDEO_PROXY_ALLOWED_HOSTS', '').split(',')
            if host.strip()
        }
        parsed = urlparse(video_url)
        if not parsed.hostname or parsed.scheme not in {'http', 'https'}:
            return abort(400)
        if allowed_hosts and parsed.hostname not in allowed_hosts:
            current_app.logger.warning("Blocked legacy video proxy host: %s", parsed.hostname)
            return abort(403)
        if not allowed_hosts:
            current_app.logger.warning("Legacy video proxy enabled without allowed hosts")
            return abort(403)
            
        current_app.logger.info(f"Proxying video: {video_url}")
            
        # 获取视频内容
        timeout = float(os.getenv('LEGACY_VIDEO_PROXY_TIMEOUT', '10'))
        headers = {}
        if 'Range' in request.headers:
            headers['Range'] = request.headers['Range']
        response = requests.get(video_url, stream=True, timeout=timeout, headers=headers)
        if response.status_code not in (200, 206):
            current_app.logger.error(f"Failed to fetch video: {response.status_code}")
            return abort(404)
            
        # 返回视频流
        def generate():
            for chunk in response.iter_content(chunk_size=8192):
                yield chunk
                
        return Response(
            generate(),
            status=response.status_code,
            content_type=response.headers.get('content-type', 'video/mp4'),
            headers={
                'Accept-Ranges': 'bytes',
                'Content-Length': response.headers.get('content-length'),
                'Content-Range': response.headers.get('content-range'),
                'Cache-Control': 'no-cache'
            }
        )
        
    except Exception as e:
        current_app.logger.error(f"Video proxy error: {str(e)}")
        return abort(500)
