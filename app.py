from flask import Flask, request, jsonify, send_from_directory, redirect, url_for, Response, stream_with_context, send_file, abort
from flask_cors import CORS
from app.services.video.upload import UploadVideoService
from app.services.video.search import SearchVideoService
from app.services.video.integrated_search import IntegratedSearchService
from app.utils.minio_uploader import MinioFileUploader
import json
import tempfile
import os
import queue
import shutil
import threading
import time
import uuid
import hashlib
import fcntl
import yaml
from urllib.parse import urljoin, urlparse
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from PIL import Image
import io

# 获取当前文件所在目录的绝对路径
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_CHUNK_DIR = os.getenv('UPLOAD_CHUNK_DIR', '/tmp/vision_upload_chunks')
MEDIA_CACHE_DIR = os.getenv('MEDIA_CACHE_DIR', os.path.join(BASE_DIR, 'data', 'media_cache'))
MEDIA_CACHE_MAX_AGE = int(os.getenv('MEDIA_CACHE_MAX_AGE', '3600'))
TASK_STATUS_DIR = os.getenv('TASK_STATUS_DIR', os.path.join(BASE_DIR, 'data', 'task_status'))
SCENE_MINING_CONFIG_PATH = os.getenv(
    'SCENE_MINING_CONFIG_PATH',
    os.path.join(BASE_DIR, 'app', 'algorithm', 'scene_mining', 'config-qwen-gemini.yaml')
)

# 确保static目录存在
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

app = Flask(__name__,
            static_url_path='',
            static_folder='static'
            )
CORS(app)  # 启用CORS支持

# 打印调试信息
print(f"Base Directory: {BASE_DIR}")
print(f"Static Directory: {STATIC_DIR}")


def _scene_mining_max_concurrent_videos():
    try:
        with open(SCENE_MINING_CONFIG_PATH, 'r', encoding='utf-8') as config_file:
            config = yaml.safe_load(config_file) or {}
        value = int((config.get('concurrency') or {}).get('max_concurrent_videos', 10))
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        app.logger.warning("Failed to read scene mining concurrency config, using default 10: %s", exc)
        value = 10
    return max(1, min(value, 50))


MAX_CONCURRENT_ADD_TASKS = _scene_mining_max_concurrent_videos()
ADD_PROCESS_SEMAPHORE = threading.BoundedSemaphore(MAX_CONCURRENT_ADD_TASKS)


@app.before_request
def log_upload_request():
    if request.path.startswith('/api/upload'):
        print(
            f"Upload request start: remote={request.remote_addr}, "
            f"path={request.path}, content_length={request.content_length}"
        )


@app.after_request
def disable_html_cache(response):
    if request.path in {'/', '/upload', '/process', '/search'} or request.path.endswith('.html'):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['ETag'] = ''
    return response


def _to_internal_url(video_url):
    parsed = urlparse(video_url or '')
    if parsed.scheme:
        return video_url
    internal_base_url = os.getenv('APP_INTERNAL_BASE_URL', 'http://127.0.0.1:30501')
    return urljoin(internal_base_url.rstrip('/') + '/', str(video_url).lstrip('/'))


def _media_cache_paths(bucket_name, object_name):
    safe_base = secure_filename(os.path.basename(object_name))
    ext = os.path.splitext(safe_base)[1] or '.bin'
    digest = hashlib.sha256(f'{bucket_name}/{object_name}'.encode('utf-8')).hexdigest()
    cache_path = os.path.join(MEDIA_CACHE_DIR, f'{digest}{ext}')
    return cache_path, f'{cache_path}.json', f'{cache_path}.lock'


def _ensure_media_cache(uploader, bucket_name, object_name, stat):
    os.makedirs(MEDIA_CACHE_DIR, exist_ok=True)
    cache_path, meta_path, lock_path = _media_cache_paths(bucket_name, object_name)
    expected_meta = {
        'bucket': bucket_name,
        'object': object_name,
        'etag': getattr(stat, 'etag', None),
        'size': stat.size,
    }

    with open(lock_path, 'w', encoding='utf-8') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            meta_matches = False
            if os.path.exists(cache_path) and os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as meta_file:
                    cached_meta = json.load(meta_file)
                meta_matches = (
                    cached_meta.get('etag') == expected_meta['etag']
                    and cached_meta.get('size') == expected_meta['size']
                    and os.path.getsize(cache_path) == stat.size
                )
            if meta_matches:
                return cache_path

            tmp_path = f'{cache_path}.{uuid.uuid4()}.tmp'
            try:
                uploader.minio_client.fget_object(bucket_name, object_name, tmp_path)
                os.replace(tmp_path, cache_path)
                with open(meta_path, 'w', encoding='utf-8') as meta_file:
                    json.dump(expected_meta, meta_file, ensure_ascii=False)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            return cache_path
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _public_media_url(value):
    raw_value = str(value or '').strip()
    if not raw_value:
        return ''
    if raw_value.startswith('/media/'):
        return raw_value

    bucket_name = os.getenv('OSS_BUCKET_NAME', '').strip()
    parsed = urlparse(raw_value)
    if parsed.scheme in {'http', 'https'} and bucket_name:
        prefix = f'/{bucket_name}/'
        if parsed.path.startswith(prefix):
            return f"/media{parsed.path}"
    return raw_value


def _parse_media_url(media_url):
    raw_value = str(media_url or '').strip()
    parsed = urlparse(raw_value)
    path = parsed.path if parsed.scheme else raw_value
    prefix = '/media/'
    if not path.startswith(prefix):
        raise ValueError(f"只支持 /media/<bucket>/<object> 路径: {raw_value}")
    rest = path[len(prefix):]
    bucket_name, sep, object_name = rest.partition('/')
    if not sep or not bucket_name or not object_name:
        raise ValueError(f"媒体路径格式无效: {raw_value}")
    return bucket_name, object_name


def _check_media_object(uploader, media_url, warm_cache=False):
    bucket_name, object_name = _parse_media_url(media_url)
    public_url = f"/media/{bucket_name}/{object_name}"
    stat = uploader.minio_client.stat_object(bucket_name, object_name)
    cache_path, _, _ = _media_cache_paths(bucket_name, object_name)
    cached = os.path.exists(cache_path) and os.path.getsize(cache_path) == stat.size
    if warm_cache and not cached:
        cache_path = _ensure_media_cache(uploader, bucket_name, object_name, stat)
        cached = os.path.exists(cache_path) and os.path.getsize(cache_path) == stat.size
    return {
        'url': public_url,
        'bucket': bucket_name,
        'object': object_name,
        'exists': True,
        'size': stat.size,
        'content_type': stat.content_type or 'application/octet-stream',
        'etag': getattr(stat, 'etag', None),
        'cached': cached,
    }


def _format_add_result(video_url, action_type_desc, m_id, video_record=None):
    video_record = video_record or {}
    tags = video_record.get('tags') or []
    if not isinstance(tags, list):
        tags = []
    return {
        'video_url': _public_media_url(video_record.get('path') or video_url),
        'thumbnail_url': _public_media_url(video_record.get('thumbnail_path', '')),
        'action_type_desc': action_type_desc,
        'm_id': m_id,
        'title': video_record.get('title', ''),
        'tags': tags,
        'summary': video_record.get('summary_txt') or '',
    }


def _load_processed_video_record(add_service, m_id, video_url):
    candidates = []
    if m_id:
        candidates.extend(add_service.video_dao.get_by_m_id(m_id) or [])
    if video_url:
        candidates.extend(add_service.video_dao.get_by_path(video_url) or [])
        internal_url = _to_internal_url(video_url)
        if internal_url != video_url:
            candidates.extend(add_service.video_dao.get_by_path(internal_url) or [])

    best_record = {}
    for record in candidates:
        if not best_record:
            best_record = record
        if record.get('summary_txt') or record.get('tags'):
            return record
    return best_record


def _add_video_with_concurrency(add_service, video_url, action_type, process_video_url=None, progress_callback=None):
    if progress_callback:
        progress_callback("queued", f"等待处理资源（最多并发 {MAX_CONCURRENT_ADD_TASKS} 个视频）...")
    ADD_PROCESS_SEMAPHORE.acquire()
    try:
        return add_service.add(
            video_url,
            action_type,
            process_video_url=process_video_url,
            progress_callback=progress_callback,
        )
    finally:
        ADD_PROCESS_SEMAPHORE.release()


def _task_status_path(task_id):
    safe_task_id = secure_filename(str(task_id or ''))
    if not safe_task_id:
        raise ValueError("task_id 不能为空")
    return os.path.join(TASK_STATUS_DIR, f'{safe_task_id}.json')


def _read_task_status(task_id):
    status_path = _task_status_path(task_id)
    if not os.path.exists(status_path):
        return None
    with open(status_path, 'r', encoding='utf-8') as status_file:
        return json.load(status_file)


def _write_task_status(task_id, status):
    os.makedirs(TASK_STATUS_DIR, exist_ok=True)
    status_path = _task_status_path(task_id)
    tmp_path = f'{status_path}.{uuid.uuid4()}.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as status_file:
        json.dump(status, status_file, ensure_ascii=False)
    os.replace(tmp_path, status_path)


def _update_task_status(task_id, **changes):
    now_ms = int(time.time() * 1000)
    status = _read_task_status(task_id) or {
        'task_id': task_id,
        'status': 'queued',
        'stage': 'queued',
        'message': '已提交',
        'created_at': now_ms,
        'updated_at': now_ms,
        'history': [],
    }
    status.update(changes)
    status['updated_at'] = now_ms
    if 'stage' in changes or 'message' in changes:
        history = status.get('history') or []
        history.append({
            'stage': status.get('stage'),
            'message': status.get('message'),
            'timestamp': now_ms,
        })
        status['history'] = history[-50:]
    _write_task_status(task_id, status)
    return status


@app.route('/')
def index():
    """返回主页"""
    try:
        print(f"Trying to serve index.html from {STATIC_DIR}")
        if not os.path.exists(os.path.join(STATIC_DIR, 'index.html')):
            print("Warning: index.html not found!")
            return "Error: index.html not found", 404
        return send_from_directory(STATIC_DIR, 'index.html')
    except Exception as e:
        print(f"Error serving index.html: {str(e)}")
        return str(e), 500


@app.route('/upload')
def upload():
    """返回上传页面"""
    try:
        print(f"Trying to serve index.html from {STATIC_DIR}")
        if not os.path.exists(os.path.join(STATIC_DIR, 'upload.html')):
            print("Warning: index.html not found!")
            return "Error: index.html not found", 404
        return send_from_directory(STATIC_DIR, 'upload.html')
    except Exception as e:
        print(f"Error serving index.html: {str(e)}")
        return str(e), 500


@app.route('/process')
def process():
    """返回添加页面"""
    try:
        print(f"Trying to serve add.html from {STATIC_DIR}")
        if not os.path.exists(os.path.join(STATIC_DIR, 'process.html')):
            print("Warning: add.html not found!")
            return "Error: add.html not found", 404
        return send_from_directory(STATIC_DIR, 'process.html')
    except Exception as e:
        print(f"Error serving add.html: {str(e)}")
        return str(e), 500


@app.route('/search')
def search():
    """返回搜索页面"""
    try:
        print(f"Trying to serve search.html from {STATIC_DIR}")
        if not os.path.exists(os.path.join(STATIC_DIR, 'search.html')):
            print("Warning: search.html not found!")
            return "Error: search.html not found", 404
        return send_from_directory(STATIC_DIR, 'search.html')
    except Exception as e:
        print(f"Error serving search.html: {str(e)}")
        return str(e), 500


@app.route('/api/upload/config', methods=['GET'])
def upload_config():
    return jsonify({
        'status': 'success',
        'data': {
            'max_concurrent_videos': MAX_CONCURRENT_ADD_TASKS,
            'source': SCENE_MINING_CONFIG_PATH,
        }
    })


@app.route('/media/<bucket_name>/<path:object_name>')
def serve_media(bucket_name, object_name):
    """通过 Flask 同源代理 MinIO 对象，并用本地缓存稳定支持浏览器 Range 播放。"""
    uploader = MinioFileUploader()
    try:
        stat = uploader.minio_client.stat_object(bucket_name, object_name)
    except Exception as exc:
        app.logger.error("Media object not found: %s/%s: %s", bucket_name, object_name, exc)
        abort(404)

    content_type = stat.content_type or 'application/octet-stream'
    try:
        cache_path = _ensure_media_cache(uploader, bucket_name, object_name, stat)
    except Exception as exc:
        app.logger.error("Failed to cache media object %s/%s: %s", bucket_name, object_name, exc)
        abort(502)

    response = send_file(
        cache_path,
        mimetype=content_type,
        conditional=True,
        as_attachment=False,
        download_name=os.path.basename(object_name),
        max_age=MEDIA_CACHE_MAX_AGE,
    )
    response.headers['Accept-Ranges'] = 'bytes'
    response.headers['Content-Disposition'] = f'inline; filename="{os.path.basename(object_name)}"'
    return response


@app.route('/api/media/health', methods=['GET', 'POST'])
def media_health():
    """检查 /media 对象是否存在，并可选择预热本地 Range 缓存。"""
    payload = request.get_json(silent=True) or {}
    warm_cache = str(
        payload.get('warm_cache', request.args.get('warm_cache', 'false'))
    ).lower() == 'true'

    if request.method == 'POST':
        urls = payload.get('urls') or []
    else:
        urls = request.args.getlist('url')

    if isinstance(urls, str):
        urls = [urls]
    if not urls:
        return jsonify({'status': 'error', 'message': '缺少 url 或 urls 参数'}), 400
    if len(urls) > 500:
        return jsonify({'status': 'error', 'message': '一次最多检查 500 个媒体 URL'}), 400

    uploader = MinioFileUploader()
    results = []
    for media_url in urls:
        try:
            results.append(_check_media_object(uploader, media_url, warm_cache=warm_cache))
        except Exception as exc:
            result = {'url': str(media_url or ''), 'exists': False, 'error': str(exc)}
            try:
                bucket_name, object_name = _parse_media_url(media_url)
                result.update({'bucket': bucket_name, 'object': object_name})
            except ValueError:
                pass
            results.append(result)

    missing_count = sum(1 for item in results if not item.get('exists'))
    return jsonify({
        'status': 'success' if missing_count == 0 else 'partial',
        'total': len(results),
        'missing_count': missing_count,
        'results': results,
    }), 200


# 添加静态文件路由
@app.route('/<path:path>')
def serve_static(path):
    """服务静态文件"""
    print(f"Requested path: {path}")
    return send_from_directory(STATIC_DIR, path)


@app.route('/api/upload', methods=['POST'])
def upload_video():
    """处理视频上传"""
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': '未找到上传文件'
        }), 400

    video_file = request.files['file']
    if not video_file:
        return jsonify({
            'status': 'error',
            'message': '请选择要上传的视频文件'
        }), 400

    try:
        upload_service = UploadVideoService()
        result = upload_service.upload(video_file)

        if not result:
            return jsonify({
                'status': 'error',
                'message': '视频上传服务返回空结果'
            }), 500

        if not all(key in result for key in ['file_name', 'video_url', 'title']):
            return jsonify({
                'status': 'error',
                'message': f'处理结果格式异常: {result}'
            }), 500

        return jsonify({
            'status': 'success',
            'data': result
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/upload/chunk', methods=['POST'])
def upload_video_chunk():
    """分片上传视频，避免公网代理对大文件单次 POST 超时。"""
    chunk_file = request.files.get('chunk') or request.files.get('file')
    if not chunk_file:
        return jsonify({
            'status': 'error',
            'message': '未找到上传分片'
        }), 400

    try:
        upload_id = request.form.get('upload_id') or str(uuid.uuid4())
        chunk_index = int(request.form.get('chunk_index', '0'))
        total_chunks = int(request.form.get('total_chunks', '1'))
        original_name = request.form.get('file_name') or chunk_file.filename
        filename = secure_filename(original_name)

        if not filename:
            return jsonify({'status': 'error', 'message': '文件名无效'}), 400
        if chunk_index < 0 or total_chunks <= 0 or chunk_index >= total_chunks:
            return jsonify({'status': 'error', 'message': '分片参数无效'}), 400

        safe_upload_id = ''.join(ch for ch in upload_id if ch.isalnum() or ch in ('-', '_'))
        if not safe_upload_id:
            safe_upload_id = str(uuid.uuid4())

        upload_dir = os.path.join(UPLOAD_CHUNK_DIR, safe_upload_id)
        os.makedirs(upload_dir, exist_ok=True)
        chunk_path = os.path.join(upload_dir, f'{chunk_index:08d}.part')
        chunk_file.save(chunk_path)
        print(
            f"Upload chunk saved: upload_id={safe_upload_id}, "
            f"chunk={chunk_index + 1}/{total_chunks}, path={chunk_path}"
        )

        received_chunks = [
            name for name in os.listdir(upload_dir)
            if name.endswith('.part')
        ]
        if len(received_chunks) < total_chunks:
            return jsonify({
                'status': 'uploading',
                'upload_id': safe_upload_id,
                'received_chunks': len(received_chunks),
                'total_chunks': total_chunks
            })

        missing_chunks = [
            index for index in range(total_chunks)
            if not os.path.exists(os.path.join(upload_dir, f'{index:08d}.part'))
        ]
        if missing_chunks:
            return jsonify({
                'status': 'uploading',
                'upload_id': safe_upload_id,
                'received_chunks': len(received_chunks),
                'total_chunks': total_chunks,
                'missing_chunks': missing_chunks[:20]
            })

        assembled_path = os.path.join(upload_dir, filename)
        with open(assembled_path, 'wb') as output_file:
            for index in range(total_chunks):
                part_path = os.path.join(upload_dir, f'{index:08d}.part')
                with open(part_path, 'rb') as input_file:
                    shutil.copyfileobj(input_file, output_file)

        try:
            with open(assembled_path, 'rb') as file_obj:
                file_storage = FileStorage(
                    stream=file_obj,
                    filename=filename,
                    content_type=chunk_file.content_type
                )
                result = UploadVideoService().upload(file_storage)
        finally:
            shutil.rmtree(upload_dir, ignore_errors=True)
            print(f"Upload chunks cleaned: upload_id={safe_upload_id}")

        return jsonify({
            'status': 'success',
            'data': result
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/process', methods=['POST'])
def process_raw_id():
    """处理raw_id"""
    data = request.get_json()
    if not data or 'raw_id' not in data:
        return jsonify({
            'status': 'error',
            'message': '请提供raw_id'
        }), 400

    raw_id = data['raw_id']
    if not raw_id or not raw_id.strip():
        return jsonify({
            'status': 'error',
            'message': '请输入有效的raw_id'
        }), 400

    try:
        # 处理raw_id
        upload_service = UploadVideoService()
        result = upload_service.process_by_raw_id(raw_id.strip())

        if not result:
            return jsonify({
                'status': 'error',
                'message': '视频上传服务返回空结果'
            }), 500

        if not all(key in result for key in ['file_name', 'video_url', 'title']):
            return jsonify({
                'status': 'error',
                'message': f'处理结果格式异常: {result}'
            }), 500

        return jsonify({
            'status': 'success',
            'data': result
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/search', methods=['POST'])
def search_videos():
    """处理视频搜索"""
    try:
        search_type = request.form.get('search_type')
        page = max(1, int(request.form.get('page', 1)))
        page_size = max(1, min(int(request.form.get('page_size', 6)), 50))
        top_k = max(1, min(int(request.form.get('top_k', 10)), 100))
        
        # 获取新增的筛选字段
        vconfig_id = request.form.get('vconfig_id')
        collect_start_time = request.form.get('collect_start_time')
        if collect_start_time and collect_start_time.strip():
            collect_start_time = int(collect_start_time)
        else:
            collect_start_time = None
            
        collect_end_time = request.form.get('collect_end_time')
        if collect_end_time and collect_end_time.strip():
            collect_end_time = int(collect_end_time)
        else:
            collect_end_time = None
        
        # 构建过滤条件字典
        filter_params = {
            'vconfig_id': vconfig_id,
            'collect_start_time': collect_start_time,
            'collect_end_time': collect_end_time
        }
        # 删除值为None的键
        filter_params = {k: v for k, v in filter_params.items() if v is not None}

        search_service = SearchVideoService()

        if search_type == 'smart':
            text_query = request.form.get('text_query', '').strip()
            if not text_query:
                return jsonify({
                    'msg': '请输入搜索关键词',
                    'code': 400,
                    'data': None
                }), 400
            integrated_service = IntegratedSearchService(search_service=search_service)
            results, total = integrated_service.search(
                query=text_query,
                page=page,
                page_size=page_size,
                top_k=top_k,
                **filter_params  # 传入过滤参数
            )

        elif search_type == 'text':
            text_query = request.form.get('text_query', '').strip()
            search_mode = request.form.get('search_mode', 'frame')
            if not text_query:
                return jsonify({
                    'msg': '请输入搜索关键词',
                    'code': 400,
                    'data': None
                }), 400

            try:
                results, total = search_service.search_by_text(
                    text_query,
                    page=page,
                    page_size=page_size,
                    search_mode=search_mode,
                    top_k=top_k,
                    **filter_params  # 传入过滤参数
                )

                # 如果结果为None，返回空列表
                if results is None:
                    results = []
                    total = 0

                # 确保结果可以被JSON序列化
                if results:
                    results = [{
                        'm_id': video.get('m_id', ''),
                        'title': str(video.get('title', '未知')),
                        'path': video.get('path', ''),
                        'thumbnail_path': video.get('thumbnail_path', ''),
                        'tags': list(video.get('tags', [])) if video.get('tags') else [],
                        'summary_txt': str(video.get('summary_txt', '')),
                        'timestamp': video.get('timestamp', 0),
                        'similarity': str(video.get('similarity', '0.0000')),
                        'vconfig_id': str(video.get('vconfig_id', '')),
                        'collect_start_time': video.get('collect_start_time'),
                        'collect_end_time': video.get('collect_end_time'),
                        'feature_type': video.get('feature_type'),
                        'sampled_seconds': video.get('sampled_seconds'),
                        'sampled_frame_count': video.get('sampled_frame_count'),
                    } for video in results]
            except Exception as e:
                print(f"Text search error: {str(e)}")
                return jsonify({
                    'msg': f'文本搜索失败: {str(e)}',
                    'code': 500,
                    'data': None
                }), 500

        elif search_type == 'image':
            image_file = request.files.get('image_file')
            image_url = request.form.get('image_url', '').strip()

            if not image_file and not image_url:
                return jsonify({
                    'msg': '请上传图片或输入图片URL',
                    'code': 400,
                    'data': None
                }), 400

            if image_file:
                # 将文件内容转换为PIL Image对象
                image_data = image_file.read()
                image = Image.open(io.BytesIO(image_data))
            else:
                image = None

            if image_url:
                image_url = _to_internal_url(image_url)

            results, total = search_service.search_by_image(
                image_file=image,
                image_url=image_url,
                page=page,
                page_size=page_size,
                top_k=top_k,
                **filter_params  # 传入过滤参数
            )

        elif search_type == 'tags':
            tags_input = request.form.get('tags', '').strip()
            if not tags_input:
                return jsonify({
                    'msg': '请输入搜索标签',
                    'code': 400,
                    'data': None
                }), 400

            tags = [tag.strip() for tag in tags_input.split(',') if tag.strip()]
            if not tags:
                return jsonify({
                    'msg': '请输入有效的标签',
                    'code': 400,
                    'data': None
                }), 400

            search_mode = request.form.get('search_mode', 'exact')
            if search_mode in ('semantic', 'tags', 'tag_semantic'):
                results, total = search_service.search_by_text(
                    tags_input,
                    page=page,
                    page_size=page_size,
                    search_mode='tags',
                    top_k=top_k,
                    **filter_params
                )
            else:
                results, total = search_service.search_by_tags(
                    tags=tags,
                    page=page,
                    page_size=page_size,
                    top_k=top_k,
                    **filter_params  # 传入过滤参数
                )
        
        elif search_type == 'filter':
            # 仅使用筛选条件进行搜索
            if not filter_params:
                return jsonify({
                    'msg': '请至少提供一个筛选条件',
                    'code': 400,
                    'data': None
                }), 400
                
            results, total = search_service.search_by_filter(
                page=page,
                page_size=page_size,
                top_k=top_k,
                **filter_params
            )

        else:
            return jsonify({
                'msg': '不支持的搜索类型',
                'code': 400,
                'data': None
            }), 400

        if not results:
            return jsonify({
                'msg': 'success',
                'code': 0,
                'data': {
                    'total': 0,
                    'list': []
                }
            })

        # 格式化返回结果
        formatted_results = []
        for video in results:
            formatted_video = {
                'm_id': video.get('m_id', ''),
                'title': video.get('title', '未知'),
                'video_url': _public_media_url(video.get('path', '')),
                'thumbnail_url': _public_media_url(video.get('thumbnail_path', '')),
                'tags': video.get('tags', []),
                'summary': video.get('summary_txt', ''),
                'timestamp': video.get('timestamp', 0),
                'similarity': str(video.get('similarity', '0.0000')),
                'vconfig_id': video.get('vconfig_id', ''),
                'collect_start_time': video.get('collect_start_time'),
                'collect_end_time': video.get('collect_end_time'),
                'feature_type': video.get('feature_type'),
                'sampled_seconds': video.get('sampled_seconds'),
                'sampled_frame_count': video.get('sampled_frame_count'),
            }
            formatted_results.append(formatted_video)

        return jsonify({
            'msg': 'success',
            'code': 0,
            'data': {
                'total': total,
                'list': formatted_results
            }
        })

    except ValueError as e:
        return jsonify({
            'msg': f'参数错误: {str(e)}',
            'code': 400,
            'data': None
        }), 400
    except Exception as e:
        print(f"Search error: {str(e)}")
        return jsonify({
            'msg': f'搜索失败: {str(e)}',
            'code': 500,
            'data': None
        }), 500


@app.route('/api/add', methods=['POST'])
def add_video():
    """处理视频添加"""
    data = request.get_json()
    if not data or 'video_url' not in data:
        return jsonify({
            'status': 'error',
            'message': '请提供视频URL'
        }), 400

    video_url = data['video_url']
    action_type = data.get('action_type', 3)

    if not video_url or not video_url.strip():
        return jsonify({
            'status': 'error',
            'message': '请输入有效的视频URL'
        }), 400

    try:
        # 创建服务实例
        from app.services.video.add import AddVideoService
        add_service = AddVideoService()

        # 处理视频
        action_type = int(action_type or 3)
        m_id = _add_video_with_concurrency(
            add_service,
            video_url,
            action_type,
            process_video_url=_to_internal_url(video_url),
        )
        video_record = _load_processed_video_record(add_service, m_id, video_url)

        # 获取处理类型描述
        action_type_desc = {
            1: "视频内容挖掘",
            2: "视频内容总结",
            3: "内容挖掘和总结"
        }.get(action_type, "未知操作")

        return jsonify({
            'status': 'success',
            'data': _format_add_result(video_url, action_type_desc, m_id, video_record)
        })

    except ValueError as ve:
        return jsonify({
            'status': 'error',
            'message': str(ve)
        }), 400
    except Exception as e:
        print(f"Add video error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'处理失败: {str(e)}'
        }), 500


@app.route('/api/add/task', methods=['POST'])
def add_video_task():
    """提交后台视频处理任务，供前端断线后轮询状态。"""
    data = request.get_json()
    if not data or 'video_url' not in data:
        return jsonify({
            'status': 'error',
            'message': '请提供视频URL'
        }), 400

    video_url = str(data['video_url']).strip()
    action_type = data.get('action_type', 3)
    if not video_url:
        return jsonify({
            'status': 'error',
            'message': '请输入有效的视频URL'
        }), 400

    try:
        action_type = int(action_type or 3)
    except ValueError:
        return jsonify({
            'status': 'error',
            'message': 'action_type must be an integer'
        }), 400

    task_id = str(uuid.uuid4())
    action_type_desc = {
        1: "视频内容挖掘",
        2: "视频内容总结",
        3: "内容挖掘和总结"
    }.get(action_type, "未知操作")
    _update_task_status(
        task_id,
        status='queued',
        stage='queued',
        message='已提交',
        video_url=_public_media_url(video_url),
        action_type=action_type,
        action_type_desc=action_type_desc,
    )

    def progress_callback(stage, message=None, detail=None):
        if isinstance(message, dict) and detail is None:
            detail = message
            message = None
        _update_task_status(
            task_id,
            status='processing',
            stage=str(stage or 'processing'),
            message=str(message or '处理中...'),
            detail=detail or {},
        )

    def worker():
        try:
            from app.services.video.add import AddVideoService
            _update_task_status(task_id, status='processing', stage='processing', message='处理中...')
            add_service = AddVideoService()
            m_id = _add_video_with_concurrency(
                add_service,
                video_url,
                action_type,
                process_video_url=_to_internal_url(video_url),
                progress_callback=progress_callback,
            )
            video_record = _load_processed_video_record(add_service, m_id, video_url)
            _update_task_status(
                task_id,
                status='success',
                stage='complete',
                message='处理完成',
                result=_format_add_result(video_url, action_type_desc, m_id, video_record),
            )
        except Exception as exc:
            app.logger.exception("Add video task failed: %s", task_id)
            _update_task_status(
                task_id,
                status='error',
                stage='error',
                message=f'处理失败: {exc}',
            )

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({
        'status': 'success',
        'data': {
            'task_id': task_id,
            'status_url': f'/api/add/task/{task_id}',
        }
    }), 202


@app.route('/api/add/task/<task_id>', methods=['GET'])
def get_add_video_task(task_id):
    status = _read_task_status(task_id)
    if not status:
        return jsonify({
            'status': 'error',
            'message': '任务不存在'
        }), 404
    return jsonify({
        'status': 'success',
        'data': status,
    })


@app.route('/api/add/stream', methods=['POST'])
def add_video_stream():
    """流式返回视频处理阶段进度。"""
    data = request.get_json()
    if not data or 'video_url' not in data:
        return jsonify({
            'status': 'error',
            'message': '请提供视频URL'
        }), 400

    video_url = str(data['video_url']).strip()
    action_type = data.get('action_type', 3)
    if not video_url:
        return jsonify({
            'status': 'error',
            'message': '请输入有效的视频URL'
        }), 400

    try:
        action_type = int(action_type or 3)
    except ValueError:
        return jsonify({
            'status': 'error',
            'message': 'action_type must be an integer'
        }), 400

    action_type_desc = {
        1: "视频内容挖掘",
        2: "视频内容总结",
        3: "内容挖掘和总结"
    }.get(action_type, "未知操作")

    def generate():
        events: queue.Queue[dict] = queue.Queue()
        done_marker = object()

        def send_event(event_type, stage, message, detail=None):
            events.put({
                "type": event_type,
                "stage": stage,
                "message": message,
                "detail": detail or {},
                "timestamp": int(time.time() * 1000),
            })

        def progress_callback(stage, message=None, detail=None):
            if isinstance(message, dict) and detail is None:
                detail = message
                message = None
            stage_text = str(stage or "processing")
            message_text = str(message or "处理中...")
            send_event("progress", stage_text, message_text, detail)

        def worker():
            try:
                from app.services.video.add import AddVideoService
                add_service = AddVideoService()
                m_id = _add_video_with_concurrency(
                    add_service,
                    video_url,
                    action_type,
                    process_video_url=_to_internal_url(video_url),
                    progress_callback=progress_callback,
                )
                video_record = _load_processed_video_record(add_service, m_id, video_url)
                events.put({
                    "type": "result",
                    "status": "success",
                    "data": _format_add_result(video_url, action_type_desc, m_id, video_record),
                    "timestamp": int(time.time() * 1000),
                })
            except Exception as exc:
                events.put({
                    "type": "error",
                    "status": "error",
                    "message": f"处理失败: {exc}",
                    "timestamp": int(time.time() * 1000),
                })
            finally:
                events.put(done_marker)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        send_event("progress", "processing", "处理中...")

        while True:
            item = events.get()
            if item is done_marker:
                break
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return Response(
        stream_with_context(generate()),
        content_type='application/x-ndjson; charset=utf-8',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


if __name__ == '__main__':
    host = os.getenv('SERVER_HOST', '0.0.0.0')
    port = int(os.getenv('SERVER_PORT', '5000'))
    debug = os.getenv('FLASK_ENV', 'production') != 'production'
    app.run(host=host, port=port, debug=debug)
