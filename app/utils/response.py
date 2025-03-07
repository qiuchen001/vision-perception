from functools import wraps
from flask import jsonify
from ..utils.logger import logger


def api_response(success_data):
    """统一的成功响应格式"""
    return {
        "msg": "success",
        "code": 0,
        "data": success_data
    }


def error_response(error_msg, code=500):
    """统一的错误响应格式"""
    return {
        "msg": "error",
        "code": code,
        "data": {
            "error": error_msg
        }
    }


def api_handler(f):
    """API 统一异常处理装饰器"""

    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            return jsonify(f(*args, **kwargs)), 200
        except ValueError as e:
            logger.error(f"Value Error in {f.__name__}: {str(e)}")
            return jsonify(error_response(str(e), 400)), 400
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            return jsonify(error_response(str(e))), 500

    return decorated
