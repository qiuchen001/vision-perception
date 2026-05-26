import importlib.util
from pathlib import Path


_app_path = Path(__file__).with_name("app.py")
_spec = importlib.util.spec_from_file_location("vision_perception_app", _app_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Failed to load Flask app from {_app_path}")

_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
application = _module.app
