import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = REPO_ROOT.parent.parent
if str(COMFYUI_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFYUI_ROOT))


def load_package():
    if "wan3_api" in sys.modules:
        return sys.modules["wan3_api"]
    spec = importlib.util.spec_from_file_location(
        "wan3_api",
        REPO_ROOT / "__init__.py",
        submodule_search_locations=[str(REPO_ROOT)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules["wan3_api"] = package
    spec.loader.exec_module(package)
    return package


load_package()

