import argparse
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = REPO_ROOT.parent.parent
sys.path.insert(0, str(COMFYUI_ROOT))


def load_package():
    spec = importlib.util.spec_from_file_location(
        "wan3_api_smoke",
        REPO_ROOT / "__init__.py",
        submodule_search_locations=[str(REPO_ROOT)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    return package


def main():
    parser = argparse.ArgumentParser(description="Paid Wan 3.0 smoke test using the configured provider")
    parser.add_argument("--confirm-paid", action="store_true")
    parser.add_argument("--prompt", default="A calm lake reflecting morning light, cinematic composition.")
    args = parser.parse_args()
    if not args.confirm_paid:
        parser.error("This creates a billable task. Re-run with --confirm-paid to continue.")
    package = load_package()
    node = package.NODE_CLASS_MAPPINGS["Wan 3.0 API Text To Video"]()
    video_url, video_id, task_id = node.generate(
        model_version="3.0",
        prompt=args.prompt,
        resolution="480P",
        aspect_ratio="16:9",
        duration=2,
    )
    print(f"TaskId={task_id}")
    print(f"VideoId={video_id}")
    print(f"VideoUrl={video_url}")


if __name__ == "__main__":
    main()
