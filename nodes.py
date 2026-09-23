from __future__ import annotations

import datetime as dt
import json
import os
import time
import urllib.parse
import uuid

import requests

import folder_paths

from .config import load_json_config, load_oss_config, load_tencent_config, load_provider, load_vapeur_config, load_kuaizi_config, load_opc_config
from .opc import (
    OpcClient, OpcTaskError, OpcSubmissionUncertain,
    build_payload as opc_payload, validate_media as opc_validate_media, video_result as opc_result,
)
from .kuaizi import (
    KuaiziClient, KuaiziTaskError, KuaiziSubmissionUncertain,
    build_payload as kuaizi_payload, video_result as kuaizi_result,
)
from .vapeur import (
    VapeurClient, VapeurTaskError, VapeurSubmissionUncertain,
    build_payload as vapeur_payload, video_result as vapeur_result,
)
from .media import audio_to_blob, first_image_blob, image_batch_to_blobs, video_to_blob
from .models import (
    AUDIO_GENERATION_OPTIONS,
    ASPECT_RATIOS,
    DURATIONS,
    MODEL_VERSIONS,
    RESOLUTIONS,
    SUPER_RESOLUTIONS,
    MediaBlob,
    TaskSubmission,
    WanVideoRequest,
)
from .oss_client import OssClient
from .tencent_vod import (
    FAILED_STATUSES,
    TERMINAL_STATUSES,
    TencentVodClient,
    TencentVodTaskError,
    sanitize_task,
    task_error,
    video_result,
)


NODE_CATEGORY = "Wan 3.0 API"
NODE_PREFIX = "Wan 3.0 API"
DEFAULT_VIDEO_FILENAME_PREFIX = "video/Wan3_%year%%month%%day%_%hour%%minute%%second%"


def _session_id() -> str:
    return f"wan3-{uuid.uuid4().hex[:24]}"


def _object_key(prefix: str, session_id: str, extension: str) -> str:
    date = dt.datetime.now().strftime("%Y%m%d")
    return f"{prefix}/wan3/{date}/{session_id}/{uuid.uuid4().hex}.{extension}"


def _file_info(uploaded_url: str, category: str, usage: str) -> dict:
    return {
        "Type": "Url",
        "Category": category,
        "Url": uploaded_url,
        "Usage": usage,
    }


def _request(
    model_version,
    prompt,
    resolution,
    aspect_ratio,
    duration,
    negative_prompt,
    enhance_prompt,
    seed,
    session_id,
    super_resolution="Disabled",
    audio_generation="Enabled",
) -> WanVideoRequest:
    return WanVideoRequest(
        model_version=str(model_version),
        prompt=str(prompt or ""),
        resolution=str(resolution),
        aspect_ratio=str(aspect_ratio),
        duration=int(duration),
        session_id=session_id,
        negative_prompt=str(negative_prompt or ""),
        enhance_prompt=str(enhance_prompt or "Disabled"),
        seed=None if seed is None or int(seed) < 0 else int(seed),
        super_resolution=str(super_resolution or "Disabled"),
        audio_generation=str(audio_generation or "Enabled"),
    )


def _cleanup(oss: OssClient, object_keys: list[str]) -> None:
    for object_key in object_keys:
        try:
            oss.delete(object_key)
        except Exception as exc:
            print(f"[{NODE_PREFIX}] OSS cleanup warning for {object_key}: {exc}")


def _generate(request: WanVideoRequest, media: list[tuple[MediaBlob, str, str]], *, prompt_required: bool):
    data = load_json_config()
    provider = load_provider(data)
    if provider == "opc":
        api_config = load_opc_config(data)
        client_type = OpcClient
        result_parser = opc_result
        opc_payload(request)
        opc_validate_media(
            [(item.get("Category"), item.get("Usage")) for item in request.file_infos]
            + [(category, usage) for _, category, usage in media]
        )
    elif provider == "kuaizi":
        api_config = load_kuaizi_config(data)
        client_type = KuaiziClient
        result_parser = kuaizi_result
        kuaizi_payload(request)
    elif provider == "vapeur":
        api_config = load_vapeur_config(data)
        client_type = VapeurClient
        result_parser = vapeur_result
        vapeur_payload(api_config, request)
    else:
        api_config = load_tencent_config(data)
        client_type = TencentVodClient
        result_parser = video_result
    if not media:
        with client_type(api_config) as client:
            submission = client.create_video(request, prompt_required=prompt_required)
            task = client.wait_for_task(submission.task_id)
            return result_parser(task, submission)

    oss_config = load_oss_config(data)
    if oss_config.signed_url_expires < api_config.max_wait_seconds + 600:
        raise ValueError(
            "oss_signed_url_expires must be at least the provider max_wait_seconds + 600 seconds."
        )
    object_keys: list[str] = []
    submission: TaskSubmission | None = None
    terminal = False
    submission_uncertain = False
    with OssClient(oss_config) as oss:
        try:
            for index, (blob, category, usage) in enumerate(media, start=1):
                object_key = _object_key(oss_config.prefix, request.session_id, blob.extension)
                try:
                    uploaded = oss.upload(object_key, blob, timeout=api_config.request_timeout)
                except Exception as exc:
                    raise RuntimeError(f"OSS {category} upload {index} failed: {exc}") from exc
                object_keys.append(uploaded.object_key)
                request.file_infos.append(_file_info(uploaded.url, category, usage))

            with client_type(api_config) as client:
                submission = client.create_video(request, prompt_required=prompt_required)
                try:
                    task = client.wait_for_task(submission.task_id)
                    terminal = True
                except (TencentVodTaskError, VapeurTaskError, KuaiziTaskError, OpcTaskError) as exc:
                    terminal = exc.terminal
                    raise
                return result_parser(task, submission)
        except (VapeurSubmissionUncertain, KuaiziSubmissionUncertain, OpcSubmissionUncertain):
            submission_uncertain = True
            raise
        finally:
            if oss_config.cleanup_after_task and not submission_uncertain and (submission is None or terminal):
                _cleanup(oss, object_keys)


def _common_required(*, aspect_ratio: bool) -> dict:
    required = {
        "model_version": (MODEL_VERSIONS, {"default": "3.0"}),
        "prompt": ("STRING", {"multiline": True, "default": ""}),
        "resolution": (RESOLUTIONS, {"default": "720P"}),
    }
    if aspect_ratio:
        required["aspect_ratio"] = (ASPECT_RATIOS, {"default": "16:9"})
    required["duration"] = (DURATIONS, {"default": 5})
    return required


def _common_optional() -> dict:
    return {
        "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
        "enhance_prompt": (["Disabled", "Enabled"], {"default": "Disabled"}),
        "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "step": 1}),
        "super_resolution": (SUPER_RESOLUTIONS, {"default": "Disabled"}),
        "audio_generation": (AUDIO_GENERATION_OPTIONS, {"default": "Enabled"}),
    }


class WanTextToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": _common_required(aspect_ratio=True), "optional": _common_optional()}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_id", "task_id")
    FUNCTION = "generate"
    CATEGORY = NODE_CATEGORY

    def generate(
        self,
        model_version,
        prompt,
        resolution,
        aspect_ratio,
        duration,
        negative_prompt="",
        enhance_prompt="Disabled",
        seed=-1,
        super_resolution="Disabled",
        audio_generation="Enabled",
    ):
        request = _request(
            model_version, prompt, resolution, aspect_ratio, duration,
            negative_prompt, enhance_prompt, seed, _session_id(), super_resolution,
            audio_generation,
        )
        result = _generate(request, [], prompt_required=True)
        return (result.video_url, result.video_id, result.task_id)


class WanFrameToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        optional = _common_optional()
        optional.update({"first_frame": ("IMAGE",), "last_frame": ("IMAGE",)})
        return {"required": _common_required(aspect_ratio=False), "optional": optional}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_id", "task_id")
    FUNCTION = "generate"
    CATEGORY = NODE_CATEGORY

    def generate(
        self,
        model_version,
        prompt,
        resolution,
        duration,
        negative_prompt="",
        enhance_prompt="Disabled",
        seed=-1,
        super_resolution="Disabled",
        audio_generation="Enabled",
        first_frame=None,
        last_frame=None,
    ):
        if first_frame is None and last_frame is None:
            raise ValueError("Connect first_frame, last_frame, or both.")
        media = []
        if first_frame is not None:
            media.append((first_image_blob(first_frame), "Image", "FirstFrame"))
        if last_frame is not None:
            media.append((first_image_blob(last_frame), "Image", "LastFrame"))
        request = _request(
            model_version, prompt, resolution, "adaptive", duration,
            negative_prompt, enhance_prompt, seed, _session_id(), super_resolution,
            audio_generation,
        )
        result = _generate(request, media, prompt_required=False)
        return (result.video_url, result.video_id, result.task_id)


class WanReferenceToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        optional = _common_optional()
        optional["reference_images"] = ("IMAGE",)
        optional.update({f"reference_video_{index}": ("VIDEO",) for index in range(1, 6)})
        optional.update({f"reference_audio_{index}": ("AUDIO",) for index in range(1, 6)})
        optional["reference_file_url"] = ("STRING", {"default": "", "tooltip": "OPC: public document URL; requires enhance_prompt Enabled."})
        optional["reference_link_url"] = ("STRING", {"default": "", "tooltip": "OPC: public webpage URL; requires enhance_prompt Enabled."})
        return {"required": _common_required(aspect_ratio=True), "optional": optional}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_id", "task_id")
    FUNCTION = "generate"
    CATEGORY = NODE_CATEGORY

    def generate(
        self,
        model_version,
        prompt,
        resolution,
        aspect_ratio,
        duration,
        negative_prompt="",
        enhance_prompt="Disabled",
        seed=-1,
        super_resolution="Disabled",
        audio_generation="Enabled",
        reference_images=None,
        **kwargs,
    ):
        image_blobs = image_batch_to_blobs(reference_images, maximum=10)
        video_inputs = [kwargs.get(f"reference_video_{index}") for index in range(1, 6)]
        audio_inputs = [kwargs.get(f"reference_audio_{index}") for index in range(1, 6)]
        video_blobs = [video_to_blob(item) for item in video_inputs if item is not None]
        audio_blobs = [audio_to_blob(item) for item in audio_inputs if item is not None]
        file_url = str(kwargs.get("reference_file_url") or "").strip()
        link_url = str(kwargs.get("reference_link_url") or "").strip()
        if file_url or link_url or (audio_blobs and not image_blobs and not video_blobs):
            if load_provider(load_json_config()) != "opc":
                raise ValueError("File/link inputs and audio-only reference mode currently require provider opc.")
        if not image_blobs and not video_blobs and not audio_blobs and not file_url and not link_url:
            raise ValueError("Reference mode requires at least one image, video, audio, file, or link.")
        video_duration = sum(blob.duration or 0 for blob in video_blobs)
        audio_duration = sum(blob.duration or 0 for blob in audio_blobs)
        if video_duration > 15:
            raise ValueError("Reference video total duration must not exceed 15 seconds.")
        if audio_duration > 15:
            raise ValueError("Reference audio total duration must not exceed 15 seconds.")
        if video_blobs and video_duration + int(duration) > 30:
            raise ValueError("Reference video total duration plus output duration must not exceed 30 seconds.")
        media = (
            [(blob, "Image", "Reference") for blob in image_blobs]
            + [(blob, "Video", "Reference") for blob in video_blobs]
            + [(blob, "Audio", "Reference") for blob in audio_blobs]
        )
        request = _request(
            model_version, prompt, resolution, aspect_ratio, duration,
            negative_prompt, enhance_prompt, seed, _session_id(), super_resolution,
            audio_generation,
        )
        if file_url:
            request.file_infos.append(_file_info(file_url, "File", "Reference"))
        if link_url:
            request.file_infos.append(_file_info(link_url, "Link", "Reference"))
        result = _generate(request, media, prompt_required=False)
        return (result.video_url, result.video_id, result.task_id)


class WanQueryTask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_id": ("STRING", {"default": ""}),
                "wait_for_completion": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("status", "video_url", "video_id", "task_id", "result_json")
    FUNCTION = "query"
    CATEGORY = NODE_CATEGORY

    def query(self, task_id, wait_for_completion=True):
        task_id = str(task_id or "").strip()
        if not task_id:
            raise ValueError("task_id is required.")
        data = load_json_config()
        provider = load_provider(data)
        if provider in {"vapeur", "kuaizi", "opc"}:
            client_type, config_loader, result_parser = {
                "vapeur": (VapeurClient, load_vapeur_config, vapeur_result),
                "kuaizi": (KuaiziClient, load_kuaizi_config, kuaizi_result),
                "opc": (OpcClient, load_opc_config, opc_result),
            }[provider]
            api_config = config_loader(data)
            with client_type(api_config) as client:
                detail = client.wait_for_task(task_id) if wait_for_completion else client.describe_task(task_id)
                status = client.check_task(detail, task_id)
                output = detail.get("output") or {}
                url = str(output.get("video_url") or "")
                if status == "SUCCEEDED":
                    url = result_parser(detail, TaskSubmission(task_id, str(detail.get("request_id") or ""))).video_url
                result_json = json.dumps(sanitize_task(detail), ensure_ascii=False, separators=(",", ":"))
                return (status, url, task_id if url else "", task_id, result_json)
        config = load_tencent_config(data)
        with TencentVodClient(config) as client:
            if wait_for_completion:
                task = client.wait_for_task(task_id)
            else:
                detail = client.describe_task(task_id)
                task = client.extract_task(detail)
                if task is None:
                    raise RuntimeError(f"Tencent VOD task {task_id} returned no task details.")
                error = task_error(task)
                task_status = client.status(task)
                if task_status in FAILED_STATUSES or (task_status in TERMINAL_STATUSES and error):
                    raise TencentVodTaskError(
                        f"Tencent VOD task {task_id} failed with status={task_status}: {error or '-'}",
                        task_id,
                        True,
                    )
            status = client.status(task)
        output = task.get("Output") or {}
        file_infos = output.get("FileInfos") or []
        first = next(
            (item for item in file_infos if isinstance(item, dict) and (item.get("FileUrl") or item.get("Url"))),
            {},
        )
        video_url = str(first.get("FileUrl") or first.get("Url") or "")
        video_id = str(first.get("FileId") or first.get("VideoId") or "")
        result_json = json.dumps(sanitize_task(task), ensure_ascii=False, separators=(",", ":"))
        return (status, video_url, video_id, task_id, result_json)


def _saved_result(filename, subfolder, folder_type):
    return {"filename": filename, "subfolder": subfolder, "type": folder_type}


def _local_media_url(filename, subfolder, folder_type):
    return "/view?" + urllib.parse.urlencode({
        "filename": filename,
        "subfolder": subfolder,
        "type": folder_type,
    })


class WanPreviewVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video_url": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    OUTPUT_NODE = True
    FUNCTION = "download"
    CATEGORY = NODE_CATEGORY

    def download(self, video_url):
        video_url = str(video_url or "").strip()
        if not video_url:
            raise ValueError("video_url is empty; query the task_id to recover the result.")
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            DEFAULT_VIDEO_FILENAME_PREFIX,
            output_dir,
        )
        os.makedirs(full_output_folder, exist_ok=True)
        file = f"{filename}_{counter:05}_.mp4"
        file_path = os.path.join(full_output_folder, file)
        try:
            with requests.get(video_url, stream=True, timeout=(15, 600)) as response:
                response.raise_for_status()
                with open(file_path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if not os.path.getsize(file_path):
                raise RuntimeError("Downloaded video is empty.")
        except Exception as exc:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise RuntimeError(
                "Video download failed; the Temporary URL may have expired. "
                "Query the task_id and retry."
            ) from exc
        preview_url = _local_media_url(file, subfolder, "output")
        return {
            "ui": {
                "images": [_saved_result(file, subfolder, "output")],
                "video_url": [preview_url],
                "animated": (True,),
            },
            "result": (file_path,),
        }


NODE_CLASS_MAPPINGS = {
    f"{NODE_PREFIX} Text To Video": WanTextToVideo,
    f"{NODE_PREFIX} Frame To Video": WanFrameToVideo,
    f"{NODE_PREFIX} Reference To Video": WanReferenceToVideo,
    f"{NODE_PREFIX} Query Task": WanQueryTask,
    f"{NODE_PREFIX} Preview Video": WanPreviewVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {key: key for key in NODE_CLASS_MAPPINGS}
