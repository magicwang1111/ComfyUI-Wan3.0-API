from __future__ import annotations

import io
import os
import tempfile
import wave

import numpy
from PIL import Image

from .models import MediaBlob


MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 50 * 1024 * 1024
MAX_AUDIO_BYTES = 15 * 1024 * 1024


def _check_size(data: bytes, maximum: int, label: str) -> None:
    if len(data) > maximum:
        raise ValueError(f"{label} exceeds the {maximum // (1024 * 1024)} MB limit.")


def _image_blob(image: Image.Image) -> MediaBlob:
    width, height = image.size
    if not (256 <= width <= 5760 and 256 <= height <= 5760):
        raise ValueError("Images must have width and height between 256 and 5760 pixels.")
    if not 0.4 <= width / height <= 2.5:
        raise ValueError("Image aspect ratio must be between 0.4 and 2.5.")
    with io.BytesIO() as buffer:
        image.convert("RGB").save(buffer, format="JPEG", quality=95, subsampling=0)
        data = buffer.getvalue()
    _check_size(data, MAX_IMAGE_BYTES, "Image")
    return MediaBlob(data=data, content_type="image/jpeg", extension="jpg")


def image_batch_to_blobs(tensor, maximum: int = 10) -> list[MediaBlob]:
    if tensor is None:
        return []
    array = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else numpy.asarray(tensor)
    if array.ndim == 3:
        array = array[numpy.newaxis, ...]
    if array.ndim != 4 or not array.shape[0]:
        raise ValueError("IMAGE input must contain one or more images.")
    if array.shape[0] > maximum:
        raise ValueError(f"At most {maximum} images are supported.")
    array = numpy.clip(array * 255.0, 0, 255).astype(numpy.uint8)
    return [_image_blob(Image.fromarray(frame)) for frame in array]


def first_image_blob(tensor) -> MediaBlob:
    blobs = image_batch_to_blobs(tensor, maximum=1)
    if not blobs:
        raise ValueError("IMAGE input is empty.")
    return blobs[0]


def video_to_blob(video) -> MediaBlob:
    if video is None or not callable(getattr(video, "save_to", None)):
        raise ValueError("VIDEO input must support save_to(path).")
    duration_getter = getattr(video, "get_duration", None)
    if not callable(duration_getter):
        raise ValueError("VIDEO input must expose get_duration() for duration validation.")
    duration = float(duration_getter())
    if not 2 <= duration <= 15:
        raise ValueError("Each reference video must be between 2 and 15 seconds.")
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        video.save_to(path)
        with open(path, "rb") as handle:
            data = handle.read()
    finally:
        if os.path.exists(path):
            os.remove(path)
    _check_size(data, MAX_VIDEO_BYTES, "Video")
    if not data:
        raise ValueError("VIDEO input produced an empty MP4 file.")
    return MediaBlob(data=data, content_type="video/mp4", extension="mp4", duration=duration)


def audio_to_blob(audio) -> MediaBlob:
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("AUDIO input must contain waveform and sample_rate.")
    waveform = audio["waveform"]
    if hasattr(waveform, "detach"):
        waveform = waveform.detach()
    if hasattr(waveform, "cpu"):
        waveform = waveform.cpu()
    if hasattr(waveform, "numpy"):
        waveform = waveform.numpy()
    waveform = numpy.asarray(waveform)
    if waveform.ndim == 3:
        waveform = waveform[0]
    elif waveform.ndim == 1:
        waveform = waveform[numpy.newaxis, :]
    if waveform.ndim != 2 or not waveform.shape[-1]:
        raise ValueError("AUDIO waveform must contain samples.")
    sample_rate = int(audio["sample_rate"])
    if sample_rate <= 0:
        raise ValueError("AUDIO sample_rate must be greater than 0.")
    duration = waveform.shape[-1] / sample_rate
    if not 2 <= duration <= 15:
        raise ValueError("Each reference audio must be between 2 and 15 seconds.")
    pcm = (numpy.clip(waveform.astype(numpy.float32), -1.0, 1.0).T * 32767).astype(numpy.int16)
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(int(waveform.shape[0]))
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())
        data = buffer.getvalue()
    _check_size(data, MAX_AUDIO_BYTES, "Audio")
    return MediaBlob(data=data, content_type="audio/wav", extension="wav", duration=duration)
