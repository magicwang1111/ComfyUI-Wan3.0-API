from contextlib import contextmanager
from dataclasses import dataclass, replace
import math
from pathlib import Path
import re
import shutil
import tempfile

import av
import folder_paths
from PIL import Image


@dataclass(frozen=True)
class ImageFile:
    path: Path
    size: tuple[int, int]
    mode: str
    format: str
    resize_sizes: tuple = ()

    def read_image(self):
        # Match LoadImage's 8-bit PyAV decode, without its float32 tensor.
        with av.open(str(self.path)) as container:
            frame = next(container.decode(video=0))
            image = Image.fromarray(frame.to_ndarray(format="rgb24"))
        for size in self.resize_sizes:
            resized = image.resize(size, Image.Resampling.LANCZOS)
            image.close()
            image = resized
        return image

    @contextmanager
    def upload_path(self):
        suffix = ".jpg" if self.format == "JPEG" else ".png"
        original = self.mode == "RGB" and not self.resize_sizes
        if original and self.path.suffix.lower() in ({".jpg", ".jpeg"} if self.format == "JPEG" else {".png"}):
            yield str(self.path)
            return
        with tempfile.TemporaryDirectory(prefix="reference_image_") as directory:
            path = Path(directory) / ("image" + suffix if original else "image.png")
            if original:
                shutil.copyfile(self.path, path)
            else:
                with self.read_image() as image:
                    image.save(path, format="PNG")
            yield str(path)


def _plain_png(path):
    # EXIF may follow IDAT; inspect chunks without decompressing image pixels.
    with path.open("rb") as source:
        source.seek(8)
        while header := source.read(8):
            if len(header) != 8:
                return False
            length, kind = int.from_bytes(header[:4], "big"), header[4:]
            if kind in {b"eXIf", b"acTL", b"tRNS"}:
                return False
            if kind == b"IEND":
                return True
            if kind == b"IHDR":
                header_data = source.read(length)
                if len(header_data) != 13 or header_data[8] != 8:
                    return False
                source.seek(4, 1)
                continue
            source.seek(length + 4, 1)
    return False


def _resize_size(inputs, size):
    if inputs.get("upscale_method") != "lanczos" or inputs.get("device", "cpu") != "cpu":
        return None
    if "mask" in inputs or inputs.get("keep_proportion") not in {"stretch", "resize", "total_pixels"}:
        return None
    width, height, divisible = (inputs.get(key) for key in ("width", "height", "divisible_by"))
    if any(type(value) is not int or value < 0 for value in (width, height, divisible)):
        return None
    original_width, original_height = size
    mode = inputs["keep_proportion"]
    if mode == "total_pixels":
        pixels = width * height
        aspect = original_width / original_height
        width, height = int(math.sqrt(pixels * aspect)), int(math.sqrt(pixels / aspect))
    elif mode == "resize":
        if width == 0 and height == 0:
            width, height = size
        elif width == 0:
            width = round(original_width * (height / original_height))
        elif height == 0:
            height = round(original_height * (width / original_width))
        else:
            ratio = min(width / original_width, height / original_height)
            width, height = round(original_width * ratio), round(original_height * ratio)
    else:
        width, height = width or original_width, height or original_height
    if divisible > 1:
        width, height = width - width % divisible, height - height % divisible
    return (width, height) if width > 0 and height > 0 else None


def _source_files(dynprompt, link, visiting):
    if not isinstance(link, list) or len(link) != 2 or link[1] != 0:
        return None
    node_id = link[0]
    if node_id in visiting or not dynprompt.has_node(node_id):
        return None
    node = dynprompt.get_node(node_id)
    inputs = node["inputs"]
    if node["class_type"] == "LoadImage" and set(inputs) == {"image"}:
        filename = inputs["image"]
        if not isinstance(filename, str):
            return None
        root = Path(folder_paths.get_input_directory()).resolve()
        filename, annotated_root = folder_paths.annotated_filepath(filename)
        if annotated_root is not None and Path(annotated_root).resolve() != root:
            return None
        path = (root / filename).resolve()
        if Path(filename).is_absolute() or not path.is_relative_to(root):
            raise ValueError("Reference image must be inside the current input directory.")
        try:
            with Image.open(path) as image:
                if image.format not in {"JPEG", "PNG"} or image.mode not in {"RGB", "RGBA"}:
                    return None
                if image.format == "JPEG" and image.getexif().get(274, 1) != 1:
                    return None
                if image.format == "PNG" and not _plain_png(path):
                    return None
                return [ImageFile(path, image.size, image.mode, image.format)]
        except (OSError, Image.DecompressionBombError):
            return None

    visiting = visiting | {node_id}
    if node["class_type"] == "ImageResizeKJv2":
        files = _source_files(dynprompt, inputs.get("image"), visiting)
        if files is None:
            return None
        size = _resize_size(inputs, files[0].size)
        if size is None:
            return None
        return [replace(source, size=size, resize_sizes=source.resize_sizes + (size,)) for source in files]

    if node["class_type"] != "BatchImagesNode" or not inputs:
        return None
    if any(re.fullmatch(r"images\.image\d+", key) is None for key in inputs):
        return None
    files = []
    for key in sorted(inputs, key=lambda key: int(key.removeprefix("images.image"))):
        sources = _source_files(dynprompt, inputs[key], visiting)
        if sources is None:
            return None
        files.extend(sources)
    # Mixed-size BatchImages resizes pixels; preserve that operation by falling back.
    if any(source.size != files[0].size for source in files):
        return None
    return files


def image_input_files(dynprompt, unique_id, name):
    if dynprompt is None or unique_id is None:
        return None
    link = dynprompt.get_node(unique_id)["inputs"].get(name)
    return _source_files(dynprompt, link, set())


def required_image_inputs(dynprompt, unique_id, images):
    return [name for name, image in images.items()
            if image is None and image_input_files(dynprompt, unique_id, name) is None]
