import ast
import importlib
import io
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zlib

import httpx
import numpy as np
from PIL import Image
import requests
import torch

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parent.parent
sys.path.insert(0, str(COMFY_ROOT))
from comfy.cli_args import args
args.cpu = True
import execution
import nodes as core_nodes
import comfy.model_management as model_management
from comfy.utils import common_upscale
from comfy_execution.graph import DynamicPrompt
from comfy_extras.nodes_post_processing import BatchImagesNode

import importlib.util
if "wan3_api" not in sys.modules:
    spec = importlib.util.spec_from_file_location("wan3_api", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules["wan3_api"] = package
    spec.loader.exec_module(package)
plugin = importlib.import_module("wan3_api.nodes")
image_files = importlib.import_module("wan3_api.image_files")


def kj_resize_class():
    path = COMFY_ROOT / "custom_nodes/comfyui-kjnodes/nodes/image_nodes.py"
    if not path.exists():
        raise unittest.SkipTest("Install KJNodes to check resize equivalence.")
    source = ast.parse(path.read_text(encoding="utf-8"))
    node = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "ImageResizeKJv2")
    namespace = dict(torch=torch, math=math, model_management=model_management,
                     common_upscale=common_upscale, PromptServer=None, MAX_RESOLUTION=16384)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["ImageResizeKJv2"]


class Collect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("STRING",)}}
    RETURN_TYPES = ()
    FUNCTION = "collect"
    OUTPUT_NODE = True
    def collect(self, value):
        return {"ui": {"value": [value]}}


class ImageFileTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.owner = self.root / "alice"
        self.owner.mkdir()
        self.start_patch(patch.object(image_files.folder_paths, "get_input_directory", return_value=str(self.owner)))
        self.start_patch(patch.object(httpx.Client, "send", side_effect=AssertionError("Unexpected HTTP")))
        self.start_patch(patch.object(requests.Session, "send", side_effect=AssertionError("Unexpected HTTP")))
        self.start_patch(patch.dict(core_nodes.NODE_CLASS_MAPPINGS, {
            "LoadImage": core_nodes.LoadImage, "BatchImagesNode": BatchImagesNode, "Collect": Collect,
        }))

    def start_patch(self, patcher):
        self.addCleanup(patcher.stop)
        return patcher.start()

    def picture(self, name="photo.png", size=(381, 529), mode="RGB", **options):
        pixels = np.random.default_rng(4).integers(0, 256, (size[1], size[0], len(mode)), dtype=np.uint8)
        path = self.owner / name
        Image.fromarray(pixels, mode).save(path, **options)
        return path

    def graph(self, filename="photo.png"):
        return {
            "source": {"class_type": "LoadImage", "inputs": {"image": filename}},
            "target": {"class_type": "Target", "inputs": {"image": ["source", 0]}},
        }

    def resized_graph(self, filename="photo.png", **options):
        graph = self.graph(filename)
        graph["resize"] = {"class_type": "ImageResizeKJv2", "inputs": {
            "image": ["source", 0], "width": 512, "height": 512, "divisible_by": 8,
            "keep_proportion": "total_pixels", "upscale_method": "lanczos",
            "pad_color": "0, 0, 0", "crop_position": "center", "device": "cpu", **options}}
        graph["target"]["inputs"]["image"] = ["resize", 0]
        return graph

    async def execute(self, graph):
        graph["collect"] = {"class_type": "Collect", "inputs": {"value": ["target", 0]}}
        server = SimpleNamespace(client_id=None, send_sync=Mock())
        executor = execution.PromptExecutor(server, cache_type=execution.CacheType.NONE,
                                            cache_args={"ram": 0, "ram_inactive": 0})
        await executor.execute_async(graph, "file-upload-test", execute_outputs=["collect"])
        self.assertTrue(executor.success, executor.status_messages)
        return executor

    def test_original_bytes_without_decoding(self):
        for extension in ("png", "jpg"):
            with self.subTest(extension=extension):
                path = self.picture("photo." + extension)
                with patch.object(Image.Image, "load", side_effect=AssertionError("Decoded pixels")):
                    source = image_files.image_input_files(DynamicPrompt(self.graph(path.name)), "target", "image")[0]
                    with source.upload_path() as upload:
                        self.assertEqual(Path(upload).read_bytes(), path.read_bytes())

    def test_preprocessed_pixels_match_load_and_kj_resize(self):
        resize = kj_resize_class()()
        for extension, mode in (("png", "RGB"), ("jpg", "RGB"), ("png", "RGBA")):
            path = self.picture("photo." + extension, mode=mode)
            tensor, _ = core_nodes.LoadImage().load_image(path.name)
            for settings in (
                {}, {"keep_proportion": "resize"}, {"keep_proportion": "stretch"},
                {"keep_proportion": "resize", "width": 0, "height": 300},
                {"keep_proportion": "resize", "width": 0, "height": 0, "divisible_by": 0},
            ):
                with self.subTest(extension=extension, mode=mode, settings=settings):
                    graph = self.resized_graph(path.name, **settings)
                    kwargs = dict(graph["resize"]["inputs"], image=tensor, unique_id=None)
                    expected = resize.resize(**kwargs)[0]
                    expected_pixels = np.clip(expected[0].numpy() * 255, 0, 255).astype(np.uint8)
                    source = image_files.image_input_files(DynamicPrompt(graph), "target", "image")[0]
                    with source.read_image() as result:
                        np.testing.assert_array_equal(np.asarray(result), expected_pixels)

    def test_rgba_upload_drops_alpha_like_load_image_and_cleans_temporary_file(self):
        path = self.picture(mode="RGBA")
        expected = core_nodes.LoadImage().load_image(path.name)[0][0].numpy()
        source = image_files.image_input_files(DynamicPrompt(self.graph()), "target", "image")[0]
        with source.upload_path() as upload:
            with Image.open(upload) as image:
                self.assertEqual(image.mode, "RGB")
                np.testing.assert_array_equal(np.asarray(image), (expected * 255).astype(np.uint8))
        self.assertFalse(Path(upload).exists())

    def test_changed_filename_suffix_preserves_mime_and_bytes(self):
        path = self.picture("photo.jpg")
        renamed = path.with_suffix(".png")
        path.rename(renamed)
        source = image_files.image_input_files(DynamicPrompt(self.graph()), "target", "image")[0]
        with source.upload_path() as upload:
            self.assertEqual(Path(upload).suffix, ".jpg")
            self.assertEqual(Path(upload).read_bytes(), renamed.read_bytes())
        self.assertFalse(Path(upload).exists())

    def test_batch_order_and_mixed_dimensions(self):
        first, second = self.picture(), self.picture("second.jpg")
        graph = self.graph()
        graph["second"] = {"class_type": "LoadImage", "inputs": {"image": second.name}}
        graph["batch"] = {"class_type": "BatchImagesNode", "inputs": {
            "images.image2": ["second", 0], "images.image0": ["source", 0], "images.image1": ["second", 0]}}
        graph["target"]["inputs"]["image"] = ["batch", 0]
        sources = image_files.image_input_files(DynamicPrompt(graph), "target", "image")
        self.assertEqual([s.path for s in sources], [first, second, second])
        self.picture("second.jpg", size=(320, 320))
        self.assertIsNone(image_files.image_input_files(DynamicPrompt(graph), "target", "image"))

    def test_unhandled_processing_uses_tensor_path(self):
        self.picture()
        for options in ({"upscale_method": "bilinear"}, {"keep_proportion": "crop"},
                        {"keep_proportion": "pad"}, {"device": "gpu"},
                        {"mask": ["source", 1]}, {"width": ["other", 0]}):
            graph = self.resized_graph(**options)
            self.assertEqual(image_files.required_image_inputs(DynamicPrompt(graph), "target", {"image": None}), ["image"])
        graph = self.graph()
        graph["target"]["inputs"]["image"] = ["source", 1]
        self.assertIsNone(image_files.image_input_files(DynamicPrompt(graph), "target", "image"))

    def test_orientation_animation_and_late_exif_use_tensor_path(self):
        exif = Image.Exif()
        exif[274] = 8
        rotated = self.picture("rotated.jpg", exif=exif)
        animated = self.owner / "animated.png"
        Image.new("RGB", (320, 320), "red").save(animated, save_all=True, append_images=[Image.new("RGB", (320, 320), "blue")])
        path = self.picture()
        data = exif.tobytes()[6:]
        chunk = b"eXIf" + data
        chunk = len(data).to_bytes(4, "big") + chunk + zlib.crc32(chunk).to_bytes(4, "big")
        raw = path.read_bytes()
        path.write_bytes(raw[:-12] + chunk + raw[-12:])
        for filename in (rotated.name, animated.name, path.name):
            with patch.object(Image.Image, "load", side_effect=AssertionError("Decoded pixels")):
                self.assertIsNone(image_files.image_input_files(DynamicPrompt(self.graph(filename)), "target", "image"))

    def test_account_directory_isolation(self):
        other = self.root / "bob"
        other.mkdir()
        path = other / "private.jpg"
        Image.new("RGB", (320, 320)).save(path)
        for filename in ("../bob/private.jpg", str(path)):
            with self.assertRaisesRegex(ValueError, "current input directory"):
                image_files.image_input_files(DynamicPrompt(self.graph(filename)), "target", "image")
        try:
            os.symlink(path, self.owner / "linked.jpg")
        except OSError:
            return
        with self.assertRaisesRegex(ValueError, "current input directory"):
            image_files.image_input_files(DynamicPrompt(self.graph("linked.jpg")), "target", "image")

    async def test_reference_batch_executes_without_image_tensors(self):
        self.picture()
        self.picture("second.png")
        resize_class = kj_resize_class()
        self.start_patch(patch.dict(core_nodes.NODE_CLASS_MAPPINGS, {
            "Target": plugin.WanReferenceToVideo, "ImageResizeKJv2": resize_class}))
        graph = self.resized_graph()
        graph["second"] = {"class_type": "LoadImage", "inputs": {"image": "second.png"}}
        graph["resize2"] = {"class_type": "ImageResizeKJv2", "inputs": dict(graph["resize"]["inputs"], image=["second", 0])}
        graph["batch"] = {"class_type": "BatchImagesNode", "inputs": {
            "images.image0": ["resize2", 0], "images.image1": ["resize", 0]}}
        graph["target"]["inputs"] = dict(reference_images=["batch", 0],
            model_version="3.0", prompt="demo", resolution="720P", aspect_ratio="16:9", duration=5)
        captured = []
        def generate(request, media, **kwargs):
            captured.extend(media)
            return SimpleNamespace(video_url="video", video_id="id", task_id="task")
        self.start_patch(patch.object(plugin, "_generate", side_effect=generate))
        with patch.object(core_nodes.LoadImage, "load_image", side_effect=AssertionError("LoadImage executed")), \
             patch.object(resize_class, "resize", side_effect=AssertionError("Resize executed")), \
             patch.object(BatchImagesNode, "execute", side_effect=AssertionError("Batch executed")):
            await self.execute(graph)
        self.assertEqual([item[2] for item in captured], ["Reference", "Reference"])
        for blob, _, _ in captured:
            self.assertEqual(blob.content_type, "image/jpeg")
            with Image.open(io.BytesIO(blob.data)) as image:
                self.assertEqual(image.size, (432, 600))

    async def test_first_and_last_frames_upload_original_bytes(self):
        first = self.picture("first.jpg")
        last = self.picture("last.png")
        self.start_patch(patch.dict(core_nodes.NODE_CLASS_MAPPINGS, {"Target": plugin.WanFrameToVideo}))
        captured = []
        def generate(request, media, **kwargs):
            captured.extend(media)
            return SimpleNamespace(video_url="video", video_id="id", task_id="task")
        self.start_patch(patch.object(plugin, "_generate", side_effect=generate))
        graph = {
            "first": {"class_type": "LoadImage", "inputs": {"image": first.name}},
            "last": {"class_type": "LoadImage", "inputs": {"image": last.name}},
            "target": {"class_type": "Target", "inputs": dict(
                first_frame=["first", 0], last_frame=["last", 0],
                model_version="3.0", prompt="", resolution="720P", duration=5)},
        }
        with patch.object(core_nodes.LoadImage, "load_image", side_effect=AssertionError("LoadImage executed")):
            await self.execute(graph)
        self.assertEqual([(item[0].content_type, item[2]) for item in captured],
                         [("image/jpeg", "FirstFrame"), ("image/png", "LastFrame")])
        self.assertEqual([item[0].data for item in captured], [first.read_bytes(), last.read_bytes()])

    def test_file_path_retains_image_count_and_dimension_limits(self):
        self.picture()
        source = image_files.image_input_files(DynamicPrompt(self.graph()), "target", "image")[0]
        with self.assertRaisesRegex(ValueError, "At most 10"):
            plugin.image_files_to_blobs([source] * 11)
        path = self.picture("small.png", size=(128, 128))
        small = image_files.image_input_files(DynamicPrompt(self.graph(path.name)), "target", "image")[0]
        with self.assertRaisesRegex(ValueError, "between 256"):
            plugin.image_files_to_blobs([small])
