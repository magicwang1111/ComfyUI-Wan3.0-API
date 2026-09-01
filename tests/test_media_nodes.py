import json
from types import SimpleNamespace
from unittest import mock

import numpy
import pytest

from wan3_api import media, nodes
from wan3_api.config import OssConfig, TencentConfig
from wan3_api.models import MediaBlob, TaskSubmission, UploadedObject, VideoResult
from wan3_api.tencent_vod import TencentVodTaskError


class FakeTensor:
    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeVideo:
    def __init__(self, duration=3):
        self.duration = duration

    def get_duration(self):
        return self.duration

    def save_to(self, path):
        with open(path, "wb") as handle:
            handle.write(b"mp4")


class VideoWithoutDuration:
    def save_to(self, path):
        pass


def image_tensor(count=1, width=256, height=256):
    return FakeTensor(numpy.zeros((count, height, width, 3), dtype=numpy.float32))


def audio(duration=3, sample_rate=8000):
    return {
        "waveform": FakeTensor(numpy.zeros((1, 1, int(duration * sample_rate)), dtype=numpy.float32)),
        "sample_rate": sample_rate,
    }


def fake_result():
    return VideoResult("https://video", "file", "task", "request", "FINISH", {})


def test_image_video_audio_conversion_and_limits():
    assert media.first_image_blob(image_tensor()).content_type == "image/jpeg"
    assert media.video_to_blob(FakeVideo()).duration == 3
    assert media.audio_to_blob(audio()).duration == 3
    with pytest.raises(ValueError, match="At most 10"):
        media.image_batch_to_blobs(image_tensor(count=11), maximum=10)
    with pytest.raises(ValueError, match="between 2 and 15"):
        media.video_to_blob(FakeVideo(duration=1))
    with pytest.raises(ValueError, match="get_duration"):
        media.video_to_blob(VideoWithoutDuration())


def test_node_mappings_and_public_contracts():
    assert len(nodes.NODE_CLASS_MAPPINGS) == 5
    text = nodes.WanTextToVideo.INPUT_TYPES()
    assert text["required"]["model_version"][0] == ["3.0", "3.0-prime"]
    assert text["required"]["resolution"][0] == ["480P", "720P", "1080P"]
    assert text["required"]["duration"][0] == list(range(2, 31))
    assert text["optional"]["super_resolution"][0] == ["Disabled", "2K", "4K"]
    assert nodes.WanTextToVideo.RETURN_NAMES == ("video_url", "video_id", "task_id")
    preview_inputs = nodes.WanPreviewVideo.INPUT_TYPES()["required"]
    assert list(preview_inputs) == ["video_url"]


def test_text_node_does_not_upload_oss():
    with mock.patch.object(nodes, "_generate", return_value=fake_result()) as generate:
        result = nodes.WanTextToVideo().generate("3.0", "demo", "480P", "16:9", 2)
    assert result == ("https://video", "file", "task")
    assert generate.call_args.args[1] == []


def test_node_maps_super_resolution_separately():
    captured = {}

    def fake_generate(request, media_items, **kwargs):
        captured["request"] = request
        return fake_result()

    with mock.patch.object(nodes, "_generate", side_effect=fake_generate):
        nodes.WanTextToVideo().generate(
            "3.0", "demo", "1080P", "16:9", 5, super_resolution="2K"
        )
    assert captured["request"].resolution == "1080P"
    assert captured["request"].super_resolution == "2K"


def test_frame_node_maps_first_and_last():
    captured = {}

    def fake_generate(request, media_items, **kwargs):
        captured["request"] = request
        captured["media"] = media_items
        return fake_result()

    with mock.patch.object(nodes, "_generate", side_effect=fake_generate):
        nodes.WanFrameToVideo().generate(
            "3.0", "", "720P", 5,
            first_frame=image_tensor(),
            last_frame=image_tensor(),
        )
    assert captured["request"].aspect_ratio == "adaptive"
    assert [item[2] for item in captured["media"]] == ["FirstFrame", "LastFrame"]


def test_reference_validation_and_order():
    captured = {}

    def fake_generate(request, media_items, **kwargs):
        captured["media"] = media_items
        return fake_result()

    with mock.patch.object(nodes, "_generate", side_effect=fake_generate):
        nodes.WanReferenceToVideo().generate(
            "3.0", "demo", "720P", "16:9", 5,
            reference_images=image_tensor(count=2),
            reference_video_1=FakeVideo(3),
            reference_audio_1=audio(3),
        )
    assert [item[1] for item in captured["media"]] == ["Image", "Image", "Video", "Audio"]
    with pytest.raises(ValueError, match="audio cannot be used alone"):
        nodes.WanReferenceToVideo().generate(
            "3.0", "demo", "720P", "16:9", 5,
            reference_audio_1=audio(3),
        )
    with pytest.raises(ValueError, match="plus output duration"):
        nodes.WanReferenceToVideo().generate(
            "3.0", "demo", "720P", "16:9", 20,
            reference_video_1=FakeVideo(11),
        )


def test_prompt_boundary():
    with mock.patch.object(nodes, "_generate", return_value=fake_result()):
        nodes.WanTextToVideo().generate("3.0", "x" * 20000, "480P", "16:9", 2)
    request = nodes._request("3.0", "x" * 20001, "480P", "16:9", 2, "", "Disabled", -1, "id")
    from wan3_api.models import validate_request
    with pytest.raises(ValueError, match="20000"):
        validate_request(request, prompt_required=True)


def test_query_result_is_sanitized():
    task = {
        "Status": "FINISH",
        "Output": {"FileInfos": [{"FileId": "file", "FileUrl": "https://host/v.mp4?token=secret"}]},
    }
    client = mock.MagicMock()
    client.__enter__.return_value = client
    client.wait_for_task.return_value = task
    client.status.side_effect = lambda value: value["Status"]
    with mock.patch.object(nodes, "load_tencent_config"), mock.patch.object(nodes, "TencentVodClient", return_value=client):
        status, url, video_id, task_id, result_json = nodes.WanQueryTask().query("task", True)
    assert (status, url, video_id, task_id) == ("FINISH", "https://host/v.mp4?token=secret", "file", "task")
    assert "token=secret" not in result_json


@pytest.mark.parametrize("terminal,expected_deletes", [(True, 1), (False, 0)])
def test_oss_cleanup_only_after_terminal_task(terminal, expected_deletes):
    oss_config = OssConfig("endpoint", "id", "secret", "bucket", "tmp", 5000, True)
    tencent_config = TencentConfig(
        "id", "secret", "region", 123, 1, 120, 3600,
        "Temporary", "Enabled", "Enabled",
    )
    oss = mock.MagicMock()
    oss.__enter__.return_value = oss
    oss.upload.return_value = UploadedObject("tmp/object.jpg", "https://signed")
    client = mock.MagicMock()
    client.__enter__.return_value = client
    client.create_video.return_value = TaskSubmission("task", "request")
    client.wait_for_task.side_effect = TencentVodTaskError("stopped", "task", terminal)
    request = nodes._request("3.0", "demo", "720P", "adaptive", 5, "", "Disabled", -1, "session")
    with (
        mock.patch.object(nodes, "load_json_config", return_value={}),
        mock.patch.object(nodes, "load_oss_config", return_value=oss_config),
        mock.patch.object(nodes, "load_tencent_config", return_value=tencent_config),
        mock.patch.object(nodes, "OssClient", return_value=oss),
        mock.patch.object(nodes, "TencentVodClient", return_value=client),
        pytest.raises(TencentVodTaskError),
    ):
        nodes._generate(request, [(MediaBlob(b"x", "image/jpeg", "jpg"), "Image", "Reference")], prompt_required=False)
    assert oss.delete.call_count == expected_deletes
