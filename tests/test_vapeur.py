from dataclasses import replace
from unittest import mock

import pytest
import requests

from wan3_api import config, nodes
from wan3_api.models import MediaBlob, TaskSubmission, UploadedObject, WanVideoRequest
from wan3_api.vapeur import (
    VapeurClient, VapeurError, VapeurTaskError, VapeurSubmissionUncertain, VapeurTransientError,
    build_payload, video_result,
)


def settings(**kwargs):
    return config.load_vapeur_config({"vapeur_api_key": "test-secret", **kwargs})


def request():
    return WanVideoRequest("3.0", "demo", "480P", "16:9", 2, "session")


def test_config_defaults_and_isolation():
    with mock.patch.dict("os.environ", {}, clear=True):
        assert config.load_provider({}) == "tencent"
        assert config.load_provider({"provider": "Vapeur"}) == "vapeur"
        assert settings().region == "cn"
        with pytest.raises(ValueError, match="provider"):
            config.load_provider({"provider": "invalid"})
        with pytest.raises(ValueError, match="vapeur_api_key"):
            config.load_vapeur_config({})
    with mock.patch.dict("os.environ", {"VAPEUR_API_KEY": "env-key", "VAPEUR_REGION": "glb"}, clear=True):
        assert config.load_vapeur_config({}).api_key == "env-key"
        assert config.load_vapeur_config({}).region == "glb"
        assert settings().api_key == "test-secret"


@pytest.mark.parametrize("values", [
    {"vapeur_region": "us"}, {"vapeur_base_url": "https://api.vapeur.ai/v1"},
    {"vapeur_base_url": "http://api.vapeur.ai"}, {"vapeur_poll_interval": 0},
])
def test_invalid_config(values):
    with pytest.raises(ValueError):
        settings(**values)


@pytest.mark.parametrize("region", ["cn", "glb"])
@pytest.mark.parametrize("version,model", [("3.0", "wan3.0-video"), ("3.0-prime", "wan3.0-video-prime")])
def test_all_four_models_and_parameters(region, version, model):
    req = replace(request(), model_version=version, duration=-1, seed=0,
                  enhance_prompt="Enabled", audio_generation="Disabled", negative_prompt="blur")
    payload = build_payload(settings(vapeur_region=region), req)
    assert payload["model"] == f"{model}-{region}"
    assert payload["input"] == {"prompt": "demo", "media": [], "negative_prompt": "blur"}
    assert payload["parameters"] == {
        "resolution": "480P", "ratio": "16:9", "duration": -1, "seed": 0,
        "audio": False, "prompt_extend": True, "watermark": False,
    }


@pytest.mark.parametrize("category,usage,kind", [
    ("Image", "FirstFrame", "first_frame"), ("Image", "LastFrame", "last_frame"),
    ("Image", "Reference", "reference_image"), ("Video", "Reference", "reference_video"),
    ("Audio", "Reference", "reference_audio"),
])
def test_media_mapping(category, usage, kind):
    req = replace(request(), file_infos=[nodes._file_info("https://signed", category, usage)])
    assert build_payload(settings(), req)["input"]["media"] == [{"type": kind, "url": "https://signed"}]


def test_unsupported_parameters_rejected_before_upload():
    with mock.patch.object(nodes, "load_json_config", return_value={"provider": "vapeur", "vapeur_api_key": "test"}), mock.patch.object(nodes, "OssClient") as oss:
        with pytest.raises(ValueError, match="super_resolution"):
            nodes._generate(replace(request(), super_resolution="4K"), [(MediaBlob(b"x", "image/jpeg", "jpg"), "Image", "FirstFrame")], prompt_required=True)
    oss.assert_not_called()
    with pytest.raises(ValueError, match="seed"):
        build_payload(settings(), replace(request(), seed=2147483648))


def test_submit_and_query_wire_contract():
    with VapeurClient(settings()) as client:
        response = mock.Mock(status_code=200)
        response.json.return_value = {"output": {"task_id": "task", "task_status": "PENDING"}, "request_id": "req"}
        client.session.request = mock.Mock(return_value=response)
        assert client.create_video(request(), prompt_required=True) == TaskSubmission("task", "req")
        args, kwargs = client.session.request.call_args
        assert args == ("POST", "https://api.vapeur.ai/qwen/api/v1/services/aigc/video-generation/generation")
        assert kwargs["headers"]["Authorization"] == "Bearer test-secret"
        assert kwargs["allow_redirects"] is False
        assert kwargs["json"]["model"] == "wan3.0-video-cn"
        client.describe_task("task/id")
        assert client.session.request.call_args.args == ("GET", "https://api.vapeur.ai/qwen/api/v1/tasks/video/task%2Fid")


def test_poll_success_and_result():
    detail = {"output": {"task_status": "SUCCEEDED", "video_url": "https://host/v.mp4?token=x"}}
    with VapeurClient(settings()) as client, mock.patch("wan3_api.vapeur.time.sleep"):
        client.describe_task = mock.Mock(side_effect=[{"output": {"task_status": "PENDING"}}, {"output": {"task_status": "RUNNING"}}, detail])
        result = video_result(client.wait_for_task("task"), TaskSubmission("task", "req"))
        assert result.video_url == detail["output"]["video_url"]
        assert result.task_id == result.video_id == "task"
    with pytest.raises(VapeurTaskError, match="no video_url"):
        video_result({"output": {"task_status": "SUCCEEDED"}}, TaskSubmission("task", "req"))


@pytest.mark.parametrize("status,terminal", [("FAILED", True), ("CANCELED", True), ("UNKNOWN", False)])
def test_task_errors(status, terminal):
    with VapeurClient(settings()) as client:
        client.describe_task = mock.Mock(return_value={"output": {"task_status": status}})
        with pytest.raises(VapeurTaskError) as exc:
            client.wait_for_task("task")
    assert exc.value.terminal is terminal
    assert exc.value.task_id == "task"


def test_timeout_and_http_error_preserve_task_id():
    with VapeurClient(settings()) as client:
        client.describe_task = mock.Mock(return_value={"output": {"task_status": "RUNNING"}})
        with mock.patch("wan3_api.vapeur.time.monotonic", side_effect=[0, 3601]), pytest.raises(VapeurTaskError) as exc:
            client.wait_for_task("task")
        assert not exc.value.terminal
        client.describe_task.side_effect = VapeurError("HTTP 500")
        with pytest.raises(VapeurTaskError, match="task") as exc:
            client.wait_for_task("task")
        assert not exc.value.terminal


def test_api_error_redaction_and_no_post_retry():
    with VapeurClient(settings()) as client:
        response = mock.Mock(status_code=500)
        response.json.return_value = {"error": {"message": "bad test-secret"}}
        client.session.request = mock.Mock(return_value=response)
        with pytest.raises(VapeurSubmissionUncertain) as exc:
            client.create_video(request(), prompt_required=True)
        assert "test-secret" not in str(exc.value)
        assert client.session.request.call_count == 1
        client.session.request.side_effect = requests.Timeout()
        with pytest.raises(VapeurSubmissionUncertain, match="Do not automatically resubmit"):
            client.create_video(request(), prompt_required=True)


@pytest.mark.parametrize("wait,status", [(False, "RUNNING"), (False, "SUCCEEDED"), (True, "SUCCEEDED")])
def test_query_node_routes_to_vapeur(wait, status):
    detail = {"output": {"task_status": status}}
    if status == "SUCCEEDED":
        detail["output"]["video_url"] = "https://host/v.mp4?token=secret"
    with mock.patch.object(nodes, "load_json_config", return_value={"provider": "vapeur", "vapeur_api_key": "test"}), mock.patch.object(VapeurClient, "describe_task", return_value=detail), mock.patch.object(nodes, "TencentVodClient") as tencent:
        result = nodes.WanQueryTask().query("task", wait)
    assert result[0] == status
    assert "token=secret" not in result[4]
    tencent.assert_not_called()


@pytest.mark.parametrize("error,deletes", [
    (VapeurTaskError("failed", "task", True), 1),
    (VapeurTaskError("timeout", "task", False), 0),
    (VapeurSubmissionUncertain("unknown"), 0),
])
def test_vapeur_oss_retention(error, deletes):
    oss = mock.MagicMock()
    oss.__enter__.return_value = oss
    oss.upload.return_value = UploadedObject("tmp/x", "https://signed")
    client = mock.MagicMock()
    client.__enter__.return_value = client
    if isinstance(error, VapeurSubmissionUncertain):
        client.create_video.side_effect = error
    else:
        client.create_video.return_value = TaskSubmission("task", "req")
        client.wait_for_task.side_effect = error
    with (
        mock.patch.object(nodes, "load_json_config", return_value={"provider": "vapeur", "vapeur_api_key": "test"}),
        mock.patch.object(nodes, "load_oss_config", return_value=config.OssConfig("host", "id", "secret", "bucket", "tmp", 86400, True)),
        mock.patch.object(nodes, "OssClient", return_value=oss),
        mock.patch.object(nodes, "VapeurClient", return_value=client),
        pytest.raises(type(error)),
    ):
        nodes._generate(request(), [(MediaBlob(b"x", "image/jpeg", "jpg"), "Image", "FirstFrame")], prompt_required=True)
    assert oss.delete.call_count == deletes


@pytest.mark.parametrize("failure", [requests.ReadTimeout(), requests.ConnectionError(), 408, 429, 500, 502, 503, 504])
def test_transient_query_errors_retry_without_resubmitting(failure):
    success = mock.Mock(status_code=200)
    success.json.return_value = {"output": {"task_status": "SUCCEEDED"}}
    failed = mock.Mock(status_code=failure) if isinstance(failure, int) else failure
    with VapeurClient(settings()) as client, mock.patch("wan3_api.vapeur.time.sleep") as sleep:
        client.session.request = mock.Mock(side_effect=[failed, success])
        assert client.wait_for_task("existing-task")["output"]["task_status"] == "SUCCEEDED"
        assert [call.args[0] for call in client.session.request.call_args_list] == ["GET", "GET"]
        assert all(call.args[1].endswith("/existing-task") for call in client.session.request.call_args_list)
        sleep.assert_called_once_with(5)


def test_query_retries_are_bounded_and_nonterminal():
    with VapeurClient(settings()) as client, mock.patch("wan3_api.vapeur.time.sleep") as sleep:
        client.session.request = mock.Mock(side_effect=requests.ReadTimeout())
        with pytest.raises(VapeurTaskError, match="existing-task") as exc:
            client.wait_for_task("existing-task")
        assert not exc.value.terminal
        assert "submission outcome" not in str(exc.value)
        assert client.session.request.call_count == 6
        assert [call.args[0] for call in sleep.call_args_list] == [5, 10, 20, 30, 30]


def test_retry_stops_at_wait_deadline():
    with VapeurClient(settings()) as client, mock.patch("wan3_api.vapeur.time.sleep") as sleep:
        client.describe_task = mock.Mock(side_effect=VapeurTransientError("timeout"))
        with mock.patch("wan3_api.vapeur.time.monotonic", side_effect=[0, 3599, 3600]):
            with pytest.raises(VapeurTaskError, match="timed out") as exc:
                client.wait_for_task("task")
        assert not exc.value.terminal
        assert client.describe_task.call_count == 1
        sleep.assert_called_once_with(1)


def test_successful_poll_resets_consecutive_errors():
    with VapeurClient(settings()) as client, mock.patch("wan3_api.vapeur.time.sleep") as sleep:
        client.describe_task = mock.Mock(side_effect=[
            VapeurTransientError("timeout"), {"output": {"task_status": "RUNNING"}},
            VapeurTransientError("timeout"), {"output": {"task_status": "SUCCEEDED"}},
        ])
        client.wait_for_task("task")
        assert [call.args[0] for call in sleep.call_args_list] == [5, 5, 5]


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_permanent_query_error_is_not_retried(status):
    response = mock.Mock(status_code=status)
    response.json.return_value = {"error": {"message": "invalid request"}}
    with VapeurClient(settings()) as client, mock.patch("wan3_api.vapeur.time.sleep") as sleep:
        client.session.request = mock.Mock(return_value=response)
        with pytest.raises(VapeurTaskError):
            client.wait_for_task("task")
        assert client.session.request.call_count == 1
        sleep.assert_not_called()
