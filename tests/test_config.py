import json
from pathlib import Path
from unittest import mock

import pytest

from wan3_api import config


def test_bom_json_and_environment_fallback(tmp_path):
    path = tmp_path / "config.local.json"
    path.write_text(json.dumps({"tencent_secret_id": "json-id"}), encoding="utf-8-sig")
    data = config.load_json_config(path)
    with mock.patch.dict(
        "os.environ",
        {
            "TENCENTCLOUD_SECRET_ID": "env-id",
            "TENCENTCLOUD_SECRET_KEY": "env-key",
            "TENCENT_SUB_APP_ID": "123",
        },
        clear=True,
    ):
        loaded = config.load_tencent_config(data)
    assert loaded.secret_id == "json-id"
    assert loaded.secret_key == "env-key"
    assert loaded.sub_app_id == 123


def test_primary_config_overrides_legacy_and_reloads(tmp_path):
    legacy = tmp_path / "config.local.json"
    primary = tmp_path / "config.json"
    legacy.write_text(json.dumps({"provider": "tencent", "oss_bucket": "keep", "kuaizi_api_key": "local-key"}), encoding="utf-8")
    primary.write_text(json.dumps({"provider": "kuaizi", "kuaizi_api_key": ""}), encoding="utf-8-sig")
    with mock.patch.object(config, "CONFIG_PATH", legacy), mock.patch.object(config, "PRIMARY_CONFIG_PATH", primary):
        data = config.load_json_config()
        assert data == {"provider": "kuaizi", "oss_bucket": "keep", "kuaizi_api_key": "local-key"}
        primary.write_text(json.dumps({"provider": "vapeur"}), encoding="utf-8")
        assert config.load_provider() == "vapeur"
        assert config.load_json_config(legacy)["provider"] == "tencent"


def test_legacy_config_without_primary(tmp_path):
    legacy = tmp_path / "config.local.json"
    legacy.write_text('{"provider": "vapeur"}', encoding="utf-8")
    with mock.patch.object(config, "CONFIG_PATH", legacy), mock.patch.object(config, "PRIMARY_CONFIG_PATH", tmp_path / "missing.json"):
        assert config.load_provider() == "vapeur"


def test_empty_json_value_falls_back_to_environment():
    with mock.patch.dict(
        "os.environ",
        {
            "OSS_ACCESS_KEY_ID": "env-id",
            "OSS_ACCESS_KEY_SECRET": "env-secret",
            "OSS_BUCKET": "bucket",
        },
        clear=True,
    ):
        loaded = config.load_oss_config({"oss_access_key_id": ""})
    assert loaded.access_key_id == "env-id"
    assert loaded.bucket == "bucket"


@pytest.mark.parametrize("prefix", ["../bad", "/", "./bad"])
def test_unsafe_oss_prefix_is_rejected(prefix):
    data = {
        "oss_access_key_id": "id",
        "oss_access_key_secret": "secret",
        "oss_bucket": "bucket",
        "oss_prefix": prefix,
    }
    with pytest.raises(ValueError, match="oss_prefix"):
        config.load_oss_config(data)


def test_local_config_is_ignored():
    ignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert "config.local.json" in ignore.splitlines()


def test_oss_endpoint_rejects_paths():
    data = {
        "oss_endpoint": "oss-cn-hangzhou.aliyuncs.com/path",
        "oss_access_key_id": "id",
        "oss_access_key_secret": "secret",
        "oss_bucket": "bucket",
    }
    with pytest.raises(ValueError, match="hostname"):
        config.load_oss_config(data)
