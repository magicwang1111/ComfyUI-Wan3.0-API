import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_examples_are_valid_and_secret_free():
    examples = sorted((REPO_ROOT / "examples").glob("*.json"))
    assert len(examples) == 6
    forbidden = ("oss_access_key_secret", "tencent_secret_key", "Authorization")
    for path in examples:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
        assert payload["version"] == 0.4
        assert payload["nodes"]
        assert not any(token in text for token in forbidden)

