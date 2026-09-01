from __future__ import annotations

import base64
import datetime as dt
import email.utils
import hashlib
import hmac
import time
import urllib.parse

import requests

from .config import OssConfig
from .models import MediaBlob, UploadedObject


class OssError(RuntimeError):
    pass


class OssClient:
    def __init__(self, config: OssConfig):
        self.config = config
        self.session = requests.Session()

    def close(self) -> None:
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _resource(self, object_key: str) -> str:
        return f"/{self.config.bucket}/{object_key}"

    def _object_url(self, object_key: str) -> str:
        quoted = "/".join(urllib.parse.quote(part, safe="") for part in object_key.split("/"))
        return f"https://{self.config.bucket}.{self.config.endpoint}/{quoted}"

    def _signature(self, value: str) -> str:
        digest = hmac.new(
            self.config.access_key_secret.encode("utf-8"),
            value.encode("utf-8"),
            hashlib.sha1,
        ).digest()
        return base64.b64encode(digest).decode("ascii")

    def _authorization(self, method: str, object_key: str, content_type: str, date: str) -> str:
        string_to_sign = f"{method}\n\n{content_type}\n{date}\n{self._resource(object_key)}"
        return f"OSS {self.config.access_key_id}:{self._signature(string_to_sign)}"

    @staticmethod
    def _request_id(response: requests.Response) -> str:
        return response.headers.get("x-oss-request-id", "-")

    def upload(self, object_key: str, blob: MediaBlob, timeout: int = 120) -> UploadedObject:
        date = email.utils.format_datetime(dt.datetime.now(dt.timezone.utc), usegmt=True)
        headers = {
            "Authorization": self._authorization("PUT", object_key, blob.content_type, date),
            "Content-Type": blob.content_type,
            "Date": date,
        }
        response = self.session.put(
            self._object_url(object_key),
            headers=headers,
            data=blob.data,
            timeout=timeout,
        )
        if response.status_code not in {200, 201}:
            raise OssError(
                f"OSS upload failed: HTTP {response.status_code}, "
                f"RequestId={self._request_id(response)}."
            )
        return UploadedObject(object_key=object_key, url=self.signed_get_url(object_key))

    def signed_get_url(self, object_key: str, now: int | None = None) -> str:
        expires = int(now if now is not None else time.time()) + self.config.signed_url_expires
        string_to_sign = f"GET\n\n\n{expires}\n{self._resource(object_key)}"
        query = urllib.parse.urlencode({
            "OSSAccessKeyId": self.config.access_key_id,
            "Expires": str(expires),
            "Signature": self._signature(string_to_sign),
        })
        return f"{self._object_url(object_key)}?{query}"

    def delete(self, object_key: str, timeout: int = 30) -> None:
        date = email.utils.format_datetime(dt.datetime.now(dt.timezone.utc), usegmt=True)
        headers = {
            "Authorization": self._authorization("DELETE", object_key, "", date),
            "Date": date,
        }
        response = self.session.delete(self._object_url(object_key), headers=headers, timeout=timeout)
        if response.status_code not in {200, 204, 404}:
            raise OssError(
                f"OSS cleanup failed: HTTP {response.status_code}, "
                f"RequestId={self._request_id(response)}."
            )

