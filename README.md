# ComfyUI Wan 3.0 API

ComfyUI nodes for Wan 3.0 and Wan 3.0 Prime video generation through Tencent VOD, Vapeur, Kuaizi, or OPC. Select the provider in `config.json` (legacy `config.local.json` is also supported). Local images, videos, and audios use a private Alibaba Cloud OSS prefix. The Preview node saves the original MP4 under ComfyUI `output/video`.

Video generation is a paid operation on all providers. OSS hosts input media separately from the Wan generation charge.

## Security first

Do not reuse credentials that have appeared in chat, logs, screenshots, or source control. Rotate them first, then place the new values only in `config.json` or `config.local.json`. Both files are ignored by Git.

Use an Alibaba Cloud RAM identity restricted to `PutObject` and `DeleteObject` under the configured temporary prefix. Keep the bucket private. Configure an OSS lifecycle rule that deletes objects under `<oss_prefix>/wan3/` after one day; signed URL expiry does not delete objects.

## Provider selection

Both configuration files are read on each generation/query. Non-empty values in `config.json` override the same keys in `config.local.json`; other legacy settings, including OSS credentials, remain available. Set `provider` to `tencent`, `vapeur`, `kuaizi`, or `opc`; omitting it preserves Tencent behavior (unless `WAN3_PROVIDER` is set). Restart ComfyUI once after installing this code update. Subsequent configuration changes do not require a restart, but generation nodes must execute again to avoid ComfyUI reusing a cached result.

### OPC

Add these fields to `config.local.json`. Remove any `provider` or `opc_*` overrides from `config.json` if this local file should control OPC selection.

```json
{
  "provider": "opc",
  "opc_base_url": "https://model-router.edu-aliyun.com",
  "opc_api_key": "YOUR_OPC_API_KEY",
  "opc_poll_interval": 5,
  "opc_request_timeout": 120,
  "opc_max_wait_seconds": 3600
}
```

OPC submits the native Wan `model` / `input` / `parameters` JSON structure to `POST /v1/videos/generations` with Bearer authentication and `X-DashScope-Async: enable`. Models `3.0` and `3.0-prime` map to `qwen/wan3.0-video/v1` and `qwen/wan3.0-video-prime/v1`, as listed by this platform's `/v1/models` endpoint. Non-empty JSON values override `OPC_API_KEY`, `OPC_BASE_URL`, `OPC_POLL_INTERVAL`, `OPC_REQUEST_TIMEOUT`, and `OPC_MAX_WAIT_SECONDS`.

Text To Video requires a prompt. Frame To Video supports a first frame or a first/last-frame pair. Reference To Video supports up to 10 images, 5 videos, and 5 audios, including audio-only input. Prompt is optional when media is supplied. Local media uses the existing OSS upload and cleanup settings, so existing single-image OPC workflows now require OSS configuration too; text-only generation needs no OSS.

Reference To Video also accepts an OPC-only `reference_file_url` (public document URL) or `reference_link_url` (public webpage URL). File and link are mutually exclusive; either can accompany image/video/audio references. Set `enhance_prompt=Enabled` when using file/link. These URLs are sent directly without uploading local files. First/last frames cannot be mixed with reference media, files, or links; a last frame requires a first frame.

Resolution and aspect ratio map to `parameters.resolution` and `parameters.ratio`, retaining native `adaptive` behavior. Duration supports fixed 2-30 seconds and smart duration (`-1`). `audio_generation` controls `parameters.audio`, `enhance_prompt` controls `parameters.prompt_extend`, and a nonnegative seed maps to `parameters.seed` (`-1` omits it). Watermark is disabled. Keep `super_resolution=Disabled` and `negative_prompt` empty. Existing local-media validation limits and aspect-ratio choices below still apply; this update does not expose every upstream limit or ratio.

The supplied async-task documentation confirms `GET /v1/tasks/{task_id}` with `Authorization: Bearer ...` and `Content-Type: application/json`. The task ID returned by creation is a path parameter, not a JSON body; the client URL-encodes it without adding or removing a provider prefix. Queries have no request body. The client retries temporary query failures and never automatically resubmits generation.

Generation nodes automatically poll after submission. To resume an existing task, use Query Task with the original `task_id` and `provider=opc`: enable `wait_for_completion` to poll every 5 seconds by default (up to 3600 seconds), or disable it for one query. Connect its `video_url` output to Preview Video to download a completed result. Query Task must use the task's original provider.

The [official Wan 3.0 API reference](https://docs.bailian.console.aliyun.com/zh/model-studio/wan3-video-generation-api-reference.md) documents the native media and parameter fields. On 2026-09-23, invalid-input diagnostics confirmed OPC forwards nested `input.media` and returns prefixed task IDs plus native `output.task_status`, `output.code`, and `output.message`. Both diagnostic tasks reached `FAILED` without producing video. Upstream `request_id` and router `mr_req_id` are retained in errors. Successful multimodal generation and download remain unverified. Download successful video URLs promptly: upstream task IDs and result URLs are valid for 24 hours.

### Kuaizi

Set these fields in `config.json` to use Kuaizi with your own enabled API key:

```json
{
  "provider": "kuaizi",
  "kuaizi_base_url": "https://aiopenapi.kuaizi.cn",
  "kuaizi_api_key": "YOUR_KUAIZI_API_KEY",
  "kuaizi_poll_interval": 15,
  "kuaizi_request_timeout": 120,
  "kuaizi_max_wait_seconds": 3600
}
```

The base URL accepts either the origin above or the same URL ending in `/ai-open-platform-api`; that prefix is added exactly once. The client sends `Authorization: Bearer ...` and `X-DashScope-Async: enable` when creating tasks. Creation uses `POST /api/v1/services/aigc/video-generation/video-synthesis`; queries use `GET /api/v1/tasks/{task_id}` under that prefix.

Node model `3.0` maps to `wan3.0-video`, and `3.0-prime` maps to `wan3.0-video-prime`, with no region suffix. Existing text, first/last-frame, and image/video/audio reference nodes are supported. Local media still uses the existing OSS configuration; text-only generation does not require OSS. This integration does not add file/webpage input nodes.

Keep `super_resolution` and `enhance_prompt` set to `Disabled`, and `negative_prompt` empty: these options are not in the supplied Kuaizi v1.2 protocol and are rejected before upload rather than silently ignored. `audio_generation` maps to `parameters.audio`; watermark is disabled. Fixed/smart duration, aspect ratio, resolution, and optional seed retain the existing node controls.

Polling defaults to 15 seconds (minimum 15). Temporary query failures retry with bounded backoff, without resubmitting generation. Creation errors include the platform error code and `request_id`; `429 Throttling` requires waiting for capacity, while `429 InsufficientBalance` requires topping up. Creation is never automatically retried, avoiding duplicate paid jobs. HTTP 200 with `output.task_status=FAILED` is treated as a failed task. Inputs are retained if submission is uncertain or polling times out; they are cleaned after a known terminal state.

The Query Task node must use the task's original provider. It returns the Kuaizi task ID as both `video_id` and `task_id`. According to the supplied platform documentation, the first successful URL may be temporary; download it promptly or query the same task later for the permanent stored URL. Kuaizi bills successful tasks by input-video plus output-video duration.

Environment alternatives: `KUAIZI_API_KEY`, `KUAIZI_BASE_URL`, `KUAIZI_POLL_INTERVAL`, `KUAIZI_REQUEST_TIMEOUT`, and `KUAIZI_MAX_WAIT_SECONDS`. Non-empty JSON settings take precedence.

### Vapeur

Add these fields to your existing local configuration to use Vapeur:

```json
{
  "provider": "vapeur",
  "vapeur_base_url": "https://api.vapeur.ai",
  "vapeur_api_key": "YOUR_API_KEY",
  "vapeur_region": "cn",
  "vapeur_poll_interval": 5,
  "vapeur_request_timeout": 120,
  "vapeur_max_wait_seconds": 3600
}
```

The base URL must not end in `/v1`. Keep existing Tencent and OSS fields in the same file; only the selected provider's credentials are required. Text-to-video does not require OSS.

| Node model_version | vapeur_region | API model |
| --- | --- | --- |
| `3.0` | `cn` | `wan3.0-video-cn` |
| `3.0-prime` | `cn` | `wan3.0-video-prime-cn` |
| `3.0` | `glb` | `wan3.0-video-glb` |
| `3.0-prime` | `glb` | `wan3.0-video-prime-glb` |

Vapeur supports native 480P/720P/1080P. Keep `super_resolution` set to `Disabled`; 2K/4K requests fail before media upload. `audio_generation` maps to `parameters.audio`, `enhance_prompt` to `parameters.prompt_extend`, and watermark is disabled. Existing frame/reference nodes map their OSS URLs to `first_frame`, `last_frame`, `reference_image`, `reference_video`, and `reference_audio`.

The Query Task node uses the configured provider; switch back to the task's original provider when recovering an older task. Vapeur returns `PENDING`, `RUNNING`, or `SUCCEEDED`; failed/canceled/unknown tasks raise an actionable error. The returned `video_id` equals `task_id` because Vapeur does not provide a separate video ID. A submitted task ID is printed immediately for recovery. Polling errors/timeouts and uncertain submissions retain input objects for OSS lifecycle cleanup; submissions are never automatically retried. While waiting for completion, query timeouts, connection errors, and HTTP 408/429/500/502/503/504 are retried up to five consecutive times with exponential backoff capped at 30 seconds, within the configured total wait limit. A successful query resets the retry count; authentication and other permanent errors fail immediately. Single-query mode still performs only one request.

Vapeur API references: [submit video](https://vapeur.apifox.cn/508059339e0), [query video](https://vapeur.apifox.cn/499420766e0). Submission uses `POST /qwen/api/v1/services/aigc/video-generation/generation`; querying uses `GET /qwen/api/v1/tasks/video/{taskId}`.

## Configuration

For a fresh install, copy `config.example.json` to `config.json` and fill in rotated credentials. Existing `config.local.json` does not need to be renamed or copied:

```json
{
  "oss_endpoint": "oss-cn-hangzhou.aliyuncs.com",
  "oss_access_key_id": "",
  "oss_access_key_secret": "",
  "oss_bucket": "example-bucket",
  "oss_prefix": "ComfyUI-tmp",
  "oss_signed_url_expires": 86400,
  "oss_cleanup_after_task": true,
  "tencent_secret_id": "",
  "tencent_secret_key": "",
  "tencent_region": "ap-guangzhou",
  "tencent_sub_app_id": 0,
  "tencent_poll_interval": 5.0,
  "tencent_request_timeout": 120,
  "tencent_max_wait_seconds": 3600,
  "tencent_storage_mode": "Temporary",
  "tencent_input_compliance_check": "Enabled",
  "tencent_output_compliance_check": "Enabled"
}
```

The same values can be supplied through the environment variables documented in `config.example.json` and `config.py`. Non-empty JSON values take precedence over environment variables. JSON is read as `utf-8-sig`, so a Windows UTF-8 BOM is accepted.

`oss_signed_url_expires` must be at least the selected provider's `max_wait_seconds + 600`. Input objects are deleted after the task reaches a terminal state. If local polling times out while the remote task is still running, the objects are retained for the OSS lifecycle rule instead of being deleted underneath the task.

## Nodes

- **Wan 3.0 API Text To Video** supports `3.0` and `3.0-prime`, native 480P/720P/1080P output, optional Tencent-only 2K/4K super-resolution, ratios from the supplied Wan guide, and smart (`-1`) or fixed 2-30 second duration. It does not use OSS.
- **Wan 3.0 API Frame To Video** accepts a first frame, last frame, or both (OPC requires a first frame when using a last frame). Frames are uploaded to OSS and submitted as `FirstFrame` / `LastFrame`; aspect ratio is fixed to `adaptive`.
- **Wan 3.0 API Reference To Video** accepts an IMAGE batch of up to 10 images, five VIDEO sockets, and five AUDIO sockets. OPC additionally supports audio-only references and public file/webpage URLs. Reference video and audio are each limited to 15 seconds total.
- **Wan 3.0 API Query Task** performs one status query or waits for completion, allowing recovery from an interrupted workflow.
- **Wan 3.0 API Preview Video** downloads the temporary result without re-encoding and shows a responsive video preview.

All generation nodes return `video_url`, `video_id`, and `task_id`. Keep the TaskId when diagnosing or recovering an interrupted job.

All generation nodes expose `audio_generation`, defaulting to `Enabled`. Tencent receives this as `OutputConfig.AudioGeneration`; set it to `Disabled` for video without a generated audio track. The optional `seed` remains available: use a fixed value for more reproducible output, `-1` to omit the field, or ComfyUI's control-after-generate setting to randomize/increment it between runs.

## Limits

- Prompt: at most 20,000 characters.
- Image: 256-5760 pixels per side, aspect ratio 0.4-2.5, at most 10 MB after JPEG conversion.
- Reference video: 2-15 seconds each, 50 MB each, at most 15 seconds total.
- Reference audio: 2-15 seconds each, 15 MB each, at most 15 seconds total.
- With reference video, input video duration plus requested output duration must not exceed 30 seconds.
- Set `duration` to `-1` for smart duration, allowing the model to choose a suitable length from the prompt and reference media. Fixed duration choices remain 2-30 seconds.
- `resolution` contains only native 480P/720P/1080P choices. Set `super_resolution` to `2K` or `4K` to request Tencent's super-resolution output; leave it `Disabled` to send the native resolution unchanged.

## Install and test

Install `requirements.txt`, restart ComfyUI, and import a workflow from `examples/`.

Run the offline suite from this repository:

```powershell
D:\miniconda3\envs\comfyui\python.exe -m pytest -q tests --rootdir=tests --import-mode=importlib
```

The paid smoke test is disabled unless `--confirm-paid` is provided:

```powershell
D:\miniconda3\envs\comfyui\python.exe scripts\live_smoke.py --confirm-paid
```

The smoke script performs one 480P, 2-second text-to-video request using the configured provider. Frame, tail-frame, and multimodal checks remain manual because every creation request can incur a charge.

## Documentation notes

The supplied VOD guide's 2026-09-01 update identifies `ModelName=Wan` and versions `3.0` / `3.0-prime`. A later Wan section contains copied Hailuo/H3 values; this implementation follows the Wan update record and the generic `CreateAigcVideoTask` wire contract. Tail-frame-only generation is implemented from that update and should be verified against the enabled Tencent account before production use.
