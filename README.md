# ComfyUI Wan 3.0 API

ComfyUI nodes for Wan 3.0 and Wan 3.0 Prime video generation through Tencent VOD or Vapeur. Select the provider in `config.local.json`. Local images, videos, and audios are uploaded to a private Alibaba Cloud OSS prefix and passed to the selected provider as short-lived signed URLs. The Preview node saves the original MP4 under ComfyUI `output/video`.

Video generation is a paid operation on both providers. OSS hosts input media separately from the Wan generation charge.

## Security first

Do not reuse credentials that have appeared in chat, logs, screenshots, or source control. Rotate them first, then place the new values only in `config.local.json`. That file is ignored by Git.

Use an Alibaba Cloud RAM identity restricted to `PutObject` and `DeleteObject` under the configured temporary prefix. Keep the bucket private. Configure an OSS lifecycle rule that deletes objects under `<oss_prefix>/wan3/` after one day; signed URL expiry does not delete objects.

## Provider selection

`config.local.json` is read on each generation/query. Set `provider` to `tencent` or `vapeur`; omitting it preserves Tencent behavior. Restart ComfyUI once after installing this code update. Subsequent configuration changes do not require a restart, but generation nodes must execute again (for example, change the seed) to avoid ComfyUI reusing a cached result.

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

The Query Task node uses the configured provider; switch back to the task's original provider when recovering an older task. Vapeur returns `PENDING`, `RUNNING`, or `SUCCEEDED`; failed/canceled/unknown tasks raise an actionable error. The returned `video_id` equals `task_id` because Vapeur does not provide a separate video ID. A submitted task ID is printed immediately for recovery. Polling errors/timeouts and uncertain submissions retain input objects for OSS lifecycle cleanup; submissions are never automatically retried.

Vapeur API references: [submit video](https://vapeur.apifox.cn/508059339e0), [query video](https://vapeur.apifox.cn/499420766e0). Submission uses `POST /qwen/api/v1/services/aigc/video-generation/generation`; querying uses `GET /qwen/api/v1/tasks/video/{taskId}`.

## Configuration

Copy `config.example.json` to `config.local.json` and fill in rotated credentials:

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
- **Wan 3.0 API Frame To Video** accepts a first frame, last frame, or both. Frames are uploaded to OSS and submitted as `FirstFrame` / `LastFrame`; aspect ratio is fixed to `adaptive`.
- **Wan 3.0 API Reference To Video** accepts an IMAGE batch of up to 10 images, five VIDEO sockets, and five AUDIO sockets. Audio cannot be used alone. Reference video and audio are each limited to 15 seconds total.
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
