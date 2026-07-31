# Traffic Accident Monitoring Prototype

This folder contains the small backend/frontend prototype used in Chapter 6.
It is not a production traffic platform. The goal is to show that the trained
VADCLIP model can be plugged into a simple monitoring workflow: upload a video
segment, run inference, create an event, and save evidence for review.

## What It Does

- Receives uploaded video segments through `POST /api/segments`.
- Can read enabled camera/file sources and push one recent segment per camera.
- Stores segment metadata in SQLite.
- Processes queued segments with the VADCLIP backend.
- Supports `pth`, `onnx`, and `mock` inference modes.
- Uses the A2 score timeline by default.
- Creates an event when `max_score >= threshold_accident`.
- Saves an evidence package with clip, thumbnail, metadata, scores, hash, and
  source segment reference.
- Shows events in a simple HTML dashboard.

For enabled camera sources, each camera keeps its own timer. With the default
configuration, the worker writes the latest 60-second window as a segment every
50 seconds and then sends that segment to the same processing queue used by
manual uploads. This keeps camera capture separate from model inference and
avoids running the model on every frame continuously.

## Files

```text
system/
  backend/
    app/                  FastAPI app
    configs/settings.yaml main runtime config
    configs/cameras.yaml  sample camera entries
    run.py                backend entrypoint
  frontend/index.html     dashboard page
```

Runtime files are created under `system/backend/storage/` and
`system/backend/accident_system.db`. These files are not meant to be submitted.

## Model Backend

The backend is selected in `backend/configs/settings.yaml`:

```yaml
model:
  backend: "pth"        # pth, onnx, or mock
  score_source: "a2"    # a1, a2, or fusion
```

For direct PyTorch checkpoint inference:

```yaml
model:
  backend: "pth"
  pth_model_path: "../../models/model_current.pth"
  pth_device: "auto"
  clip_download_root: "../../models/clip_cache"
```

For ONNX inference, use:

```text
models/clip_vit_b16_image.onnx
models/vadclip_2class.onnx
```

```yaml
model:
  backend: "onnx"
  onnx_provider: "auto"
  clip_onnx_path: "../../models/clip_vit_b16_image.onnx"
  vad_onnx_path: "../../models/vadclip_2class.onnx"
  score_source: "a2"
  inference_mode: 2
  spatial_top_k: 3
```

`mock` is only for checking the dashboard and storage workflow when model files
are not available. It should not be used for reporting model predictions.

## Run

From `final_code`:

```powershell
cd system\backend
pip install -r requirements.txt
python run.py
```

Open:

```text
http://localhost:8000
```

## Upload a Segment

```powershell
curl.exe -X POST "http://localhost:8000/api/segments" `
  -F "camera_id=CAM_TEST_001" `
  -F "file=@..\..\data\sample_videos\demo.mp4"
```

Useful endpoints:

```text
GET  /api/cameras
GET  /api/segments
GET  /api/events
GET  /api/events/{event_id}
GET  /api/events/{event_id}/clip
GET  /api/events/{event_id}/scores
POST /api/events/{event_id}/confirm
POST /api/events/{event_id}/false-alarm
```

## Evidence Output

Detected accident events are stored as:

```text
storage/evidence/{camera_id}/{YYYY-MM-DD}/{event_id}/
  accident_clip.mp4
  thumbnail.jpg
  metadata.json
  scores.csv
  sha256.txt
  source_segment_ref.txt
```

Normal segments are deleted after processing by default. Their metadata stays
in the database.

## Reset Local Test Data

Stop the backend first, then run from `system/backend`:

```powershell
Remove-Item -LiteralPath .\accident_system.db -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath .\storage -Recurse -Force -ErrorAction SilentlyContinue
```

The database and storage folders will be recreated on the next run.

## Notes

- `threshold_accident` and evidence clip settings are in `settings.yaml`.
- `segment_seconds` controls the length of each camera-pushed segment.
- `segment_interval_seconds` controls how often each camera sends a segment.
- `camera_id` is used for grouping segments and events.
- This prototype uses SQLite and an in-process worker, which is enough for the
  thesis demo but should be replaced for a real multi-camera deployment.
