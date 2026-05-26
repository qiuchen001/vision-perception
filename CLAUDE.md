# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Chinese-language automotive driving video intelligence platform. It ingests dashcam videos, runs a multi-agent LLM-based scene analysis pipeline ("scene mining"), generates textual summaries and semantic tags, stores everything as vectors in Milvus, and exposes multi-modal search across the video corpus.

## Development Commands

**Local setup:**
```bash
conda create -y -p ./venv python=3.12
conda activate ./venv
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
cp .env_sample .env
# Edit .env with real credentials
python app.py
```

**Production (gunicorn):**
```bash
gunicorn --workers 2 --threads 8 --bind 0.0.0.0:30501 --timeout 3600 wsgi:application
```

**Docker (full stack with GPU VLMs):**
```bash
cp .env_sample .env
./scripts/start_docker_auto_gpu.sh  # selects idle GPU via nvidia-smi, then docker compose up --build
```

**Database initialization and backfill:**
```bash
python -m app.scripts.create_database
python -m app.scripts.backfill_features --limit 10   # trial run
python -m app.scripts.backfill_features              # full backfill
python -m app.scripts.cleanup_runtime --dry-run
python -m app.scripts.cleanup_runtime --max-age-hours 24
```

**Tests** (no runner config; run ad-hoc):
```bash
python -m pytest app/tests/
python app/utils/embedding/test_embedding.py
```

There is no linter configuration in this repo.

## Architecture

```
Browser/Client
      |
   Flask (app.py, port 30501) — all routes defined directly in app.py
      |
  ┌──────────────────────────────┐
  │  Services (app/services/)    │
  │  AddVideoService             │
  │   ├─ MiningVideoService      │ ← LangGraph multi-agent pipeline
  │   ├─ SummaryVideoService     │ ← VLM natural-language summary
  │   └─ VideoFeatureService     │ ← embedding + Milvus upsert
  │  SearchService / IntentService│
  │  UploadService               │ ← MinIO + stub record
  └──────────────────────────────┘
      |
  ┌──────────────────────────────┐
  │  DAO (app/dao/)              │
  │  VideoDAO     → primary Milvus collection (video metadata + embeddings)
  │  FeatureDAO   → text_features + visual_features collections
  └──────────────────────────────┘
      |
  External services (see docker-compose.yml):
  ├── MinIO (S3-compatible)         — video + thumbnail storage
  ├── Milvus (port 19530)           — vector DB, all collections
  ├── vLLM VLM (port 8576)          — scene mining (Qwen3.5-9B-Gemini-Distill)
  └── vLLM Embedding (port 8575)    — Qwen3-VL-Embedding-2B (2048-dim)
```

**Video processing pipeline (`POST /api/add`):**
1. `MiningVideoService` → `app/algorithm/scene_mining/adapter.py` → LangGraph DAG → VLM at port 8576 → `pred` dict with scene labels + `abnormal_event_times`
2. `SummaryVideoService` → sends mining results to VLM → natural-language summary
3. `VideoFeatureService` → ffmpeg extracts frames → embeds via Qwen3-VL-Embedding → upserts to `video_visual_features` and `video_text_features`
4. `VideoDAO.upsert_video()` → writes final record to primary Milvus collection

## Key Files

| File | Purpose |
|---|---|
| `app.py` | Flask app factory + all route definitions |
| `config/config.py` | Central `Config` class wrapping all env vars |
| `app/algorithm/scene_mining/config-qwen-gemini.yaml` | Scene mining pipeline config: VLM endpoint, FPS, category list, concurrency |
| `docker-compose.yml` | Three services: VLM (8576), embedding (8575), Flask app (30501) |
| `.env_sample` | Template for all runtime env vars — copy to `.env` |

## Critical Environment Variables

- `MILVUS_HOST/PORT/DB_NAME` — Milvus connection
- `MILVUS_VIDEO_COLLECTION_NAME` — primary video collection (not in `.env_sample`, must be set manually)
- `OSS_ENDPOINT/ACCESS_KEY/SECRET_KEY/BUCKET_NAME` — MinIO
- `QWEN3_VL_EMBEDDING_BASE_URL` — embedding service (default: `http://localhost:8575`)
- `SCENE_MINING_API_BASE_URL/MODEL_NAME` — VLM endpoint
- `SCENE_MINING_CONFIG_PATH` — path to the YAML config above
- `DASHSCOPE_API_KEY` — DashScope/Qwen API key for summary service

## Important Patterns

**Three add-video delivery modes:** synchronous (`POST /api/add`), NDJSON streaming (`POST /api/add/stream`), and async poll (`POST /api/add/task` + `GET /api/add/task/<id>`). Status is persisted as JSON files in `TASK_STATUS_DIR`.

**Search modes** (`search_type` on `POST /api/search`): `text`, `image`, `tags`, `filter`, and `smart`. The `smart` mode uses `IntentService` to classify the query then merges results. Text search supports `search_mode` sub-options: `frame`, `summary`, `tags`, `visual`.

**Embedding factory** (`app/utils/embedding/embedding_factory.py`): Singleton-per-type factory with four backends (`CLIP`, `MULTIMODAL`, `JINA_CLIP_V2`, `QWEN3_VL`). Active model controlled by `EMBEDDING_MODEL` env var (default: `qwen3-vl`).

**Scene mining LangGraph pipeline** (`app/algorithm/scene_mining/`): Multi-agent DAG — supervisor fans out to simple workers (time-of-day, weather, road type) and complex workers (pedestrian anomalies, trajectory conflicts) with optional YOLO pre-filtering. Prompts per category are in `app/algorithm/scene_mining/skills/`.

**Concurrency control:** `threading.BoundedSemaphore` limits simultaneous video processing jobs; limit comes from `concurrency.max_concurrent_videos` in the YAML config.

**Media proxy:** All MinIO URLs are normalized to `/media/<bucket>/<object>`. The Flask app caches them locally with `fcntl.LOCK_EX` file locks and serves with Range-request support for in-browser video playback.

**`app/routes/main.py`** defines a Blueprint stub but is **not registered** in `app.py` — all active routes are in `app.py` directly.

## Milvus Collections

1. **Primary video collection** (`MILVUS_VIDEO_COLLECTION_NAME`) — video metadata + two FLOAT_VECTOR(2048) fields (`embedding`, `summary_embedding`) with COSINE FLAT index
2. **`video_text_features`** — one row per `(m_id, feature_type)` where `feature_type` is `"tags"` or `"summary"`
3. **`video_visual_features`** — one row per video; weighted-average-pooled frame embedding
4. **Frame-level collection** (`MILVUS_VIDEO_FRAME_COLLECTION_NAME`) — per-frame vectors for `search_mode=frame`
