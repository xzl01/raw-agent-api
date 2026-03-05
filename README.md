# Raw Agent API

A Web API that processes Sony RAW (.ARW) files, uses an LLM Agent (via Tool Calling) to determine image adjustments, outputs edited JPGs, supports batch processing, and includes unit tests.

An interactive single-page frontend is bundled and served directly by the API server.

---

## Features

- **LLM-guided processing** — GPT-4o analyses EXIF metadata and recommends optimal exposure, contrast, saturation and white-balance adjustments.
- **Single & batch endpoints** — `/process` for one file, `/batch-process` for many.
- **Interactive frontend** — drag-and-drop web UI served at the root URL, no separate install needed.
- **One-click launcher** — `start.sh` sets up a virtual environment, installs dependencies and launches the server automatically.

---

## Quick Start (one-click)

```bash
# 1. Clone the repository
git clone https://github.com/xzl01/raw-agent-api.git
cd raw-agent-api

# 2. Set your OpenAI API key
export OPENAI_API_KEY=sk-...

# 3. Launch everything
chmod +x start.sh
./start.sh
```

The script will:
1. Detect your Python version (3.9+ required).
2. Create a `.venv` virtual environment if one doesn't exist.
3. Install all dependencies from `requirements.txt`.
4. Start the Uvicorn server on `http://127.0.0.1:8000`.
5. Open your default browser to the frontend automatically.

Optional environment variables:

| Variable | Default | Description |
|---|---|---|
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `8000` | Bind port |
| `OPENAI_API_KEY` | *(required)* | OpenAI key for the LLM agent |

---

## Manual Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY=sk-...
uvicorn main:app --reload
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Interactive frontend (HTML) |
| `GET`  | `/health` | Health check — returns `{"status":"ok"}` |
| `POST` | `/process` | Process a single `.ARW` file |
| `POST` | `/batch-process` | Process multiple `.ARW` files |
| `GET`  | `/docs` | Swagger / OpenAPI documentation |

### `POST /process`

```
Content-Type: multipart/form-data
Field: file  (binary, .ARW)
```

Response:
```json
{
  "filename": "photo.arw",
  "adjustments": {
    "exposure": 1.2,
    "contrast": 1.1,
    "saturation": 1.0,
    "white_balance_red": 1.05,
    "white_balance_blue": 0.95
  },
  "jpeg_b64": "<base64-encoded JPEG>",
  "message": "Processing successful."
}
```

### `POST /batch-process`

```
Content-Type: multipart/form-data
Field: files[]  (binary, multiple .ARW files)
```

Response:
```json
{
  "results": [ /* same shape as /process, one entry per file */ ]
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

All 39 tests should pass without a real OpenAI key or real `.ARW` files — the test suite mocks both.

---

## Project Structure

```
raw-agent-api/
├── main.py            # FastAPI app entry point, CORS, static-file mounting
├── start.sh           # One-click local launcher
├── requirements.txt
├── frontend/
│   └── index.html     # Single-page interactive UI
├── api/
│   └── routes.py      # HTTP route definitions
├── agent/
│   ├── llm_client.py  # OpenAI API wrapper
│   ├── photographer.py# Photographer agent (tool calling)
│   └── tools_schema.py# Tool JSON schema
├── core/
│   ├── raw_engine.py  # rawpy decode + metadata extraction
│   └── image_tools.py # Pillow-based image adjustments
└── tests/
    ├── conftest.py
    ├── test_agent.py
    ├── test_api.py
    └── test_raw_engine.py
```