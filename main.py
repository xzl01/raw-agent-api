"""FastAPI application instance and route wiring."""

import pathlib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router

app = FastAPI(
    title="RAW Agent API",
    description="Web API that processes Sony RAW (.ARW) files using an LLM Agent.",
    version="0.1.0",
)

# Allow all origins so the bundled frontend (opened as a local file or served
# from any port) can reach the API without CORS errors.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Serve the bundled single-page frontend at the root.
_FRONTEND_DIR = pathlib.Path(__file__).parent / "frontend"

if _FRONTEND_DIR.is_dir():
    app.mount("/frontend", StaticFiles(directory=str(_FRONTEND_DIR)), name="frontend")

    @app.get("/", include_in_schema=False)
    async def serve_index() -> FileResponse:
        """Redirect browser clients to the interactive frontend."""
        return FileResponse(str(_FRONTEND_DIR / "index.html"))
