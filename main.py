"""FastAPI application instance and route wiring."""

from fastapi import FastAPI

from api.routes import router

app = FastAPI(
    title="RAW Agent API",
    description="Web API that processes Sony RAW (.ARW) files using an LLM Agent.",
    version="0.1.0",
)

app.include_router(router)
