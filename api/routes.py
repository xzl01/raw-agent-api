"""FastAPI route definitions for RAW image processing."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from agent.photographer import PhotographerAgent
from core.image_tools import apply_adjustments, image_to_jpeg_bytes
from core.raw_engine import decode_raw_bytes

router = APIRouter()

_agent = PhotographerAgent()


@router.post("/process", summary="Process a single RAW (.ARW) file")
async def process_single(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a single .ARW file, apply LLM-guided adjustments, return the JPEG.

    Returns a JSON response containing:
    - ``filename``: original filename
    - ``adjustments``: the parameters chosen by the agent
    - ``jpeg_b64``: base64-encoded JPEG bytes of the processed image
    - ``message``: success message
    """
    raw_bytes = await file.read()
    return _process_raw_bytes(raw_bytes, filename=file.filename or "image.arw")


@router.post("/batch-process", summary="Process multiple RAW (.ARW) files")
async def process_batch(files: List[UploadFile] = File(...)) -> JSONResponse:
    """Upload multiple .ARW files and process each with LLM-guided adjustments.

    Returns a JSON response containing a list of per-file results (same shape
    as the single-file ``/process`` endpoint).
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    results = []
    for upload in files:
        raw_bytes = await upload.read()
        try:
            result = _process_raw_bytes(
                raw_bytes, filename=upload.filename or "image.arw"
            )
            results.append(result.body)  # JSONResponse.body is bytes
        except HTTPException as exc:
            results.append(
                {
                    "filename": upload.filename,
                    "error": exc.detail,
                }
            )

    # Decode each body from bytes to dict for the aggregate response
    import json

    decoded: List[Dict[str, Any]] = []
    for item in results:
        if isinstance(item, bytes):
            decoded.append(json.loads(item))
        else:
            decoded.append(item)

    return JSONResponse(content={"results": decoded})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_raw_bytes(raw_bytes: bytes, filename: str) -> JSONResponse:
    """Core processing pipeline: decode → agent → adjust → encode."""
    try:
        image, metadata = decode_raw_bytes(raw_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to decode RAW file '{filename}': {exc}",
        ) from exc

    try:
        adjustments = _agent.determine_adjustments(metadata)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Agent failed to determine adjustments: {exc}",
        ) from exc

    try:
        adjusted = apply_adjustments(image, **adjustments)
        jpeg_bytes = image_to_jpeg_bytes(adjusted)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Image processing failed: {exc}",
        ) from exc

    jpeg_b64 = base64.b64encode(jpeg_bytes).decode("ascii")

    return JSONResponse(
        content={
            "filename": filename,
            "adjustments": adjustments,
            "jpeg_b64": jpeg_b64,
            "message": "Processing successful.",
        }
    )
