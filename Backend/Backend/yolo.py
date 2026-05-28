from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import cv2
import fitz
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

# --- Setup ---
BACKEND_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_ROOT / ".env")

app = FastAPI(title="Sarla P&ID Expert System v2", version="0.7.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schemas ---

class CategoryCounts(BaseModel):
    text: int = 0
    motor: int = 0
    pump: int = 0
    tank: int = 0
    valve: int = 0

class PageDetectionResult(BaseModel):
    page_index: int
    counts: CategoryCounts
    model_results: list[Any]

class DetectionResponse(BaseModel):
    filename: str
    source_type: str
    page_count: int
    models_used: list[str]
    pages: list[PageDetectionResult]

# --- Helper Functions ---

def load_image_frames(file_bytes: bytes, filename: str | None) -> tuple[list[Image.Image], str]:
    if filename and filename.lower().endswith(".pdf"):
        document = fitz.open(stream=file_bytes, filetype="pdf")
        frames = []
        for page in document:
            # Ultra-high res (4.0) to ensure check valves are visible
            pixmap = page.get_pixmap(matrix=fitz.Matrix(4.0, 4.0), alpha=False)
            frames.append(Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples))
        document.close()
        return frames, "pdf"
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")], "image"

def image_to_base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")

def get_micromanaged_tiles(image: Image.Image, tile_size: int = 700, overlap: int = 250):
    """Smaller tiles with high overlap to catch small check valves."""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    tiles = []
    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            tiles.append(Image.fromarray(img_array[y_start:y_end, x_start:x_end]))
            if x_end == w: break
        if y_end == h: break
    return tiles

# --- AI Logic ---

def get_expert_prompt():
    return (
        "You are a P&ID Auditor. Identify and count every single mechanical symbol: "
        "1. VALVES (CRITICAL): Count both 'Control Valves' (bow-tie symbols) AND 'Check Valves'. "
        "Check valves are small Z-shaped, N-shaped, or triangle symbols integrated directly into the pipe line. Do not miss them. "
        "2. PUMPS: Horizontal capsule/cylindrical shapes connected to lines. "
        "3. TANKS: Large vertical or horizontal vessels/reactors. "
        "4. MOTORS: Circular/Square driver units attached to equipment. "
        "STRICT RULES: Ignore all text/tags. Return ONLY JSON: {'motor': X, 'pump': X, 'tank': X, 'valve': X}"
    )

async def call_gemini_tile_scan(tile: Image.Image, config: dict) -> CategoryCounts:
    encoded = image_to_base64_png(tile)
    payload = {
        "model": config["gemini_model"],
        "messages": [{"role": "user", "content": [{"type": "text", "text": get_expert_prompt()}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}]}],
        "response_format": {"type": "json_object"}, "temperature": 0
    }
    headers = {"Authorization": f"Bearer {config['api_key']}", "Content-Type": "application/json"}
    try:
        resp = await asyncio.to_thread(requests.post, f"{config['base_url']}/chat/completions", headers=headers, json=payload, timeout=60)
        parsed = json.loads(resp.json()["choices"][0]["message"]["content"])
        return CategoryCounts(motor=int(parsed.get("motor", 0)), pump=int(parsed.get("pump", 0)), tank=int(parsed.get("tank", 0)), valve=int(parsed.get("valve", 0)))
    except: return CategoryCounts()

async def call_claude_global_audit(full_image: Image.Image, tile_sum: CategoryCounts, config: dict) -> CategoryCounts:
    print(f"  --> Claude Final Audit: Verifying all valves (including check valves)...")
    full_image.thumbnail((2200, 2200))
    encoded = image_to_base64_png(full_image)
    
    prompt = (
        f"Final Audit. Preliminary tile-scan found: {tile_sum.json()}. "
        "Scan the FULL image. Pay extreme attention to VALVES. "
        "There are often 'Check Valves' which look like small Z-shapes or triangles on the lines. "
        "Ensure you count both the large bow-tie control valves AND the small check valves. "
        "Pumps are horizontal capsules. Tanks are large vessels. "
        "Return ONLY the final unique count JSON."
    )
    payload = {
        "model": config["claude_model"],
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}]}],
        "response_format": {"type": "json_object"}, "temperature": 0
    }
    headers = {"Authorization": f"Bearer {config['api_key']}", "Content-Type": "application/json"}
    try:
        resp = await asyncio.to_thread(requests.post, f"{config['base_url']}/chat/completions", headers=headers, json=payload, timeout=90)
        parsed = json.loads(resp.json()["choices"][0]["message"]["content"])
        return CategoryCounts(motor=int(parsed.get("motor", 0)), pump=int(parsed.get("pump", 0)), tank=int(parsed.get("tank", 0)), valve=int(parsed.get("valve", 0)))
    except: return tile_sum

# --- Endpoints ---

@app.post("/upload")
async def upload_stage1(file: UploadFile = File(...)):
    file_bytes = await file.read()
    frames, source_type = load_image_frames(file_bytes, file.filename)
    summaries = [{"page_index": i, "width": f.width, "height": f.height, "preview_png_base64": image_to_base64_png(f)} for i, f in enumerate(frames, start=1)]
    return {"filename": file.filename, "source_type": source_type, "page_count": len(frames), "frames": summaries}

@app.post("/detect", response_model=DetectionResponse)
async def detect_components(file: UploadFile = File(...)):
    file_bytes = await file.read()
    frames, source_type = load_image_frames(file_bytes, file.filename)
    
    config = {
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "gemini_model": os.getenv("OPENROUTER_QWEN_MODEL", "google/gemini-2.0-flash-001"),
        "claude_model": os.getenv("OPENROUTER_CLAUDE_MODEL", "anthropic/claude-3.5-sonnet")
    }

    page_results = []
    for page_idx, frame in enumerate(frames, start=1):
        # 1. Micromanaged Tiling (Smaller tiles to see small check valves)
        tiles = get_micromanaged_tiles(frame)
        print(f"Page {page_idx}: {len(tiles)} tiles generated.")
        
        tile_tasks = [call_gemini_tile_scan(tile, config) for tile in tiles]
        raw_tile_results = await asyncio.gather(*tile_tasks)
        
        tile_sum = CategoryCounts()
        for r in raw_tile_results:
            tile_sum.motor += r.motor
            tile_sum.pump += r.pump
            tile_sum.tank += r.tank
            tile_sum.valve += r.valve

        # 2. Global Audit
        final_verified_counts = await call_claude_global_audit(frame, tile_sum, config)

        page_results.append(PageDetectionResult(
            page_index=page_idx,
            counts=final_verified_counts,
            model_results=[
                {"model": config["gemini_model"], "role": "tiled_scan", "raw": tile_sum},
                {"model": config["claude_model"], "role": "final_audit", "final": final_verified_counts}
            ]
        ))

    return DetectionResponse(
        filename=file.filename or "file", source_type=source_type, page_count=len(frames),
        models_used=[config["gemini_model"], config["claude_model"]],
        pages=page_results
    )