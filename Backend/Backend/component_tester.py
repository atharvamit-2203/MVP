from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

if __package__:
	from .active_learning import load_model, predict_candidates
else:
	current_dir = Path(__file__).resolve().parent
	if str(current_dir) not in sys.path:
		sys.path.insert(0, str(current_dir))
	from active_learning import load_model, predict_candidates


UI_PATH = Path(__file__).resolve().parent / "static" / "component_tester.html"

app = FastAPI(title="Sarla Component Tester", version="1.0.0")
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=False,
	allow_methods=["*"],
	allow_headers=["*"],
)


def _model_status() -> dict[str, Any]:
	model_blob = load_model()
	if model_blob is None:
		return {"loaded": False, "labels": []}
	return {"loaded": True, "labels": model_blob.get("labels", [])}


def _predict_from_image_bytes(file_bytes: bytes, filename: str | None = None) -> dict[str, Any]:
	image_array = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
	if image_array is None:
		raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.")

	height, width = image_array.shape[:2]
	candidates = [{"bbox": [0, 0, width, height], "vertex_count": 0, "name": filename or "component"}]
	results = predict_candidates(image_array, candidates)
	if not results:
		raise HTTPException(status_code=500, detail="Model returned no prediction.")

	result = results[0]
	return {
		"filename": filename or "uploaded-file",
		"width": width,
		"height": height,
		"predicted": result.get("predicted"),
		"prob": result.get("prob"),
		"uncertainty": result.get("uncertainty"),
		"bbox": result.get("bbox"),
	}


@app.get("/health")
async def health() -> dict[str, Any]:
	status = _model_status()
	return {"status": "ok", **status}


@app.get("/tester", response_class=HTMLResponse)
async def tester_ui() -> HTMLResponse:
	if not UI_PATH.exists():
		raise HTTPException(status_code=404, detail="Tester UI not found")
	return HTMLResponse(UI_PATH.read_text(encoding="utf-8"))


@app.get("/labels")
async def labels() -> dict[str, list[str]]:
	status = _model_status()
	return {"labels": list(status["labels"])}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
	if load_model() is None:
		raise HTTPException(status_code=503, detail="Train the model first before testing predictions.")

	file_bytes = await file.read()
	if not file_bytes:
		raise HTTPException(status_code=400, detail="Uploaded file is empty.")
	return _predict_from_image_bytes(file_bytes, file.filename)


@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...)) -> dict[str, Any]:
	if load_model() is None:
		raise HTTPException(status_code=503, detail="Train the model first before testing predictions.")

	if not files:
		raise HTTPException(status_code=400, detail="At least one image is required.")

	results: list[dict[str, Any]] = []
	for file in files:
		file_bytes = await file.read()
		if not file_bytes:
			results.append({"filename": file.filename or "uploaded-file", "error": "Empty file"})
			continue
		try:
			results.append(_predict_from_image_bytes(file_bytes, file.filename))
		except HTTPException as exc:
			results.append({"filename": file.filename or "uploaded-file", "error": str(exc.detail)})
	return {"results": results}


if __name__ == "__main__":
	import uvicorn

	uvicorn.run(app, host="127.0.0.1", port=8002, reload=False)
