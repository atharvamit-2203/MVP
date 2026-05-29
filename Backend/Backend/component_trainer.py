from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

if __package__:
	from .active_learning import ANNOTATIONS_DIR, train_model
else:
	import sys

	current_dir = Path(__file__).resolve().parent
	if str(current_dir) not in sys.path:
		sys.path.insert(0, str(current_dir))
	from active_learning import ANNOTATIONS_DIR, train_model


UI_PATH = Path(__file__).resolve().parent / "static" / "component_trainer.html"
ALLOWED_LABELS = ("motor", "pump", "tank", "valve", "other")

app = FastAPI(title="Sarla Component Trainer", version="1.0.0")
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=False,
	allow_methods=["*"],
	allow_headers=["*"],
)


def _sample_image_name(source_name: str) -> str:
	stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
	return f"{stamp}_{Path(source_name).name}"


def _annotation_entry(image_name: str, label: str, width: int, height: int) -> dict[str, Any]:
	return {
		"timestamp": datetime.utcnow().isoformat() + "Z",
		"image": image_name,
		"annotations": [
			{
				"label": label,
				"bbox": [0, 0, width, height],
			},
		],
	}

def _save_annotation_entry(image_name: str, annotations: list[dict[str, Any]]) -> None:
	ann_path = ANNOTATIONS_DIR / "annotations.jsonl"
	entry = {
		"timestamp": datetime.utcnow().isoformat() + "Z",
		"image": image_name,
		"annotations": annotations,
	}
	with open(ann_path, "a", encoding="utf-8") as fh:
		fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_recent_samples(limit: int = 20) -> list[dict[str, Any]]:
	ann_path = ANNOTATIONS_DIR / "annotations.jsonl"
	if not ann_path.exists():
		return []

	samples: list[dict[str, Any]] = []
	with open(ann_path, "r", encoding="utf-8") as fh:
		for line in fh:
			line = line.strip()
			if not line:
				continue
			try:
				entry = json.loads(line)
			except Exception:  # noqa: BLE001
				continue

			labels = [ann.get("label") for ann in entry.get("annotations", []) if ann.get("label")]
			samples.append(
				{
					"timestamp": entry.get("timestamp"),
					"image": entry.get("image"),
					"labels": labels,
				}
			)

	return samples[-limit:]


@app.get("/health")
async def health() -> dict[str, Any]:
	return {
		"status": "ok",
		"annotations_dir": str(ANNOTATIONS_DIR),
		"labels": list(ALLOWED_LABELS),
	}


@app.get("/labels")
async def labels() -> dict[str, list[str]]:
	return {"labels": list(ALLOWED_LABELS)}


@app.get("/samples")
async def samples(limit: int = 20) -> dict[str, Any]:
	if limit < 1:
		raise HTTPException(status_code=400, detail="limit must be at least 1")
	return {"samples": _load_recent_samples(limit=limit)}


@app.get("/trainer", response_class=HTMLResponse)
async def trainer_ui() -> HTMLResponse:
	if not UI_PATH.exists():
		raise HTTPException(status_code=404, detail="Trainer UI not found")
	return HTMLResponse(UI_PATH.read_text(encoding="utf-8"))


@app.post("/samples")
async def add_sample(
	image: UploadFile = File(...),
	label: str = Form(...),
	train_after_save: bool = Form(False),
) -> dict[str, Any]:
	if label not in ALLOWED_LABELS:
		raise HTTPException(status_code=400, detail=f"label must be one of {list(ALLOWED_LABELS)}")

	file_bytes = await image.read()
	if not file_bytes:
		raise HTTPException(status_code=400, detail="Uploaded image is empty.")

	ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
	image_name = _sample_image_name(image.filename or "component.png")
	image_path = ANNOTATIONS_DIR / image_name

	with open(image_path, "wb") as fh:
		fh.write(file_bytes)

	loaded = cv2.imread(str(image_path))
	if loaded is None:
		image_path.unlink(missing_ok=True)
		raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.")

	height, width = loaded.shape[:2]
	_save_annotation_entry(image_path.name, _annotation_entry(image_name=image_path.name, label=label, width=width, height=height)["annotations"])

	result: dict[str, Any] | None = None
	if train_after_save:
		result = train_model()

	return {
		"status": "saved",
		"image": image_path.name,
		"label": label,
		"width": width,
		"height": height,
		"trained": result,
	}


@app.post("/samples/batch")
async def add_samples_batch(
	images: list[UploadFile] = File(...),
	label: str = Form(...),
	train_after_save: bool = Form(False),
) -> dict[str, Any]:
	if label not in ALLOWED_LABELS:
		raise HTTPException(status_code=400, detail=f"label must be one of {list(ALLOWED_LABELS)}")

	if not images:
		raise HTTPException(status_code=400, detail="At least one image is required.")

	ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
	saved: list[dict[str, Any]] = []

	for image in images:
		file_bytes = await image.read()
		if not file_bytes:
			continue

		image_name = _sample_image_name(image.filename or "component.png")
		image_path = ANNOTATIONS_DIR / image_name
		with open(image_path, "wb") as fh:
			fh.write(file_bytes)

		loaded = cv2.imread(str(image_path))
		if loaded is None:
			image_path.unlink(missing_ok=True)
			continue

		height, width = loaded.shape[:2]
		_save_annotation_entry(image_path.name, [_annotation_entry(image_path.name, label, width, height)["annotations"][0]])
		saved.append({"image": image_path.name, "width": width, "height": height})

	if not saved:
		raise HTTPException(status_code=400, detail="No valid images were saved.")

	result: dict[str, Any] | None = None
	if train_after_save:
		result = train_model()

	return {
		"status": "saved",
		"saved": saved,
		"label": label,
		"trained": result,
	}


@app.post("/annotations")
async def add_multi_annotations(
	image: UploadFile = File(...),
	annotations: str = Form(...),
	train_after_save: bool = Form(False),
) -> dict[str, Any]:
	file_bytes = await image.read()
	if not file_bytes:
		raise HTTPException(status_code=400, detail="Uploaded image is empty.")

	try:
		parsed_annotations = json.loads(annotations)
	except json.JSONDecodeError as exc:
		raise HTTPException(status_code=400, detail="annotations must be valid JSON.") from exc

	if not isinstance(parsed_annotations, list) or not parsed_annotations:
		raise HTTPException(status_code=400, detail="annotations must be a non-empty list.")

	cleaned_annotations: list[dict[str, Any]] = []
	for annotation in parsed_annotations:
		if not isinstance(annotation, dict):
			continue
		label = annotation.get("label")
		bbox = annotation.get("bbox")
		if label not in ALLOWED_LABELS or not isinstance(bbox, list) or len(bbox) != 4:
			continue
		try:
			cleaned_annotations.append({"label": str(label), "bbox": [int(v) for v in bbox]})
		except (TypeError, ValueError):
			continue

	if not cleaned_annotations:
		raise HTTPException(status_code=400, detail="No valid annotations were provided.")

	ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
	image_name = _sample_image_name(image.filename or "component.png")
	image_path = ANNOTATIONS_DIR / image_name
	with open(image_path, "wb") as fh:
		fh.write(file_bytes)

	loaded = cv2.imread(str(image_path))
	if loaded is None:
		image_path.unlink(missing_ok=True)
		raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.")

	_save_annotation_entry(image_path.name, cleaned_annotations)

	result: dict[str, Any] | None = None
	if train_after_save:
		result = train_model()

	return {
		"status": "saved",
		"image": image_path.name,
		"annotations": cleaned_annotations,
		"trained": result,
	}


@app.post("/train")
async def train() -> dict[str, Any]:
	return train_model()


if __name__ == "__main__":
	import uvicorn

	uvicorn.run(app, host="127.0.0.1", port=8001, reload=False)