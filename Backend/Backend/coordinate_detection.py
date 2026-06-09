from __future__ import annotations

import base64
import io
import json
import re
from typing import Any
import logging

import requests
from PIL import Image

logger = logging.getLogger(__name__)


def build_coordinate_detection_prompt() -> str:
	return (
		"You are an expert P&ID (Piping and Instrumentation Diagram) analyst with exceptional precision in component detection and coordinate extraction. "
		"Analyze this image systematically and detect ALL components with pixel-perfect accuracy.\n\n"
		"DETECTION REQUIREMENTS:\n"
		"1. Scan the entire image in a grid pattern - left to right, top to bottom\n"
		"2. Detect EVERY visible component regardless of size or label presence\n"
		"3. Look for standard P&ID symbols:\n"
		"   - Circles with M/MTR: Motors\n"
		"   - Circles with P/PUMP: Pumps\n"
		"   - Triangles/Diamonds: Valves (control valves, check valves, gate valves, etc.)\n"
		"   - Large rectangles/ovals: Tanks, Vessels, Reactors, Drums\n"
		"   - Small circles: Instruments, Sensors, Transmitters\n"
		"   - Bow-tie shapes: Butterfly valves\n"
		"   - T-shaped symbols: Gate valves\n"
		"4. For each component, extract:\n"
		"   - Component name/label (exact text from the diagram)\n"
		"   - Component type (motor, pump, valve, tank, vessel, sensor, instrument, or other)\n"
		"   - Precise bounding box (x, y, width, height) in pixels\n\n"
		"COORDINATE EXTRACTION RULES:\n"
		"- Coordinates MUST be relative to the TOP-LEFT corner of the image (0,0)\n"
		"- x: horizontal position of left edge\n"
		"- y: vertical position of top edge\n"
		"- width: horizontal span from left to right edge\n"
		"- height: vertical span from top to bottom edge\n"
		"- Bounding boxes must TIGHTLY enclose each component with minimal padding\n"
		"- Do NOT include connecting lines or pipes in the bounding box\n"
		"- For text labels, use the bounding box of the text itself\n"
		"- For symbols, use the bounding box of the symbol shape only\n\n"
		"COMPONENT NAMING:\n"
		"- Use exact label text if visible (e.g., 'P-101', 'V-201', 'TK-301')\n"
		"- If no label, use descriptive generic names: 'Motor', 'Pump', 'Valve', 'Tank', 'Vessel', 'Sensor'\n"
		"- Add numbers to distinguish similar components: 'Motor 1', 'Motor 2', etc.\n\n"
		"QUALITY CHECKS:\n"
		"- All coordinates must be non-negative integers\n"
		"- All widths and heights must be positive (minimum 5 pixels)\n"
		"- Bounding boxes must be within image dimensions\n"
		"- No overlapping bounding boxes for distinct components\n"
		"- Minimum component size: 10x10 pixels\n"
		"- Maximum reasonable size: 80% of image dimensions\n\n"
		"Return ONLY valid JSON with this exact structure:\n"
		'{\n'
		'  "custom": {},\n'
		'  "params": {},\n'
		'  "props": {},\n'
		'  "root": {\n'
		'    "children": [\n'
		'      {\n'
		'        "meta": {"name": "component_name"},\n'
		'        "position": {"x": 0, "y": 0, "width": 0, "height": 0},\n'
		'        "type": "ia.symbol.type"\n'
		'      }\n'
		'    ],\n'
		'    "meta": {"name": "root"},\n'
		'    "type": "ia.container.coord"\n'
		'  }\n'
		'}\n\n'
		"Type mappings:\n"
		"- vessel/tank/reactor/drum -> ia.symbol.vessel\n"
		"- motor/mtr -> ia.symbol.motor\n"
		"- pump/pmp -> ia.symbol.pump\n"
		"- valve/cv/xv/pcv/fcv/lcv/tcv/psv/nrv/sdv -> ia.symbol.valve\n"
		"- sensor/instrument/transmitter/pt/lt/ft/tt -> ia.symbol.sensor\n"
		"- other -> ia.symbol.other\n\n"
		"CRITICAL: Return ONLY the JSON. No explanations, no markdown code blocks, no additional text."
	)


def image_to_base64_png(image: Image.Image) -> str:
	buffer = io.BytesIO()
	image.save(buffer, format="PNG")
	return base64.b64encode(buffer.getvalue()).decode("ascii")


def resize_for_model(image: Image.Image, max_edge: int = 1600) -> tuple[Image.Image, float, float]:
	original_width, original_height = image.size
	prepared = image.convert("RGB")
	prepared.thumbnail((max_edge, max_edge))
	resized_width, resized_height = prepared.size
	scale_x = original_width / resized_width if resized_width > 0 else 1.0
	scale_y = original_height / resized_height if resized_height > 0 else 1.0
	return prepared, scale_x, scale_y


def extract_message_text(message: Any) -> str:
	if isinstance(message, dict):
		for key in ("content", "output_text", "parsed", "text"):
			if key in message:
				candidate = message[key]
				if isinstance(candidate, str):
					return candidate
				if isinstance(candidate, (dict, list)):
					return coerce_message_content(candidate)

		for tool_call_key in ("tool_calls", "function_call"):
			if tool_call_key in message:
				candidate = message[tool_call_key]
				if isinstance(candidate, list):
					for tool_call in candidate:
						if not isinstance(tool_call, dict):
							continue
						function_block = tool_call.get("function")
						if isinstance(function_block, dict):
							for nested_key in ("arguments", "content", "text"):
								if isinstance(function_block.get(nested_key), str):
									return function_block[nested_key]
						for nested_key in ("arguments", "content", "text"):
							if isinstance(tool_call.get(nested_key), str):
								return tool_call[nested_key]
				elif isinstance(candidate, dict):
					function_block = candidate.get("function")
					if isinstance(function_block, dict):
						for nested_key in ("arguments", "content", "text"):
							if isinstance(function_block.get(nested_key), str):
								return function_block[nested_key]
					for nested_key in ("arguments", "content", "text"):
						if isinstance(candidate.get(nested_key), str):
							return candidate[nested_key]

	return coerce_message_content(message)


def coerce_message_content(raw_content: Any) -> str:
	if isinstance(raw_content, str):
		return raw_content

	if isinstance(raw_content, dict):
		if "content" in raw_content:
			return coerce_message_content(raw_content["content"])
		return json.dumps(raw_content)

	if isinstance(raw_content, list):
		parts: list[str] = []
		for part in raw_content:
			if isinstance(part, str):
				parts.append(part)
			elif isinstance(part, dict):
				if isinstance(part.get("text"), str):
					parts.append(part["text"])
				elif isinstance(part.get("content"), str):
					parts.append(part["content"])
		return "\n".join(parts)

	return str(raw_content)


def validate_coordinates(position: dict[str, Any], image_width: int, image_height: int) -> bool:
	"""Validate that coordinates are within reasonable bounds."""
	try:
		x = int(position.get("x", 0))
		y = int(position.get("y", 0))
		width = int(position.get("width", 0))
		height = int(position.get("height", 0))
		
		# Check for non-negative coordinates
		if x < 0 or y < 0 or width <= 0 or height <= 0:
			return False
		
		# Check minimum size (10x10 pixels)
		if width < 10 or height < 10:
			return False
		
		# Check maximum reasonable size (80% of image)
		max_width = image_width * 0.8
		max_height = image_height * 0.8
		if width > max_width or height > max_height:
			return False
		
		# Check that bounding box is within image bounds
		if x + width > image_width or y + height > image_height:
			return False
		
		return True
	except (ValueError, TypeError):
		return False


def scale_coordinates(position: dict[str, Any], scale_x: float, scale_y: float) -> dict[str, Any]:
	"""Scale coordinates back to original image dimensions."""
	try:
		x = int(float(position.get("x", 0)) * scale_x)
		y = int(float(position.get("y", 0)) * scale_y)
		width = int(float(position.get("width", 0)) * scale_x)
		height = int(float(position.get("height", 0)) * scale_y)
		return {"x": x, "y": y, "width": width, "height": height}
	except (ValueError, TypeError):
		return position


def parse_json_payload(raw_content: str) -> dict[str, Any]:
	trimmed = raw_content.strip()
	for candidate in (
		trimmed,
		re.sub(r"^```(?:json)?\s*|\s*```$", "", trimmed, flags=re.IGNORECASE | re.DOTALL),
	):
		try:
			parsed = json.loads(candidate)
			if isinstance(parsed, dict):
				return parsed
		except json.JSONDecodeError:
			continue

	json_start_tokens = ["[", "{"]
	for start_token in json_start_tokens:
		start_index = trimmed.find(start_token)
		if start_index == -1:
			continue
		end_token = "}" if start_token == "{" else "]"
		end_index = trimmed.rfind(end_token)
		if end_index != -1 and end_index > start_index:
			fragment = trimmed[start_index : end_index + 1]
			try:
				parsed = json.loads(fragment)
				if isinstance(parsed, dict):
					return parsed
			except json.JSONDecodeError:
				continue

	raise ValueError(f"Model response did not contain valid JSON. Raw output: {raw_content[:1000]}")


def detect_coordinates(
	image: Image.Image,
	config: dict[str, str],
) -> dict[str, Any]:
	# Vision model detection disabled - system uses local detection (opencv, paddleocr, ollama)
	# Coordinates are generated by analyze_pid_image_async in local_detection.py
	logger.warning("External coordinate detection disabled - using local detection pipeline")
	return {
		"root": {
			"children": []
		}
	}


async def detect_coordinates_async(image: Image.Image) -> dict[str, Any]:
	"""Async wrapper for coordinate detection using local detection."""
	from local_detection import analyze_pid_image_async
	from image_processing import resize_for_fast_processing
	
	original_width, original_height = image.size
	resized_image, scale_x, scale_y = resize_for_fast_processing(image)
	result = await analyze_pid_image_async(resized_image)
	coordinates = result["coordinates"]
	
	# Scale coordinates back to original image dimensions
	if "root" in coordinates and "children" in coordinates["root"]:
		for child in coordinates["root"]["children"]:
			if "position" in child:
				position = child["position"]
				try:
					x = int(float(position.get("x", 0)) * scale_x)
					y = int(float(position.get("y", 0)) * scale_y)
					width = int(float(position.get("width", 0)) * scale_x)
					height = int(float(position.get("height", 0)) * scale_y)
					
					# Validate scaled coordinates
					if x >= 0 and y >= 0 and width > 0 and height > 0:
						if x + width <= original_width and y + height <= original_height:
							child["position"] = {"x": x, "y": y, "width": width, "height": height}
				except (ValueError, TypeError):
					logger.warning(f"Failed to scale coordinates for child: {child.get('meta', {})}")
	
	return coordinates
