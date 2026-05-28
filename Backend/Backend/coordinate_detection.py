from __future__ import annotations

import base64
import io
import json
import re
from typing import Any

import requests
from PIL import Image


def build_coordinate_detection_prompt() -> str:
	return (
		"Analyze this P&ID (Piping and Instrumentation Diagram) image and detect all components with their exact positions. "
		"For each component, identify: "
		"1. Component name/label (text visible near or on the component) "
		"2. Component type (vessel, motor, pump, valve, sensor, tank, or other) "
		"3. Bounding box coordinates (x, y, width, height) in pixels "
		"Return strict JSON only with this exact structure: "
		'{'
		'  "custom": {},'
		'  "params": {},'
		'  "props": {},'
		'  "root": {'
		'    "children": ['
		'      {'
		'        "meta": {"name": "component_name"},'
		'        "position": {"x": 0, "y": 0, "width": 0, "height": 0},'
		'        "type": "ia.symbol.type"'
		'      }'
		'    ],'
		'    "meta": {"name": "root"},'
		'    "type": "ia.container.coord"'
		'  }'
		'}'
		"Use these type mappings: "
		"vessel/tank -> ia.symbol.vessel, "
		"motor -> ia.symbol.motor, "
		"pump -> ia.symbol.pump, "
		"valve -> ia.symbol.valve, "
		"sensor/instrument -> ia.symbol.sensor, "
		"other -> ia.symbol.other. "
		"Coordinates should be relative to the image top-left corner (0,0)."
	)


def image_to_base64_png(image: Image.Image) -> str:
	buffer = io.BytesIO()
	image.save(buffer, format="PNG")
	return base64.b64encode(buffer.getvalue()).decode("ascii")


def resize_for_model(image: Image.Image, max_edge: int = 1600) -> Image.Image:
	prepared = image.convert("RGB")
	prepared.thumbnail((max_edge, max_edge))
	return prepared


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
	prepared_image = resize_for_model(image)
	encoded_image = image_to_base64_png(prepared_image)
	
	payload = {
		"model": config["qwen_model"],
		"max_tokens": 4000,
		"messages": [
			{
				"role": "system",
				"content": "You are an expert at analyzing P&ID diagrams and detecting component positions with high precision.",
			},
			{
				"role": "user",
				"content": [
					{"type": "text", "text": build_coordinate_detection_prompt()},
					{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}},
				],
			},
		],
		"temperature": 0.1,
	}

	headers = {
		"Authorization": f"Bearer {config['api_key']}",
		"Content-Type": "application/json",
		"HTTP-Referer": config["site_url"],
		"X-Title": config["app_name"],
	}

	response = requests.post(
		f"{config['base_url']}/chat/completions",
		headers=headers,
		json=payload,
		timeout=180,
	)
	if response.status_code >= 400:
		raise ValueError(f"OpenRouter model failed: {response.status_code} {response.text}")
	
	response_json = response.json()
	choices = response_json.get("choices") or []
	if not choices:
		raise ValueError("OpenRouter model returned no choices.")
	
	message = choices[0].get("message", {})
	raw_content = extract_message_text(message)
	
	try:
		parsed = parse_json_payload(raw_content)
		# Validate structure
		if "root" not in parsed:
			raise ValueError("Missing 'root' key in response")
		if "children" not in parsed["root"]:
			raise ValueError("Missing 'children' key in root")
		return parsed
	except ValueError as exc:
		# Return empty structure on failure
		return {
			"custom": {},
			"params": {},
			"props": {},
			"root": {
				"children": [],
				"meta": {"name": "root"},
				"type": "ia.container.coord",
			},
		}
