"""
Configuration for the new 4-phase P&ID analysis pipeline
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
PROJECT_ROOT = BASE_DIR.parent  # Go up to project root
REFERENCE_LIBRARY_DIR = BASE_DIR / "reference_library"
OUTPUT_DIR = BASE_DIR / "pipeline_outputs"

# Create directories if they don't exist
REFERENCE_LIBRARY_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Florence-2 Configuration
FLORENCE_MODEL_PATH = os.getenv("FLORENCE_MODEL_PATH", "microsoft/Florence-2-base")
FLORENCE_DEVICE = os.getenv("FLORENCE_DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")

# Grounding DINO Configuration
GROUNDING_DINO_CONFIG_PATH = os.getenv("GROUNDING_DINO_CONFIG_PATH", str(PROJECT_ROOT / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"))
GROUNDING_DINO_CHECKPOINT_PATH = os.getenv("GROUNDING_DINO_CHECKPOINT_PATH", str(PROJECT_ROOT / "GroundingDINO/weights/groundingdino_swint_ogc.pth"))
GROUNDING_DINO_BOX_THRESHOLD = float(os.getenv("GROUNDING_DINO_BOX_THRESHOLD", "0.25"))  # Increased to reduce false positives
GROUNDING_DINO_TEXT_THRESHOLD = float(os.getenv("GROUNDING_DINO_TEXT_THRESHOLD", "0.20"))  # Increased to reduce false positives

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))

# OpenAI Configuration (fallback)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))

# PaddleOCR Configuration
PADDLEOCR_LANG = os.getenv("PADDLEOCR_LANG", "en")
PADDLEOCR_USE_GPU = os.getenv("PADDLEOCR_USE_GPU", "false").lower() == "true"
OCR_MIN_TEXT_CONFIDENCE = float(os.getenv("OCR_MIN_TEXT_CONFIDENCE", "0.5"))

# OpenCV Configuration
TEMPLATE_MATCH_THRESHOLD = float(os.getenv("TEMPLATE_MATCH_THRESHOLD", "0.8"))
MIN_COMPONENT_AREA = int(os.getenv("MIN_COMPONENT_AREA", "100"))

# ISA 5.1 Rules Configuration
VALID_INSTRUMENT_TAGS = ["PT", "LT", "FT", "TT", "AT", "PD", "LD", "FD", "TD", "AD", "PIC", "LIC", "FIC", "TIC", "AIC", "PI", "LI", "FI", "TI", "AI", "PS", "LS", "FS", "TS", "AS", "PIT", "LIT", "FIT", "AIT"]
VALID_EQUIPMENT_PREFIXES = ["P", "V", "T", "A", "F", "C", "E", "M", "K"]

# Reference Library Categories
REFERENCE_CATEGORIES = {
    "pumps": ["centrifugal", "gear", "reciprocating", "screw", "vane"],
    "valves": ["gate", "globe", "ball", "butterfly", "check", "control", "relief", "safety"],
    "vessels": ["vertical", "horizontal", "spherical", "tank", "reactor", "separator"],
    "motors": ["induction", "synchronous", "dc", "servo", "stepper"],
    "pipes": ["process", "utility", "instrument", "drain", "vent"]
}
