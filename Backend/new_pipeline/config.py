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
TEMPLATE_DIR = PROJECT_ROOT / "Template"  # Ignition Designer templates directory

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
TEMPLATE_MATCH_THRESHOLD = float(os.getenv("TEMPLATE_MATCH_THRESHOLD", "0.5"))  # Lowered for better detection accuracy
MIN_COMPONENT_AREA = int(os.getenv("MIN_COMPONENT_AREA", "100"))

# ISA 5.1 Rules Configuration
# TV, PV, TE, etc. are TAGS, not instruments - they should not be rendered as components
# Instruments themselves should also not be rendered as components
VALID_INSTRUMENT_TAGS = ["PT", "LT", "FT", "TT", "AT", "PD", "LD", "FD", "TD", "AD", "PIC", "LIC", "FIC", "TIC", "AIC", "PI", "LI", "FI", "TI", "AI", "PS", "LS", "FS", "TS", "AS", "PIT", "LIT", "FIT", "AIT"]
TAG_ONLY = ["TV", "PV", "LV", "FV", "AV", "TE", "LE", "FE", "AE", "TT", "LT", "FT", "AT", "PT"]  # These are tags only, not instruments
INSTRUMENT_TYPES = ["instrument", "sensor", "gauge", "meter", "transmitter", "indicator", "controller"]  # These should not be rendered as components
VALID_EQUIPMENT_PREFIXES = ["P", "V", "T", "A", "F", "C", "E", "M", "K"]

# Reference Library Categories
REFERENCE_CATEGORIES = {
    "pumps": ["centrifugal", "gear", "reciprocating", "screw", "vane"],
    "valves": ["gate", "globe", "ball", "butterfly", "check", "control", "relief", "safety"],
    "vessels": ["vertical", "horizontal", "spherical", "tank", "reactor", "separator"],
    "motors": ["induction", "synchronous", "dc", "servo", "stepper"],
    "pipes": ["process", "utility", "instrument", "drain", "vent"]
}

# Standard Component Types (for Ignition Designer JSON output)
# TANKS, VALVES, MOTORS, and PUMPS are standard components
STANDARD_COMPONENT_TYPES = ["motor", "pipe", "pump", "valve", "tank", "vessel"]

# Standard Component Subtypes (these will be treated as standard components)
STANDARD_SUBTYPES = {
    "valves": ["gate", "globe", "ball", "butterfly", "check", "control", "relief", "safety", "three_way", "angle", "plug", "diaphragm", "needle", "solenoid", "pressure", "temperature"],
    "pumps": ["centrifugal", "gear", "reciprocating", "screw", "vane", "motor_pump", "pump_double", "pump_left", "pump_right", "vertical_pump", "arrow", "arrow_head"],
    "motors": ["induction", "synchronous", "dc", "servo", "stepper", "motor_gear"],
    "tanks": ["storage", "spherical", "horizontal", "vertical", "separator_tank", "water_tank", "square_tank"],
    "pipes": ["process", "utility", "instrument", "drain", "vent", "arrow_pipe", "l_shape_pipe", "m_shape_pipe"]
}

# Customized Component Subtypes (these will be treated as customized with template paths)
# All components except TANKS, VALVES, MOTORS, and PUMPS are customized
CUSTOMIZED_SUBTYPES = {
    "general": ["baghouse", "baghouse_hopper", "baghouse_hopper_single", "cyclone", "cyclone_separator", "cyclone_separator1", "preheater_cyclone",
                "turbine", "boiler", "heat_exchanger", "compressor", "reactor",
                "chimney", "chimney_without_smoke", "chimney_without_smoke_grey", "exhaust_stack", "exhaust_stack_1",
                "stacker", "stacker_reclaimer", "conveyor", "belt_conveyor", "chain_conveyor", "rectangular_conveyor", "spiral", "duct",
                "crusher", "calciner", "furnace", "kiln", "rotary_kiln",
                "separator", "square_tank", "water_tank", "tank_6", "storage_tank", "separator_tank",
                "three_way_valve", "mobile_panel", "addition", "and_gate", "or_gate", "not_gate",
                "clinker_silo", "packers", "lorry", "hag", "whrs"]
}

# Ignition Designer JSON Output Configuration
IGNITION_OUTPUT_FORMAT = "ia.display.view"  # Type for customized components (this works in Ignition Designer)
IGNITION_STANDARD_PREFIX = "ia.symbol."  # Prefix for standard components

# Template path mapping based on actual Template folder structure
TEMPLATE_PATH_MAPPING = {
    # Baghouse components
    "baghouse": "Template/Baghouse_Hopper/Baghouse_Hopper",
    "baghouse_hopper": "Template/Baghouse_Hopper/Baghouse_Hopper",
    "baghouse_hopper_single": "Template/Baghouse_Hopper/Baghouse_Hopper_Single",
    # Cyclone components
    "cyclone": "Template/Cyclone/Cyclone",
    "cyclone_separator": "Template/Cyclone/Cyclone_Separator",
    "cyclone_separator1": "Template/Cyclone/Cyclone_Separator1",
    "preheater_cyclone": "Template/Cyclone/Preheater_Cyclone",
    # Chimney components
    "chimney": "Template/Chimney/Chimney",
    "chimney_without_smoke": "Template/Chimney/Chimney_WithoutSmoke",
    "chimney_without_smoke_grey": "Template/Chimney/Chimney_WithoutSmoke_Grey",
    "exhaust_stack": "Template/Chimney/ExhaustStack",
    "exhaust_stack_1": "Template/Chimney/ExhaustStack_1",
    # Conveyor components
    "conveyor": "Template/Conveyors/BeltConveyor",
    "belt_conveyor": "Template/Conveyors/BeltConveyor",
    "chain_conveyor": "Template/Conveyors/ChainConveyor",
    "rectangular_conveyor": "Template/Conveyors/RectangularConveyor",
    "duct": "Template/Conveyors/Duct",
    "spiral": "Template/Conveyors/Spiral_Single",
    # Furnace/Kiln components
    "furnace": "Template/Furnace/Kiln",
    "kiln": "Template/Furnace/Kiln",
    "kiln_1": "Template/Furnace/Kiln_1",
    "kiln_2": "Template/Furnace/Kiln_2",
    "rotary_kiln": "Template/Furnace/Rotary_Kiln",
    # Other customized components
    "turbine": "Template/Turbine/Turbine",
    "boiler": "Template/Steam_Operations/Boiler",
    "crusher": "Template/Crusher/Crusher",
    "calciner": "Template/Calciner/Calciner",
    "stacker": "Template/Stacker/Stacker",
    "stacker_reclaimer": "Template/Stacker/Stacker",
    "separator": "Template/Seperator/Seperator",
    "clinker_silo": "Template/Clinker_Silo",
    "packers": "Template/Packers",
    "lorry": "Template/Lorry",
    "hag": "Template/Hag",
    "whrs": "Template/WHRS",
    # Tank components (customized variants)
    "square_tank": "Template/Tanks/Square_Tank",
    "water_tank": "Template/Tanks/WaterTank",
    "tank_6": "Template/Tanks/Tank_6",
    "separator_tank": "Template/Tanks/Separator_tank",
    "storage_tank": "Template/Tanks/Storage Tank",
    # Valve components (customized variants)
    "three_way_valve": "Template/Valves/ThreeWayValve",
    # Pump components (standard but with template paths for reference)
    "motor_pump": "Template/Pumps/Motor_Pump",
    "motor_pump_left": "Template/Pumps/Motor_Pump _Left",
    "motor_pump_inverted": "Template/Pumps/Motor_Pump_Inverted",
    "motor_pump_simple": "Template/Pumps/Motor_Pump_Simple",
    "motor_gear": "Template/Pumps/Motor_gear",
    "motor_gear_single": "Template/Pumps/Motor_gear_Single",
    "motor_gear_box": "Template/Pumps/Motor_gear_box",
    "pump": "Template/Pumps/Pump",
    "pump_1": "Template/Pumps/Pump_1",
    "pump_double": "Template/Pumps/Pump_Double",
    "pump_double_left": "Template/Pumps/Pump_Double_Left",
    "pump_down_right": "Template/Pumps/Pump_Down_Right",
    "pump_feedback": "Template/Pumps/Pump_Feedback",
    "pump_left": "Template/Pumps/Pump_Left",
    "pump_right": "Template/Pumps/Pump_Right",
    "pump_two_way": "Template/Pumps/Pump_Two_way",
    "vertical_pump": "Template/Pumps/Vertical_Pump",
    "arrow": "Template/Pumps/Arrow",
    "arrow_head": "Template/Pumps/Arrow_head",
    "vertical_arrow": "Template/Pumps/VerticalArrow",
    # Standard components (for reference, though they use standard ignition types)
    "valve": "Template/Valves/Valve",
    # Pipe templates
    "arrow_pipe": "Template/Pipe/ArrowPipe",
    "arrow_pipe_black": "Template/Pipe/ArrowPipeBlack",
    "arrow_pipe_1": "Template/Pipe/ArrowPipe_1",
    "arrow_pipe_blue": "Template/Pipe/ArrowPipe_Blue",
    "arrow_pipe_green": "Template/Pipe/ArrowPipe_Green",
    "arrow_pipe_gray": "Template/Pipe/ArrowPipe_Gry",
    "l_shape_pipe": "Template/Pipe/LShapePipe",
    "m_shape_pipe": "Template/Pipe/MShapePipe",
    "without_arrow_pipe": "Template/Pipe/WithoutArrowPipe",
    "without_arrow_pipe_black": "Template/Pipe/WithoutArrowPipeBlack",
    "without_arrow_pipe_skyblue": "Template/Pipe/WithoutArrowPipeSkyblue",
    "without_arrow_pipe_skyblue_dotline": "Template/Pipe/WithoutArrowPipeSkyblue_Dotline",
    "without_arrow_pipe_yellow_dotline": "Template/Pipe/WithoutArrowPipeYellow_Dotline",
    "without_arrow_pipe_gray": "Template/Pipe/WithoutArrowPipe_Gry"
}

# Template image mapping for template matching
TEMPLATE_IMAGE_MAPPING = {
    "cyclone": "Template/Cyclone/Cyclone/thumbnail.png",
    "cyclone_separator": "Template/Cyclone/Cyclone_Separator/thumbnail.png",
    "turbine": "Template/Turbine/Turbine/thumbnail.png",
    "boiler": "Template/Steam_Operations/Boiler/thumbnail.png",
    "conveyor": "Template/Conveyors/BeltConveyor/thumbnail.png",
    "crusher": "Template/Crusher/Crusher/thumbnail.png",
    "furnace": "Template/Furnace/Kiln/thumbnail.png",
    "calciner": "Template/Calciner/Calciner/thumbnail.png",
    "stacker": "Template/Stacker/Stacker/thumbnail.png",
    "separator": "Template/Seperator/Seperator/thumbnail.png",
    "square_tank": "Template/Tanks/Square_Tank/thumbnail.png",
    "water_tank": "Template/Tanks/WaterTank/thumbnail.png",
    "tank_6": "Template/Tanks/Tank_6/thumbnail.png",
    "separator_tank": "Template/Tanks/Separator_tank/thumbnail.png",
    "storage_tank": "Template/Tanks/Storage Tank/thumbnail.png",
    "motor_pump": "Template/Pumps/Motor_Pump/thumbnail.png",
    "pump": "Template/Pumps/Pump/thumbnail.png",
    "valve": "Template/Valves/Valve/thumbnail.png",
    # Pipe template images
    "arrow_pipe": "Template/Pipe/ArrowPipe/thumbnail.png",
    "arrow_pipe_black": "Template/Pipe/ArrowPipeBlack/thumbnail.png",
    "arrow_pipe_1": "Template/Pipe/ArrowPipe_1/thumbnail.png",
    "arrow_pipe_blue": "Template/Pipe/ArrowPipe_Blue/thumbnail.png",
    "arrow_pipe_green": "Template/Pipe/ArrowPipe_Green/thumbnail.png",
    "arrow_pipe_gray": "Template/Pipe/ArrowPipe_Gry/thumbnail.png",
    "l_shape_pipe": "Template/Pipe/LShapePipe/thumbnail.png",
    "m_shape_pipe": "Template/Pipe/MShapePipe/thumbnail.png",
    "without_arrow_pipe": "Template/Pipe/WithoutArrowPipe/thumbnail.png",
    "without_arrow_pipe_black": "Template/Pipe/WithoutArrowPipeBlack/thumbnail.png",
    "without_arrow_pipe_skyblue": "Template/Pipe/WithoutArrowPipeSkyblue/thumbnail.png",
    "without_arrow_pipe_skyblue_dotline": "Template/Pipe/WithoutArrowPipeSkyblue_Dotline/thumbnail.png",
    "without_arrow_pipe_yellow_dotline": "Template/Pipe/WithoutArrowPipeYellow_Dotline/thumbnail.png",
    "without_arrow_pipe_gray": "Template/Pipe/WithoutArrowPipe_Gry/thumbnail.png"
}
