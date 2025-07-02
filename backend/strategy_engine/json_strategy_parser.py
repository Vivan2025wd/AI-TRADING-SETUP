import json
from pathlib import Path
from typing import Dict, Any

# Base directory where strategy JSON files are stored
STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtester" / "strategies"
STRATEGY_DIR.mkdir(parents=True, exist_ok=True)

def load_strategy_for_symbol(symbol: str) -> Dict[str, Any]:
    """
    Load a default strategy JSON file for the given symbol.
    The file is expected to be named as e.g. 'BTCUSDT_strategy_default.json'
    """
    filename = f"{symbol}_strategy_default.json"
    file_path = STRATEGY_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Strategy file not found: {file_path}")

    with open(file_path, "r") as f:
        strategy = json.load(f)

    validate_strategy(strategy)
    return strategy

def load_strategy_from_file(symbol: str, strategy_id: str) -> Dict[str, Any]:
    """
    Load a user strategy JSON file and return it as a dict.
    Expects files named like: BTCUSDT_strategy_001.json
    """
    file_path = STRATEGY_DIR / f"{symbol}_strategy_{strategy_id}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Strategy file not found: {file_path}")

    with open(file_path, "r") as f:
        strategy = json.load(f)

    validate_strategy(strategy)
    return strategy

def load_strategy_from_json_string(json_str: str) -> Dict[str, Any]:
    """
    Load and validate a strategy from a raw JSON string (e.g., from frontend)
    """
    try:
        strategy = json.loads(json_str)
        validate_strategy(strategy)
        return strategy
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format")
    except Exception as e:
        raise ValueError(f"Strategy validation failed: {str(e)}")

def validate_strategy(strategy: Dict[str, Any]):
    """
    Validate that the JSON strategy contains required keys
    """
    if "symbol" not in strategy:
        raise ValueError("Missing 'symbol' in strategy")
    if "indicators" not in strategy:
        raise ValueError("Missing 'indicators' in strategy")
    if not isinstance(strategy["indicators"], dict):
        raise ValueError("'indicators' should be a dictionary")

def parse_strategy_json(json_str: str) -> Dict[str, Any]:
    """
    Frontend entry: parse JSON string and return validated strategy dict.
    """
    return load_strategy_from_json_string(json_str)
