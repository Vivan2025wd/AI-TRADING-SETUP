import json
from pathlib import Path
from typing import Dict, Any

# Base directory where strategy JSON files are stored
STRATEGY_DIR = Path(__file__).resolve().parent.parent / "storage" / "strategies"
STRATEGY_DIR.mkdir(parents=True, exist_ok=True)


def validate_strategy(strategy: Dict[str, Any]) -> None:
    """
    Ensure strategy dict contains required keys and correct types.
    Raises ValueError if invalid.
    """
    if "symbol" not in strategy:
        raise ValueError("Missing 'symbol' in strategy")
    if "indicators" not in strategy:
        raise ValueError("Missing 'indicators' in strategy")
    if not isinstance(strategy["indicators"], dict):
        raise ValueError("'indicators' should be a dictionary")


def save_strategy_to_file(symbol: str, strategy_id: str, strategy: Dict[str, Any]) -> None:
    """
    Save the strategy dict to a JSON file.
    Also saves a `_strategy_default.json` file if strategy_id is 'default'.
    """
    validate_strategy(strategy)
    filename = f"{symbol}_strategy_{strategy_id}.json"
    file_path = STRATEGY_DIR / filename

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(strategy, f, indent=4)
    except Exception as e:
        raise IOError(f"Failed to save strategy file {file_path}: {e}")


def load_strategy_from_file(symbol: str, strategy_id: str) -> Dict[str, Any]:
    """
    Load a specific user strategy JSON file.
    """
    file_path = STRATEGY_DIR / f"{symbol}_strategy_{strategy_id}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Strategy file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        strategy = json.load(f)

    validate_strategy(strategy)
    return strategy


def load_strategy_from_json_string(json_str: str) -> Dict[str, Any]:
    """
    Load and validate a strategy from a JSON string, typically from frontend input.
    """
    try:
        strategy = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")

    validate_strategy(strategy)
    return strategy


def parse_strategy_json(json_str: str) -> Dict[str, Any]:
    """
    Entry point to parse and validate JSON strategy string.
    """
    return load_strategy_from_json_string(json_str)


def load_strategy_for_symbol(symbol: str) -> Dict[str, Any]:
    """
    Loads the default strategy for a given symbol.
    If default is missing, loads the first available strategy file.
    Raises FileNotFoundError if none are found.
    """
    symbol = symbol.upper()
    default_path = STRATEGY_DIR / f"{symbol}_strategy_default.json"

    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            strategy = json.load(f)
            validate_strategy(strategy)
            return strategy

    # Fallback: any other strategy file
    candidates = list(STRATEGY_DIR.glob(f"{symbol}_strategy_*.json"))
    for candidate in candidates:
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                strategy = json.load(f)
                validate_strategy(strategy)
                return strategy
        except Exception:
            continue  # skip invalid files

    raise FileNotFoundError(f"No valid strategy found for symbol: {symbol}")
