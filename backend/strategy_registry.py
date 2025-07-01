import os
import json
from pathlib import Path
from typing import Dict, Optional, List

STRATEGY_DIR = Path("backend/backtester/strategies")
STRATEGY_REGISTRY_FILE = STRATEGY_DIR / "registry.json"

STRATEGY_DIR.mkdir(parents=True, exist_ok=True)

def list_strategies() -> Dict[str, list]:
    """
    Lists all strategy JSON files grouped by symbol.
    Returns: { symbol: [strategy_file_names] }
    """
    registry = {}
    for file in STRATEGY_DIR.glob("*.json"):
        if file.name == "registry.json":
            continue
        try:
            with open(file, "r") as f:
                data = json.load(f)
                symbol = data.get("symbol", "UNKNOWN")
                if symbol not in registry:
                    registry[symbol] = []
                registry[symbol].append(file.name)
        except Exception as e:
            print(f"[ERROR] Failed to read {file}: {e}")
    return registry


def load_strategy(symbol: str, strategy_id: str) -> Optional[dict]:
    """
    Loads a specific strategy JSON file.
    """
    file_path = STRATEGY_DIR / f"{symbol}_strategy_{strategy_id}.json"
    if file_path.exists():
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load strategy {file_path}: {e}")
    return None


def save_strategy(symbol: str, strategy_id: str, data: dict) -> bool:
    """
    Saves a strategy JSON to disk and registers it in registry.json.
    """
    file_path = STRATEGY_DIR / f"{symbol}_strategy_{strategy_id}.json"
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        register_strategy(symbol, strategy_id)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save strategy {file_path}: {e}")
        return False


def delete_strategy(symbol: str, strategy_id: str) -> bool:
    """
    Deletes a saved strategy file.
    """
    file_path = STRATEGY_DIR / f"{symbol}_strategy_{strategy_id}.json"
    try:
        if file_path.exists():
            file_path.unlink()
            _unregister_strategy(symbol, strategy_id)
            return True
        return False
    except Exception as e:
        print(f"[ERROR] Failed to delete {file_path}: {e}")
        return False


def get_strategy_by_symbol(symbol: str) -> List[dict]:
    """
    Loads all strategies for a given symbol.
    """
    strategies = []
    for file in STRATEGY_DIR.glob(f"{symbol}_strategy_*.json"):
        try:
            with open(file, "r") as f:
                strategies.append(json.load(f))
        except Exception as e:
            print(f"[ERROR] Failed to read strategy {file}: {e}")
    return strategies


# ✅ NEW — Register strategy ID into local registry
def register_strategy(symbol: str, strategy_id: str):
    """
    Adds strategy to local registry.json file.
    """
    registry = {}
    if STRATEGY_REGISTRY_FILE.exists():
        with open(STRATEGY_REGISTRY_FILE, "r") as f:
            registry = json.load(f)

    if symbol not in registry:
        registry[symbol] = []

    if strategy_id not in registry[symbol]:
        registry[symbol].append(strategy_id)

    with open(STRATEGY_REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def _unregister_strategy(symbol: str, strategy_id: str):
    """
    Removes strategy from registry.json.
    """
    if not STRATEGY_REGISTRY_FILE.exists():
        return

    with open(STRATEGY_REGISTRY_FILE, "r") as f:
        registry = json.load(f)

    if symbol in registry and strategy_id in registry[symbol]:
        registry[symbol].remove(strategy_id)
        if not registry[symbol]:
            del registry[symbol]

    with open(STRATEGY_REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


# ✅ NEW — Get list of registered strategy IDs
def get_registered_strategies(symbol: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Returns the strategy registry. If symbol is provided, returns only for that symbol.
    """
    if not STRATEGY_REGISTRY_FILE.exists():
        return {}

    with open(STRATEGY_REGISTRY_FILE, "r") as f:
        registry = json.load(f)

    if symbol:
        return {symbol: registry.get(symbol, [])}

    return registry
