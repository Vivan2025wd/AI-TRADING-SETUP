import json
from pathlib import Path
from typing import Dict, Optional, List

# --- Directories and Paths ---
STRATEGY_DIR = Path("backend/storage/strategies")
STRATEGY_REGISTRY_FILE = STRATEGY_DIR / "registry.json"

# Ensure strategy directory exists
STRATEGY_DIR.mkdir(parents=True, exist_ok=True)


# 🔍 Load a specific strategy file
def load_strategy(symbol: str, strategy_id: str) -> Optional[dict]:
    file_path = STRATEGY_DIR / f"{symbol}_strategy_{strategy_id}.json"
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load strategy '{file_path.name}': {e}")
        return None


# 💾 Save a strategy file and register it
def save_strategy(symbol: str, strategy_id: str, strategy_data: dict) -> bool:
    file_path = STRATEGY_DIR / f"{symbol}_strategy_{strategy_id}.json"
    try:
        with open(file_path, "w") as f:
            json.dump(strategy_data, f, indent=2)
        register_strategy(symbol, strategy_id)
        return True
    except Exception as e:
        print(f"[ERROR] Could not save strategy {file_path.name}: {e}")
        return False


# 🗑️ Delete strategy file and unregister it
def delete_strategy(symbol: str, strategy_id: str) -> bool:
    file_path = STRATEGY_DIR / f"{symbol}_strategy_{strategy_id}.json"
    try:
        if file_path.exists():
            file_path.unlink()
            _unregister_strategy(symbol, strategy_id)
            return True
        return False
    except Exception as e:
        print(f"[ERROR] Could not delete strategy {file_path.name}: {e}")
        return False


# 📚 List all strategy files grouped by symbol
def list_strategies() -> Dict[str, List[str]]:
    registry = {}
    for file in STRATEGY_DIR.glob("*_strategy_*.json"):
        if file.name == "registry.json":
            continue
        try:
            with open(file, "r") as f:
                data = json.load(f)
                symbol = data.get("symbol", "UNKNOWN")
                registry.setdefault(symbol, []).append(file.stem)
        except Exception as e:
            print(f"[ERROR] Failed to read strategy file {file.name}: {e}")
    return registry


# 📂 Get all strategies for a given symbol
def get_strategy_by_symbol(symbol: str) -> List[dict]:
    strategies = []
    for file in STRATEGY_DIR.glob(f"{symbol}_strategy_*.json"):
        try:
            with open(file, "r") as f:
                strategies.append(json.load(f))
        except Exception as e:
            print(f"[ERROR] Failed to read strategy {file.name}: {e}")
    return strategies


# ✅ Register strategy ID into registry.json
def register_strategy(symbol: str, strategy_id: str):
    registry = {}
    if STRATEGY_REGISTRY_FILE.exists():
        try:
            with open(STRATEGY_REGISTRY_FILE, "r") as f:
                registry = json.load(f)
        except Exception as e:
            print(f"[ERROR] Could not load registry.json: {e}")
            registry = {}

    registry.setdefault(symbol, [])
    if strategy_id not in registry[symbol]:
        registry[symbol].append(strategy_id)

    try:
        with open(STRATEGY_REGISTRY_FILE, "w") as f:
            json.dump(registry, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Could not update registry.json: {e}")


# ❌ Remove strategy from registry.json
def _unregister_strategy(symbol: str, strategy_id: str):
    if not STRATEGY_REGISTRY_FILE.exists():
        return

    try:
        with open(STRATEGY_REGISTRY_FILE, "r") as f:
            registry = json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not load registry.json: {e}")
        return

    if symbol in registry and strategy_id in registry[symbol]:
        registry[symbol].remove(strategy_id)
        if not registry[symbol]:
            del registry[symbol]

    try:
        with open(STRATEGY_REGISTRY_FILE, "w") as f:
            json.dump(registry, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Could not write to registry.json: {e}")


# 📖 Get strategy registry fully or by symbol
def get_registered_strategies(symbol: Optional[str] = None) -> Dict[str, List[str]]:
    if not STRATEGY_REGISTRY_FILE.exists():
        return {}

    try:
        with open(STRATEGY_REGISTRY_FILE, "r") as f:
            registry = json.load(f)
        return {symbol: registry.get(symbol, [])} if symbol else registry
    except Exception as e:
        print(f"[ERROR] Could not read registry.json: {e}")
        return {}
