import inspect
import importlib.util
from pathlib import Path
from fastapi import APIRouter
from backend.agents.generic_agent import GenericAgent

router = APIRouter()

# Correct path to: backend/agents/
AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"

@router.get("/")
async def list_agents():
    """
    Dynamically list all agent class names in backend/agents/*.py,
    excluding 'generic_agent.py' and '__init__.py'.

    Example output: ["BTC", "ETH", "SOL"]
    """
    agent_names = []

    for file in AGENTS_DIR.glob("*.py"):
        if file.name in {"__init__.py", "generic_agent.py"}:
            continue

        module_name = f"backend.agents.{file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                print(f"❌ Failed to load agent file {file.name}: {e}")
                continue

            for name, cls in inspect.getmembers(module, inspect.isclass):
                if issubclass(cls, GenericAgent) and cls is not GenericAgent:
                    agent_name = cls.__name__.replace("Agent", "")  # e.g. BTCUSDTAgent → BTCUSDT
                    agent_names.append(agent_name)

    return sorted(agent_names)
