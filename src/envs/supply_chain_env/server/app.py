import os

try:
    from core.env_server import create_fastapi_app
except ImportError:
    # Dummy mock for type checking
    def create_fastapi_app(*args, **kwargs): return None

from .environment import SupplyChainEnvironment

# Automatically inject the environment configurations based on OS env vars
task_name = os.environ.get("SUPPLY_CHAIN_TASK", "hard")

env = SupplyChainEnvironment(task_name=task_name)
app = create_fastapi_app(env)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
