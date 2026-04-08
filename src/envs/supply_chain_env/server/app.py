import os

from fastapi import FastAPI
from pydantic import BaseModel

from .environment import SupplyChainEnvironment
from ..models import SupplyChainAction


class ResetRequest(BaseModel):
    task_name: str | None = None


def create_app(task_name: str | None = None) -> FastAPI:
    app = FastAPI(title="Supply Chain Environment", version="0.1.0")
    default_task_name = task_name or os.environ.get("SUPPLY_CHAIN_TASK", "hard")
    app.state.env = SupplyChainEnvironment(task_name=default_task_name)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/reset")
    def reset(request: ResetRequest | None = None) -> dict:
        selected_task = request.task_name if request and request.task_name else default_task_name
        app.state.env = SupplyChainEnvironment(task_name=selected_task)
        return app.state.env.reset().model_dump()

    @app.post("/step")
    def step(action: SupplyChainAction) -> dict:
        return app.state.env.step(action).model_dump()

    @app.get("/state")
    def state() -> dict:
        return app.state.env.state.model_dump()

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("envs.supply_chain_env.server.app:app", host="0.0.0.0", port=8000)
