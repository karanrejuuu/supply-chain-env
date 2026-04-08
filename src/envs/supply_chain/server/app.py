from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
from .environment import SupplyChainEnv

app = FastAPI(title="Supply Chain Environment")

# Global environment instance
env = SupplyChainEnv()


class ResetRequest(BaseModel):
    task: Optional[str] = "easy"


class StepRequest(BaseModel):
    action_type: str = "do_nothing"
    order_quantity: int = 20
    supplier_choice: str = "standard"
    transport_mode: str = "normal"


@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()):
    global env
    task = request.task if request.task in ("easy", "medium", "hard") else "easy"
    env = SupplyChainEnv(task=task)
    state = env.reset()
    return state


@app.post("/step")
async def step(request: StepRequest):
    action = request.dict()
    obs, reward, done = env.step(action)
    return {
        "observation": obs,
        "reward": round(reward, 4),
        "done": done,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
