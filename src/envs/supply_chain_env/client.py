from __future__ import annotations

from typing import Any

import httpx

from .models import StepResult, SupplyChainAction, SupplyChainObservation, SupplyChainState


class SupplyChainEnvClient:
    """Simple HTTP client for the FastAPI supply chain server."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 10.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def health(self) -> dict[str, Any]:
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()

    def reset(self, task_name: str | None = None) -> SupplyChainObservation:
        payload = {"task_name": task_name} if task_name else {}
        response = self._client.post("/reset", json=payload)
        response.raise_for_status()
        return SupplyChainObservation(**response.json())

    def step(self, action: SupplyChainAction) -> StepResult:
        response = self._client.post("/step", json=action.model_dump(exclude_none=True))
        response.raise_for_status()
        observation = SupplyChainObservation(**response.json())
        return StepResult(observation=observation, reward=observation.reward, done=observation.done)

    def state(self) -> SupplyChainState:
        response = self._client.get("/state")
        response.raise_for_status()
        return SupplyChainState(**response.json())

    def __enter__(self) -> "SupplyChainEnvClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
