import requests


class SupplyChainClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def reset(self, task: str = "easy") -> dict:
        response = requests.post(
            f"{self.base_url}/reset",
            json={"task": task},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def step(self, action: dict) -> dict:
        response = requests.post(
            f"{self.base_url}/step",
            json=action,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def health(self) -> dict:
        response = requests.get(f"{self.base_url}/health", timeout=5)
        response.raise_for_status()
        return response.json()
