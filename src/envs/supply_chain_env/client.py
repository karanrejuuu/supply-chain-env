from typing import Any

try:
    from core.http_env_client import HTTPEnvClient
    from core.types import StepResult
except ImportError:
    class HTTPEnvClient: pass
    class StepResult: pass

from .models import SupplyChainAction, SupplyChainObservation, SupplyChainState

class SupplyChainEnvClient(HTTPEnvClient):
    """
    OpenEnv HTTP Client implementation for the Supply Chain simulator.
    Usage:
        client = SupplyChainEnvClient(base_url="http://localhost:8000")
        obs_result = client.reset()
    """
    
    def _step_payload(self, action: SupplyChainAction) -> dict:
        return {
            "action_type": action.action_type,
            "route_id": action.route_id,
            "supplier_id": action.supplier_id,
            "warehouse_id": action.warehouse_id,
            "quantity": action.quantity
        }
    
    def _parse_result(self, payload: dict) -> Any:
        obs_dict = {
            "inventory_levels": payload.get("inventory_levels", {}),
            "pending_orders": payload.get("pending_orders", []),
            "disruption_status": payload.get("disruption_status", []),
            "cost_so_far": payload.get("cost_so_far", 0.0),
            "time_step": payload.get("time_step", 0),
            "supplier_info": payload.get("supplier_info", {}),
            "route_info": payload.get("route_info", {}),
            "done": payload.get("done", False),
            "reward": payload.get("reward", 0.0)
        }
        
        obs = SupplyChainObservation(**obs_dict)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False)
        )
    
    def _parse_state(self, payload: dict) -> SupplyChainState:
        return SupplyChainState(**payload)
