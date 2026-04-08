from types import SimpleNamespace

from fastapi import FastAPI

from envs.supply_chain_env.models import SupplyChainAction
from envs.supply_chain_env.server.app import app
from envs.supply_chain_env.server.environment import SupplyChainEnvironment
from envs.supply_chain_env.server.grader import grade
from scripts.policies import llm_policy


class _DummyMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _DummyChoice:
    def __init__(self, content: str) -> None:
        self.message = _DummyMessage(content)


class _DummyCompletions:
    def __init__(self, content: str) -> None:
        self._content = content

    def create(self, **_kwargs):
        return SimpleNamespace(choices=[_DummyChoice(self._content)])


class _DummyChat:
    def __init__(self, content: str) -> None:
        self.completions = _DummyCompletions(content)


class _DummyClient:
    def __init__(self, content: str) -> None:
        self.chat = _DummyChat(content)


def test_env_reset_step():
    env = SupplyChainEnvironment(task_name="easy")
    observation = env.reset()
    assert observation.time_step == 0
    assert isinstance(observation.inventory_levels, dict)

    next_observation = env.step(SupplyChainAction(action_type="wait"))
    assert next_observation.time_step == 1
    assert env.state.step_count == 1
    assert env.state.score is not None


def test_done_logic():
    env = SupplyChainEnvironment(task_name="easy")
    env.reset()
    env._sim.warehouses = {"warehouse-A": 0, "warehouse-B": 0}
    for supplier in env._sim.suppliers.values():
        supplier["active"] = False
    for route in env._sim.routes.values():
        route["active"] = False

    for _ in range(8):
        observation = env.step(SupplyChainAction(action_type="wait"))
        if env.state.orders_missed:
            break

    assert env.state.orders_missed >= 1
    assert not observation.done
    assert env.state.step_count < env._task.max_steps


def test_grader():
    score = grade(
        {
            "orders_fulfilled": 3,
            "total_orders": 4,
            "total_cost": 25.0,
            "time_step": 5,
            "max_steps": 10,
        }
    )
    assert 0.0 <= score <= 1.0
    assert score == 0.76


def test_llm_policy_parsing():
    env = SupplyChainEnvironment(task_name="easy")
    observation = env.reset()
    client = _DummyClient('{"action_type":"expedite","supplier_id":"supplier-1"}')
    action = llm_policy(observation, client=client, model_name="dummy-model", structured_output=True)
    assert action.action_type == "expedite"
    assert action.supplier_id == "supplier-1"


def test_server_app_exists():
    assert isinstance(app, FastAPI)
