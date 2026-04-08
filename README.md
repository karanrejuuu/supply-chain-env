# Supply Chain Disruption OpenEnv

Real-world OpenEnv environment for operational supply-chain disruption management.  
Agents must recover service levels under supplier failures, route blocks, and inventory imbalances using typed actions over `reset()`, `step()`, and `state()`.

## Environment Motivation

This simulates a task humans do in logistics operations: deciding when to reroute, expedite, reallocate stock, or wait during disruptions. It is not a toy game and is designed for deterministic, reproducible agent evaluation.

## OpenEnv API and Spaces

### Action Space (`SupplyChainAction`)

- `action_type`: one of `reroute | expedite | reallocate | wait`
- `route_id`: required for `reroute`
- `supplier_id`: required for `expedite`
- `warehouse_id`, `quantity`: required for `reallocate`

### Observation Space (`SupplyChainObservation`)

- `inventory_levels`: warehouse inventory map
- `pending_orders`: typed order list with deadlines and fulfillment flags
- `disruption_status`: active disruptions
- `supplier_info`, `route_info`: controllable topology state
- `time_step`, `max_steps`, `cost_so_far`
- `orders_fulfilled`, `orders_missed`, `total_orders`
- `reward`, `done`

### Reward Model (`SupplyChainReward`)

- Typed reward representation in `models.py` for normalized evaluation dimensions:
  `value`, `progress`, `efficiency`, `resilience`, `penalties` (all deterministic and bounded).

## Tasks and Grading

Three built-in tasks with clear difficulty progression:

- `easy`: single supplier delay
- `medium`: multi-warehouse rebalance
- `hard`: cascading supplier + route disruptions

Deterministic grader returns score in `[0.0, 1.0]` based on:

- fulfillment rate (weighted highest)
- cost efficiency
- speed

Reward shaping provides trajectory feedback (partial progress) and penalties for wasteful loops/costly behavior.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell activation:

```powershell
.\.venv\Scripts\activate
```

## Run the Environment

```bash
python -m uvicorn --app-dir src envs.supply_chain_env.server.app:app --host 0.0.0.0 --port 8000
```

Routes:

- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`

## Baseline Inference (`inference.py`)

The required root inference script uses OpenAI client calls and prints strict structured logs:

- `[START]`
- `[STEP]`
- `[END]`

Set variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` (or `OPENAI_API_KEY`)

Run:

```bash
python inference.py
```

## Docker and HF Spaces

Build and run:

```bash
docker build -t supply-chain-env .
docker run --rm -p 8000:8000 supply-chain-env
```

The container starts Uvicorn and serves the OpenEnv API, ready for HF Spaces deployment.

## Validation

Pre-submission checks:

```bash
openenv validate
python -m pytest
```

Run the bundled validator against your HF Space URL:

```bash
bash validate-submission.sh https://your-space.hf.space
```

Mandatory runtime variables for judging:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
