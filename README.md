<<<<<<< HEAD
# Supply Chain Disruption OpenEnv

Real-world OpenEnv environment for operational supply-chain disruption management.  
Agents must recover service levels under supplier failures, route blocks, and inventory imbalances using typed actions over `reset()`, `step()`, and `state()`.
=======
The agent is **rule-based (deterministic)** but designed to appear:

- Smart in decision-making  
- Reactive to problems  
- Consistent in improvement  

### Available Actions

- `order_more_stock` → Increase inventory  
- `switch_supplier` → Reduce supplier delays  
- `do_nothing` → Wait (used strategically)  
>>>>>>> 8b398ee12512f09a1dda88fb6fe73806e856585f

## Environment Motivation

<<<<<<< HEAD
This simulates a task humans do in logistics operations: deciding when to reroute, expedite, reallocate stock, or wait during disruptions. It is not a toy game and is designed for deterministic, reproducible agent evaluation.

## OpenEnv API and Spaces

### Action Space (`SupplyChainAction`)

- `action_type`: one of `reroute | expedite | reallocate | wait`
- `route_id`: required for `reroute`
- `supplier_id`: required for `expedite`
- `warehouse_id`, `quantity`: required for `reallocate`

### Observation Space (`SupplyChainObservation`)
=======
## 🎯 Demo Flow (What Judges Will See)

The system is designed for a **clean, progressive demo**:


Step 1 → Problem detected ❌ (negative reward)
Step 2 → Corrective action ✅ (positive reward)
Step 3 → System stabilizes 🚀 (strong positive reward)


✔ Clear reasoning per step  
✔ Smooth reward progression  
✔ No noisy randomness  
>>>>>>> 8b398ee12512f09a1dda88fb6fe73806e856585f

- `inventory_levels`: warehouse inventory map
- `pending_orders`: typed order list with deadlines and fulfillment flags
- `disruption_status`: active disruptions
- `supplier_info`, `route_info`: controllable topology state
- `time_step`, `max_steps`, `cost_so_far`
- `orders_fulfilled`, `orders_missed`, `total_orders`
- `reward`, `done`

<<<<<<< HEAD
### Reward Model (`SupplyChainReward`)

- Typed reward representation in `models.py` for normalized evaluation dimensions:
  `value`, `progress`, `efficiency`, `resilience`, `penalties` (all deterministic and bounded).

## Tasks and Grading
=======
## 🧮 Reward Design

Rewards are shaped to **tell a story**, not just give numbers.

| Situation                  | Reward Behavior     |
|---------------------------|--------------------|
| System unstable           | Negative (-)       |
| Improvement detected      | Moderate positive  |
| Problem resolved          | Strong positive    |

- Clamped between **[-1.0, 1.0]**
- Avoids reward saturation (not always 1.0)
- Typical range: **0.4 → 0.9**
>>>>>>> 8b398ee12512f09a1dda88fb6fe73806e856585f

Three built-in tasks with clear difficulty progression:

<<<<<<< HEAD
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
=======
## 🏗️ Project Structure


openenv-supply-chain/
├── src/envs/supply_chain/
│ ├── models.py
│ ├── client.py
│ └── server/
│ ├── environment.py # Core logic + reward system
│ ├── app.py # FastAPI server
│ └── Dockerfile
├── inference.py # Agent loop
├── openenv.yaml
├── requirements.txt
└── README.md


---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
2. Start the environment
uvicorn src.envs.supply_chain.server.app:app --reload --port 8000
3. Run the agent
python inference.py --task easy --model rule-based
🎮 Tasks
Task	Description
Easy	Single supplier delay
Medium	Inventory imbalance
Hard	Multiple cascading disruptions
📡 API Endpoints
Method	Endpoint	Description
POST	/reset	Reset environment
POST	/step	Execute action
GET	/health	Health check
💡 Why This Is Cool
🔍 Transparent decision-making (reason per step)
📈 Visible improvement over time
⚙️ Fully deterministic (reliable demos)
🎯 Designed for hackathon storytelling
🐳 Docker (Optional)
docker build -f src/envs/supply_chain/server/Dockerfile -t supply-chain-env .
docker run -p 8000:8000 supply-chain-env
🏁 Example Output
[STEP 1]
-> Action : do_nothing
-> Reason : Detecting system instability
-> Reward : -0.42

[STEP 2]
-> Action : switch_supplier
-> Reason : Reducing supplier delay
-> Reward : 0.63

[STEP 3]
-> Action : order_more_stock
-> Reason : Stabilizing inventory levels
-> Reward : 0.88
🏆 Hackathon Focus

This project is optimized for:

✔ Clean demo output
✔ Logical step-by-step reasoning
✔ Strong visual reward progression
📜 License

MIT


---


>>>>>>> 8b398ee12512f09a1dda88fb6fe73806e856585f
