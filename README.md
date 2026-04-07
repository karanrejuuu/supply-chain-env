# Supply Chain Disruption Agent

> An **OpenEnv**-compatible reinforcement learning environment that simulates
> real-world supply chain disruptions and challenges AI agents to optimise
> recovery across a multi-warehouse, multi-supplier network.

---

## 📦 Problem Statement

Global supply chains are fragile. A single supplier delay in Shenzhen can halt
production in Detroit. A blocked shipping lane can cascade into empty shelves
within days. The 2021 Suez Canal blockage, COVID-era semiconductor shortages,
and the 2023 Red Sea disruptions collectively cost the global economy over
**$80 billion**.

This environment models those dynamics:

| Disruption Type      | Real-World Example                             |
|----------------------|------------------------------------------------|
| Supplier delay       | COVID-19 factory shutdowns (2020–2021)         |
| Transport blockage   | Suez Canal blockage (Ever Given, 2021)         |
| Inventory imbalance  | Semiconductor shortage hitting auto industry   |
| Cascading failure    | Red Sea rerouting + port congestion (2023)     |

An AI agent must **diagnose**, **prioritise**, and **resolve** these disruptions
through corrective actions to satisfy customer demand as efficiently as possible.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   SupplyChainEnv                    │
│        (OpenEnv-compatible reset/step API)          │
│                                                     │
│   ┌───────────────────────────────────────────┐     │
│   │          SupplyChainSimulator              │     │
│   │  ┌───────────┐  ┌──────────┐  ┌────────┐ │     │
│   │  │ Warehouses │  │ Suppliers│  │ Routes │ │     │
│   │  └───────────┘  └──────────┘  └────────┘ │     │
│   │  ┌───────────┐  ┌──────────────────────┐ │     │
│   │  │  Orders   │  │    Disruptions       │ │     │
│   │  └───────────┘  └──────────────────────┘ │     │
│   └───────────────────────────────────────────┘     │
│                                                     │
│   Reward Shaping ─── Grader ─── Task Definitions    │
└─────────────────────────────────────────────────────┘
```

| Module            | Purpose                                         |
|-------------------|-------------------------------------------------|
| `environment.py`  | OpenEnv interface (reset/step), reward shaping   |
| `simulator.py`    | Core simulation engine (state, actions, time)    |
| `models.py`       | Pydantic schemas for Action & Observation        |
| `tasks.py`        | 3 task definitions (easy/medium/hard)            |
| `grader.py`       | Deterministic scoring (0.0 – 1.0)               |

---

## 📋 Task Descriptions

### Easy — Single Supplier Delay
- **Scenario**: Primary supplier is delayed; warehouse-A runs short
- **Goal**: Expedite the supplier and fulfil all orders before deadlines
- **Max Steps**: 10
- **Disruptions**: 1 (supplier delay)

### Medium — Multi-Warehouse Imbalance
- **Scenario**: warehouse-A is overstocked, warehouse-C is nearly empty; orders target warehouse-C
- **Goal**: Reallocate inventory and manage transport to fulfil all orders
- **Max Steps**: 15
- **Disruptions**: 0 (logistics challenge)

### Hard — Cascading Disruptions
- **Scenario**: Supplier failure + route blockage + low inventory across all warehouses
- **Goal**: Dynamically adapt — expedite supplier, unblock routes, reallocate stock
- **Max Steps**: 20
- **Disruptions**: 2 (supplier delay + route block)

---

## 🎮 Action Schema

| Field          | Type           | Description                          |
|----------------|----------------|--------------------------------------|
| `action_type`  | `str` (required) | `reroute` · `expedite` · `reallocate` · `wait` |
| `route_id`     | `str?`         | Target route (for reroute)           |
| `supplier_id`  | `str?`         | Target supplier (for expedite)       |
| `warehouse_id` | `str?`         | Target warehouse (for reallocate)    |
| `quantity`     | `int?`         | Units to transfer (for reallocate)   |

---

## 🔭 Observation Schema

| Field               | Type        | Description                                    |
|---------------------|-------------|------------------------------------------------|
| `inventory_levels`  | `dict`      | Inventory count per warehouse                  |
| `pending_orders`    | `list`      | Pending orders with destination, qty, deadline  |
| `disruption_status` | `list`      | Active disruptions with type, target, severity  |
| `cost_so_far`       | `float`     | Cumulative cost of all actions                 |
| `time_step`         | `int`       | Current simulation step                        |
| `supplier_info`     | `dict`      | Supplier capacity, reliability, active status   |
| `route_info`        | `dict`      | Route cost, delay prob, flow rate, active status|

---

## ⚡ Reward Function

| Signal                          | Reward     |
|---------------------------------|------------|
| Order fulfilled                 | **+1.0**   |
| Disruption resolved             | **+0.5**   |
| Inventory reallocated           | **+0.3**   |
| Terminal: all orders done       | **+2.0**   |
| Missed deadline                 | **−1.5**   |
| Unnecessary/invalid action      | **−0.2**   |
| Repeated wait (×2)              | **−0.3**   |
| Repeated wait (×3+)            | **−0.5**   |
| Time pressure (per step)        | **−0.05**  |

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10+
- pip

### Install Locally

```bash
cd supply-chain-env
pip install -r requirements.txt
```

### Run the Baseline Agent

```bash
# Heuristic policy (no API key needed)
python scripts/run_inference.py

# LLM policy (requires HF_TOKEN)
export HF_TOKEN=your_token_here
python scripts/run_inference.py --mode llm
```

---

## 🐳 Docker

```bash
# Build
docker build -t supply-chain-env .

# Run
docker run supply-chain-env
```

---

## 📊 Baseline Results

| Task     | Score  | Fulfilled | Cost   | Steps |
|----------|--------|-----------|--------|-------|
| Easy     | 0.9620 | 2/2       | 30.00  | 1     |
| Medium   | 0.9787 | 3/3       | 20.00  | 1     |
| Hard     | 0.9520 | 4/4       | 60.00  | 3     |
| **Avg**  | **0.9642** | —     | —      | —     |

*Results from the built-in heuristic policy. The heuristic acts as an expert upper-bound baseline.*

---

## 🏆 Grading Formula

```
score = 0.5 × fulfilment_rate + 0.3 × cost_efficiency + 0.2 × speed
```

- **Fulfilment rate**: orders fulfilled / total orders
- **Cost efficiency**: 1 − (total_cost / max_plausible_cost)
- **Speed**: 1 − (steps_used / max_steps)

Score is deterministic and always in **[0.0, 1.0]**.

---

## 📁 Project Structure

```
supply-chain-env/
├── env/
│   ├── __init__.py         # Package exports
│   ├── environment.py      # SupplyChainEnv (OpenEnv interface)
│   ├── simulator.py        # SupplyChainSimulator (core engine)
│   ├── models.py           # Pydantic Action & Observation models
│   ├── tasks.py            # Easy / Medium / Hard task definitions
│   └── grader.py           # Deterministic grader (0.0–1.0)
├── scripts/
│   └── run_inference.py    # Baseline inference (heuristic + LLM)
├── openenv.yaml            # OpenEnv manifest
├── Dockerfile              # Container build
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 📜 License

MIT — free to use, modify, and distribute.
