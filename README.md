# Supply Chain Disruption Agent

> An **OpenEnv**-compatible reinforcement learning environment that simulates
> real-world supply chain disruptions and challenges agents to recover efficiently.

---

## 📦 Project Overview

Modern supply chains are fragile. A delayed shipment, a supplier going offline, or a
blocked transport route can cascade into stock-outs and financial losses within hours.
This environment models those dynamics and asks an AI agent to take corrective actions
under uncertainty, with the goal of satisfying customer demand as quickly and cheaply
as possible.

---

## 🌍 Real-World Motivation

| Disruption Type     | Real Example                                  |
|---------------------|-----------------------------------------------|
| Supplier delay      | COVID-19 factory shutdowns in 2020–2021       |
| Transport blockage  | Suez Canal blockage (Ever Given, 2021)        |
| Inventory stock-out | Semiconductor shortage hitting auto industry  |

By training agents on this environment, organisations can develop automated
decision-support systems that respond to disruptions faster than human planners.

---

## 🔭 Observation Space

| Field              | Type    | Values              | Description                        |
|--------------------|---------|---------------------|------------------------------------|
| `inventory`        | `int`   | ≥ 0                 | Units currently in stock           |
| `demand`           | `int`   | ≥ 0                 | Units required to satisfy orders   |
| `supplier_status`  | `str`   | `"ok"` / `"delayed"`| Primary supplier availability      |
| `transport_status` | `str`   | `"ok"` / `"blocked"`| Logistics route status             |
| `time_elapsed`     | `int`   | ≥ 0                 | Steps taken in the current episode |
| `cost`             | `float` | ≥ 0.0               | Cumulative cost of actions taken   |

---

## 🎮 Action Space

| Action                | Effect                                       | Cost   |
|-----------------------|----------------------------------------------|--------|
| `order_more_stock`    | inventory += 30                              | +50    |
| `switch_supplier`     | supplier_status → "ok"                       | +30    |
| `reroute_transport`   | transport_status → "ok"                      | +20    |
| `delay_order`         | No improvement; incurs reward penalty        | 0      |
| `do_nothing`          | No improvement; incurs reward penalty        | 0      |

---

## 📋 Task Descriptions

### Easy
- **Initial state**: inventory=10, demand=50, supplier=ok, transport=ok
- **Optimal solution**: `["order_more_stock"]`
- **Challenge**: Straightforward reorder; infrastructure is intact.

### Medium
- **Initial state**: inventory=10, demand=50, supplier=**delayed**, transport=ok
- **Optimal solution**: `["switch_supplier", "order_more_stock"]`
- **Challenge**: Must fix the supplier before restocking is effective.

### Hard
- **Initial state**: inventory=5, demand=80, supplier=**delayed**, transport=**blocked**
- **Optimal solution**: `["switch_supplier", "reroute_transport", "order_more_stock"]`
- **Challenge**: Cascading failures requiring coordinated multi-step recovery.

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10+
- pip

### Install locally

```bash
# Clone / download the project
cd supply-chain-env

# Install dependencies
pip install -r requirements.txt
```

### Run the baseline agent

```bash
python scripts/run_inference.py
```

---

## 🐳 Docker

### Build the image

```bash
docker build -t supply-chain .
```

### Run the container

```bash
docker run supply-chain
```

---

## 🖥️ Example Output

```
============================================================
  Supply Chain Disruption Agent — Task: HARD
============================================================

Initial State:
  {'inventory': 5, 'demand': 80, 'supplier_status': 'delayed',
   'transport_status': 'blocked', 'time_elapsed': 0, 'cost': 0.0}

  Step 01 | Action: switch_supplier        | Reward: +0.50 | Done: False
  Step 02 | Action: reroute_transport      | Reward: +0.50 | Done: False
  Step 03 | Action: order_more_stock       | Reward: +1.70 | Done: True

============================================================
  Episode Complete
============================================================

Final State:
  {'inventory': 35, 'demand': 80, 'supplier_status': 'ok',
   'transport_status': 'ok', 'time_elapsed': 3, 'cost': 100.0}

Total Steps  : 3
Total Reward : 2.7000
Actions Taken: ['switch_supplier', 'reroute_transport', 'order_more_stock']

Grader Score : 1.0000  (optimal: ['switch_supplier', 'reroute_transport', 'order_more_stock'])
============================================================
```

---

## 📁 Project Structure

```
supply-chain-env/
├── env/
│   ├── __init__.py         # Package exports
│   ├── environment.py      # SupplyChainEnv (OpenEnv interface)
│   ├── models.py           # Pydantic Observation & Action models
│   ├── tasks.py            # Easy / Medium / Hard task definitions
│   └── grader.py           # Deterministic grader (0.0–1.0)
├── scripts/
│   └── run_inference.py    # Rule-based baseline agent
├── openenv.yaml            # OpenEnv manifest
├── Dockerfile              # Container build instructions
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🏆 Reward Function Summary

| Condition                                      | Reward  |
|------------------------------------------------|---------|
| Action matches next optimal step               | +0.3    |
| Action improves system state                   | +0.2    |
| Demand satisfied (terminal)                    | +1.0    |
| Action does not match optimal step             | −0.3    |
| `delay_order`                                  | −0.2    |
| `do_nothing`                                   | −0.3    |

---

## 📜 License

MIT — free to use, modify, and distribute.
