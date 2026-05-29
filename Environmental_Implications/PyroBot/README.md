# PyroBot: Autonomous Closed-Loop AI Pyrolysis Recipe Inverse Design Framework

**PyroBot** is an autonomous, target-driven co-pyrolysis recipe discovery and thermochemical path optimization engine. By coupling pre-trained high-fidelity deep neural networks ([`bpDNN2Ea`](models/bpDNN2Ea) and [`bpDNN2Yield`](models/bpDNN2Yield)) with physics-informed differential kinetics simulators and the state-of-the-art **`Qwen3.6-35B-A3B-Instruct`** Mixture-of-Experts (MoE) brain, **PyroBot** transitions co-pyrolysis design from traditional brute-force experimental search to autonomous closed-loop scientific discovery.

Designed for high-performance deployment on the **Wuhan University (WHU) HPC Cluster** (192 physical CPU cores, GPU partitions).

---

## 📂 Project Architecture

```text
PyroBot/
├── README.md                              # Unified User & HPC Cluster execution guide
├── requirements.txt                       # Consolidated environment dependencies
├── config.json                            # Global configurations & chemical safeguards
├── pyrolysis_Bot.md                      # PNAS-level Agentic AI architecture & proposal
│
├── data/                                  # Immutable Data Layer (Read-Only)
│   ├── raw/                               # Raw sewage sludge baselines
│   └── reference/                         # 118 literature feedstocks compilation
│
├── models/                                # pre-trained Model Weights Layer
│   ├── bpDNN2Ea/                          # activation energy network weights
│   └── bpDNN2Yield/                       # product yields (Char/Liquid/Gas) weights
│
├── core/                                  # Computational Core Engine (Mathematical & Physics ODE Solvers)
│   ├── missing_value_handler.py           # Missing data imputer
│   ├── dnn_surrogates.py                  # MATLAB NN loading and vectorized inference APIs
│   ├── continuous_optimizer.py            # Bounded Brent & SLSQP continuous optimizers
│   └── tg_differential_simulator.py       # ODE solid-state kinetics virtual TG/DTG curves solver
│
├── agent/                                 # Orchestration Layer (Autonomous LLM Agent)
│   ├── prompt_templates.py                # System few-shot CoT and feedback loop templates
│   └── pyrobot_orchestrator.py            # Closed-loop LangChain agent decision logic
│
├── slurm_jobs/                            # HPC Jobs Layer (SLURM Templates)
│   ├── run_mc_predictions.sh              # 192-core parallel Monte Carlo baseline scan
│   ├── run_blending_optimizations.sh      # 192-core dual-scenario recipe optimizers
│   └── run_pyrobot_agent.sh               # Local Qwen3.6 MoE server & agent bootstrapper
│
└── run_pyrobot.py                         # Central orchestrator entry CLI / Conversational Chatbot
```

---

## 🚀 WHU HPC Cluster Execution Guide

### 1. high-performance Monte Carlo Scanning
Generate 2,000,000 baseline sewage sludge samples, run neural network inference on 192 cores in parallel, and export uncertainty violin charts:
```bash
sbatch slurm_jobs/run_mc_predictions.sh

# Monitor log in real time
tail -f slurm_jobs/mc_predictions_*.log
```
Outputs are written to `results/mc_predictions/`.

### 2. Dual-Scenario simplex Continuous Optimization Sweep
Runs continuous SLSQP optimizations across active candidate promoters under Scenario A ($50\%$ sludge lock, $25\%$ individual additive caps) and Scenario B ($80\%$ sludge lock, $10\%$ individual additive caps) sequentially:
```bash
sbatch slurm_jobs/run_blending_optimizations.sh

# Monitor log in real time
tail -f slurm_jobs/blending_optimizations_*.log
```
Outputs (recipe CSV spreadsheets and ranking figures) are written to `results/optimized_blends/`.

### 3. Unified Local Qwen Serving & Autonomous Agent Orchestration
The Slurm script `slurm_jobs/run_pyrobot_agent.sh` dynamically activates the virtual environment, redirects Hugging Face caches to `/scratch` to prevent home quota overflows, and spawns the local high-performance **llama.cpp** model server (supporting dynamic GPU CUDA-acceleration and 192-core CPU execution locks). 

This unified script supports three dynamic run-time execution formats via shell parameters:

*   **Interactive Conversational Chatbot Mode** (Launches the interactive terminal Chatbot shell inside active interactive allocations, e.g. `salloc`):
    ```bash
    bash slurm_jobs/run_pyrobot_agent.sh --chatbot
    ```
*   **Non-Interactive Custom Batch Query** (Ingests a custom target objective query directly in batch-command mode):
    ```bash
    bash slurm_jobs/run_pyrobot_agent.sh "Design a co-pyrolysis recipe with municipal sewage sludge that maximizes Biochar above 40%."
    ```
*   **Non-Interactive Default PNAS Demonstration Query** (Directly schedules a background Slurm job executing the target inverse design query from [pyrolysis_Bot.md](pyrolysis_Bot.md#L97-L99) as a zero-configuration demo):
    ```bash
    sbatch slurm_jobs/run_pyrobot_agent.sh
    
    # Monitor logs in real time
    tail -f slurm_jobs/pyrobot_agent_*.log
    ```

---

## 💻 CLI Orchestrator Usage

You can also execute individual routines locally using the central CLI wrapper:

*   **Interactive Conversational Agent Chatbot**:
    ```bash
    python run_pyrobot.py --mode agent --chatbot
    ```
*   **One-Shot Scenario A Recipe Optimizer Sweep**:
    ```bash
    python run_pyrobot.py --mode optimize --scenario A --cores 8
    ```
*   **Minimal Monte Carlo Verification**:
    ```bash
    python run_pyrobot.py --mode mc --samples 1000 --cores 4
    ```

---

## 🔬 Scientific Core Safeguards
*   **Sparsity Filter**: Clips active additive recipe fractions $< 0.1\%$ to protect physical co-pyrolysis boundaries from numerical degeneracies.
*   **basicity Index (Slagging check)**: Computes Stoichiometric oxide basicity to prevent fluid-bed stiction runaways:
    $$\text{Slagging Index} = \frac{Fe_2O_3 + CaO + MgO + Na_2O + K_2O}{SiO_2 + Al_2O_3} \le 1.2$$
*   **Exothermic Runaway Shield**: ODE solver intercepts and penalizes recipe formulations generating DTG mass-loss rates exceeding $0.8\%/^\circ\text{C}$.
