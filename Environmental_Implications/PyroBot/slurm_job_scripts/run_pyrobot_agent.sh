#!/bin/bash
#SBATCH --job-name=PyroBot_QwenAgent
#SBATCH --account=tangsiqi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=pyrobot_agent_%j.log
#SBATCH --error=pyrobot_agent_%j.err

# [Submission Guidelines] Do not hardcode the partition in the script. Submit via terminal using:
# Pure CPU test: sbatch -p 9a14a --cpus-per-task=64 run_pyrobot_agent.sh
# V100 high-perf: sbatch -p gpu --gres=gpu:2 --cpus-per-task=10 run_pyrobot_agent.sh
# A100 extreme: sbatch -p a100x4 --gres=gpu:1 --cpus-per-task=16 run_pyrobot_agent.sh

#-----------------------------------------------------------------------------#
# PyroBot: Autonomous Agent (Qwen3.6-27B) Server & Orchestrator Scheduler
# Designed for Wuhan University HPC Cluster (Multi-Partition Smart Routing)
#-----------------------------------------------------------------------------#

echo "======================================================================="
echo "Starting Slurm Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Node assigned:      $SLURM_JOB_NODELIST"
echo "Submission dir:     $SLURM_SUBMIT_DIR"
echo "Start time:         $(date)"
echo "======================================================================="

# Establish robust zero-configuration workspace anchoring
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    # Running under Slurm scheduler context (prevents spool copy path errors)
    case "$SLURM_SUBMIT_DIR" in
        */slurm_job_scripts) PROJECT_ROOT="$( dirname "$SLURM_SUBMIT_DIR" )" ;;
        *)                   PROJECT_ROOT="$SLURM_SUBMIT_DIR" ;;
    esac
else
    # Running under direct shell execution context
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    case "$SCRIPT_DIR" in
        */slurm_job_scripts) PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )" ;;
        *)                   PROJECT_ROOT="$SCRIPT_DIR" ;;
    esac
fi
cd "$PROJECT_ROOT"
echo "Active workspace root: $(pwd)"

# -----------------------------------------------------------------------------
# 1. Hugging Face Global Cache Redirection (Prevents Home Quota Overflow)
# -----------------------------------------------------------------------------
# Redirection of heavy model static weights to the project's scratch space
export HF_HOME="/scratch/tangsiqi/huggingface_cache"
mkdir -p "$HF_HOME"
echo "Redirected HF_HOME caching registry to project scratch partition: $HF_HOME"

# -----------------------------------------------------------------------------
# 2. Conda Environment Activation
# -----------------------------------------------------------------------------
echo "Initializing Anaconda..."
if [ -f "$HOME/project/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/project/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    export PATH="$HOME/project/miniconda3/bin:$HOME/anaconda3/bin:$HOME/miniconda3/bin:$PATH"
    source conda activate 2>/dev/null
fi

echo "Activating virtual environment: pyrolysis_model_dnn..."
conda activate pyrolysis_model_dnn || conda activate base

# -----------------------------------------------------------------------------
# 3. Serving Backend Deployment Strategy (Smart Hardware Routing with llama.cpp)
# -----------------------------------------------------------------------------
PORT=8000
HOST="127.0.0.1"
MODEL_ID="Qwen/Qwen3.6-27B-Instruct"

LLAMA_DIR="/home/$USER/project/software/llama.cpp"
# Default path to the downloaded Qwen 27B GGUF model
GGUF_MODEL="${LLAMA_DIR}/models/shared_weights/qwen/Qwen3.6-27B/Qwen3.6-27B-Q8_0.gguf"

SERVER_PID=0

# Load base compilation environment
module load scl/gcc13 2>/dev/null || true

# Smart hardware routing logic
if [ "$SLURM_JOB_PARTITION" == "a100x4" ]; then
    echo "[Info] A100 queue detected, launching Ampere architecture engine..."
    module load nvidia/cuda/12.9 2>/dev/null || true
    LLAMA_BIN="${LLAMA_DIR}/build_a100/bin/llama-server"
    GPU_ARGS=("--n-gpu-layers" "99" "-sm" "row" "-fa" "on")
    
elif [ "$SLURM_JOB_PARTITION" == "gpu" ]; then
    echo "[Info] V100 queue detected, launching Volta architecture engine..."
    module load nvidia/cuda/12.9 2>/dev/null || true
    LLAMA_BIN="${LLAMA_DIR}/build_v100/bin/llama-server"
    GPU_ARGS=("--n-gpu-layers" "99" "-sm" "row" "-fa" "on")
    
elif [ "$SLURM_JOB_PARTITION" == "9a14a" ]; then
    echo "[Info] Pure CPU queue detected, launching AMD AVX-512 compute core..."
    LLAMA_BIN="${LLAMA_DIR}/build_cpu/bin/llama-server"
    GPU_ARGS=("--n-gpu-layers" "0")
    
else
    echo "[Warning] Unknown or unspecified partition: $SLURM_JOB_PARTITION, fallback to default CPU."
    LLAMA_BIN="${LLAMA_DIR}/build_cpu/bin/llama-server"
    GPU_ARGS=("--n-gpu-layers" "0")
fi

# Core Fix 2: Prevent CPU thread thrashing by capping at 64 threads for the LLM backend
LLAMA_THREADS=${SLURM_CPUS_PER_TASK:-16}
if [ "$LLAMA_THREADS" -gt 64 ]; then
    LLAMA_THREADS=64
    echo "[Info] Capping llama.cpp threads to 64 to prevent NUMA thrashing."
fi

if [ -f "$LLAMA_BIN" ]; then
    if [ -f "$GGUF_MODEL" ]; then
        echo "======================================================================="
        echo "Deploying llama.cpp server (Backend: $LLAMA_BIN)..."
        echo "======================================================================="
        
        mkdir -p logs
        # Core Fix 3: Isolate backend logs to keep Agent logs clean
        "$LLAMA_BIN" \
            -m "$GGUF_MODEL" \
            --port "$PORT" \
            --host "$HOST" \
            -c 32768 \
            -t "$LLAMA_THREADS" \
            --jinja \
            --reasoning-format none \
            "${GPU_ARGS[@]}" > "logs/llama_server_${SLURM_JOB_ID:-local}.log" 2>&1 &
        SERVER_PID=$!
    else
        echo "======================================================================="
        echo "WARNING: GGUF model not found at $GGUF_MODEL."
        echo "The agent orchestrator will fall back to loading the model locally."
        echo "======================================================================="
    fi
else
    echo "======================================================================="
    echo "WARNING: llama-server binary not found at $LLAMA_BIN."
    echo "Please ensure you have compiled llama.cpp in the respective directories."
    echo "The agent orchestrator will fall back to loading the model locally."
    echo "======================================================================="
fi

# -----------------------------------------------------------------------------
# 4. Serving Health Check (Wait for Qwen3.6 Brain to Come Online)
# -----------------------------------------------------------------------------
if [ "$SERVER_PID" -gt 0 ]; then
    echo "Waiting for local model server to initialize and load weights (PID: $SERVER_PID)..."
    until curl -sf "http://$HOST:$PORT/v1/models" > /dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "CRITICAL ERROR: Serving process died. Check logs/llama_server_${SLURM_JOB_ID:-local}.log."
            exit 1
        fi
        sleep 5
    done
    echo "Local Qwen3.6-27B Model Server is online and ready!"
    export OPENAI_API_BASE="http://$HOST:$PORT/v1"
    export OPENAI_API_KEY="local-token-pyrobot"
else
    echo "Skipping background server startup. Running offline HF pipeline loader."
    export OPENAI_API_BASE="offline"
fi

# -----------------------------------------------------------------------------
# 5. Core Execution: Launch PyroBot Agent Orchestrator
# -----------------------------------------------------------------------------
echo "======================================================================="
echo "Launching Qwen3.6-27B Autonomous Scientific Agent Orchestrator..."
echo "======================================================================="

# Initialize Python agent parameters array to ensure robust space and quote preservation
AGENT_ARGS=("--mode" "agent" "--cores" "${SLURM_CPUS_PER_TASK:-16}")

# Define task-aligned output directory with Slurm ID
if [ -n "$SLURM_JOB_ID" ]; then
    OUT_DIR="results/pyrobot_agent_${SLURM_JOB_ID}"
else
    OUT_DIR="results/pyrobot_agent_local"
fi
mkdir -p "$OUT_DIR"
AGENT_ARGS+=("--out-dir" "$OUT_DIR")

# Dynamic execution mode determination based on script arguments:
# 1. run_pyrobot_agent.sh --chatbot  => Starts interactive conversational shell
# 2. run_pyrobot_agent.sh "query"    => Starts batch mode with custom natural language query
# 3. run_pyrobot_agent.sh            => Starts batch mode with default target demonstration query
if [ "$1" == "--chatbot" ]; then
    AGENT_ARGS+=("--chatbot")
    echo "Execution Mode: Interactive Scientific Chatbot Shell"
elif [ -n "$1" ]; then
    AGENT_ARGS+=("--query" "$1")
    echo "Execution Mode: Batch Command Mode (Custom query: \"$1\")"
else
    echo "Execution Mode: Batch Command Mode"
fi

python -u run_pyrobot.py "${AGENT_ARGS[@]}"

# Clean up background server processes if spawned
if [ "$SERVER_PID" -gt 0 ]; then
    echo "Shutting down local Qwen3.6 serving backend (PID: $SERVER_PID)..."
    kill "$SERVER_PID"
    wait "$SERVER_PID" 2>/dev/null
fi

# Copy Slurm log and error files to the consolidated output directory at the end and clean up originals
if [ -n "$SLURM_JOB_ID" ]; then
    cp "$SLURM_SUBMIT_DIR/pyrobot_agent_${SLURM_JOB_ID}.log" "$OUT_DIR/" 2>/dev/null
    cp "$SLURM_SUBMIT_DIR/pyrobot_agent_${SLURM_JOB_ID}.err" "$OUT_DIR/" 2>/dev/null
    rm -f "$SLURM_SUBMIT_DIR/pyrobot_agent_${SLURM_JOB_ID}.log" "$SLURM_SUBMIT_DIR/pyrobot_agent_${SLURM_JOB_ID}.err" 2>/dev/null
fi

echo "======================================================================="
echo "PyroBot Agent Job completed successfully at: $(date)"
echo "======================================================================="
