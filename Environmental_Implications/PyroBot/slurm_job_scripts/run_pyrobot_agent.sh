#!/bin/bash
#SBATCH --job-name=PyroBot_QwenAgent
#SBATCH --partition=9a14a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192                # 192 cores allocated for MoE local CPU execution or parallel utilities
#SBATCH --account=tangsiqi
#SBATCH --output=pyrobot_agent_%j.log
#SBATCH --error=pyrobot_agent_%j.err

#-----------------------------------------------------------------------------#
# PyroBot: Autonomous Agent (Qwen3.6-35B) Server & Orchestrator Scheduler
# Designed for Wuhan University HPC Cluster (9a14a Partition, CPU/GPU Modes)
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
# 3. Serving Backend Deployment Strategy (Dynamic GPU/CPU Auto-Detection with llama.cpp)
# -----------------------------------------------------------------------------
PORT=8000
HOST="127.0.0.1"
MODEL_ID="Qwen/Qwen3.6-27B-Instruct"
GGUF_MODEL="/scratch/tangsiqi/ai_models/qwen/Qwen3.6-27B/Qwen3.6-27B-Q8_0.gguf"

SERVER_PID=0
LLAMA_BIN=""

# Dynamic detection of local llama.cpp serving binary
if command -v llama-server &> /dev/null; then
    LLAMA_BIN="llama-server"
elif command -v llama-cli &> /dev/null; then
    LLAMA_BIN="llama-cli"
elif [ -f "./llama-server" ]; then
    LLAMA_BIN="./llama-server"
elif [ -f "./llama-cli" ]; then
    LLAMA_BIN="./llama-cli"
fi

if [ -n "$LLAMA_BIN" ]; then
    if [ -f "$GGUF_MODEL" ]; then
        # Check for active CUDA GPUs via nvidia-smi
        if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
            echo "======================================================================="
            echo "GPU DETECTED: Deploying llama.cpp server with CUDA GPU acceleration..."
            echo "Offloading all layers (ngl=99) of Qwen3.6-27B-Q8 to the active GPU(s)."
            echo "======================================================================="
            
            # Start llama-server with full GPU offloading for maximum throughput
            "$LLAMA_BIN" \
                --model "$GGUF_MODEL" \
                --port "$PORT" \
                --host "$HOST" \
                --threads "$SLURM_CPUS_PER_TASK" \
                --n-gpu-layers 99 &
            SERVER_PID=$!
        else
            echo "======================================================================="
            echo "CPU-ONLY NODE: Deploying llama.cpp server on CPU threads..."
            echo "Utilizing OpenMP threads: $SLURM_CPUS_PER_TASK to execute Qwen3.6 MoE..."
            echo "======================================================================="
            
            # Start llama-server in pure CPU thread mode (ngl=0)
            "$LLAMA_BIN" \
                --model "$GGUF_MODEL" \
                --port "$PORT" \
                --host "$HOST" \
                --threads "$SLURM_CPUS_PER_TASK" \
                --n-gpu-layers 0 &
            SERVER_PID=$!
        fi
    else
        echo "======================================================================="
        echo "WARNING: GGUF model not found at $GGUF_MODEL."
        echo "The agent orchestrator will fall back to loading the model"
        echo "locally using Hugging Face pipelines inside the orchestrator process."
        echo "======================================================================="
    fi
else
    echo "======================================================================="
    echo "WARNING: llama-server or llama-cli binaries not found in environment PATH."
    echo "The agent orchestrator will fall back to loading the model"
    echo "locally using Hugging Face pipelines inside the orchestrator process."
    echo "======================================================================="
fi

# -----------------------------------------------------------------------------
# 4. Serving Health Check (Wait for Qwen3.6 Brain to Come Online)
# -----------------------------------------------------------------------------
if [ "$SERVER_PID" -gt 0 ]; then
    echo "Waiting for local model server to initialize and load weights (PID: $SERVER_PID)..."
    until curl -s "http://$HOST:$PORT/v1/models" > /dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "CRITICAL ERROR: Serving process died during loading phase. Check logs above."
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
AGENT_ARGS=("--mode" "agent" "--cores" "$SLURM_CPUS_PER_TASK")

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
    echo "Execution Mode: Batch Command Mode (Default PNAS target demonstration query)"
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
