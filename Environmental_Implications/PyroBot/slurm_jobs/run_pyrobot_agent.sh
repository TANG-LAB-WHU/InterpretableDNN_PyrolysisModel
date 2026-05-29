#!/bin/bash
#SBATCH --job-name=PyroBot_QwenAgent
#SBATCH --partition=9a14a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192                # 192 cores allocated for MoE local CPU execution or parallel utilities
#SBATCH --account=tangsiqi
#SBATCH --output=slurm_jobs/pyrobot_agent_%j.log
#SBATCH --error=slurm_jobs/pyrobot_agent_%j.err

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
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"
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
# 3. Serving Backend Deployment Strategy (Dynamic GPU/CPU Auto-Detection)
# -----------------------------------------------------------------------------
PORT=8000
HOST="127.0.0.1"
MODEL_ID="Qwen/Qwen3.6-27B-Instruct"

# Check for active CUDA GPUs
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "======================================================================="
    echo "GPU DETECTED: Deploying high-throughput vLLM OpenAI API Server..."
    echo "======================================================================="
    
    # Launch vLLM server in the background
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_ID" \
        --tensor-parallel-size 1 \
        --port "$PORT" \
        --host "$HOST" \
        --disable-log-requests &
    VLLM_PID=$!
    
else
    echo "======================================================================="
    echo "CPU-ONLY NODE: Deploying localized llama.cpp or GGUF cache engine..."
    echo "Utilizing OpenMP threads: $SLURM_CPUS_PER_TASK to execute Qwen3.6 MoE..."
    echo "======================================================================="
    
    # Check if a llama.cpp binary exists in the path
    if command -v llama-cli &> /dev/null || command -v ./llama-cli &> /dev/null; then
        # Serving localized GGUF MoE quantized weights in the background
        # (Assuming GGUF version is downloaded to the project scratch)
        GGUF_MODEL="/scratch/tangsiqi/ai_models/qwen/Qwen3.6-27B/Qwen3.6-27B-Q8_0.gguf"
        if [ -f "$GGUF_MODEL" ]; then
            llama-server \
                --model "$GGUF_MODEL" \
                --port "$PORT" \
                --host "$HOST" \
                --threads "$SLURM_CPUS_PER_TASK" &
            VLLM_PID=$!
        else
            echo "Warning: GGUF model not found at $GGUF_MODEL. Falling back to native HF pipeline."
            VLLM_PID=0
        fi
    else
        echo "Llama.cpp not found. The agent orchestrator will fall back to loading the model"
        echo "locally using Hugging Face pipelines inside the orchestrator process."
        VLLM_PID=0
    fi
fi

# -----------------------------------------------------------------------------
# 4. Serving Health Check (Wait for Qwen3.6 Brain to Come Online)
# -----------------------------------------------------------------------------
if [ "$VLLM_PID" -gt 0 ]; then
    echo "Waiting for local model server to initialize and load weights (PID: $VLLM_PID)..."
    until curl -s "http://$HOST:$PORT/v1/models" > /dev/null; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
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

# Run the agent in non-interactive batch-command mode for this job
python -u run_pyrobot.py \
    --mode agent \
    --chatbot \
    --cores "$SLURM_CPUS_PER_TASK"

# Clean up background server processes if spawned
if [ "$VLLM_PID" -gt 0 ]; then
    echo "Shutting down local Qwen3.6 serving backend (PID: $VLLM_PID)..."
    kill "$VLLM_PID"
    wait "$VLLM_PID" 2>/dev/null
fi

echo "======================================================================="
echo "PyroBot Agent Job completed successfully at: $(date)"
echo "======================================================================="
