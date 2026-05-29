#!/bin/bash
#SBATCH --job-name=PyroBot_Blending_Optimizations
#SBATCH --partition=9a14a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192                # Parallel processing for continuous optimizations
#SBATCH --account=tangsiqi
#SBATCH --output=blending_optimizations_%j.log
#SBATCH --error=blending_optimizations_%j.err

#-----------------------------------------------------------------------------#
# PyroBot: Continuous simplex recipe optimizations Slurm Scheduler (192 Cores)
# Designed for Wuhan University HPC Cluster (9a14a CPU/GPU Partition)
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
        */slurm_jobs) PROJECT_ROOT="$( dirname "$SLURM_SUBMIT_DIR" )" ;;
        *)            PROJECT_ROOT="$SLURM_SUBMIT_DIR" ;;
    esac
else
    # Running under direct shell execution context
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    case "$SCRIPT_DIR" in
        */slurm_jobs) PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )" ;;
        *)            PROJECT_ROOT="$SCRIPT_DIR" ;;
    esac
fi
cd "$PROJECT_ROOT"
echo "Active workspace root: $(pwd)"

# -----------------------------------------------------------------------------
# Conda Environment Activation
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
# Thread Isolation to Prevent Intel MKL / OpenBLAS Core Thrashing
# -----------------------------------------------------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "OMP_NUM_THREADS is set to: $OMP_NUM_THREADS"
echo "Allocated CPU cores per task: $SLURM_CPUS_PER_TASK"
echo "Optimizing co-pyrolysis blending recipes under dual industrial scenarios..."

# Define task-aligned output directory with Slurm ID
if [ -n "$SLURM_JOB_ID" ]; then
    OUT_DIR="results/blending_optimizations_${SLURM_JOB_ID}"
else
    OUT_DIR="results/blending_optimizations_local"
fi
mkdir -p "$OUT_DIR"

# SCENARIO A: Integrated Regional Multi-Waste Co-Processing Model
# Sludge lock = 50%, Individual additive ratio limit = 25% (total additive space = 50%)
echo "======================================================================="
echo "Executing SCENARIO A (50% Sewage Sludge locked, 25% max per additive)"
echo "======================================================================="
python -u run_pyrobot.py \
    --mode optimize \
    --scenario A \
    --cores $SLURM_CPUS_PER_TASK \
    --out-dir "${OUT_DIR}/blending_outputs_50"

# SCENARIO B: High-Throughput Sludge Disposal & Catalytic Co-processing Model
# Sludge lock = 80%, Individual additive ratio limit = 10% (total additive space = 20%)
echo "======================================================================="
echo "Executing SCENARIO B (80% Sewage Sludge locked, 10% max per additive)"
echo "======================================================================="
python -u run_pyrobot.py \
    --mode optimize \
    --scenario B \
    --cores $SLURM_CPUS_PER_TASK \
    --out-dir "${OUT_DIR}/blending_outputs_20"

# Copy Slurm log and error files to the consolidated output directory at the end and clean up originals
if [ -n "$SLURM_JOB_ID" ]; then
    cp "$SLURM_SUBMIT_DIR/blending_optimizations_${SLURM_JOB_ID}.log" "$OUT_DIR/" 2>/dev/null
    cp "$SLURM_SUBMIT_DIR/blending_optimizations_${SLURM_JOB_ID}.err" "$OUT_DIR/" 2>/dev/null
    rm -f "$SLURM_SUBMIT_DIR/blending_optimizations_${SLURM_JOB_ID}.log" "$SLURM_SUBMIT_DIR/blending_optimizations_${SLURM_JOB_ID}.err" 2>/dev/null
fi

echo "======================================================================="
echo "Dual-Scenario recipe optimizations finished successfully at: $(date)"
echo "======================================================================="
