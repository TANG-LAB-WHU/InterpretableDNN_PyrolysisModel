#!/bin/bash
#SBATCH --job-name=PyroBot_MC_Predictions
#SBATCH --partition=9a14a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192                # 192-core high-throughput scanning
#SBATCH --account=tangsiqi
#SBATCH --output=mc_predictions_%j.log
#SBATCH --error=mc_predictions_%j.err

#-----------------------------------------------------------------------------#
# PyroBot: Unified Monte Carlo Prediction Slurm Scheduler (192 Cores)
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

echo "Active Python interpreter: $(which python)"
python --version

# -----------------------------------------------------------------------------
# Thread Isolation to Prevent Intel MKL / OpenBLAS Core Thrashing
# -----------------------------------------------------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "Allocated CPU cores per task: $SLURM_CPUS_PER_TASK"
echo "Running parallelized 2,000,000 Monte Carlo predictions (both Ea & Yield)..."

# Define task-aligned output directory with Slurm ID
if [ -n "$SLURM_JOB_ID" ]; then
    OUT_DIR="results/mc_predictions_${SLURM_JOB_ID}"
else
    OUT_DIR="results/mc_predictions_local"
fi
mkdir -p "$OUT_DIR"

# Execute central CLI orchestrator in unbuffered mode
python -u run_pyrobot.py \
    --mode mc \
    --samples 2000000 \
    --seed 2026 \
    --cores $SLURM_CPUS_PER_TASK \
    --out-dir "$OUT_DIR"

# Copy Slurm log and error files to the consolidated output directory at the end and clean up originals
if [ -n "$SLURM_JOB_ID" ]; then
    cp "$SLURM_SUBMIT_DIR/mc_predictions_${SLURM_JOB_ID}.log" "$OUT_DIR/" 2>/dev/null
    cp "$SLURM_SUBMIT_DIR/mc_predictions_${SLURM_JOB_ID}.err" "$OUT_DIR/" 2>/dev/null
    rm -f "$SLURM_SUBMIT_DIR/mc_predictions_${SLURM_JOB_ID}.log" "$SLURM_SUBMIT_DIR/mc_predictions_${SLURM_JOB_ID}.err" 2>/dev/null
fi

echo "======================================================================="
echo "Monte Carlo Prediction finished successfully at: $(date)"
echo "======================================================================="
