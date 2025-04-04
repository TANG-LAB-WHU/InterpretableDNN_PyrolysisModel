#!/bin/bash
#SBATCH --job-name=full_GPM_SHAP
#SBATCH --output=output/full_analysis/full_%j.out
#SBATCH --error=output/full_analysis/full_%j.err
#SBATCH --time=24:00:00            # 24 hours runtime
#SBATCH --account=siqi-ic          # Account name from debug script
#SBATCH --partition=IllinoisComputes  # Partition name from debug script
#SBATCH --nodes=1                  # Request single node
#SBATCH --ntasks-per-node=16        # Utilize all cores on the node (assuming 16-core node)
#SBATCH --mem=48G                    # Request more memory for full analysis
#SBATCH --mail-type=BEGIN,END,FAIL    # Email notification
#SBATCH --mail-user=siqi@illinois.edu

# Get current working directory
WORK_DIR=$(pwd)
echo "=== Starting GPM_SHAP Full Analysis Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"

# Create complete directory structure
echo "Creating complete directory structure..."
mkdir -p data/processed
mkdir -p data/raw
mkdir -p output/debug_analysis
mkdir -p output/full_analysis
mkdir -p results/training/Figures
mkdir -p results/debug/Figures
mkdir -p results/analysis/debug/data
mkdir -p results/analysis/debug/figures
mkdir -p results/analysis/full/data
mkdir -p results/analysis/full/figures
mkdir -p results/optimization
mkdir -p results/best_model/Figures
mkdir -p src/model
mkdir -p src/scripts
mkdir -p src/visualization

# Set up environment variables
export MODEL_DIR="$WORK_DIR/src/model"
export TRAINING_DIR="$WORK_DIR/results/training"
export BEST_MODEL_DIR="$WORK_DIR/results/best_model"
export OPTIMIZATION_DIR="$WORK_DIR/results/optimization"
export ANALYSIS_DIR="$WORK_DIR/results/analysis/full"

echo "Model directory: $MODEL_DIR"
echo "Training directory: $TRAINING_DIR"
echo "Best model directory: $BEST_MODEL_DIR"
echo "Optimization directory: $OPTIMIZATION_DIR"
echo "Analysis directory: $ANALYSIS_DIR"

# Check required data files
echo "Checking data files..."
DATA_MISSING=0

if [ ! -f "data/processed/CopyrolysisFeedstock.mat" ]; then
    echo "ERROR: CopyrolysisFeedstock.mat file not found"
    DATA_MISSING=1
fi

if [ ! -f "data/raw/RawInputData.xlsx" ]; then
    echo "ERROR: RawInputData.xlsx file not found"
    DATA_MISSING=1
fi

if [ $DATA_MISSING -eq 1 ]; then
    echo "ERROR: Critical data files are missing. Exiting."
    exit 1
fi

# Check model files
echo "Checking model files..."
MODEL_FILES=("bpDNN4PyroProd.m" "nnbp.m" "nncheckgrad.m" "nnconfigure.m" "nncreate.m" 
             "nneval.m" "nnff.m" "nnfindbest.m" "nninit.m" "nnpostprocess.m" "nnpredict.m" 
             "nnprepare.m" "nnpreprocess.m" "nnstopcriteria.m" "nntrain.m" "nnupdatefigure.m")

MODEL_MISSING=0
for file in "${MODEL_FILES[@]}"; do
    if [ ! -f "src/model/$file" ]; then
        echo "WARNING: Required model file not found: src/model/$file"
        MODEL_MISSING=1
    fi
done

# Check script files
echo "Checking script files..."
SCRIPT_FILES=("run_analysis.m" "calc_shap_values.m" "export_shap_to_excel.m")
SCRIPT_MISSING=0

for file in "${SCRIPT_FILES[@]}"; do
    if [ ! -f "src/scripts/$file" ]; then
        echo "WARNING: src/scripts/$file not found. Please add this file."
        SCRIPT_MISSING=1
    fi
done

# Check visualization files
echo "Checking visualization files..."
VIZ_FILES=("plot_shap_results.m" "plot_shap_beeswarm.m" 
          "plot_shap_original_summary.m" "fix_colorbar_style.m")
VIZ_MISSING=0

for file in "${VIZ_FILES[@]}"; do
    if [ ! -f "src/visualization/$file" ]; then
        echo "WARNING: src/visualization/$file not found. Creating placeholder."
        VIZ_MISSING=1
    fi
done

# Copy data files to model directory
echo "Copying data files to model directory..."
cp -f "data/processed/CopyrolysisFeedstock.mat" "src/model/"
cp -f "data/raw/RawInputData.xlsx" "src/model/"
echo "Data files copied to model directory"

# Check for SHAP_Analysis metadata and copy if available
echo "Checking for SHAP feature metadata..."
if [ -d "SHAP_Analysis" ] && [ -f "SHAP_Analysis/Data/SHAP_metadata.xlsx" ]; then
    echo "Found SHAP metadata file. Copying for feature name reference..."
    mkdir -p "results/analysis/full/data"
    cp -f "SHAP_Analysis/Data/SHAP_metadata.xlsx" "results/analysis/full/data/"
    echo "SHAP metadata copied to results directory"
elif [ -d "SHAP_Analysis_Full" ] && [ -f "SHAP_Analysis_Full/Data/SHAP_metadata.xlsx" ]; then
    echo "Found SHAP_Analysis_Full metadata file. Copying for feature name reference..."
    mkdir -p "results/analysis/full/data"
    cp -f "SHAP_Analysis_Full/Data/SHAP_metadata.xlsx" "results/analysis/full/data/"
    echo "SHAP full metadata copied to results directory"
fi

# Set up MATLAB temporary directory
export MATLAB_TEMP_DIR="/scratch/${USER}/matlab_temp_${SLURM_JOB_ID}"
mkdir -p ${MATLAB_TEMP_DIR}
echo "MATLAB temp directory: $MATLAB_TEMP_DIR"

# Load MATLAB module
echo "Loading MATLAB module..."
module load matlab/24.1

# Run MATLAB full analysis script
echo "Running full SHAP analysis script..."
echo "MATLAB will use following directories:"
echo "- Working directory: $WORK_DIR"
echo "- Scripts directory: $WORK_DIR/src/scripts"
echo "- Model directory: $WORK_DIR/src/model"
echo "- Visualization directory: $WORK_DIR/src/visualization"
echo "- Results directory: $WORK_DIR/results/analysis/full"
echo "- Training directory: $WORK_DIR/results/training"

# Use direct MATLAB command with explicit variable setting
matlab -nodisplay -nosplash -nodesktop -r "try; disp('===== STARTING FULL ANALYSIS ====='); cd('$WORK_DIR'); rootDir='$WORK_DIR'; analysisMode='full'; disp(['Analysis mode explicitly set to: ' analysisMode]); addpath(fullfile(rootDir, 'src', 'scripts')); addpath(fullfile(rootDir, 'src', 'model')); addpath(fullfile(rootDir, 'src', 'visualization')); pc = parcluster('local'); pc.NumWorkers = str2num(getenv('SLURM_NTASKS_PER_NODE')); pc.JobStorageLocation = getenv('MATLAB_TEMP_DIR'); parpool(pc, pc.NumWorkers); shapDir = fullfile(rootDir, 'results', 'analysis', 'full'); trainingDir = fullfile(rootDir, 'results', 'training'); disp(['Root directory: ' rootDir]); disp(['SHAP directory: ' shapDir]); disp(['Training directory: ' trainingDir]); disp('Running run_analysis.m with full mode...'); run(fullfile(rootDir, 'src', 'scripts', 'run_analysis.m')); delete(gcp('nocreate')); exit; catch ME; disp('===== ERROR IN FULL ANALYSIS ====='); disp(getReport(ME)); exit(1); end;"

# Check MATLAB execution status
MATLAB_STATUS=$?
if [ $MATLAB_STATUS -ne 0 ]; then
    echo "ERROR: MATLAB execution failed with exit code: $MATLAB_STATUS"
    echo "Trying export_shap_to_excel.m directly to ensure Excel files are created..."
    
    # Try running the Excel export directly if the main script failed
    matlab -nodisplay -nosplash -nodesktop -r "try; cd('$WORK_DIR'); addpath('$WORK_DIR/src/scripts'); addpath('$WORK_DIR/src/model'); addpath('$WORK_DIR/src/visualization'); shapDir = fullfile('$WORK_DIR', 'results', 'analysis', 'full'); export_shap_to_excel; exit; catch ME; disp(getReport(ME)); exit(1); end;"
    
    EXCEL_STATUS=$?
    if [ $EXCEL_STATUS -ne 0 ]; then
        echo "ERROR: Excel export also failed with exit code: $EXCEL_STATUS"
    else
        echo "Excel export completed successfully."
    fi
else
    echo "SHAP analysis completed successfully with exit code: $MATLAB_STATUS"
fi

# Clean up any incorrect SHAP outputs in src/shap directory
echo "Cleaning up any incorrect SHAP outputs in src/shap directory..."
if [ -d "src/shap/figures" ]; then
    echo "Found incorrect figures in src/shap/figures - moving to correct location..."
    mkdir -p $ANALYSIS_DIR/figures
    mv src/shap/figures/* $ANALYSIS_DIR/figures/ 2>/dev/null
    rm -rf src/shap/figures
fi

if [ -d "src/shap/data" ]; then
    echo "Found incorrect data in src/shap/data - moving to correct location..."
    mkdir -p $ANALYSIS_DIR/data
    mv src/shap/data/* $ANALYSIS_DIR/data/ 2>/dev/null
    rm -rf src/shap/data
fi

# Create a job summary
echo "Creating job summary..."
SUMMARY_FILE="output/full_analysis/full_summary_${SLURM_JOB_ID}.txt"
echo "=== Full Analysis Summary ===" > $SUMMARY_FILE
echo "Job ID: $SLURM_JOB_ID" >> $SUMMARY_FILE
echo "Completion time: $(date)" >> $SUMMARY_FILE
echo -e "\nMATLAB log file:" >> $SUMMARY_FILE
ls -lh run_analysis_log.txt >> $SUMMARY_FILE 2>/dev/null
echo -e "\nResults directory structure:" >> $SUMMARY_FILE
ls -la results/ >> $SUMMARY_FILE 2>/dev/null
echo -e "\nOptimization results:" >> $SUMMARY_FILE
ls -la results/optimization/ >> $SUMMARY_FILE 2>/dev/null
echo -e "\nBest model from optimization:" >> $SUMMARY_FILE
ls -la results/best_model/ >> $SUMMARY_FILE 2>/dev/null
echo -e "\nTraining results directory:" >> $SUMMARY_FILE
ls -la results/training >> $SUMMARY_FILE 2>/dev/null
echo -e "\nSHAP Full Analysis directory:" >> $SUMMARY_FILE
ls -la results/analysis/full >> $SUMMARY_FILE 2>/dev/null
echo -e "\nSHAP Excel outputs:" >> $SUMMARY_FILE
ls -la results/analysis/full/*.xlsx >> $SUMMARY_FILE 2>/dev/null

# Check if there are any incorrect outputs in src/shap
echo -e "\nVerifying correct output locations:" >> $SUMMARY_FILE
if [ -d "src/shap/figures" ] || [ -d "src/shap/data" ]; then
    echo "WARNING: Some SHAP results were incorrectly saved to src/shap directory." >> $SUMMARY_FILE
    echo "         These should be moved to results/analysis/full/ directory." >> $SUMMARY_FILE
    if [ -d "src/shap/figures" ]; then
        echo "Incorrect figures found:" >> $SUMMARY_FILE
        ls -la src/shap/figures >> $SUMMARY_FILE 2>/dev/null
    fi
    if [ -d "src/shap/data" ]; then
        echo "Incorrect data found:" >> $SUMMARY_FILE
        ls -la src/shap/data >> $SUMMARY_FILE 2>/dev/null
    fi
else
    echo "PASS: All SHAP results are correctly saved to results/analysis/full directory." >> $SUMMARY_FILE
    echo "      No incorrect outputs found in src/shap directory." >> $SUMMARY_FILE
fi

# Clean up temporary files
echo "Cleaning up temporary files..."
rm -rf ${MATLAB_TEMP_DIR}

echo "Full analysis completed. Check the results/ directory for output."
echo "End time: $(date)"

# End of script