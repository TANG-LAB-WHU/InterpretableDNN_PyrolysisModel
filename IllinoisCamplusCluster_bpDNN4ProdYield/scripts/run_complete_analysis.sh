#!/bin/bash
#SBATCH --job-name=GPM_matlab_shap
#SBATCH --account=siqi-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=24:00:00
#SBATCH --output=output/shap_analysis/GPM_matlab_shap.o%j
#SBATCH --error=output/shap_analysis/GPM_matlab_shap.e%j

echo "=== Checking project setup before running analysis ==="

# Create main directory structure if not exists
echo "Ensuring project directories exist..."
mkdir -p output/shap_analysis
mkdir -p output/shap_analysis/significant_features
mkdir -p output/figures_tracking_training_process
mkdir -p output/results

# Ensure all required MATLAB function files are in src/matlab
echo "Checking for MATLAB function files..."

# Check files in the src/matlab directory to ensure all functions are there
if [ -d "src/matlab" ]; then
    echo "Found src/matlab directory."
    
    # Check main MATLAB files
    REQUIRED_FILES=("bpDNN4PyroProd.m" "nnbp.m" "nncheckgrad.m" "nnconfigure.m" "nncreate.m" 
                   "nneval.m" "nnff.m" "nnfindbest.m" "nninit.m" "nnpostprocess.m"
                   "nnprepare.m" "nnpredict.m" "nnpreprocess.m" "nnstopcriteria.m"
                   "nntrain.m" "nnupdatefigure.m")
    
    MISSING_FILES=0
    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "src/matlab/$file" ]; then
            echo "WARNING: Missing required file: src/matlab/$file"
            MISSING_FILES=1
        fi
    done
    
    if [ $MISSING_FILES -eq 0 ]; then
        echo "All required MATLAB function files are present."
    else
        echo "Some required files are missing. Please ensure all MATLAB function files are in the src/matlab directory."
        exit 1
    fi
else
    echo "ERROR: src/matlab directory not found!"
    exit 1
fi

# Ensure data files exist
if [ ! -f "data/processed/CopyrolysisFeedstock.mat" ]; then
    echo "ERROR: CopyrolysisFeedstock.mat not found in data/processed directory!"
    exit 1
else
    echo "Found CopyrolysisFeedstock.mat"
fi

# Ensure extract_vars_for_shap.m is in the correct location
if [ ! -f "src/matlab/extract_vars_for_shap.m" ]; then
    echo "ERROR: extract_vars_for_shap.m not found in src/matlab directory!"
    exit 1
else
    echo "Found extract_vars_for_shap.m"
fi

# Ensure shap_analysis.py is in the correct location
if [ ! -f "src/python/shap_analysis.py" ]; then
    echo "ERROR: shap_analysis.py not found in src/python directory!"
    exit 1
else
    echo "Found shap_analysis.py"
fi

echo "Project setup check completed successfully."
echo "=== Beginning analysis workflow ==="

echo "Starting complete analysis at $(date)"

# Set up environment variables
export MATLAB_TEMP_DIR="/scratch/${USER}/matlab_temp_${SLURM_JOB_ID}"
mkdir -p ${MATLAB_TEMP_DIR}

echo "=== Phase 1: Model Parameter Optimization and Training ==="
# Load MATLAB module
module load matlab/24.1

# Copy data file to working directory for MATLAB
cp data/processed/CopyrolysisFeedstock.mat ./

# First, run model parameter optimization
echo "Running model parameter optimization..."
if [ "$SKIP_OPTIMIZATION" = "true" ]; then
    echo "Skipping optimization as requested by SKIP_OPTIMIZATION flag."
else
    echo "Running optimize_model_params.m to find optimal parameters..."
    matlab -nodisplay -nosplash -nodesktop -r "try; cd('$(pwd)'); pc = parcluster('local'); pc.NumWorkers = str2num(getenv('SLURM_NTASKS')); pc.JobStorageLocation = getenv('MATLAB_TEMP_DIR'); parpool(pc, pc.NumWorkers); addpath('src/matlab'); optimize_model_params; delete(gcp('nocreate')); exit; catch ME; disp(getReport(ME)); exit(1); end;"

    # Check if optimization was successful
    if [ $? -ne 0 ]; then
        echo "Model parameter optimization failed. Stopping analysis."
        exit 1
    fi
fi

# Next, run training with optimized parameters
echo "Training model with optimized parameters..."
matlab -nodisplay -nosplash -nodesktop -r "try; cd('$(pwd)'); pc = parcluster('local'); pc.NumWorkers = str2num(getenv('SLURM_NTASKS')); pc.JobStorageLocation = getenv('MATLAB_TEMP_DIR'); parpool(pc, pc.NumWorkers); addpath('src/matlab'); train_with_optimal_params; delete(gcp('nocreate')); exit; catch ME; disp(getReport(ME)); exit(1); end;"

# Check if training was successful
if [ $? -ne 0 ]; then
    echo "Model training with optimized parameters failed. Falling back to default training."
    matlab -nodisplay -nosplash -nodesktop -r "try; cd('$(pwd)'); pc = parcluster('local'); pc.NumWorkers = str2num(getenv('SLURM_NTASKS')); pc.JobStorageLocation = getenv('MATLAB_TEMP_DIR'); parpool(pc, pc.NumWorkers); addpath('src/matlab'); bpDNN4PyroProd; delete(gcp('nocreate')); exit; catch ME; disp(getReport(ME)); exit(1); end;"
    
    # Check if fallback training was successful
    if [ $? -ne 0 ]; then
        echo "MATLAB model training failed. Stopping analysis."
        exit 1
    fi
fi

# Now extract variables for SHAP analysis
echo "Extracting variables for SHAP analysis..."
matlab -nodisplay -nosplash -nodesktop -r "try; cd('$(pwd)'); addpath('src/matlab'); extract_vars_for_shap; exit; catch ME; disp(getReport(ME)); exit(1); end;"

# Check if MATLAB extraction was successful
if [ $? -ne 0 ]; then
    echo "Variable extraction for SHAP analysis failed. Stopping analysis."
    exit 1
fi

# Copy training figures to the tracking directory
echo "Copying training figures to tracking directory..."
if [ -d "Figures" ]; then
    cp -r Figures/* output/figures_tracking_training_process/
    # Backup Figures to results directory
    mkdir -p output/results/Figures
    cp -r Figures/* output/results/Figures/
fi

# Cleanup MATLAB temporary directory
rm -rf ${MATLAB_TEMP_DIR}

echo "=== Phase 2: SHAP Analysis ==="
# Load Python module
module load python/3.11.11

# Create a local conda environment directory
export CONDA_ENV_DIR="${HOME}/.conda/envs"
mkdir -p ${CONDA_ENV_DIR}

# Set up conda to use local environment
module load anaconda3/2024.06
export CONDA_PKGS_DIRS="${HOME}/.conda/pkgs"
mkdir -p ${CONDA_PKGS_DIRS}

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Create and activate conda environment if not exists
CONDA_ENV_NAME="shap_env"
if ! conda env list | grep -q "^${CONDA_ENV_NAME}"; then
    echo "Creating conda environment in user space..."
    conda create -y -n ${CONDA_ENV_NAME} python=3.11 -c conda-forge
fi

# Activate environment and install packages
conda activate ${CONDA_ENV_NAME}

# Install required packages if not already installed
if ! python -c "import shap" &>/dev/null; then
    echo "Installing required packages..."
    conda install -y -c conda-forge shap scikit-learn pandas matplotlib scipy joblib
fi

# Run SHAP analysis
echo "Running SHAP analysis with Python..."
python src/python/shap_analysis.py

# Check if Python execution was successful
if [ $? -ne 0 ]; then
    echo "SHAP analysis failed."
    exit 1
fi

# Deactivate conda environment
conda deactivate

echo "Complete analysis finished at $(date)"

# Create a summary of outputs
echo "=== Analysis Summary ===" > output/shap_analysis/analysis_summary.txt
echo "Analysis completed at: $(date)" >> output/shap_analysis/analysis_summary.txt
echo -e "\nParameter Optimization Results:" >> output/shap_analysis/analysis_summary.txt
ls -lh output/optimization_results/ >> output/shap_analysis/analysis_summary.txt
echo -e "\nOutput files:" >> output/shap_analysis/analysis_summary.txt
ls -lh output/shap_analysis/ >> output/shap_analysis/analysis_summary.txt
echo -e "\nSignificant Features Analysis:" >> output/shap_analysis/analysis_summary.txt
ls -lh output/shap_analysis/significant_features/ >> output/shap_analysis/analysis_summary.txt
echo -e "\nTraining process figures:" >> output/shap_analysis/analysis_summary.txt
ls -lh output/figures_tracking_training_process/ >> output/shap_analysis/analysis_summary.txt
echo -e "\nTrained Model and Parameters:" >> output/shap_analysis/analysis_summary.txt
ls -lh output/results/ >> output/shap_analysis/analysis_summary.txt
echo -e "\nBackup Figures:" >> output/shap_analysis/analysis_summary.txt
ls -lh output/results/Figures/ >> output/shap_analysis/analysis_summary.txt

# Set directory permissions
chmod -R 755 output/results/
chmod -R 755 output/optimization_results/
chmod -R 755 output/shap_analysis/
chmod -R 755 output/figures_tracking_training_process/

echo "All tasks completed successfully. Check output/shap_analysis/ for SHAP results, output/shap_analysis/significant_features/ for significant feature analysis, and output/figures_tracking_training_process/ for training visualization." 