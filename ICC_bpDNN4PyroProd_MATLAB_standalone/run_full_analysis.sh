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
SCRIPT_FILES=("run_analysis.m" "calc_shap_values.m" "export_shap_to_excel.m" "optimize_hyperparameters.m" "train_best_model.m" "run_optimized_model.m")
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

# Create complete directory structure for SHAP analysis
mkdir -p "results/analysis/full/data"
mkdir -p "results/analysis/full/figures"

# Set up MATLAB temporary directory
export MATLAB_TEMP_DIR="/scratch/${USER}/matlab_temp_${SLURM_JOB_ID}"
mkdir -p ${MATLAB_TEMP_DIR}
echo "MATLAB temp directory: $MATLAB_TEMP_DIR"

# Load MATLAB module
echo "Loading MATLAB module..."
module load matlab/24.1

# ===========================================================================
# Step 1: Run Hyperparameter Optimization
# ===========================================================================
echo "======================================"
echo "STEP 1: RUNNING HYPERPARAMETER OPTIMIZATION"
echo "======================================"

# Check if optimization results already exist
if [ -f "$OPTIMIZATION_DIR/best_model.mat" ]; then
    echo "Existing optimization results found at $OPTIMIZATION_DIR/best_model.mat"
    echo "Checking if this is a complete result..."
    
    # Run a simple MATLAB check to validate the file
    matlab -nodisplay -nosplash -nodesktop -r "try; cd('$WORK_DIR'); disp('Checking optimization results...'); addpath(fullfile('$WORK_DIR', 'src', 'scripts')); load('$OPTIMIZATION_DIR/best_model.mat', 'best_config'); if exist('best_config', 'var'); disp('Valid optimization results found.'); exit(0); else disp('Incomplete optimization results, will run optimization again.'); exit(1); end; catch ME; disp(getReport(ME)); exit(1); end;" &> /tmp/optim_check_${SLURM_JOB_ID}.log
    
    OPTIM_CHECK_STATUS=$?
    if [ $OPTIM_CHECK_STATUS -eq 0 ]; then
        echo "Valid optimization results found. Proceeding to model training step."
        SKIP_OPTIMIZATION=1
    else
        echo "Optimization results incomplete or invalid. Will run optimization again."
        SKIP_OPTIMIZATION=0
    fi
else
    echo "No existing optimization results found. Will run optimization."
    SKIP_OPTIMIZATION=0
fi

if [ $SKIP_OPTIMIZATION -eq 0 ]; then
    echo "Running hyperparameter optimization..."
    echo "This may take several hours depending on search space."
    
    # Run optimization
    matlab -nodisplay -nosplash -nodesktop -r "try; disp('===== STARTING HYPERPARAMETER OPTIMIZATION ====='); cd('$WORK_DIR'); addpath(fullfile('$WORK_DIR', 'src', 'scripts')); addpath(fullfile('$WORK_DIR', 'src', 'model')); addpath(fullfile('$WORK_DIR', 'src', 'visualization')); pc = parcluster('local'); pc.NumWorkers = str2num(getenv('SLURM_NTASKS_PER_NODE')); pc.JobStorageLocation = getenv('MATLAB_TEMP_DIR'); parpool(pc, pc.NumWorkers); disp('Running optimize_hyperparameters.m...'); useFullData = true; useAllSamples = true; useAllFeatures = true; fullOptimization = true; optimDir = '$OPTIMIZATION_DIR'; disp('Running FULL optimization with ALL parameter combinations...'); try; run(fullfile('$WORK_DIR', 'src', 'scripts', 'optimize_hyperparameters.m')); catch ME; disp('ERROR IN HYPERPARAMETER OPTIMIZATION:'); disp(getReport(ME)); exit(1); end; delete(gcp('nocreate')); exit(0); catch ME; disp('===== ERROR IN HYPERPARAMETER OPTIMIZATION ====='); disp(getReport(ME)); exit(1); end;"
    
    # Check optimization execution status
    OPTIM_STATUS=$?
    if [ $OPTIM_STATUS -ne 0 ]; then
        echo "ERROR: Hyperparameter optimization failed with exit code: $OPTIM_STATUS"
        echo "Checking for partial results and continuing with best available..."
        
        # Check if any optimization results were generated
        if [ -f "$OPTIMIZATION_DIR/best_model.mat" ]; then
            echo "Found partial optimization results at $OPTIMIZATION_DIR/best_model.mat"
            echo "Will attempt to continue with these results."
        else
            echo "ERROR: No optimization results found. Cannot continue."
            exit 1
        fi
    else
        echo "Hyperparameter optimization completed successfully with exit code: $OPTIM_STATUS"
    fi
fi

# Summarize optimization results
echo "Summarizing optimization results..."

matlab -nodisplay -nosplash -nodesktop -r "try; cd('$WORK_DIR'); addpath(fullfile('$WORK_DIR', 'src', 'scripts')); disp('Summarizing optimization results...'); if exist('$OPTIMIZATION_DIR/best_model.mat', 'file'); load('$OPTIMIZATION_DIR/best_model.mat'); disp('Best hyperparameters:'); disp(best_config); fid = fopen('$OPTIMIZATION_DIR/best_configuration.txt', 'w'); fprintf(fid, 'Best Hyperparameter Configuration:\n\n'); fields = fieldnames(best_config); for i = 1:length(fields), if isnumeric(best_config.(fields{i})), fprintf(fid, '%s: %g\n', fields{i}, best_config.(fields{i})); elseif ischar(best_config.(fields{i})), fprintf(fid, '%s: %s\n', fields{i}, best_config.(fields{i})); elseif iscell(best_config.(fields{i})), fprintf(fid, '%s: %s\n', fields{i}, mat2str(best_config.(fields{i}))); end; end; fclose(fid); disp('Saved summary to best_configuration.txt'); else disp('No optimization results file found!'); end; exit(0); catch ME; disp(getReport(ME)); exit(1); end;" &> /tmp/optim_summary_${SLURM_JOB_ID}.log

# ===========================================================================
# Step 2: Train Best Model Using the Optimized Script
# ===========================================================================
echo "======================================"
echo "STEP 2: TRAINING BEST MODEL"
echo "======================================"

# Check if optimization results are available
if [ ! -f "$OPTIMIZATION_DIR/best_model.mat" ]; then
    echo "ERROR: Optimization results not found at $OPTIMIZATION_DIR/best_model.mat"
    echo "Cannot proceed with best model training. Exiting."
    exit 1
fi

# Run the optimized model training script (skip SHAP analysis)
echo "Running the optimized model training script..."
echo "This will handle both TVT and TT strategies."

# Run the optimized script for training only
matlab -nodisplay -nosplash -nodesktop -r "cd('$WORK_DIR'); addpath(fullfile('$WORK_DIR', 'src', 'scripts')); addpath(fullfile('$WORK_DIR', 'src', 'model')); addpath(fullfile('$WORK_DIR', 'src', 'visualization')); pc = parcluster('local'); pc.NumWorkers = str2num(getenv('SLURM_NTASKS_PER_NODE')); pc.JobStorageLocation = getenv('MATLAB_TEMP_DIR'); parpool(pc, pc.NumWorkers); disp('Running run_optimized_model.m for TRAINING only...'); optimDir = '$OPTIMIZATION_DIR'; optimizationDir = '$OPTIMIZATION_DIR'; bestModelDir = '$BEST_MODEL_DIR'; trainingDir = '$TRAINING_DIR'; analysisDir = '$ANALYSIS_DIR'; useFullData = true; useAllSamples = true; useAllFeatures = true; skipSHAP = true; try; run(fullfile('$WORK_DIR', 'src', 'scripts', 'run_optimized_model.m')); catch ME; disp('ERROR IN OPTIMIZATION MODEL:'); disp(getReport(ME)); exit(1); end; delete(gcp('nocreate')); exit(0);"

# Check execution status of the training step
TRAINING_STATUS=$?
if [ $TRAINING_STATUS -ne 0 ]; then
    echo "ERROR: Model training failed with exit code: $TRAINING_STATUS"
    echo "Training script execution log saved to run_optimized_model_log.txt"
    
    # Create a summary of the error
    SUMMARY_FILE="output/full_analysis/full_summary_${SLURM_JOB_ID}.txt"
    echo "=== Full Analysis Summary ===" > $SUMMARY_FILE
    echo "Job ID: $SLURM_JOB_ID" >> $SUMMARY_FILE
    echo "Completion time: $(date)" >> $SUMMARY_FILE
    echo -e "\nERROR: Model training failed with exit code: $TRAINING_STATUS" >> $SUMMARY_FILE
    echo "The script was unable to complete model training." >> $SUMMARY_FILE
    echo "Please check run_optimized_model_log.txt for detailed error information." >> $SUMMARY_FILE
    
    # Clean up temporary files
    echo "Cleaning up temporary files..."
    rm -rf ${MATLAB_TEMP_DIR}
    rm -f /tmp/optim_check_${SLURM_JOB_ID}.log
    rm -f /tmp/optim_summary_${SLURM_JOB_ID}.log
    
    echo "Workflow terminated due to training script failure."
    echo "End time: $(date)"
    exit 1
else
    echo "Model training completed successfully with exit code: $TRAINING_STATUS"
    echo "Trained model is available at: $BEST_MODEL_DIR/best_model.mat and $TRAINING_DIR/trained_model.mat"
fi

# Check that the required model files exist before proceeding to SHAP analysis
if [ ! -f "$BEST_MODEL_DIR/best_model.mat" ] || [ ! -f "$TRAINING_DIR/trained_model.mat" ]; then
    echo "ERROR: Required model files not found after training step"
    echo "Either $BEST_MODEL_DIR/best_model.mat or $TRAINING_DIR/trained_model.mat is missing"
    echo "Cannot proceed with SHAP analysis. Exiting."
    exit 1
fi

# ===========================================================================
# Step 3: Run SHAP Analysis Using the Trained Model
# ===========================================================================
echo "======================================"
echo "STEP 3: PERFORMING SHAP ANALYSIS"
echo "======================================"

# Run the SHAP analysis script (skip training)
echo "Running the SHAP analysis script using the trained model..."

# Run the optimized script for SHAP analysis only
matlab -nodisplay -nosplash -nodesktop -r "cd('$WORK_DIR'); addpath(fullfile('$WORK_DIR', 'src', 'scripts')); addpath(fullfile('$WORK_DIR', 'src', 'model')); addpath(fullfile('$WORK_DIR', 'src', 'visualization')); pc = parcluster('local'); pc.NumWorkers = str2num(getenv('SLURM_NTASKS_PER_NODE')); pc.JobStorageLocation = getenv('MATLAB_TEMP_DIR'); parpool(pc, pc.NumWorkers); disp('Running run_optimized_model.m for SHAP ANALYSIS only...'); optimDir = '$OPTIMIZATION_DIR'; optimizationDir = '$OPTIMIZATION_DIR'; bestModelDir = '$BEST_MODEL_DIR'; trainingDir = '$TRAINING_DIR'; analysisDir = '$ANALYSIS_DIR'; useFullData = true; useAllSamples = true; useAllFeatures = true; skipTraining = true; try; run(fullfile('$WORK_DIR', 'src', 'scripts', 'run_optimized_model.m')); catch ME; disp('ERROR IN SHAP ANALYSIS:'); disp(getReport(ME)); exit(1); end; delete(gcp('nocreate')); exit(0);"

# Check execution status of the SHAP analysis step
SHAP_STATUS=$?
if [ $SHAP_STATUS -ne 0 ]; then
    echo "ERROR: SHAP analysis failed with exit code: $SHAP_STATUS"
    echo "SHAP analysis script execution log saved to run_optimized_model_log.txt"
    
    # Create a summary of the error
    SUMMARY_FILE="output/full_analysis/full_summary_${SLURM_JOB_ID}.txt"
    echo "=== Full Analysis Summary ===" > $SUMMARY_FILE
    echo "Job ID: $SLURM_JOB_ID" >> $SUMMARY_FILE
    echo "Completion time: $(date)" >> $SUMMARY_FILE
    echo -e "\nWARNING: Model training completed successfully" >> $SUMMARY_FILE
    echo -e "ERROR: But SHAP analysis failed with exit code: $SHAP_STATUS" >> $SUMMARY_FILE
    echo "The script completed model training but was unable to complete SHAP analysis." >> $SUMMARY_FILE
    echo "Please check run_optimized_model_log.txt for detailed error information." >> $SUMMARY_FILE
    
    # Clean up temporary files
    echo "Cleaning up temporary files..."
    rm -rf ${MATLAB_TEMP_DIR}
    rm -f /tmp/optim_check_${SLURM_JOB_ID}.log
    rm -f /tmp/optim_summary_${SLURM_JOB_ID}.log
    
    echo "Workflow terminated due to SHAP analysis script failure."
    echo "End time: $(date)"
    exit 1
else
    echo "SHAP analysis completed successfully with exit code: $SHAP_STATUS"
    echo "The script has completed both model training and SHAP analysis."
    echo "Results are available in the respective directories:"
    echo "- Best model: $BEST_MODEL_DIR/best_model.mat"
    echo "- Training model: $TRAINING_DIR/trained_model.mat" 
    echo "- SHAP results: $ANALYSIS_DIR/data/shap_results.mat"
    echo "- SHAP Excel file: $ANALYSIS_DIR/shap_analysis_results.xlsx (if generated)"
    echo "- SHAP figures: $ANALYSIS_DIR/figures/"
fi

# Check for required output files and create them if missing
echo "Checking for required output files..."
if [ ! -f "$ANALYSIS_DIR/data/shap_results.mat" ]; then
    echo "WARNING: SHAP results MAT file not found!"
else
    echo "SHAP results MAT file found: $ANALYSIS_DIR/data/shap_results.mat"
    # Look for Excel outputs
    EXCEL_FILES=$(ls -la $ANALYSIS_DIR/*.xlsx 2>/dev/null | wc -l)
    if [ $EXCEL_FILES -eq 0 ]; then
        echo "WARNING: No Excel output files found in $ANALYSIS_DIR"
    else
        echo "Found $EXCEL_FILES Excel output files in $ANALYSIS_DIR"
    fi
    
    # Look for figure outputs
    FIG_FILES=$(ls -la $ANALYSIS_DIR/figures/*.png $ANALYSIS_DIR/figures/*.fig 2>/dev/null | wc -l)
    if [ $FIG_FILES -eq 0 ]; then
        echo "WARNING: No figure output files found in $ANALYSIS_DIR/figures"
    else
        echo "Found $FIG_FILES figure output files in $ANALYSIS_DIR/figures"
    fi
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
echo -e "\nOptimized script log file:" >> $SUMMARY_FILE
ls -lh run_optimized_model_log.txt >> $SUMMARY_FILE 2>/dev/null
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
echo -e "\nSHAP Figure outputs:" >> $SUMMARY_FILE
ls -la results/analysis/full/figures/ >> $SUMMARY_FILE 2>/dev/null

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
rm -f /tmp/optim_check_${SLURM_JOB_ID}.log
rm -f /tmp/optim_summary_${SLURM_JOB_ID}.log
rm -f /tmp/model_check_${SLURM_JOB_ID}.log
rm -f /tmp/default_model_${SLURM_JOB_ID}.log

echo "Full analysis workflow completed. Check the results/ directory for output."
echo "End time: $(date)"

# End of script