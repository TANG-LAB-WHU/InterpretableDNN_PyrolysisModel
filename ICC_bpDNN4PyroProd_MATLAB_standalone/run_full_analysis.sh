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

# --------------------------------
# IMPROVED ERROR HANDLING
# --------------------------------
# Exit immediately if a command exits with non-zero status
set -e

# Function to log and handle errors
error_handler() {
    echo "ERROR: Command failed with exit code $1 at line $2"
    # Create a summary file if it doesn't exist
    SUMMARY_FILE="output/full_analysis/full_summary_${SLURM_JOB_ID}.txt"
    echo "=== Full Analysis Summary (ERROR) ===" > $SUMMARY_FILE
    echo "Job ID: $SLURM_JOB_ID" >> $SUMMARY_FILE
    echo "Error time: $(date)" >> $SUMMARY_FILE
    echo "Failed at line $2 with error code $1" >> $SUMMARY_FILE
    echo "Please check full_${SLURM_JOB_ID}.err for details" >> $SUMMARY_FILE
    
    # Clean up temporary MATLAB directory
    if [ -d "${MATLAB_TEMP_DIR}" ]; then
        echo "Cleaning up MATLAB temp directory..."
        rm -rf ${MATLAB_TEMP_DIR}
    fi
    
    # Exit with error code
    exit 1
}

# Set up error trap
trap 'error_handler $? $LINENO' ERR

# Get current working directory
WORK_DIR=$(pwd)
echo "=== Starting GPM_SHAP Full Analysis Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"

# --------------------------------
# IMPROVED DIRECTORY CREATION
# --------------------------------
# Create complete directory structure with error checking
echo "Creating complete directory structure..."
for dir in data/processed data/raw \
           output/debug_analysis output/full_analysis \
           results/training/Figures results/debug/Figures \
           results/analysis/debug/data results/analysis/debug/figures \
           results/analysis/full/data results/analysis/full/figures \
           results/optimization results/best_model/Figures \
           src/model src/scripts src/visualization src/shap; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to create directory: $dir"
            exit 1
        fi
    fi
done
echo "Directory structure created successfully"

# Set up environment variables
export MODEL_DIR="$WORK_DIR/src/model"
export SCRIPTS_DIR="$WORK_DIR/src/scripts"
export VISUALIZATION_DIR="$WORK_DIR/src/visualization"
export TRAINING_DIR="$WORK_DIR/results/training"
export BEST_MODEL_DIR="$WORK_DIR/results/best_model"
export OPTIMIZATION_DIR="$WORK_DIR/results/optimization"
export ANALYSIS_DIR="$WORK_DIR/results/analysis/full"

echo "Model directory: $MODEL_DIR"
echo "Scripts directory: $SCRIPTS_DIR"
echo "Visualization directory: $VISUALIZATION_DIR"
echo "Training directory: $TRAINING_DIR"
echo "Best model directory: $BEST_MODEL_DIR"
echo "Optimization directory: $OPTIMIZATION_DIR"
echo "Analysis directory: $ANALYSIS_DIR"

# --------------------------------
# IMPROVED DATA FILE CHECKING
# --------------------------------
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

# Check model files (improved to check for non-empty files)
echo "Checking model files..."
MODEL_FILES=("bpDNN4PyroProd.m" "nnbp.m" "nncheckgrad.m" "nnconfigure.m" "nncreate.m" 
             "nneval.m" "nnff.m" "nnfindbest.m" "nninit.m" "nnpostprocess.m" "nnpredict.m" 
             "nnprepare.m" "nnpreprocess.m" "nnstopcriteria.m" "nntrain.m" "nnupdatefigure.m")

MODEL_MISSING=0
for file in "${MODEL_FILES[@]}"; do
    if [ ! -f "src/model/$file" ] || [ ! -s "src/model/$file" ]; then
        echo "WARNING: Required model file not found or empty: src/model/$file"
        MODEL_MISSING=1
    fi
done

if [ $MODEL_MISSING -eq 1 ]; then
    echo "WARNING: Some model files are missing or empty. This may cause the workflow to fail."
fi

# Check script files with improved verification
echo "Checking script files..."
SCRIPT_FILES=("run_analysis.m" "calc_shap_values.m" "export_shap_to_excel.m" 
              "optimize_hyperparameters.m" "train_best_model.m" "run_optimized_model.m")
SCRIPT_MISSING=0

for file in "${SCRIPT_FILES[@]}"; do
    if [ ! -f "src/scripts/$file" ] || [ ! -s "src/scripts/$file" ]; then
        echo "WARNING: Required script file not found or empty: src/scripts/$file"
        SCRIPT_MISSING=1
    fi
done

if [ $SCRIPT_MISSING -eq 1 ]; then
    echo "WARNING: Some script files are missing or empty. This may cause the workflow to fail."
fi

# Check visualization files
echo "Checking visualization files..."
VIZ_FILES=("plot_shap_results.m" "plot_shap_beeswarm.m" 
          "plot_shap_original_summary.m" "fix_colorbar_style.m")
VIZ_MISSING=0

for file in "${VIZ_FILES[@]}"; do
    if [ ! -f "src/visualization/$file" ] || [ ! -s "src/visualization/$file" ]; then
        echo "WARNING: Required visualization file not found or empty: src/visualization/$file"
        VIZ_MISSING=1
    fi
done

if [ $VIZ_MISSING -eq 1 ]; then
    echo "WARNING: Some visualization files are missing or empty. This may cause the workflow to fail."
fi

# --------------------------------
# COPY DATA FILES
# --------------------------------
# Copy data files to model directory with verification
echo "Copying data files to model directory..."
cp -f "data/processed/CopyrolysisFeedstock.mat" "src/model/"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to copy CopyrolysisFeedstock.mat to src/model/"
    exit 1
fi

cp -f "data/raw/RawInputData.xlsx" "src/model/"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to copy RawInputData.xlsx to src/model/"
    exit 1
fi
echo "Data files copied to model directory successfully"

# --------------------------------
# IMPROVED MATLAB SETUP
# --------------------------------
# Set up MATLAB temporary directory with better scratch path
export MATLAB_TEMP_DIR="/scratch/${USER}/matlab_temp_${SLURM_JOB_ID}"
mkdir -p ${MATLAB_TEMP_DIR}
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create MATLAB temporary directory"
    echo "Directory: ${MATLAB_TEMP_DIR}"
    exit 1
fi
echo "MATLAB temp directory created: $MATLAB_TEMP_DIR"

# Create a MATLAB initialization script with proper path setup
cat > ${MATLAB_TEMP_DIR}/init_paths.m << EOF
function init_paths()
    % Get the current script directory
    currentDir = pwd;
    
    % Add all required paths
    addpath(genpath(fullfile(currentDir, 'src', 'model')));
    addpath(genpath(fullfile(currentDir, 'src', 'scripts')));
    addpath(genpath(fullfile(currentDir, 'src', 'visualization')));
    addpath(genpath(fullfile(currentDir, 'src', 'shap')));
    
    % Verify paths are added
    disp('=== PATH VERIFICATION ===');
    disp('Model path exists:');
    disp(exist(fullfile(currentDir, 'src', 'model'), 'dir'));
    disp('Scripts path exists:');
    disp(exist(fullfile(currentDir, 'src', 'scripts'), 'dir'));
    disp('Visualization path exists:');
    disp(exist(fullfile(currentDir, 'src', 'visualization'), 'dir'));
    
    % List available MATLAB script files
    disp('=== AVAILABLE SCRIPT FILES ===');
    ls(fullfile(currentDir, 'src', 'scripts'));
    
    % Report success
    disp('Paths initialized successfully');
end
EOF

# Load MATLAB module
echo "Loading MATLAB module..."
module load matlab/24.1
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to load MATLAB module. Please check if matlab/24.1 is available."
    exit 1
fi
echo "MATLAB module loaded successfully"

# --------------------------------
# STEP 1: HYPERPARAMETER OPTIMIZATION
# --------------------------------
echo "======================================"
echo "STEP 1: RUNNING HYPERPARAMETER OPTIMIZATION"
echo "======================================"

# Check if optimization results already exist
if [ -f "$OPTIMIZATION_DIR/best_model.mat" ]; then
    echo "Existing optimization results found at $OPTIMIZATION_DIR/best_model.mat"
    echo "Checking if this is a complete result..."
    
    # Run a simple MATLAB check to validate the file with improved error handling
    matlab -nodisplay -nosplash -nodesktop -r "try; cd('$WORK_DIR'); addpath('${MATLAB_TEMP_DIR}'); init_paths(); disp('Checking optimization results...'); if exist('$OPTIMIZATION_DIR/best_model.mat', 'file'); load('$OPTIMIZATION_DIR/best_model.mat', 'best_config'); if exist('best_config', 'var'); disp('Valid optimization results found.'); exit(0); else disp('Incomplete optimization results, will run optimization again.'); exit(1); end; else disp('Optimization file not accessible'); exit(1); end; catch ME; disp('ERROR:'); disp(getReport(ME)); exit(1); end;" > ${MATLAB_TEMP_DIR}/optim_check.log 2>&1
    
    OPTIM_CHECK_STATUS=$?
    if [ $OPTIM_CHECK_STATUS -eq 0 ]; then
        echo "Valid optimization results found. Proceeding to model training step."
        SKIP_OPTIMIZATION=1
    else
        echo "Optimization results incomplete or invalid. Will run optimization again."
        echo "Check error log: ${MATLAB_TEMP_DIR}/optim_check.log"
        cat ${MATLAB_TEMP_DIR}/optim_check.log
        SKIP_OPTIMIZATION=0
    fi
else
    echo "No existing optimization results found. Will run optimization."
    SKIP_OPTIMIZATION=0
fi

if [ $SKIP_OPTIMIZATION -eq 0 ]; then
    echo "Running hyperparameter optimization..."
    echo "This may take several hours depending on search space."
    
    # Run optimization with improved error handling and path setup
    matlab -nodisplay -nosplash -nodesktop -r "try; disp('===== STARTING HYPERPARAMETER OPTIMIZATION ====='); cd('$WORK_DIR'); addpath('${MATLAB_TEMP_DIR}'); init_paths(); disp('Paths initialized'); pc = parcluster('local'); pc.NumWorkers = min(str2num(getenv('SLURM_NTASKS_PER_NODE')), maxNumCompThreads); pc.JobStorageLocation = getenv('MATLAB_TEMP_DIR'); disp(['Creating parallel pool with ', num2str(pc.NumWorkers), ' workers']); pool = parpool(pc, pc.NumWorkers); disp('Parallel pool created'); disp('Running optimize_hyperparameters.m...'); useFullData = true; useAllSamples = true; useAllFeatures = true; fullOptimization = true; optimDir = '$OPTIMIZATION_DIR'; disp('Running FULL optimization with ALL parameter combinations...'); run(fullfile('$WORK_DIR', 'src', 'scripts', 'optimize_hyperparameters.m')); disp('Optimization completed'); delete(gcp('nocreate')); exit(0); catch ME; disp('===== ERROR IN HYPERPARAMETER OPTIMIZATION ====='); disp(getReport(ME)); exit(1); end;" > ${MATLAB_TEMP_DIR}/optimization.log 2>&1
    
    # Check optimization execution status
    OPTIM_STATUS=$?
    if [ $OPTIM_STATUS -ne 0 ]; then
        echo "ERROR: Hyperparameter optimization failed with exit code: $OPTIM_STATUS"
        echo "Check error log: ${MATLAB_TEMP_DIR}/optimization.log"
        cat ${MATLAB_TEMP_DIR}/optimization.log
        
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

# Summarize optimization results with improved error handling
echo "Summarizing optimization results..."
matlab -nodisplay -nosplash -nodesktop -r "try; cd('$WORK_DIR'); addpath('${MATLAB_TEMP_DIR}'); init_paths(); disp('Summarizing optimization results...'); if exist('$OPTIMIZATION_DIR/best_model.mat', 'file'); load('$OPTIMIZATION_DIR/best_model.mat'); disp('Best hyperparameters:'); disp(best_config); fid = fopen('$OPTIMIZATION_DIR/best_configuration.txt', 'w'); fprintf(fid, 'Best Hyperparameter Configuration:\n\n'); fields = fieldnames(best_config); for i = 1:length(fields), if isnumeric(best_config.(fields{i})), fprintf(fid, '%s: %g\n', fields{i}, best_config.(fields{i})); elseif ischar(best_config.(fields{i})), fprintf(fid, '%s: %s\n', fields{i}, best_config.(fields{i})); elseif iscell(best_config.(fields{i})), fprintf(fid, '%s: %s\n', fields{i}, mat2str(best_config.(fields{i}))); end; end; fclose(fid); disp('Saved summary to best_configuration.txt'); else disp('No optimization results file found!'); end; exit(0); catch ME; disp('ERROR IN SUMMARIZING OPTIMIZATION:'); disp(getReport(ME)); exit(1); end;" > ${MATLAB_TEMP_DIR}/optim_summary.log 2>&1

# --------------------------------
# STEP 2: TRAIN BEST MODEL
# --------------------------------
echo "======================================"
echo "STEP 2: TRAINING BEST MODEL"
echo "======================================"

# Check if optimization results are available with improved verification
if [ ! -f "$OPTIMIZATION_DIR/best_model.mat" ] || [ ! -s "$OPTIMIZATION_DIR/best_model.mat" ]; then
    echo "ERROR: Optimization results not found or empty at $OPTIMIZATION_DIR/best_model.mat"
    echo "Cannot proceed with best model training. Exiting."
    exit 1
fi

# Run the optimized model training script with improved path setup and error handling
echo "Running the optimized model training script..."
echo "This will handle both TVT and TT strategies."

# Run the optimized script for training only
matlab -nodisplay -nosplash -nodesktop -r "try; cd('$WORK_DIR'); addpath('${MATLAB_TEMP_DIR}'); init_paths(); disp('Paths initialized for training'); pc = parcluster('local'); pc.NumWorkers = min(str2num(getenv('SLURM_NTASKS_PER_NODE')), maxNumCompThreads); pc.JobStorageLocation = getenv('MATLAB_TEMP_DIR'); disp(['Creating parallel pool with ', num2str(pc.NumWorkers), ' workers']); pool = parpool(pc, pc.NumWorkers); disp('Running run_optimized_model.m for TRAINING only...'); optimDir = '$OPTIMIZATION_DIR'; optimizationDir = '$OPTIMIZATION_DIR'; bestModelDir = '$BEST_MODEL_DIR'; trainingDir = '$TRAINING_DIR'; analysisDir = '$ANALYSIS_DIR'; useFullData = true; useAllSamples = true; useAllFeatures = true; skipSHAP = true; disp('Beginning model training...'); run(fullfile('$WORK_DIR', 'src', 'scripts', 'run_optimized_model.m')); disp('Model training completed'); if exist('pool', 'var') && ~isempty(pool); delete(pool); end; exit(0); catch ME; disp('ERROR IN MODEL TRAINING:'); disp(getReport(ME)); if exist('pool', 'var') && ~isempty(pool); delete(pool); end; exit(1); end;" > ${MATLAB_TEMP_DIR}/training.log 2>&1

# Check execution status of the training step
TRAINING_STATUS=$?
if [ $TRAINING_STATUS -ne 0 ]; then
    echo "ERROR: Model training failed with exit code: $TRAINING_STATUS"
    echo "Check training log: ${MATLAB_TEMP_DIR}/training.log"
    cat ${MATLAB_TEMP_DIR}/training.log
    
    # Create a summary of the error
    SUMMARY_FILE="output/full_analysis/full_summary_${SLURM_JOB_ID}.txt"
    echo "=== Full Analysis Summary ===" > $SUMMARY_FILE
    echo "Job ID: $SLURM_JOB_ID" >> $SUMMARY_FILE
    echo "Completion time: $(date)" >> $SUMMARY_FILE
    echo -e "\nERROR: Model training failed with exit code: $TRAINING_STATUS" >> $SUMMARY_FILE
    echo "The script was unable to complete model training." >> $SUMMARY_FILE
    echo "Please check ${MATLAB_TEMP_DIR}/training.log for detailed error information." >> $SUMMARY_FILE
    
    # Clean up temporary files
    echo "Cleaning up temporary files..."
    rm -rf ${MATLAB_TEMP_DIR}
    
    echo "Workflow terminated due to training script failure."
    echo "End time: $(date)"
    exit 1
else
    echo "Model training completed successfully with exit code: $TRAINING_STATUS"
    echo "Trained model is available at: $BEST_MODEL_DIR/best_model.mat and $TRAINING_DIR/trained_model.mat"
fi

# Verify model files exist and are not empty
if [ ! -f "$BEST_MODEL_DIR/best_model.mat" ] || [ ! -s "$BEST_MODEL_DIR/best_model.mat" ] || 
   [ ! -f "$TRAINING_DIR/trained_model.mat" ] || [ ! -s "$TRAINING_DIR/trained_model.mat" ]; then
    echo "ERROR: Required model files not found or empty after training step"
    echo "Either $BEST_MODEL_DIR/best_model.mat or $TRAINING_DIR/trained_model.mat is missing or empty"
    echo "Cannot proceed with SHAP analysis. Exiting."
    exit 1
fi

# --------------------------------
# STEP 3: RUN SHAP ANALYSIS
# --------------------------------
echo "======================================"
echo "STEP 3: PERFORMING SHAP ANALYSIS"
echo "======================================"

# Run the SHAP analysis script with improved error handling and path setup
echo "Running the SHAP analysis script using the trained model..."

matlab -nodisplay -nosplash -nodesktop -r "try; cd('$WORK_DIR'); addpath('${MATLAB_TEMP_DIR}'); init_paths(); disp('Paths initialized for SHAP analysis'); pc = parcluster('local'); pc.NumWorkers = min(str2num(getenv('SLURM_NTASKS_PER_NODE')), maxNumCompThreads); pc.JobStorageLocation = getenv('MATLAB_TEMP_DIR'); disp(['Creating parallel pool with ', num2str(pc.NumWorkers), ' workers']); pool = parpool(pc, pc.NumWorkers); disp('Running run_optimized_model.m for SHAP ANALYSIS only...'); optimDir = '$OPTIMIZATION_DIR'; optimizationDir = '$OPTIMIZATION_DIR'; bestModelDir = '$BEST_MODEL_DIR'; trainingDir = '$TRAINING_DIR'; analysisDir = '$ANALYSIS_DIR'; useFullData = true; useAllSamples = true; useAllFeatures = true; skipTraining = true; disp('Beginning SHAP analysis...'); run(fullfile('$WORK_DIR', 'src', 'scripts', 'run_optimized_model.m')); disp('SHAP analysis completed'); if exist('pool', 'var') && ~isempty(pool); delete(pool); end; exit(0); catch ME; disp('ERROR IN SHAP ANALYSIS:'); disp(getReport(ME)); if exist('pool', 'var') && ~isempty(pool); delete(pool); end; exit(1); end;" > ${MATLAB_TEMP_DIR}/shap_analysis.log 2>&1

# Check execution status of the SHAP analysis step
SHAP_STATUS=$?
if [ $SHAP_STATUS -ne 0 ]; then
    echo "ERROR: SHAP analysis failed with exit code: $SHAP_STATUS"
    echo "Check SHAP analysis log: ${MATLAB_TEMP_DIR}/shap_analysis.log"
    cat ${MATLAB_TEMP_DIR}/shap_analysis.log
    
    # Create a summary of the error
    SUMMARY_FILE="output/full_analysis/full_summary_${SLURM_JOB_ID}.txt"
    if [ -f "$SUMMARY_FILE" ]; then
        echo -e "\nWARNING: Model training completed successfully" >> $SUMMARY_FILE
        echo -e "ERROR: But SHAP analysis failed with exit code: $SHAP_STATUS" >> $SUMMARY_FILE
        echo "The script completed model training but was unable to complete SHAP analysis." >> $SUMMARY_FILE
        echo "Please check ${MATLAB_TEMP_DIR}/shap_analysis.log for detailed error information." >> $SUMMARY_FILE
    else
        echo "=== Full Analysis Summary ===" > $SUMMARY_FILE
        echo "Job ID: $SLURM_JOB_ID" >> $SUMMARY_FILE
        echo "Completion time: $(date)" >> $SUMMARY_FILE
        echo -e "\nWARNING: Model training completed successfully" >> $SUMMARY_FILE
        echo -e "ERROR: But SHAP analysis failed with exit code: $SHAP_STATUS" >> $SUMMARY_FILE
        echo "The script completed model training but was unable to complete SHAP analysis." >> $SUMMARY_FILE
        echo "Please check ${MATLAB_TEMP_DIR}/shap_analysis.log for detailed error information." >> $SUMMARY_FILE
    fi
    
    echo "Workflow partially completed. Model training succeeded but SHAP analysis failed."
    echo "End time: $(date)"
    
    # Don't exit - we want to clean up and provide as much information as possible
else
    echo "SHAP analysis completed successfully with exit code: $SHAP_STATUS"
    echo "The script has completed both model training and SHAP analysis."
fi

# --------------------------------
# VERIFY RESULTS AND CLEAN UP
# --------------------------------
echo "======================================"
echo "STEP 4: VERIFYING RESULTS AND CLEANUP"
echo "======================================"

# Check for required output files
echo "Checking for required output files..."
if [ ! -f "$ANALYSIS_DIR/data/shap_results.mat" ]; then
    echo "WARNING: SHAP results MAT file not found at $ANALYSIS_DIR/data/shap_results.mat"
    # Check alternative location in src/shap
    if [ -f "src/shap/data/shap_results.mat" ]; then
        echo "Found SHAP results in alternate location: src/shap/data/shap_results.mat"
        echo "Moving to correct location..."
        mkdir -p "$ANALYSIS_DIR/data"
        cp "src/shap/data/shap_results.mat" "$ANALYSIS_DIR/data/"
        if [ $? -eq 0 ]; then
            echo "Successfully moved SHAP results to correct location"
        else
            echo "ERROR: Failed to move SHAP results to correct location"
        fi
    else
        echo "ERROR: SHAP results MAT file not found in any location!"
    fi
else
    echo "SHAP results MAT file found: $ANALYSIS_DIR/data/shap_results.mat"
fi

# Look for Excel outputs
EXCEL_FILES=$(ls -la $ANALYSIS_DIR/*.xlsx 2>/dev/null | wc -l)
if [ $EXCEL_FILES -eq 0 ]; then
    echo "WARNING: No Excel output files found in $ANALYSIS_DIR"
    # Try to run Excel export if SHAP results exist
    if [ -f "$ANALYSIS_DIR/data/shap_results.mat" ]; then
        echo "Attempting to run Excel export script..."
        matlab -nodisplay -nosplash -nodesktop -r "try; cd('$WORK_DIR'); addpath('${MATLAB_TEMP_DIR}'); init_paths(); disp('Running Excel export...'); load('$ANALYSIS_DIR/data/shap_results.mat'); disp('SHAP results loaded'); run(fullfile('$WORK_DIR', 'src', 'scripts', 'export_shap_to_excel.m')); disp('Excel export completed'); exit(0); catch ME; disp('ERROR IN EXCEL EXPORT:'); disp(getReport(ME)); exit(1); end;" > ${MATLAB_TEMP_DIR}/excel_export.log 2>&1
        if [ $? -eq 0 ]; then
            echo "Excel export completed successfully"
        else
            echo "WARNING: Excel export failed. See log: ${MATLAB_TEMP_DIR}/excel_export.log"
        fi
    fi
else
    echo "Found $EXCEL_FILES Excel output files in $ANALYSIS_DIR"
fi

# Look for figure outputs
FIG_FILES=$(ls -la $ANALYSIS_DIR/figures/*.png $ANALYSIS_DIR/figures/*.fig 2>/dev/null | wc -l)
if [ $FIG_FILES -eq 0 ]; then
    echo "WARNING: No figure output files found in $ANALYSIS_DIR/figures"
    # Check alternative location
    if [ -d "src/shap/figures" ]; then
        echo "Found figures in alternate location: src/shap/figures"
        echo "Moving to correct location..."
        mkdir -p "$ANALYSIS_DIR/figures"
        cp -r src/shap/figures/* "$ANALYSIS_DIR/figures/" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "Successfully moved figures to correct location"
        else
            echo "ERROR: Failed to move figures to correct location"
        fi
    fi
else
    echo "Found $FIG_FILES figure output files in $ANALYSIS_DIR/figures"
fi

# Fix any incorrect SHAP outputs in src/shap directory
echo "Checking for incorrect SHAP outputs in src/shap directory..."
if [ -d "src/shap/figures" ] || [ -d "src/shap/data" ]; then
    echo "Found outputs in src/shap directory that should be in results/analysis/full"
    
    if [ -d "src/shap/figures" ]; then
        echo "Moving figures from src/shap/figures to $ANALYSIS_DIR/figures"
        mkdir -p "$ANALYSIS_DIR/figures"
        cp -r src/shap/figures/* "$ANALYSIS_DIR/figures/" 2>/dev/null
    fi
    
    if [ -d "src/shap/data" ]; then
        echo "Moving data from src/shap/data to $ANALYSIS_DIR/data"
        mkdir -p "$ANALYSIS_DIR/data"
        cp -r src/shap/data/* "$ANALYSIS_DIR/data/" 2>/dev/null
    fi
else
    echo "No incorrect SHAP outputs found in src/shap directory."
fi

# Create a job summary
echo "Creating final job summary..."
SUMMARY_FILE="output/full_analysis/full_summary_${SLURM_JOB_ID}.txt"
echo "=== Full Analysis Summary ===" > $SUMMARY_FILE
echo "Job ID: $SLURM_JOB_ID" >> $SUMMARY_FILE
echo "Completion time: $(date)" >> $SUMMARY_FILE

echo -e "\n=== OPTIMIZATION RESULTS ===" >> $SUMMARY_FILE
if [ -f "$OPTIMIZATION_DIR/best_model.mat" ]; then
    echo "Optimization completed successfully" >> $SUMMARY_FILE
    echo "Best model configuration saved at: $OPTIMIZATION_DIR/best_model.mat" >> $SUMMARY_FILE
    if [ -f "$OPTIMIZATION_DIR/best_configuration.txt" ]; then
        echo -e "\nBest configuration details:" >> $SUMMARY_FILE
        cat "$OPTIMIZATION_DIR/best_configuration.txt" >> $SUMMARY_FILE
    fi
else
    echo "WARNING: No optimization results found" >> $SUMMARY_FILE
fi

echo -e "\n=== MODEL TRAINING RESULTS ===" >> $SUMMARY_FILE
if [ -f "$BEST_MODEL_DIR/best_model.mat" ] && [ -f "$TRAINING_DIR/trained_model.mat" ]; then
    echo "Model training completed successfully" >> $SUMMARY_FILE
    echo "Best model saved at: $BEST_MODEL_DIR/best_model.mat" >> $SUMMARY_FILE
    echo "Training model saved at: $TRAINING_DIR/trained_model.mat" >> $SUMMARY_FILE
    
    # List training figures
    TRAINING_FIGS=$(ls -la $TRAINING_DIR/Figures/*.fig 2>/dev/null | wc -l)
    echo "Training figures generated: $TRAINING_FIGS" >> $SUMMARY_FILE
else
    echo "WARNING: Model training may have failed" >> $SUMMARY_FILE
    echo "Missing one or more model files" >> $SUMMARY_FILE
fi

echo -e "\n=== SHAP ANALYSIS RESULTS ===" >> $SUMMARY_FILE
if [ -f "$ANALYSIS_DIR/data/shap_results.mat" ]; then
    echo "SHAP analysis completed successfully" >> $SUMMARY_FILE
    echo "SHAP results saved at: $ANALYSIS_DIR/data/shap_results.mat" >> $SUMMARY_FILE
    
    # List Excel files
    EXCEL_FILES=$(ls -la $ANALYSIS_DIR/*.xlsx 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo -e "\nSHAP Excel outputs:" >> $SUMMARY_FILE
        echo "$EXCEL_FILES" >> $SUMMARY_FILE
    else
        echo "No SHAP Excel outputs found" >> $SUMMARY_FILE
    fi
    
    # List figure files
    FIG_COUNT=$(ls -la $ANALYSIS_DIR/figures/*.fig $ANALYSIS_DIR/figures/*.png 2>/dev/null | wc -l)
    echo -e "\nSHAP visualization figures generated: $FIG_COUNT" >> $SUMMARY_FILE
    if [ $FIG_COUNT -gt 0 ]; then
        echo "Figures saved in: $ANALYSIS_DIR/figures/" >> $SUMMARY_FILE
    else
        echo "WARNING: No SHAP visualization figures found" >> $SUMMARY_FILE
    fi
else
    echo "WARNING: SHAP analysis may have failed" >> $SUMMARY_FILE
    echo "Missing SHAP results file: $ANALYSIS_DIR/data/shap_results.mat" >> $SUMMARY_FILE
fi

echo -e "\n=== WORKFLOW SUMMARY ===" >> $SUMMARY_FILE
echo "1. Hyperparameter Optimization: $([ -f "$OPTIMIZATION_DIR/best_model.mat" ] && echo "SUCCESS" || echo "FAILED")" >> $SUMMARY_FILE
echo "2. Best Model Training: $([ -f "$BEST_MODEL_DIR/best_model.mat" ] && echo "SUCCESS" || echo "FAILED")" >> $SUMMARY_FILE
echo "3. SHAP Analysis: $([ -f "$ANALYSIS_DIR/data/shap_results.mat" ] && echo "SUCCESS" || echo "FAILED")" >> $SUMMARY_FILE
echo "4. Excel Export: $([ $EXCEL_FILES -gt 0 ] && echo "SUCCESS" || echo "FAILED OR SKIPPED")" >> $SUMMARY_FILE

# Clean up temporary files
echo "Cleaning up temporary files..."
rm -rf ${MATLAB_TEMP_DIR}

echo "Full analysis workflow completed. Check the results/ directory for output."
echo "End time: $(date)"

# End of script