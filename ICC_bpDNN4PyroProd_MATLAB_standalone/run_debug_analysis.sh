#!/bin/bash
#SBATCH --job-name=debug_shap
#SBATCH --output=output/debug_analysis/debug_%j.out
#SBATCH --error=output/debug_analysis/debug_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=siqi-ic          # Account name from debug script
#SBATCH --partition=IllinoisComputes  # Partition name from debug script
#SBATCH --mail-type=BEGIN,END,FAIL    # Email notification
#SBATCH --mail-user=siqi@illinois.edu

# Improved run_debug_analysis.sh script with enhanced SHAP implementation

echo "===== DEBUG SHAP ANALYSIS JOB STARTED ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo ""

# Create output directories if they don't exist
mkdir -p output/debug_analysis
mkdir -p results/analysis/debug/data
mkdir -p results/analysis/debug/figures

# Set project directory to current directory (no hardcoded paths)
PROJECT_DIR=$(pwd)
echo "Project directory: $PROJECT_DIR"

# Log script version
SCRIPT_VERSION="2.0.0"
echo "Script version: $SCRIPT_VERSION (Enhanced SHAP Implementation)"
echo ""

# Create a summary file for easier reference
SUMMARY_FILE="output/debug_analysis/debug_summary_${SLURM_JOB_ID}.txt"

# Start summary file
echo "===== DEBUG SHAP ANALYSIS SUMMARY =====" > $SUMMARY_FILE
echo "Job ID: $SLURM_JOB_ID" >> $SUMMARY_FILE
echo "Node: $SLURM_JOB_NODELIST" >> $SUMMARY_FILE
echo "Start time: $(date)" >> $SUMMARY_FILE
echo "Script version: $SCRIPT_VERSION" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Check for critical files/directories
echo "Checking for required files and directories..."

# Check 1: Project directory should exist
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory $PROJECT_DIR not found!" | tee -a $SUMMARY_FILE
    exit 1
fi

# Check 2: Source directories should exist
if [ ! -d "$PROJECT_DIR/src" ]; then
    echo "ERROR: Source directory $PROJECT_DIR/src not found!" | tee -a $SUMMARY_FILE
    exit 1
fi

# Check 3: Data directories should exist
if [ ! -d "$PROJECT_DIR/data" ]; then
    echo "ERROR: Data directory $PROJECT_DIR/data not found!" | tee -a $SUMMARY_FILE
    exit 1
fi

# Check 4: Raw data file should exist
if [ ! -f "$PROJECT_DIR/data/raw/RawInputData.xlsx" ]; then
    echo "WARNING: Raw data file $PROJECT_DIR/data/raw/RawInputData.xlsx not found!" | tee -a $SUMMARY_FILE
    echo "  Some feature names may not be available." | tee -a $SUMMARY_FILE
else
    echo "Raw data file found: $PROJECT_DIR/data/raw/RawInputData.xlsx" | tee -a $SUMMARY_FILE
fi

# Look for model files - for debug mode, we don't need to search as extensively
MODEL_FOUND="no"

# Check for debug model
if [ -f "$PROJECT_DIR/results/debug/Results_trained.mat" ]; then
    echo "Found debug model: results/debug/Results_trained.mat" | tee -a $SUMMARY_FILE
    MODEL_FOUND="yes"
# Fallback to any model
elif [ -f "$PROJECT_DIR/results/best_model/best_model.mat" ]; then
    echo "Found best model: results/best_model/best_model.mat" | tee -a $SUMMARY_FILE
    MODEL_FOUND="yes"
fi

if [ "$MODEL_FOUND" == "no" ]; then
    echo "INFO: No pre-trained model found. Debug run will train a model with 20 epochs." | tee -a $SUMMARY_FILE
fi

echo "" | tee -a $SUMMARY_FILE
echo "All required directories found. Starting analysis..." | tee -a $SUMMARY_FILE
echo "" | tee -a $SUMMARY_FILE

# Create a temporary directory for MATLAB
MATLAB_TEMP_DIR="$PROJECT_DIR/tmp_matlab_debug_${SLURM_JOB_ID}"
mkdir -p $MATLAB_TEMP_DIR

echo "Created MATLAB temporary directory: $MATLAB_TEMP_DIR" | tee -a $SUMMARY_FILE
echo "Allocated memory: 32G (sufficient for debug run)" | tee -a $SUMMARY_FILE
echo "" | tee -a $SUMMARY_FILE

# Load MATLAB module
echo "Loading MATLAB module..."
module load matlab/R2021a

# Set MATLAB paths
MATLAB_PATHS="\
addpath(genpath('$PROJECT_DIR/src')); \
addpath('$PROJECT_DIR/src/model'); \
addpath('$PROJECT_DIR/src/scripts'); \
addpath('$PROJECT_DIR/src/shap'); \
addpath('$PROJECT_DIR/src/visualization'); \
cd('$PROJECT_DIR/src/scripts'); \
"

# Create MATLAB script for debug analysis
MATLAB_SCRIPT="$MATLAB_TEMP_DIR/run_debug_analysis.m"

cat << EOF > $MATLAB_SCRIPT
try
    % Set startup information
    disp('===== STARTING DEBUG ANALYSIS =====');
    disp(['Job ID: $SLURM_JOB_ID']);
    disp(['Running on node: $SLURM_JOB_NODELIST']);
    disp(['Start time: ' datestr(now)]);
    disp(' ');
    
    % Add all required paths
    $MATLAB_PATHS
    
    % Define analysis mode as 'debug'
    analysisMode = 'debug';
    disp(['Analysis mode: ' analysisMode]);
    
    % Create a diary file for logging
    diaryFile = fullfile('$PROJECT_DIR', 'output', 'debug_analysis', ['debug_matlab_log_' num2str($SLURM_JOB_ID) '.txt']);
    diary(diaryFile);
    disp(['Creating log file: ' diaryFile]);
    
    % Record diagnostic information
    disp('=== Environment Information ===');
    disp(['MATLAB Version: ' version]);
    if exist('parpool')
        poolobj = gcp('nocreate');
        if isempty(poolobj)
            disp('No parallel pool exists');
        else
            disp(['Parallel pool exists with ' num2str(poolobj.NumWorkers) ' workers']);
        end
    else
        disp('Parallel Computing Toolbox not available');
    end
    
    % Note: Debug mode runs much faster with reduced epochs (20 instead of 6000)
    disp('Debug mode will use 20 epochs instead of 6000 for ultra-fast execution');
    
    % Start timing
    totalTimer = tic;
    
    % Run the debug analysis
    disp('Calling debug_run_analysis.m...');
    debug_run_analysis;
    
    % End timing
    totalTime = toc(totalTimer);
    disp(['Total execution time: ' num2str(totalTime) ' seconds (' num2str(totalTime/60) ' minutes)']);
    
    % Export SHAP results to Excel
    disp('Exporting SHAP results to Excel...');
    export_shap_to_excel;
    
    % Verify output files
    disp('Verifying output files...');
    shapDir = fullfile('$PROJECT_DIR', 'results', 'analysis', 'debug');
    dataDir = fullfile(shapDir, 'data');
    figDir = fullfile(shapDir, 'figures');
    
    % Check for SHAP results file
    if exist(fullfile(dataDir, 'shap_results.mat'), 'file')
        disp('✓ SHAP results file found');
    else
        disp('✗ SHAP results file missing!');
    end
    
    % Count figure files
    figFiles = dir(fullfile(figDir, '*.fig'));
    pngFiles = dir(fullfile(figDir, '*.png'));
    disp(['✓ Figure files: ' num2str(length(figFiles)) ' .fig files and ' num2str(length(pngFiles)) ' .png files']);
    
    % Check Excel exports
    xlsFiles = dir(fullfile(dataDir, '*.xlsx'));
    disp(['✓ Excel exports: ' num2str(length(xlsFiles)) ' .xlsx files']);
    
    disp('===== DEBUG ANALYSIS COMPLETED SUCCESSFULLY =====');
    disp(['End time: ' datestr(now)]);
    diary off;
    exit(0);
catch ME
    % Handle errors
    disp('===== ERROR DURING DEBUG ANALYSIS =====');
    disp(['Error: ' ME.message]);
    for i = 1:length(ME.stack)
        disp(['File: ' ME.stack(i).file ', Line: ' num2str(ME.stack(i).line) ', Function: ' ME.stack(i).name]);
    end
    diary off;
    exit(1);
end
EOF

echo "Running MATLAB script..." | tee -a $SUMMARY_FILE
matlab -nodisplay -r "run('$MATLAB_SCRIPT')"
MATLAB_EXIT_CODE=$?

# Check if MATLAB completed successfully
if [ $MATLAB_EXIT_CODE -eq 0 ]; then
    echo "MATLAB completed successfully." | tee -a $SUMMARY_FILE
else
    echo "ERROR: MATLAB exited with code $MATLAB_EXIT_CODE" | tee -a $SUMMARY_FILE
fi

# Check for output files
echo "" | tee -a $SUMMARY_FILE
echo "Checking for output files..." | tee -a $SUMMARY_FILE

# Check for SHAP results file
if [ -f "$PROJECT_DIR/results/analysis/debug/data/shap_results.mat" ]; then
    echo "✓ SHAP results file found: results/analysis/debug/data/shap_results.mat" | tee -a $SUMMARY_FILE
else
    echo "✗ SHAP results file not found!" | tee -a $SUMMARY_FILE
fi

# Count figure files
if [ -d "$PROJECT_DIR/results/analysis/debug/figures" ]; then
    FIG_COUNT=$(ls -1 "$PROJECT_DIR/results/analysis/debug/figures"/*.fig 2>/dev/null | wc -l)
    PNG_COUNT=$(ls -1 "$PROJECT_DIR/results/analysis/debug/figures"/*.png 2>/dev/null | wc -l)
    echo "✓ Figure directory found with $FIG_COUNT .fig files and $PNG_COUNT .png files" | tee -a $SUMMARY_FILE
else
    echo "✗ Figure directory not found!" | tee -a $SUMMARY_FILE
fi

# Check for Excel exports
EXCEL_FILES=$(ls -1 "$PROJECT_DIR/results/analysis/debug/data"/*.xlsx 2>/dev/null | wc -l)
if [ $EXCEL_FILES -gt 0 ]; then
    echo "✓ Excel exports found: $EXCEL_FILES files" | tee -a $SUMMARY_FILE
else
    echo "✗ Excel exports not found!" | tee -a $SUMMARY_FILE
fi

# Clean up temporary directory
echo "" | tee -a $SUMMARY_FILE
echo "Cleaning up temporary files..." | tee -a $SUMMARY_FILE
rm -rf $MATLAB_TEMP_DIR
echo "Removed temporary directory: $MATLAB_TEMP_DIR" | tee -a $SUMMARY_FILE

# Final summary
echo "" | tee -a $SUMMARY_FILE
echo "===== DEBUG SHAP ANALYSIS JOB COMPLETED =====" | tee -a $SUMMARY_FILE
echo "End time: $(date)" | tee -a $SUMMARY_FILE
if [ $MATLAB_EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS" | tee -a $SUMMARY_FILE
else
    echo "Status: FAILED (MATLAB exit code: $MATLAB_EXIT_CODE)" | tee -a $SUMMARY_FILE
fi
echo "" | tee -a $SUMMARY_FILE
echo "Results can be found in:" | tee -a $SUMMARY_FILE
echo "  - Data files: $PROJECT_DIR/results/analysis/debug/data/" | tee -a $SUMMARY_FILE
echo "  - Figures: $PROJECT_DIR/results/analysis/debug/figures/" | tee -a $SUMMARY_FILE
echo "  - Log files: $PROJECT_DIR/output/debug_analysis/" | tee -a $SUMMARY_FILE
echo "" | tee -a $SUMMARY_FILE

echo "Debug analysis completed. Check $SUMMARY_FILE for a detailed summary."
