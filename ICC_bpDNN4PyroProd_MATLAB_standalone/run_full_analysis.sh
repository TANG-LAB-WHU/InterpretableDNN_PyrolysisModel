#!/bin/bash
# Full analysis script for GPM_SHAP neural network model
# This script runs the complete workflow from optimization to SHAP analysis
# Usage: sbatch run_full_analysis.sh
# 
# This script will:
# 1. Initialize the project structure and check for required files
# 2. Check and copy necessary data files
# 3. Create required directories
# 4. Run the complete workflow:
#    a. Hyperparameter optimization (finds best neural network configuration)
#    b. Best model training (trains a model with optimal hyperparameters)
#    c. Full SHAP analysis (with all samples/iterations)
# 5. Generate a comprehensive summary of results
# 
# Note: This script requires significant computational resources and time

#SBATCH --job-name=full_GPM_SHAP
#SBATCH --account=siqi-ic       # Replace with your ICC account name
#SBATCH --partition=IllinoisComputes   # Replace with your available partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16         # More cores for full version
#SBATCH --time=24:00:00              # More time for full version
#SBATCH --output=output/full_analysis/full_%j.out
#SBATCH --error=output/full_analysis/full_%j.err

echo "=== Starting GPM_SHAP Full Analysis Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"

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
# Note: No longer creating src/shap/data directory as all SHAP outputs go to results/analysis

# Get current working directory
WORK_DIR=$(pwd)
echo "Working directory: $WORK_DIR"

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
SCRIPT_FILES=("run_analysis.m" "debug_run_analysis.m" "calc_shap_values.m" "export_shap_to_excel.m" "optimize_hyperparameters.m" "train_best_model.m")
SCRIPT_MISSING=0

for file in "${SCRIPT_FILES[@]}"; do
    if [ ! -f "src/scripts/$file" ]; then
        echo "WARNING: src/scripts/$file not found. Please add this file."
        SCRIPT_MISSING=1
    fi
done

if [ $SCRIPT_MISSING -eq 1 ]; then
    echo "CRITICAL: Some required script files are missing. The full workflow may not complete successfully."
else
    echo "All required script files found."
fi

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

# Create placeholder visualization files if they don't exist
echo "Creating placeholder visualization files if needed..."
mkdir -p src/visualization

for file in "${VIZ_FILES[@]}"; do
    if [ ! -f "src/visualization/$file" ]; then
        echo "Creating placeholder for $file (will need to be replaced with actual implementation)..."
        FUNC_NAME=$(basename "$file" .m)
        
        # Create more appropriate placeholder based on function name
        case "$FUNC_NAME" in
            plot_shap_results)
                cat > "src/visualization/$file" << 'EOL'
%% Plot SHAP Results
% This script generates visualizations for SHAP analysis results
%
% Features:
% - Creates multiple visualization types:
%   1. Bar plots showing top 20 most important features per target
%   2. Bar plots showing all features sorted by importance
%   3. Force plots showing top feature contributions for individual samples
%   4. Summary plots showing global feature importance patterns

% Check if required variables exist in the workspace
if ~exist('shapValues', 'var') || ~exist('baseValue', 'var') || ~exist('varNames', 'var')
    warning('Required SHAP variables not found in workspace. Load from file or generate them first.');
    return;
end

% Convert varNames to featureNames if needed
if ~exist('featureNames', 'var')
    featureNames = varNames;
end

% Get script directory to determine output location
scriptPath = mfilename('fullpath');
[scriptDir, ~, ~] = fileparts(scriptPath);

% If this is a placeholder, simply report
warning('This is a placeholder version of plot_shap_results.');
disp('For proper visualization, replace this placeholder with the correct implementation.');
disp(['Will look for SHAP values in workspace or in results directory']);
disp(['Feature names available: ' num2str(length(featureNames))]);
disp(['SHAP values shape: ' num2str(size(shapValues))]);

% Create a simple summary plot as placeholder
if numel(shapValues) > 0
    try
        % Get dimensions
        [numSamples, numFeatures, numTargets] = size(shapValues);
        
        % For each target, create a simple feature importance plot
        for target = 1:numTargets
            % Calculate mean absolute SHAP values for feature importance
            mean_abs_shap = mean(abs(shapValues(:,:,target)), 1);
            
            % Sort features by importance
            [sorted_shap, sort_idx] = sort(mean_abs_shap, 'descend');
            
            % Create a simple figure
            figure('Name', sprintf('Feature Importance (Target %d)', target));
            bar(sorted_shap(1:min(20, end)));
            title(sprintf('Feature Importance (Target %d)', target));
            xlabel('Feature Index');
            ylabel('Mean |SHAP Value|');
            
            % Note that this is a placeholder
            text(0.5, 0.5, 'Placeholder visualization', 'Units', 'normalized', ...
                 'HorizontalAlignment', 'center', 'FontSize', 14, 'Color', 'red');
        end
    catch me
        warning('Failed to create placeholder visualization: %s', me.message);
    end
end
EOL
                ;;
            plot_shap_beeswarm)
                cat > "src/visualization/$file" << 'EOL'
%% SHAP Beeswarm Plot
% This script creates beeswarm plots for SHAP values

% Check if required variables exist in the workspace
if ~exist('shapValues', 'var') || ~exist('baseValue', 'var') || ~exist('varNames', 'var')
    warning('Required SHAP variables not found in workspace. Load from file or generate them first.');
    return;
end

% Convert varNames to featureNames if needed
if ~exist('featureNames', 'var')
    featureNames = varNames;
end

% Get script directory to determine output location
scriptPath = mfilename('fullpath');
[scriptDir, ~, ~] = fileparts(scriptPath);

% If this is a placeholder, simply report
warning('This is a placeholder version of plot_shap_beeswarm.');
disp('For proper visualization, replace this placeholder with the correct implementation.');
disp(['Will look for SHAP values in workspace or in results directory']);
disp(['Feature names available: ' num2str(length(featureNames))]);
disp(['SHAP values shape: ' num2str(size(shapValues))]);

% Create a simple summary plot as placeholder
if numel(shapValues) > 0
    try
        % Get dimensions
        [numSamples, numFeatures, numTargets] = size(shapValues);
        
        % For each target, create a simple feature importance plot
        for target = 1:numTargets
            % Calculate mean absolute SHAP values for feature importance
            mean_abs_shap = mean(abs(shapValues(:,:,target)), 1);
            
            % Sort features by importance
            [sorted_shap, sort_idx] = sort(mean_abs_shap, 'descend');
            
            % Create a simple figure
            figure('Name', sprintf('Beeswarm Plot (Target %d)', target));
            bar(sorted_shap(1:min(10, end)));
            title(sprintf('Beeswarm Plot Placeholder (Target %d)', target));
            xlabel('Feature Index');
            ylabel('Mean |SHAP Value|');
            
            % Note that this is a placeholder
            text(0.5, 0.5, 'Placeholder beeswarm plot', 'Units', 'normalized', ...
                 'HorizontalAlignment', 'center', 'FontSize', 14, 'Color', 'red');
        end
    catch me
        warning('Failed to create placeholder visualization: %s', me.message);
    end
end
EOL
                ;;
            plot_shap_original_summary)
                cat > "src/visualization/$file" << 'EOL'
%% SHAP Original Summary Plot
% This script creates original summary plots for SHAP values

% Check if required variables exist in the workspace
if ~exist('shapValues', 'var') || ~exist('baseValue', 'var') || ~exist('varNames', 'var')
    warning('Required SHAP variables not found in workspace. Load from file or generate them first.');
    return;
end

% Convert varNames to featureNames if needed
if ~exist('featureNames', 'var')
    featureNames = varNames;
end

% Get script directory to determine output location
scriptPath = mfilename('fullpath');
[scriptDir, ~, ~] = fileparts(scriptPath);

% If this is a placeholder, simply report
warning('This is a placeholder version of plot_shap_original_summary.');
disp('For proper visualization, replace this placeholder with the correct implementation.');
disp(['Will look for SHAP values in workspace or in results directory']);
disp(['Feature names available: ' num2str(length(featureNames))]);
disp(['SHAP values shape: ' num2str(size(shapValues))]);
EOL
                ;;
            fix_colorbar_style)
                cat > "src/visualization/$file" << 'EOL'
%% Fix Colorbar Style Script
% This script creates a colorbar legend file with a standardized style

% Get script directory
scriptPath = mfilename('fullpath');
[scriptDir, ~, ~] = fileparts(scriptPath);

% Get project root directory
rootDir = fileparts(fileparts(scriptDir));

% If this is a placeholder, simply report
warning('This is a placeholder version of fix_colorbar_style.');
disp('For proper visualization, replace this placeholder with the correct implementation.');
disp(['Root directory: ' rootDir]);
EOL
                ;;
            *)
                cat > "src/visualization/$file" << 'EOL'
function [] = placeholder_function()
% This is a placeholder function
% Please replace with the actual implementation
    warning('This is a placeholder function. Please replace with the actual implementation.');
    disp('For proper visualization, replace this placeholder with the correct implementation.');
end
EOL
                # Rename the function in the file to match the filename
                sed -i "s/placeholder_function/$(echo $FUNC_NAME)/g" "src/visualization/$file"
                ;;
        esac
        
        echo "Created placeholder for $file"
    fi
done

# Copy data files to model directory
echo "Copying data files to model directory..."
mkdir -p src/model
mkdir -p src/visualization
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
elif [ -d "SHAP_Analysis_Debug" ] && [ -f "SHAP_Analysis_Debug/Data/SHAP_metadata.xlsx" ]; then
    echo "Found SHAP_Analysis_Debug metadata file. Copying for feature name reference..."
    mkdir -p "results/analysis/full/data"
    cp -f "SHAP_Analysis_Debug/Data/SHAP_metadata.xlsx" "results/analysis/full/data/"
    echo "SHAP debug metadata copied to results directory"
fi

# Set up MATLAB temporary directory
export MATLAB_TEMP_DIR="/scratch/${USER}/matlab_temp_${SLURM_JOB_ID}"
mkdir -p ${MATLAB_TEMP_DIR}
echo "MATLAB temp directory: $MATLAB_TEMP_DIR"

# Check for previous optimization results
if [ -f "results/optimization/best_model.mat" ]; then
    echo "Found previous optimization results"
fi

# Load MATLAB module
echo "Loading MATLAB module..."
module load matlab/24.1

# Run MATLAB full analysis script
echo "Running full SHAP analysis workflow..."
echo "MATLAB will use following directories:"
echo "- Working directory: $WORK_DIR"
echo "- Scripts directory: $WORK_DIR/src/scripts"
echo "- Model directory: $WORK_DIR/src/model"
echo "- Visualization directory: $WORK_DIR/src/visualization"
echo "- Optimization directory: $WORK_DIR/results/optimization"
echo "- Best model directory: $WORK_DIR/results/best_model"
echo "- Training directory: $WORK_DIR/results/training"
echo "- Full analysis directory: $WORK_DIR/results/analysis/full"

# Ensure proper directory variables and structure for MATLAB script
matlab -nodisplay -nosplash -nodesktop -r "try; cd('$WORK_DIR'); addpath('$WORK_DIR/src/scripts'); addpath('$WORK_DIR/src/model'); addpath('$WORK_DIR/src/visualization'); pc = parcluster('local'); pc.NumWorkers = str2num(getenv('SLURM_NTASKS')); pc.JobStorageLocation = getenv('MATLAB_TEMP_DIR'); parpool(pc, pc.NumWorkers); rootDir='$WORK_DIR'; 
% Setup common paths
fprintf('===== RUNNING FULL WORKFLOW =====\n');
optimizationDir = fullfile(rootDir, 'results', 'optimization');
bestModelDir = fullfile(rootDir, 'results', 'best_model');
trainingDir = fullfile(rootDir, 'results', 'training');
shapDir = fullfile(rootDir, 'results', 'analysis', 'full');
full_results_dir = shapDir;

% Step 1: Hyperparameter Optimization
fprintf('\n===== STEP 1: RUNNING HYPERPARAMETER OPTIMIZATION =====\n');
if exist(fullfile(optimizationDir, 'best_model.mat'), 'file')
    fprintf('Found existing optimization results. Skipping optimization step.\n');
else
    fprintf('Starting hyperparameter optimization...\n');
    optimize_hyperparameters;
end

% Step 2: Train Best Model
fprintf('\n===== STEP 2: TRAINING MODEL WITH BEST HYPERPARAMETERS =====\n');
train_best_model;

% Step 3: SHAP Analysis
fprintf('\n===== STEP 3: RUNNING FULL SHAP ANALYSIS =====\n');
analysisMode = 'full';
run_analysis;

delete(gcp('nocreate')); exit; catch ME; disp(getReport(ME)); exit(1); end;"

# Check MATLAB execution status
MATLAB_STATUS=$?
if [ $MATLAB_STATUS -ne 0 ]; then
    echo "ERROR: MATLAB execution failed with exit code: $MATLAB_STATUS"
    echo "Trying export_shap_to_excel.m directly to ensure Excel files are created..."
    
    # Try running the Excel export directly if the main script failed
    matlab -nodisplay -nosplash -nodesktop -r "try; cd('$WORK_DIR'); addpath('$WORK_DIR/src/scripts'); addpath('$WORK_DIR/src/model'); addpath('$WORK_DIR/src/visualization'); shapDir = fullfile('$WORK_DIR', 'results', 'analysis', 'full'); fprintf('Running Excel export directly with shapDir: %s\n', shapDir); export_shap_to_excel; exit; catch ME; disp(getReport(ME)); exit(1); end;"
    
    EXCEL_STATUS=$?
    if [ $EXCEL_STATUS -ne 0 ]; then
        echo "ERROR: Excel export also failed with exit code: $EXCEL_STATUS"
    else
        echo "Excel export completed successfully."
    fi
else
    echo "SHAP analysis completed successfully with exit code: $MATLAB_STATUS"
fi

# Clean up any SHAP outputs that might have been saved to src/shap wrongly
echo "Cleaning up any incorrect SHAP outputs in src/shap directory..."
if [ -d "src/shap/figures" ]; then
    echo "Found incorrect figures in src/shap/figures - removing..."
    rm -rf src/shap/figures
fi

if [ -d "src/shap/data" ]; then
    echo "Found incorrect data in src/shap/data - removing..."
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
ls -lh optimization_log.txt >> $SUMMARY_FILE 2>/dev/null
ls -lh train_best_model_log.txt >> $SUMMARY_FILE 2>/dev/null
echo -e "\nResults directory structure:" >> $SUMMARY_FILE
ls -la results/ >> $SUMMARY_FILE 2>/dev/null
echo -e "\nOptimization results:" >> $SUMMARY_FILE
ls -la results/optimization >> $SUMMARY_FILE 2>/dev/null
echo -e "\nBest model directory:" >> $SUMMARY_FILE
ls -la results/best_model >> $SUMMARY_FILE 2>/dev/null
echo -e "\nTraining results directory:" >> $SUMMARY_FILE
ls -la results/training >> $SUMMARY_FILE 2>/dev/null
echo -e "\nSHAP Full Analysis directory:" >> $SUMMARY_FILE
ls -la results/analysis/full >> $SUMMARY_FILE 2>/dev/null
echo -e "\nSHAP Full Analysis data directory:" >> $SUMMARY_FILE
ls -la results/analysis/full/data >> $SUMMARY_FILE 2>/dev/null
echo -e "\nSHAP Full Analysis figures directory:" >> $SUMMARY_FILE
ls -la results/analysis/full/figures >> $SUMMARY_FILE 2>/dev/null
echo -e "\nSHAP Excel outputs:" >> $SUMMARY_FILE
ls -la results/analysis/full/data/*.xlsx >> $SUMMARY_FILE 2>/dev/null

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

# Verify all key files exist
echo -e "\nVerifying workflow completion:" >> $SUMMARY_FILE
WORKFLOW_COMPLETE=true

# Check optimization results
if [ ! -f "results/optimization/best_model.mat" ]; then
    echo "WARNING: Optimization results (best_model.mat) not found. Optimization may have failed." >> $SUMMARY_FILE
    WORKFLOW_COMPLETE=false
fi

# Check best model training results
if [ ! -f "results/best_model/best_model.mat" ]; then
    echo "WARNING: Best model file (best_model.mat) not found. Model training may have failed." >> $SUMMARY_FILE
    WORKFLOW_COMPLETE=false
fi

# Check training results
if [ ! -f "results/training/Results_trained.mat" ]; then
    echo "WARNING: Training results file (Results_trained.mat) not found. Model training may have failed." >> $SUMMARY_FILE
    WORKFLOW_COMPLETE=false
fi

# Check SHAP analysis results
if [ ! -f "results/analysis/full/data/shap_results.mat" ]; then
    echo "WARNING: SHAP results file (shap_results.mat) not found. SHAP analysis may have failed." >> $SUMMARY_FILE
    WORKFLOW_COMPLETE=false
fi

# Overall workflow status
if [ "$WORKFLOW_COMPLETE" = true ]; then
    echo "SUCCESS: All workflow stages completed successfully. Full analysis workflow is complete." >> $SUMMARY_FILE
else
    echo "WARNING: Some workflow stages may have failed. Check the logs for more information." >> $SUMMARY_FILE
fi

# Clean up MATLAB temporary directory
echo "Cleaning up temporary files..."
rm -rf ${MATLAB_TEMP_DIR}

echo "Full analysis completed. Check the results/ directory for output."
echo "End time: $(date)"