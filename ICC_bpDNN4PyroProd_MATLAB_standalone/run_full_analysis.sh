#!/bin/bash
#SBATCH --job-name=full_wf
#SBATCH --output=output/full_analysis/full_%j.out
#SBATCH --error=output/full_analysis/full_%j.err
#SBATCH --time=00:1:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --account=siqi-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=siqi@illinois.edu

# Complete workflow implementation script (Hyperparameter optimization + Best model training + SHAP analysis)
SCRIPT_VERSION="3.0.0"

echo "===== COMPLETE PYROLYSIS MODEL WORKFLOW STARTED ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "Script version: $SCRIPT_VERSION (Complete workflow implementation)"
echo ""

# Set project directory to current directory (no hardcoded paths)
PROJECT_DIR=$(pwd)
echo "Project directory: $PROJECT_DIR"
echo ""

# Create output directories if they don't exist
mkdir -p output/full_analysis
mkdir -p results/analysis/full/data
mkdir -p results/analysis/full/figures
mkdir -p results/training/Figures
mkdir -p results/best_model/Figures
mkdir -p results/optimization
mkdir -p results/debug/Figures

# Create a summary file for easier reference
SUMMARY_FILE="output/full_analysis/full_summary_${SLURM_JOB_ID}.txt"

# Start summary file
echo "===== COMPLETE WORKFLOW SUMMARY =====" > $SUMMARY_FILE
echo "Job ID: $SLURM_JOB_ID" >> $SUMMARY_FILE
echo "Node: $SLURM_JOB_NODELIST" >> $SUMMARY_FILE
echo "Start time: $(date)" >> $SUMMARY_FILE
echo "Script version: $SCRIPT_VERSION" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE
echo "Workflow steps:" >> $SUMMARY_FILE
echo "1. Hyperparameter Optimization" >> $SUMMARY_FILE
echo "2. Best Model Training" >> $SUMMARY_FILE
echo "3. SHAP Analysis" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Check for critical files/directories
echo "Checking for required files and directories..." | tee -a $SUMMARY_FILE

# Basic environment checks
if [ ! -d "$PROJECT_DIR/src" ] || [ ! -d "$PROJECT_DIR/data" ]; then
    echo "ERROR: Critical directories missing!" | tee -a $SUMMARY_FILE
    exit 1
fi

# Check for raw data file
if [ ! -f "$PROJECT_DIR/data/raw/RawInputData.xlsx" ]; then
    echo "WARNING: Raw data file not found. Feature names may not be available." | tee -a $SUMMARY_FILE
fi

# Create a temporary directory for MATLAB
MATLAB_TEMP_DIR="$PROJECT_DIR/tmp_matlab_${SLURM_JOB_ID}"
mkdir -p $MATLAB_TEMP_DIR
echo "Created MATLAB temporary directory: $MATLAB_TEMP_DIR" | tee -a $SUMMARY_FILE

# Load MATLAB module with fallback mechanism
echo "Loading MATLAB module..." | tee -a $SUMMARY_FILE
module load matlab/24.1 || module load matlab
MATLAB_LOAD_STATUS=$?

if [ $MATLAB_LOAD_STATUS -ne 0 ]; then
    echo "WARNING: Failed to load MATLAB module. Using system-wide installation if available." | tee -a $SUMMARY_FILE
fi

# Verify MATLAB is available
if ! command -v matlab &> /dev/null; then
    echo "ERROR: MATLAB command not found!" | tee -a $SUMMARY_FILE
    exit 1
else
    MATLAB_CMD="$(which matlab)"
    echo "MATLAB command found at: $MATLAB_CMD" | tee -a $SUMMARY_FILE
fi

echo "" | tee -a $SUMMARY_FILE
echo "===== WORKFLOW STEP DETERMINATION =====" | tee -a $SUMMARY_FILE

# Determine which steps to run
NEED_OPTIMIZATION=false
NEED_TRAINING=false
NEED_ANALYSIS=true

# Check for model files (in order of preference)
if [ -f "$PROJECT_DIR/results/best_model/best_model.mat" ]; then
    echo "✓ Found trained model: results/best_model/best_model.mat" | tee -a $SUMMARY_FILE
    echo "  - Skipping optimization and training steps" | tee -a $SUMMARY_FILE
elif [ -f "$PROJECT_DIR/results/optimization/best_model.mat" ]; then
    echo "✓ Found optimization results: results/optimization/best_model.mat" | tee -a $SUMMARY_FILE
    echo "  - Skipping optimization step" | tee -a $SUMMARY_FILE
    NEED_TRAINING=true
else
    echo "! No optimization results found. Will run complete workflow." | tee -a $SUMMARY_FILE
    NEED_OPTIMIZATION=true
    NEED_TRAINING=true
fi

echo "" | tee -a $SUMMARY_FILE
echo "Workflow plan:" | tee -a $SUMMARY_FILE
if [ "$NEED_OPTIMIZATION" = true ]; then echo "- Will run hyperparameter optimization" | tee -a $SUMMARY_FILE; fi
if [ "$NEED_TRAINING" = true ]; then echo "- Will run best model training" | tee -a $SUMMARY_FILE; fi
echo "- Will run SHAP analysis" | tee -a $SUMMARY_FILE
echo "" | tee -a $SUMMARY_FILE

# Create the MATLAB workflow script
MATLAB_SCRIPT="$MATLAB_TEMP_DIR/complete_workflow.m"

cat << EOF > $MATLAB_SCRIPT
try
    % Set up environment and logging
    rootDir = '$PROJECT_DIR';
    srcDir = fullfile(rootDir, 'src');
    scriptDir = fullfile(srcDir, 'scripts');
    addpath(genpath(srcDir));
    
    % Create a diary file for logging
    diaryFile = fullfile('$PROJECT_DIR/output/full_analysis', 'matlab_log_$SLURM_JOB_ID.txt');
    diary(diaryFile);
    fprintf('===== COMPLETE WORKFLOW STARTED =====\n');
    fprintf('Job ID: $SLURM_JOB_ID\n');
    fprintf('Current directory: %s\n', pwd);
    fprintf('Starting time: %s\n\n', datestr(now));
    
    % Change to scripts directory where main scripts are located
    cd(scriptDir);
    fprintf('Changed to scripts directory: %s\n\n', pwd);
    
    % Track execution time for each step
    totalTimer = tic;
    
    % STEP 1: Hyperparameter Optimization
    if $NEED_OPTIMIZATION
        fprintf('===== STEP 1: HYPERPARAMETER OPTIMIZATION =====\n');
        fprintf('DEBUG: About to execute step1Timer = tic\n');
        step1Timer = tic;
        fprintf('DEBUG: step1Timer defined? %d\n', exist('step1Timer', 'var'));
        
        fprintf('Running optimize_hyperparameters.m...\n');
        optimize_hyperparameters;
        fprintf('Hyperparameter optimization completed.\n');
        
        % Verify optimization results were created
        optResultFile = fullfile(rootDir, 'results', 'optimization', 'best_model.mat');
        if exist(optResultFile, 'file')
            % Verify file integrity and content
            try
                optResults = load(optResultFile);
                requiredFields = {'bestParams', 'optimizationResults'};
                missingFields = setdiff(requiredFields, fieldnames(optResults));
                
                if ~isempty(missingFields)
                    error('Optimization results missing required fields: %s', strjoin(missingFields, ', '));
                end
                
                fprintf('✓ Optimization results created and validated: %s\n', optResultFile);
                fprintf('  Best hyperparameters found: \n');
                disp(optResults.bestParams);
                
                % Create completion marker using the create_completion_marker function
                metadata = struct(...
                    'jobId', '$SLURM_JOB_ID', ...
                    'bestObjective', optResults.optimizationResults.bestObjective, ...
                    'totalIterations', optResults.optimizationResults.totalConfigurations ...
                );
                create_completion_marker('optimization', optResultFile, metadata);
                fprintf('✓ Created transition marker for optimization step\n');
            catch ME
                error('Failed to validate optimization results: %s', ME.message);
            end
        else
            error('Optimization failed: No results file created');
        end

        fprintf('DEBUG: Verification complete. About to execute toc(step1Timer)\n');
        fprintf('DEBUG: step1Timer still defined? %d\n', exist('step1Timer', 'var'));
        step1Time = toc(step1Timer);
        fprintf('Optimization time: %.2f seconds (%.2f minutes)\n\n', step1Time, step1Time/60);
    else
        fprintf('===== STEP 1: HYPERPARAMETER OPTIMIZATION =====\n');
        fprintf('Skipping - Optimization results already exist\n\n');
        
        % Still verify existing results before proceeding
        optResultFile = fullfile(rootDir, 'results', 'optimization', 'best_model.mat');
        try
            optResults = load(optResultFile);
            fprintf('✓ Using existing optimization results: %s\n', optResultFile);
            if isfield(optResults, 'bestParams')
                fprintf('  Using these hyperparameters: \n');
                disp(optResults.bestParams);
            end
        catch ME
            warning('Could not validate existing optimization results: %s', ME.message);
            fprintf('  Will attempt to proceed with training anyway\n');
        end
    end
    
    % STEP 2: Best Model Training
    if $NEED_TRAINING
        fprintf('===== STEP 2: BEST MODEL TRAINING =====\n');
        step2Timer = tic;
        
        % Ensure optimization results are available for training
        optResultFile = fullfile(rootDir, 'results', 'optimization', 'best_model.mat');
        if ~exist(optResultFile, 'file') && $NEED_OPTIMIZATION == false
            error('Cannot train best model: Optimization results not found at %s', optResultFile);
        end
        
        fprintf('Running train_best_model.m...\n');
        % Pass important variables to ensure proper handover
        optimResultPath = optResultFile;  % Make path available to train_best_model
        train_best_model;
        fprintf('Best model training completed.\n');
        
        % Verify model was created and is valid
        modelFile = fullfile(rootDir, 'results', 'best_model', 'best_model.mat');
        if exist(modelFile, 'file')
            try
                % Attempt to load and validate the model
                trainedModel = load(modelFile);
                requiredFields = {'net', 'trainResults', 'params'};
                missingFields = setdiff(requiredFields, fieldnames(trainedModel));
                
                if ~isempty(missingFields)
                    error('Trained model missing required fields: %s', strjoin(missingFields, ', '));
                end
                
                fprintf('✓ Trained model created and validated: %s\n', modelFile);
                
                % Copy model to the standard location expected by analysis
                standardModelPath = fullfile(rootDir, 'results', 'training', 'trained_model.mat');
                copyfile(modelFile, standardModelPath);
                fprintf('✓ Copied model to standard location for analysis: %s\n', standardModelPath);
                
                % Create completion marker using the create_completion_marker function
                metadata = struct(...
                    'jobId', '$SLURM_JOB_ID', ...
                    'finalPerformance', trainedModel.trainResults.best_perf ...
                );
                
                % Safely add neuronsInLayers field based on which structure is available
                if isfield(trainedModel, 'trainResults') && isfield(trainedModel.trainResults, 'neuronsInLayers')
                    metadata.neuronsInLayers = trainedModel.trainResults.neuronsInLayers;
                elseif isfield(trainedModel, 'net') && isfield(trainedModel.net, 'layers')
                    % Extract from net.layers structure
                    metadata.neuronsInLayers = arrayfun(@(x) x.size(1), trainedModel.net.layers, 'UniformOutput', false);
                else
                    % Default fallback if structure is unknown
                    metadata.neuronsInLayers = {8};  % Assume single hidden layer with 8 neurons as fallback
                end
                
                create_completion_marker('training', modelFile, metadata);
                fprintf('✓ Created transition marker for training step\n');
            catch ME
                error('Failed to validate trained model: %s', ME.message);
            end
        else
            error('Training failed: No model file created');
        end
        
        step2Time = toc(step2Timer);
        fprintf('Training time: %.2f seconds (%.2f minutes)\n\n', step2Time, step2Time/60);
    else
        fprintf('===== STEP 2: BEST MODEL TRAINING =====\n');
        fprintf('Skipping - Trained model already exists\n\n');
        
        % Still verify existing model before proceeding
        modelFile = fullfile(rootDir, 'results', 'best_model', 'best_model.mat');
        try
            trainedModel = load(modelFile);
            fprintf('✓ Using existing trained model: %s\n', modelFile);
            
            % Ensure model is also in standard location for analysis
            standardModelPath = fullfile(rootDir, 'results', 'training', 'trained_model.mat');
            if ~exist(standardModelPath, 'file')
                copyfile(modelFile, standardModelPath);
                fprintf('✓ Copied model to standard location for analysis: %s\n', standardModelPath);
            end
        catch ME
            warning('Could not validate existing trained model: %s', ME.message);
            fprintf('  Will attempt to proceed with analysis anyway\n');
        end
    end
    
    % STEP 3: SHAP Analysis
    fprintf('===== STEP 3: SHAP ANALYSIS =====\n');
    step3Timer = tic;
    
    % First verify model availability for analysis
    standardModelPath = fullfile(rootDir, 'results', 'training', 'trained_model.mat');
    bestModelPath = fullfile(rootDir, 'results', 'best_model', 'best_model.mat');
    
    if exist(standardModelPath, 'file')
        fprintf('✓ Using model at standard location: %s\n', standardModelPath);
        modelToUse = standardModelPath;
    elseif exist(bestModelPath, 'file')
        fprintf('✓ Using best model: %s\n', bestModelPath);
        
        % Copy to standard location expected by analysis
        copyfile(bestModelPath, standardModelPath);
        fprintf('  Copied to standard location: %s\n', standardModelPath);
    else
        % No model fallback - enforce workflow consistency by stopping with error
        error(['No trained model found at standard locations. ' ...
               'Workflow requires a trained model from either optimization+training ' ...
               'or direct training step. Check that previous steps completed successfully.']);
    end
    
    fprintf('Running run_analysis.m...\n');
    trainedModelPath = modelToUse;  % Make path available to run_analysis
    analysisMode = 'full';
    run_analysis;
    fprintf('SHAP analysis completed.\n');
    
    % Verify SHAP results were created
    shapFile = fullfile(rootDir, 'results', 'analysis', 'full', 'data', 'shap_results.mat');
    if exist(shapFile, 'file')
        fprintf('✓ SHAP results created: %s\n', shapFile);
        
        % Create completion marker using the create_completion_marker function
        metadata = struct(...
            'jobId', '$SLURM_JOB_ID', ...
            'analysisMode', analysisMode, ...
            'modelUsed', modelToUse ...
        );
        create_completion_marker('analysis', shapFile, metadata);
        fprintf('✓ Created completion marker for analysis step\n');
    else
        error('Analysis failed: No SHAP results file created');
    end
    
    step3Time = toc(step3Timer);
    fprintf('Analysis time: %.2f seconds (%.2f minutes)\n\n', step3Time, step3Time/60);
    
    % Try to export SHAP results to Excel
    fprintf('Exporting SHAP results to Excel...\n');
    try
        export_shap_to_excel;
        fprintf('✓ Excel export completed\n');
    catch xlsErr
        fprintf('! Excel export failed: %s\n', xlsErr.message);
        fprintf('Creating CSV exports instead...\n');
        
        % Create CSV files as alternative
        try
            analysisDataDir = fullfile(rootDir, 'results', 'analysis', 'full', 'data');
            shapResultsFile = fullfile(analysisDataDir, 'shap_results.mat');
            
            if exist(shapResultsFile, 'file')
                % Load the SHAP data
                shapData = load(shapResultsFile);
                
                if isfield(shapData, 'shapValues') && isfield(shapData, 'featureNames')
                    % Create CSV file
                    csvFile = fullfile(analysisDataDir, 'shap_values_summary.csv');
                    
                    % Calculate mean absolute SHAP value for each feature
                    meanAbsShap = mean(abs(shapData.shapValues), 1);
                    
                    % Sort features by importance
                    [sortedImportance, sortIndex] = sort(meanAbsShap, 'descend');
                    
                    % Write to CSV
                    fid = fopen(csvFile, 'w');
                    if fid > 0
                        % Write header
                        fprintf(fid, 'Feature,Mean Absolute SHAP Value\n');
                        
                        % Write data
                        for i = 1:length(sortIndex)
                            if sortIndex(i) <= length(shapData.featureNames)
                                fprintf(fid, '%s,%f\n', shapData.featureNames{sortIndex(i)}, sortedImportance(i));
                            end
                        end
                        fclose(fid);
                        fprintf('✓ Created CSV export: %s\n', csvFile);
                    end
                end
            end
        catch csvErr
            fprintf('! CSV export failed: %s\n', csvErr.message);
        end
    end
    
    % Calculate total workflow time
    totalTime = toc(totalTimer);
    fprintf('\n===== COMPLETE WORKFLOW FINISHED =====\n');
    fprintf('Total execution time: %.2f seconds (%.2f minutes)\n', totalTime, totalTime/60);
    if $NEED_OPTIMIZATION && $NEED_TRAINING
        fprintf('All three workflow steps completed successfully\n');
    elseif $NEED_TRAINING
        fprintf('Best model training and SHAP analysis completed successfully\n');
    else
        fprintf('SHAP analysis completed successfully\n');
    end
    
    % Create final workflow completion marker
    workflowCompletionFile = fullfile(rootDir, 'results', 'workflow_complete.mat');
    workflowSummary = struct(...
        'jobId', '$SLURM_JOB_ID', ...
        'completedAt', datestr(now), ...
        'executionTime', totalTime, ...
        'stepsCompleted', struct(...
            'optimization', $NEED_OPTIMIZATION, ...
            'training', $NEED_TRAINING, ...
            'analysis', true ...
        ) ...
    );
    create_completion_marker('workflow', workflowCompletionFile, workflowSummary);
    fprintf('✓ Created final workflow completion marker\n');
    
    diary off;
    exit(0);
catch ME
    % Handle errors with comprehensive error reporting
    fprintf('===== ERROR DURING WORKFLOW =====\n');
    fprintf('Error message: %s\n', ME.message);
    
    % Get stack trace
    if ~isempty(ME.stack)
        fprintf('Stack trace:\n');
        for i = 1:length(ME.stack)
            fprintf('File: %s, Line: %d, Function: %s\n', ...
                ME.stack(i).file, ME.stack(i).line, ME.stack(i).name);
        end
    end
    
    % Write detailed error report
    fprintf('Detailed error report:\n%s\n', getReport(ME, 'extended'));
    
    % Create error marker file
    errorFile = fullfile('$PROJECT_DIR/output/full_analysis', 'workflow_error.txt');
    fid = fopen(errorFile, 'w');
    if fid > 0
        fprintf(fid, 'Workflow failed at %s\n', datestr(now));
        fprintf(fid, 'Error message: %s\n', ME.message);
        fprintf(fid, 'Stack trace:\n');
        if ~isempty(ME.stack)
            for k = 1:length(ME.stack)
                fprintf(fid, '  Function: %s, Line: %d, File: %s\n', ...
                    ME.stack(k).name, ME.stack(k).line, ME.stack(k).file);
            end
        end
        fclose(fid);
    end
    
    diary off;
    exit(1);
end
EOF

# Run the MATLAB workflow script
echo "===== STARTING COMPLETE WORKFLOW =====" | tee -a $SUMMARY_FILE
echo "Running MATLAB script..." | tee -a $SUMMARY_FILE
$MATLAB_CMD -nodisplay -r "run('$MATLAB_SCRIPT')"
MATLAB_EXIT_CODE=$?

if [ $MATLAB_EXIT_CODE -eq 0 ]; then
    echo "MATLAB workflow completed successfully." | tee -a $SUMMARY_FILE
else
    echo "ERROR: MATLAB workflow failed with exit code $MATLAB_EXIT_CODE" | tee -a $SUMMARY_FILE
fi

# Check for output files from each step
echo "" | tee -a $SUMMARY_FILE
echo "===== WORKFLOW OUTPUT VERIFICATION =====" | tee -a $SUMMARY_FILE

# Step 1: Check optimization results
if [ "$NEED_OPTIMIZATION" = true ]; then
    if [ -f "$PROJECT_DIR/results/optimization/best_model.mat" ]; then
        echo "✓ Step 1: Optimization results created" | tee -a $SUMMARY_FILE
    else
        echo "✗ Step 1: Optimization results missing" | tee -a $SUMMARY_FILE
    fi
fi

# Step 2: Check training results
if [ "$NEED_TRAINING" = true ]; then
    if [ -f "$PROJECT_DIR/results/best_model/best_model.mat" ]; then
        echo "✓ Step 2: Best model created" | tee -a $SUMMARY_FILE
    else
        echo "✗ Step 2: Best model missing" | tee -a $SUMMARY_FILE
    fi
fi

# Step 3: Check analysis results
if [ -f "$PROJECT_DIR/results/analysis/full/data/shap_results.mat" ]; then
    echo "✓ Step 3: SHAP results created" | tee -a $SUMMARY_FILE
else
    echo "✗ Step 3: SHAP results missing" | tee -a $SUMMARY_FILE
fi

# Check for Excel or CSV exports
EXCEL_FILES=$(ls -1 "$PROJECT_DIR/results/analysis/full/data"/*.xlsx 2>/dev/null | wc -l)
CSV_FILES=$(ls -1 "$PROJECT_DIR/results/analysis/full/data"/*.csv 2>/dev/null | wc -l)
if [ $EXCEL_FILES -gt 0 ]; then
    echo "✓ Excel exports created: $EXCEL_FILES files" | tee -a $SUMMARY_FILE
elif [ $CSV_FILES -gt 0 ]; then
    echo "✓ CSV exports created: $CSV_FILES files" | tee -a $SUMMARY_FILE
else
    echo "✗ No data exports (Excel/CSV) found" | tee -a $SUMMARY_FILE
fi

# Check figures
FIG_COUNT=$(find "$PROJECT_DIR/results" -name "*.fig" | wc -l)
PNG_COUNT=$(find "$PROJECT_DIR/results" -name "*.png" | wc -l)
echo "✓ Figure files created: $FIG_COUNT .fig files, $PNG_COUNT .png files" | tee -a $SUMMARY_FILE

# Clean up temporary directory
echo "" | tee -a $SUMMARY_FILE
echo "Cleaning up temporary files..." | tee -a $SUMMARY_FILE
rm -rf $MATLAB_TEMP_DIR
echo "Removed temporary directory: $MATLAB_TEMP_DIR" | tee -a $SUMMARY_FILE

# Final summary
echo "" | tee -a $SUMMARY_FILE
echo "===== COMPLETE WORKFLOW JOB FINISHED =====" | tee -a $SUMMARY_FILE
echo "End time: $(date)" | tee -a $SUMMARY_FILE
if [ $MATLAB_EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS" | tee -a $SUMMARY_FILE
else
    echo "Status: FAILED (MATLAB exit code: $MATLAB_EXIT_CODE)" | tee -a $SUMMARY_FILE
fi

echo "" | tee -a $SUMMARY_FILE
echo "Results locations:" | tee -a $SUMMARY_FILE
echo "  - Optimization results: $PROJECT_DIR/results/optimization/" | tee -a $SUMMARY_FILE
echo "  - Best model: $PROJECT_DIR/results/best_model/" | tee -a $SUMMARY_FILE
echo "  - SHAP analysis: $PROJECT_DIR/results/analysis/full/" | tee -a $SUMMARY_FILE
echo "  - Log files: $PROJECT_DIR/output/full_analysis/" | tee -a $SUMMARY_FILE
echo "" | tee -a $SUMMARY_FILE

echo "Complete workflow finished. Check $SUMMARY_FILE for a detailed summary."
