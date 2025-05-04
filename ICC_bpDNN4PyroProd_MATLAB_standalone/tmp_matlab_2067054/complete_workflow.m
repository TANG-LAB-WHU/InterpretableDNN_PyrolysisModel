try
    % Set up environment and logging
    rootDir = '/projects/illinois/eng/cee/siqi/MATLAB_Jobs_new4bpDNN/ICC_bpDNN4PyroProd_MATLAB_standalone';
    srcDir = fullfile(rootDir, 'src');
    scriptDir = fullfile(srcDir, 'scripts');
    addpath(genpath(srcDir));
    
    % Create a diary file for logging
    diaryFile = fullfile('/projects/illinois/eng/cee/siqi/MATLAB_Jobs_new4bpDNN/ICC_bpDNN4PyroProd_MATLAB_standalone/output/full_analysis', 'matlab_log_2067054.txt');
    diary(diaryFile);
    fprintf('===== COMPLETE WORKFLOW STARTED =====\n');
    fprintf('Job ID: 2067054\n');
    fprintf('Current directory: %s\n', pwd);
    fprintf('Starting time: %s\n\n', datestr(now));
    
    % Change to scripts directory where main scripts are located
    cd(scriptDir);
    fprintf('Changed to scripts directory: %s\n\n', pwd);
    
    % Track execution time for each step
    totalTimer = tic;
    
    % STEP 1: Hyperparameter Optimization
    if true
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
                    'jobId', '2067054', ...
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
    if true
        fprintf('===== STEP 2: BEST MODEL TRAINING =====\n');
        step2Timer = tic;
        
        % Ensure optimization results are available for training
        optResultFile = fullfile(rootDir, 'results', 'optimization', 'best_model.mat');
        if ~exist(optResultFile, 'file') && true == false
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
                    'jobId', '2067054', ...
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
            'jobId', '2067054', ...
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
    if true && true
        fprintf('All three workflow steps completed successfully\n');
    elseif true
        fprintf('Best model training and SHAP analysis completed successfully\n');
    else
        fprintf('SHAP analysis completed successfully\n');
    end
    
    % Create final workflow completion marker
    workflowCompletionFile = fullfile(rootDir, 'results', 'workflow_complete.mat');
    workflowSummary = struct(...
        'jobId', '2067054', ...
        'completedAt', datestr(now), ...
        'executionTime', totalTime, ...
        'stepsCompleted', struct(...
            'optimization', true, ...
            'training', true, ...
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
    errorFile = fullfile('/projects/illinois/eng/cee/siqi/MATLAB_Jobs_new4bpDNN/ICC_bpDNN4PyroProd_MATLAB_standalone/output/full_analysis', 'workflow_error.txt');
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
