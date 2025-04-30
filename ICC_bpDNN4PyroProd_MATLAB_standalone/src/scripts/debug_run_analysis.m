%% Debug Run Analysis - Ultra-fast debugging for HPC environments
% This script runs a debug version of the analysis with enhanced diagnostics
% specifically designed for HPC environments like the ICC cluster.
%
% It performs a series of checks and diagnostics to help identify common issues
% that can occur on HPC clusters.

% Start timing execution
debugStartTime = tic;

% Enable verbose output - useful for debugging
verbose = true;

% Setup logging to file
logFile = fullfile(pwd, 'debug_run_analysis_log.txt');
diary(logFile);
fprintf('DEBUG ANALYSIS LOG - Started at %s\n\n', datestr(now));
fprintf('%-30s: %s\n', 'Working directory', pwd);
fprintf('%-30s: %s\n', 'MATLAB version', version);
fprintf('%-30s: %s\n', 'Computer architecture', computer);

try
    % Get the root directory of the project
    rootDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    fprintf('%-30s: %s\n\n', 'Root directory', rootDir);
    
    % Print environment information
    fprintf('===== ENVIRONMENT INFORMATION =====\n');
    
    % Get username
    [~, username] = system('whoami');
    fprintf('%-30s: %s', 'Username', username);
    
    % Check for SLURM environment variables
    slurmJobId = getenv('SLURM_JOB_ID');
    if ~isempty(slurmJobId)
        fprintf('%-30s: %s\n', 'SLURM Job ID', slurmJobId);
        fprintf('%-30s: %s\n', 'SLURM Job Name', getenv('SLURM_JOB_NAME'));
        fprintf('%-30s: %s\n', 'SLURM Nodelist', getenv('SLURM_JOB_NODELIST'));
        fprintf('%-30s: %s\n', 'SLURM Allocated CPUs', getenv('SLURM_JOB_CPUS_PER_NODE'));
    else
        fprintf('Not running in a SLURM environment\n');
    end
    
    % Check available disk space
    if isunix
        [~, diskInfo] = system('df -h .');
        fprintf('\nDisk Space Information:\n%s\n', diskInfo);
    end
    
    % Manual path setup with verification
    fprintf('\n===== PATH VERIFICATION =====\n');
    
    % Define important directories
    src_dirs = {'model', 'scripts', 'visualization', 'shap'};
    all_dirs_exist = true;
    
    % Check source directories
    for i = 1:length(src_dirs)
        dir_path = fullfile(rootDir, 'src', src_dirs{i});
        if exist(dir_path, 'dir')
            fprintf('%-30s: EXISTS\n', ['src/' src_dirs{i}]);
            if ~contains(path, dir_path)
                addpath(dir_path);
                fprintf('%-30s: ADDED TO PATH\n', ['src/' src_dirs{i}]);
            else
                fprintf('%-30s: ALREADY IN PATH\n', ['src/' src_dirs{i}]);
            end
        else
            fprintf('%-30s: MISSING!\n', ['src/' src_dirs{i}]);
            all_dirs_exist = false;
        end
    end
    
    if ~all_dirs_exist
        fprintf('\nWARNING: Some source directories are missing. This may cause issues.\n');
    end
    
    % Set up a flag to indicate we're in debug mode
    analysisMode = 'debug';
    fprintf('\nRunning in DEBUG mode with ultra-fast settings\n');
    
    % Create all required directories
    fprintf('\n===== CREATING DIRECTORIES =====\n');
    required_dirs = {
        fullfile(rootDir, 'results', 'debug'),
        fullfile(rootDir, 'results', 'debug', 'Figures'),
        fullfile(rootDir, 'results', 'analysis', 'debug', 'data'),
        fullfile(rootDir, 'results', 'analysis', 'debug', 'figures')
    };
    
    for i = 1:length(required_dirs)
        if ~exist(required_dirs{i}, 'dir')
            mkdir(required_dirs{i});
            fprintf('Created: %s\n', required_dirs{i});
        else
            fprintf('Already exists: %s\n', required_dirs{i});
        end
    end
    
    % Check for model conflicts - this is crucial
    fprintf('\n===== CHECKING FOR FILE CONFLICTS =====\n');
    conflict_found = false;
    
    % Check calc_shap_values.m - known to have multiple versions
    fprintf('Checking for calc_shap_values.m conflicts...\n');
    scripts_shap = fullfile(rootDir, 'src', 'scripts', 'calc_shap_values.m');
    dedicated_shap = fullfile(rootDir, 'src', 'shap', 'calc_shap_values.m');
    
    if exist(scripts_shap, 'file') && exist(dedicated_shap, 'file')
        fprintf('CONFLICT DETECTED: calc_shap_values.m exists in both:\n');
        fprintf('  1. %s\n', scripts_shap);
        fprintf('  2. %s\n', dedicated_shap);
        
        % Compare file dates to determine which is newer
        scripts_info = dir(scripts_shap);
        dedicated_info = dir(dedicated_shap);
        
        if scripts_info.datenum > dedicated_info.datenum
            fprintf('The version in src/scripts is newer (modified: %s)\n', datestr(scripts_info.datenum));
            fprintf('Will use this version during analysis.\n');
        else
            fprintf('The version in src/shap is newer (modified: %s)\n', datestr(dedicated_info.datenum));
            fprintf('Will use this version during analysis.\n');
        end
        conflict_found = true;
    elseif exist(scripts_shap, 'file')
        fprintf('Found calc_shap_values.m in src/scripts only.\n');
    elseif exist(dedicated_shap, 'file')
        fprintf('Found calc_shap_values.m in src/shap only.\n');
    else
        fprintf('WARNING: calc_shap_values.m not found in expected locations!\n');
    end
    
    fprintf('\n===== STARTING DEBUG ANALYSIS =====\n');
    fprintf('Debug mode will run with minimal epochs and simplified settings\n');
    fprintf('This is designed for ultra-fast execution to diagnose issues\n');
    
    % Call run_analysis with debug mode
    try
        % First check if we have a copy of run_analysis in path
        if exist('run_analysis', 'file')
            fprintf('Found run_analysis.m in path, proceeding with debug analysis...\n');
            % Run the analysis with debug mode set
            run_analysis;
        else
            fprintf('ERROR: run_analysis.m not found in path!\n');
            fprintf('Checking for file in src/scripts...\n');
            
            run_analysis_path = fullfile(rootDir, 'src', 'scripts', 'run_analysis.m');
            if exist(run_analysis_path, 'file')
                fprintf('Found at %s, adding to path...\n', run_analysis_path);
                addpath(fileparts(run_analysis_path));
                run_analysis;
            else
                error('Cannot find run_analysis.m file. Analysis cannot proceed.');
            end
        end
    catch run_err
        fprintf('\nERROR running analysis: %s\n', run_err.message);
        
        % Check for common error patterns and provide helpful messages
        if contains(run_err.message, 'No trained model found')
            fprintf('\nThis error occurs because no trained model was found.\n');
            fprintf('Checking for expected model locations...\n');
            
            model_locations = {
                fullfile(rootDir, 'results', 'best_model', 'best_model.mat'),
                fullfile(rootDir, 'results', 'training', 'Results_trained.mat'),
                fullfile(rootDir, 'results', 'optimization', 'best_model.mat')
            };
            
            for i = 1:length(model_locations)
                if exist(model_locations{i}, 'file')
                    fprintf('Model found at: %s\n', model_locations{i});
                else
                    fprintf('No model at: %s\n', model_locations{i});
                end
            end
            
            fprintf('\nPossible solutions:\n');
            fprintf('1. Run train_best_model.m first to generate a model file\n');
            fprintf('2. Run optimize_hyperparameters.m to generate an optimization model\n');
            fprintf('3. Ensure run_analysis.m contains code to create a synthetic model\n');
            
        elseif contains(run_err.message, 'Index exceeds')
            fprintf('\nThis appears to be a matrix dimension mismatch error.\n');
            fprintf('Common causes:\n');
            fprintf('1. Inconsistent feature dimensions between training and analysis\n');
            fprintf('2. Incorrect handling of TT vs TVT training strategies\n');
            fprintf('3. Data preprocessing differences between training and analysis\n');
        end
        
        % Always provide stack trace for detailed debugging
        fprintf('\n===== STACK TRACE =====\n');
        for k = 1:length(run_err.stack)
            fprintf('File: %s\nLine: %d\nFunction: %s\n\n', ...
                run_err.stack(k).file, run_err.stack(k).line, run_err.stack(k).name);
        end
        
        % Create a debug error file with detailed information
        debug_error_file = fullfile(rootDir, 'output', 'debug_analysis', 'debug_error_detailed.txt');
        
        % Ensure output directory exists
        output_dir = fullfile(rootDir, 'output', 'debug_analysis');
        if ~exist(output_dir, 'dir')
            mkdir(output_dir);
        end
        
        fid = fopen(debug_error_file, 'w');
        if fid ~= -1
            fprintf(fid, 'Debug Analysis Error Report\n');
            fprintf(fid, 'Generated: %s\n\n', datestr(now));
            fprintf(fid, 'Error Message: %s\n\n', run_err.message);
            fprintf(fid, 'Stack Trace:\n');
            for k = 1:length(run_err.stack)
                fprintf(fid, 'File: %s\nLine: %d\nFunction: %s\n\n', ...
                    run_err.stack(k).file, run_err.stack(k).line, run_err.stack(k).name);
            end
            
            fprintf(fid, '\nEnvironment Information:\n');
            fprintf(fid, 'MATLAB Version: %s\n', version);
            fprintf(fid, 'Computer: %s\n', computer);
            fprintf(fid, 'User: %s\n', username);
            if ~isempty(slurmJobId)
                fprintf(fid, 'SLURM Job ID: %s\n', slurmJobId);
                fprintf(fid, 'SLURM Node: %s\n', getenv('SLURM_JOB_NODELIST'));
            end
            
            fclose(fid);
            fprintf('Detailed error report written to: %s\n', debug_error_file);
        end
        
        % Rethrow error
        rethrow(run_err);
    end
    
    % If we got here, analysis completed successfully
    executionTime = toc(debugStartTime);
    
    fprintf('\n===== DEBUG ANALYSIS COMPLETED SUCCESSFULLY =====\n');
    fprintf('Execution time: %.2f seconds (%.2f minutes)\n', executionTime, executionTime/60);
    
    % Verify output files were created
    fprintf('\n===== VERIFYING OUTPUT FILES =====\n');
    
    % Define key output files that should exist after successful run
    debug_output_files = {
        fullfile(rootDir, 'results', 'debug', 'Results_trained.mat'),
        fullfile(rootDir, 'results', 'analysis', 'debug', 'data', 'shap_results.mat')
    };
    
    all_outputs_exist = true;
    for i = 1:length(debug_output_files)
        if exist(debug_output_files{i}, 'file')
            fprintf('%-50s: EXISTS\n', debug_output_files{i});
            
            % Get file size
            file_info = dir(debug_output_files{i});
            fprintf('  Size: %.2f MB, Date: %s\n', file_info.bytes/1e6, datestr(file_info.datenum));
            
            % For MAT files, check key variables
            if endsWith(debug_output_files{i}, '.mat')
                try
                    matObj = matfile(debug_output_files{i});
                    var_info = whos(matObj);
                    fprintf('  Contains %d variables:\n', length(var_info));
                    for v = 1:min(5, length(var_info))  % Show first 5 variables
                        fprintf('    %s (%s): %.2f KB\n', var_info(v).name, var_info(v).class, var_info(v).bytes/1024);
                    end
                    if length(var_info) > 5
                        fprintf('    ... and %d more variables\n', length(var_info) - 5);
                    end
                catch mat_err
                    fprintf('  WARNING: Could not inspect MAT file contents: %s\n', mat_err.message);
                end
            end
        else
            fprintf('%-50s: MISSING!\n', debug_output_files{i});
            all_outputs_exist = false;
        end
    end
    
    if ~all_outputs_exist
        fprintf('\nWARNING: Some expected output files are missing!\n');
    else
        fprintf('\nAll expected output files were created successfully.\n');
    end
    
    % Check figure files
    figure_dir = fullfile(rootDir, 'results', 'analysis', 'debug', 'figures');
    if exist(figure_dir, 'dir')
        figure_files = dir(fullfile(figure_dir, '*.png'));
        fprintf('\nFound %d PNG figures in %s\n', length(figure_files), figure_dir);
        if length(figure_files) > 0
            fprintf('First few figures:\n');
            for i = 1:min(5, length(figure_files))
                fprintf('  %s (%.2f KB)\n', figure_files(i).name, figure_files(i).bytes/1024);
            end
        else
            fprintf('WARNING: No PNG figures were generated!\n');
        end
    else
        fprintf('WARNING: Figure directory does not exist!\n');
    end
    
    % Create debug summary file
    summary_file = fullfile(rootDir, 'output', 'debug_analysis', sprintf('debug_summary_%s.txt', slurmJobId));
    fid = fopen(summary_file, 'w');
    if fid ~= -1
        fprintf(fid, 'DEBUG ANALYSIS SUMMARY\n');
        fprintf(fid, '=====================\n\n');
        fprintf(fid, 'Job completed at: %s\n', datestr(now));
        fprintf(fid, 'Execution time: %.2f seconds (%.2f minutes)\n', executionTime, executionTime/60);
        fprintf(fid, 'Status: SUCCESS\n\n');
        
        fprintf(fid, 'Output directories:\n');
        fprintf(fid, '  Model: %s\n', fullfile(rootDir, 'results', 'debug'));
        fprintf(fid, '  Analysis data: %s\n', fullfile(rootDir, 'results', 'analysis', 'debug', 'data'));
        fprintf(fid, '  Analysis figures: %s\n', fullfile(rootDir, 'results', 'analysis', 'debug', 'figures'));
        
        fprintf(fid, '\nNext steps:\n');
        fprintf(fid, '1. Review debug results in %s\n', fullfile(rootDir, 'results', 'analysis', 'debug'));
        fprintf(fid, '2. If debug results look good, run full analysis\n');
        fprintf(fid, '3. If issues persist, check %s for detailed logs\n', logFile);
        
        fclose(fid);
        fprintf('\nDebug summary written to: %s\n', summary_file);
    end
    
    fprintf('\n===== DEBUG ANALYSIS WORKFLOW COMPLETE =====\n');
    fprintf('All diagnostics completed successfully.\n');
    fprintf('To run the full analysis, execute run_analysis.m with analysisMode = ''full''\n');
    
catch main_err
    % Handle any unexpected errors in the debug script itself
    fprintf('\n===== CRITICAL ERROR IN DEBUG SCRIPT =====\n');
    fprintf('Error message: %s\n', main_err.message);
    
    % Print stack trace
    for k = 1:length(main_err.stack)
        fprintf('File: %s\nLine: %d\nFunction: %s\n\n', ...
            main_err.stack(k).file, main_err.stack(k).line, main_err.stack(k).name);
    end
    
    % Calculate execution time even on error
    executionTime = toc(debugStartTime);
    fprintf('Script execution failed after %.2f seconds (%.2f minutes)\n', executionTime, executionTime/60);
    
    % Write error to output file
    output_dir = fullfile(rootDir, 'output', 'debug_analysis');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    error_file = fullfile(output_dir, 'debug_critical_error.txt');
    fid = fopen(error_file, 'w');
    if fid ~= -1
        fprintf(fid, 'CRITICAL DEBUG SCRIPT ERROR\n');
        fprintf(fid, 'Generated: %s\n\n', datestr(now));
        fprintf(fid, 'Error Message: %s\n\n', main_err.message);
        fprintf(fid, 'Stack Trace:\n');
        for k = 1:length(main_err.stack)
            fprintf(fid, 'File: %s\nLine: %d\nFunction: %s\n\n', ...
                main_err.stack(k).file, main_err.stack(k).line, main_err.stack(k).name);
        end
        fclose(fid);
        fprintf('Critical error report written to: %s\n', error_file);
    end
    
    % Don't rethrow - this is the end of the script anyway
end

% Always close the diary
diary off;

fprintf('\nDebug log written to: %s\n', logFile);