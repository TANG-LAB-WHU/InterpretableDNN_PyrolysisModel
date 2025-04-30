% Script to check all temporary files generated during analysis
% Open a file for writing the output
log_file = 'temp_files_check.log';
diary(log_file);

fprintf('=== Checking Temporary Analysis Files ===\n');
fprintf('Log file: %s\n', fullfile(pwd, log_file));

% Get script path
curr_script_path = mfilename('fullpath');
fprintf('Current script path: %s\n', curr_script_path);

% Get directory of current script
curr_script_dir = fileparts(curr_script_path);
fprintf('Current script directory: %s\n', curr_script_dir);

% Get project root (assuming script is in src/scripts)
project_root = fileparts(fileparts(curr_script_dir));
fprintf('Project root directory: %s\n', project_root);

% Define paths to check
temp_analysis_script = fullfile(curr_script_dir, 'temp_analysis.m');
temp_globals_file = fullfile(curr_script_dir, 'temp_globals.mat');

% Check the temp_analysis.m file
fprintf('\nChecking temp_analysis.m file...\n');
if exist(temp_analysis_script, 'file')
    file_info = dir(temp_analysis_script);
    fprintf('  File exists: %s\n', temp_analysis_script);
    fprintf('  Size: %d bytes\n', file_info.bytes);
    fprintf('  Last modified: %s\n', datetime(file_info.datenum, 'ConvertFrom', 'datenum'));
    
    % Check file content
    fprintf('  Checking file content...\n');
    try
        fid = fopen(temp_analysis_script, 'r');
        if fid ~= -1
            content = fread(fid, '*char')';
            fclose(fid);
            
            % Check for key components
            has_shapdir = contains(content, 'shapDir =');
            has_modelfilesdir = contains(content, 'modelFilesDir =');
            has_try_catch = contains(content, 'try') && contains(content, 'catch');
            has_calc_shap = contains(content, 'calc_shap_values');
            
            fprintf('  File contains key components:\n');
            fprintf('    - shapDir declaration: %s\n', yesno(has_shapdir));
            fprintf('    - modelFilesDir declaration: %s\n', yesno(has_modelfilesdir));
            fprintf('    - try-catch block: %s\n', yesno(has_try_catch));
            fprintf('    - calc_shap_values call: %s\n', yesno(has_calc_shap));
            
            % Check for potentially problematic hardcoded paths
            fprintf('  Checking for hardcoded paths...\n');
            if contains(content, 'GPM_SHAP_FinalVersion')
                fprintf('    WARNING: Found reference to GPM_SHAP_FinalVersion!\n');
                
                % Find the specific lines
                lines = strsplit(content, '\n');
                for j = 1:length(lines)
                    if contains(lines{j}, 'GPM_SHAP_FinalVersion')
                        fprintf('      Line: %s\n', strtrim(lines{j}));
                    end
                end
            else
                fprintf('    No references to GPM_SHAP_FinalVersion found.\n');
            end
        else
            fprintf('  Could not open file for reading\n');
        end
    catch ME
        fprintf('  Error reading file: %s\n', ME.message);
    end
else
    fprintf('  File does not exist: %s\n', temp_analysis_script);
end

% Check the temp_globals.mat file
fprintf('\nChecking temp_globals.mat file...\n');
if exist(temp_globals_file, 'file')
    file_info = dir(temp_globals_file);
    fprintf('  File exists: %s\n', temp_globals_file);
    fprintf('  Size: %d bytes\n', file_info.bytes);
    fprintf('  Last modified: %s\n', datetime(file_info.datenum, 'ConvertFrom', 'datenum'));
    
    % Check file content
    fprintf('  Loading file to check variables...\n');
    try
        data = load(temp_globals_file);
        vars = fieldnames(data);
        fprintf('  File contains variables: %s\n', strjoin(vars, ', '));
        
        % Check for PS and TS
        has_ps = isfield(data, 'PS');
        has_ts = isfield(data, 'TS');
        has_ps_global = isfield(data, 'PS_global');
        has_ts_global = isfield(data, 'TS_global');
        
        fprintf('  Contains preprocessing variables:\n');
        fprintf('    - PS: %s\n', yesno(has_ps));
        fprintf('    - TS: %s\n', yesno(has_ts));
        fprintf('    - PS_global: %s\n', yesno(has_ps_global));
        fprintf('    - TS_global: %s\n', yesno(has_ts_global));
    catch ME
        fprintf('  Error loading file: %s\n', ME.message);
    end
else
    fprintf('  File does not exist: %s\n', temp_globals_file);
end

% Try running the calc_shap_values script
fprintf('\nChecking calc_shap_values.m...\n');
calc_shap_path = fullfile(curr_script_dir, 'calc_shap_values.m');
if exist(calc_shap_path, 'file')
    fprintf('  File exists: %s\n', calc_shap_path);
    
    % Check file content for any issues
    try
        fid = fopen(calc_shap_path, 'r');
        if fid ~= -1
            content = fread(fid, '*char')';
            fclose(fid);
            
            % Check for hardcoded paths
            if contains(content, 'GPM_SHAP_FinalVersion')
                fprintf('  WARNING: Found reference to GPM_SHAP_FinalVersion!\n');
                
                % Find the specific lines
                lines = strsplit(content, '\n');
                for j = 1:length(lines)
                    if contains(lines{j}, 'GPM_SHAP_FinalVersion')
                        fprintf('    Line: %s\n', strtrim(lines{j}));
                    end
                end
            else
                fprintf('  No references to GPM_SHAP_FinalVersion found.\n');
            end
        else
            fprintf('  Could not open file for reading\n');
        end
    catch ME
        fprintf('  Error reading file: %s\n', ME.message);
    end
else
    fprintf('  File does not exist: %s\n', calc_shap_path);
end

% Check if the results directory exists
resultsDir = fullfile(project_root, 'results');
debugDir = fullfile(resultsDir, 'debug');
analysisDir = fullfile(resultsDir, 'analysis', 'debug');

fprintf('\nChecking results directories...\n');
fprintf('  Results directory: %s (exists: %d)\n', resultsDir, exist(resultsDir, 'dir'));
fprintf('  Debug directory: %s (exists: %d)\n', debugDir, exist(debugDir, 'dir'));
fprintf('  Analysis directory: %s (exists: %d)\n', analysisDir, exist(analysisDir, 'dir'));

fprintf('\n=== Temporary Files Check Complete ===\n');
fprintf('Check the log file for detailed results: %s\n', fullfile(pwd, log_file));

% Close the log file
diary off;

% Helper function to convert logical to Yes/No
function str = yesno(value)
    if value
        str = 'Yes';
    else
        str = 'No';
    end
end 