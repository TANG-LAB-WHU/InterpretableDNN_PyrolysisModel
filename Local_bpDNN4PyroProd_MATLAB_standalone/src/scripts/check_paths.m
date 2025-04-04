% Simple script to check for hardcoded paths
fid = fopen('path_check_results.txt', 'w');
if fid == -1
    error('Could not create output file');
end

fprintf(fid, 'Path Check Results\n');
fprintf(fid, '=================\n\n');

% Get current script directory and project root
script_dir = fileparts(mfilename('fullpath'));
fprintf(fid, 'Script directory: %s\n', script_dir);

project_root = fileparts(fileparts(script_dir));
fprintf(fid, 'Project root: %s\n\n', project_root);

% Check only specific files for GPM_SHAP_FinalVersion
target_files = {
    fullfile(script_dir, 'debug_run_analysis.m'),
    fullfile(script_dir, 'run_analysis.m'),
    fullfile(script_dir, 'temp_analysis.m')
};

for i = 1:length(target_files)
    file_path = target_files{i};
    fprintf(fid, 'Checking file: %s\n', file_path);
    
    if exist(file_path, 'file')
        fprintf(fid, '  File exists\n');
        
        % Open and read the file
        try
            file_id = fopen(file_path, 'r');
            if file_id ~= -1
                content = fread(file_id, '*char')';
                fclose(file_id);
                
                % Check for the hardcoded path
                if contains(content, 'GPM_SHAP_FinalVersion')
                    fprintf(fid, '  FOUND hardcoded path reference\n');
                    
                    % Find the specific lines
                    lines = strsplit(content, '\n');
                    for j = 1:length(lines)
                        if contains(lines{j}, 'GPM_SHAP_FinalVersion')
                            fprintf(fid, '    Line %d: %s\n', j, strtrim(lines{j}));
                        end
                    end
                else
                    fprintf(fid, '  No hardcoded path reference found\n');
                end
            else
                fprintf(fid, '  Could not open file\n');
            end
        catch ME
            fprintf(fid, '  Error reading file: %s\n', ME.message);
        end
    else
        fprintf(fid, '  File does not exist\n');
    end
    
    fprintf(fid, '\n');
end

% Check if temporary analysis script was created
temp_script = fullfile(script_dir, 'temp_analysis.m');
fprintf(fid, 'Checking if temporary analysis script exists: %s\n', temp_script);
if exist(temp_script, 'file')
    fprintf(fid, '  File exists with size %d bytes\n', dir(temp_script).bytes);
else
    fprintf(fid, '  File does not exist\n');
end

fclose(fid);
fprintf('Check complete, results written to path_check_results.txt\n'); 