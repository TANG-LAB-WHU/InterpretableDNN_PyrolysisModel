% MATLAB diagnostic script
% Open a file for writing the diagnostic information
diary('diagnostic_output.txt');
fprintf('==== MATLAB Diagnostic Script ====\n');

% Get script path and determine project root
curr_script_path = mfilename('fullpath');
fprintf('Current script path: %s\n', curr_script_path);

% Get directory of current script
curr_script_dir = fileparts(curr_script_path);
fprintf('Current script directory: %s\n', curr_script_dir);

% Get project root (assuming script is in src/scripts)
project_root = fileparts(fileparts(curr_script_dir));
fprintf('Project root directory: %s\n', project_root);

% Output MATLAB version
v = version;
fprintf('MATLAB version: %s\n', v);

% Check path
p = path;
fprintf('Path contains scripts directory: %d\n', contains(p, curr_script_dir));

% Display directories and check if they exist
directories = {
    'src/scripts', 
    'src/shap', 
    'src/dnn', 
    'src/utils', 
    'src/model',
    'src/visualization',
    'results/training',
    'results/debug',
    'results/optimization',
    'debug_dir'
};

fprintf('\n==== Checking Directories ====\n');
for i = 1:length(directories)
    dir_path = fullfile(project_root, directories{i});
    exists = exist(dir_path, 'dir');
    fprintf('%s: %s (exists: %d)\n', directories{i}, dir_path, exists);
    
    % If directory exists, list files
    if exists
        files = dir(dir_path);
        if ~isempty(files)
            fprintf('  Contains files:\n');
            for j = 1:min(5, length(files))
                if ~files(j).isdir
                    fprintf('    - %s\n', files(j).name);
                end
            end
            if length(files) > 5
                fprintf('    - ... (%d more files)\n', length(files) - 5);
            end
        else
            fprintf('  Directory is empty\n');
        end
    end
end

% Check specific files
check_files = {
    fullfile(project_root, 'src', 'model', 'CopyrolysisFeedstock.mat'),
    fullfile(project_root, 'data', 'processed', 'CopyrolysisFeedstock.mat'),
    fullfile(project_root, 'results', 'training', 'Results_trained.mat'),
    fullfile(project_root, 'results', 'debug', 'Results_trained.mat')
};

fprintf('\n==== Checking Critical Files ====\n');
for i = 1:length(check_files)
    file_exists = exist(check_files{i}, 'file');
    if file_exists
        fprintf('%s: Exists\n', check_files{i});
    else
        fprintf('%s: Not found\n', check_files{i});
    end
end

% Check data
fprintf('\n==== Checking Data Files ====\n');
% First try processed directory
dataDir = fullfile(project_root, 'data', 'processed');
feedstock_file = fullfile(dataDir, 'CopyrolysisFeedstock.mat');

% If not there, check model directory
if ~exist(feedstock_file, 'file')
    feedstock_file = fullfile(project_root, 'src', 'model', 'CopyrolysisFeedstock.mat');
end

if exist(feedstock_file, 'file')
    fprintf('Found feedstock file: %s\n', feedstock_file);
    try
        data = load(feedstock_file);
        fields = fieldnames(data);
        fprintf('  Loaded successfully. Contains fields:\n');
        for i = 1:min(5, length(fields))
            fprintf('    - %s\n', fields{i});
        end
        if length(fields) > 5
            fprintf('    - ... (%d more fields)\n', length(fields) - 5);
        end
    catch ME
        fprintf('  Error loading file: %s\n', ME.message);
    end
else
    fprintf('Feedstock file not found in data/processed or src/model\n');
end

% Check for training results
fprintf('\n==== Checking Training Results ====\n');
results_file = fullfile(project_root, 'results', 'training', 'Results_trained.mat');

if exist(results_file, 'file')
    fprintf('Found results file: %s\n', results_file);
    try
        data = load(results_file);
        fields = fieldnames(data);
        fprintf('  Loaded successfully. Contains fields:\n');
        for i = 1:min(5, length(fields))
            fprintf('    - %s\n', fields{i});
        end
        if length(fields) > 5
            fprintf('    - ... (%d more fields)\n', length(fields) - 5);
        end
    catch ME
        fprintf('  Error loading file: %s\n', ME.message);
    end
else
    fprintf('Results file not found in results/training\n');
end

fprintf('\n==== Diagnostic Complete ====\n');
diary off;

% Helper function to recursively find files with a specific extension
function files = findfiles(directory, pattern)
    files = {};
    items = dir(fullfile(directory, pattern));
    for i = 1:length(items)
        if ~items(i).isdir
            files{end+1} = fullfile(directory, items(i).name);
        end
    end
    
    % Get subdirectories
    subdirs = dir(directory);
    for i = 1:length(subdirs)
        if subdirs(i).isdir && ~strcmp(subdirs(i).name, '.') && ~strcmp(subdirs(i).name, '..')
            subdir_files = findfiles(fullfile(directory, subdirs(i).name), pattern);
            files = [files, subdir_files];
        end
    end
end 