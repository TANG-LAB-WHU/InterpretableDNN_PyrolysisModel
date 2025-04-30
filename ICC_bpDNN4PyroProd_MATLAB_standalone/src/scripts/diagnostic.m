<<<<<<< HEAD
function diagnostic()
    % This function diagnoses common issues with the model execution
    
    diary('matlab_diagnostic.log');
    fprintf('==== MATLAB Diagnostic ====\n');

    % Check current directory
    rootDir = pwd;
    fprintf('Root directory: %s\n', rootDir);
    
    % Check directory structure
    fprintf('\n=== Directory Structure ===\n');
    checkDir = @(dir) fprintf('Directory %s exists: %d\n', dir, exist(dir, 'dir'));
    
    checkDir(fullfile(rootDir, 'src', 'model'));
    checkDir(fullfile(rootDir, 'src', 'scripts'));
    checkDir(fullfile(rootDir, 'src', 'visualization'));
    checkDir(fullfile(rootDir, 'src', 'shap'));
    checkDir(fullfile(rootDir, 'results', 'analysis', 'full', 'data'));
    checkDir(fullfile(rootDir, 'results', 'analysis', 'full', 'figures'));
    
    % Check key files
    fprintf('\n=== Key Files ===\n');
    requiredFiles = {
        'src/scripts/run_analysis.m',
        'src/model/bpDNN4PyroProd.m',
        'data/processed/CopyrolysisFeedstock.mat',
        'data/raw/RawInputData.xlsx'
    };

    for i = 1:length(requiredFiles)
        filePath = fullfile(rootDir, requiredFiles{i});
        fprintf('File %s exists: %d\n', requiredFiles{i}, exist(filePath, 'file'));
    end
    
    % Check MATLAB paths and path conflicts
    fprintf('\n=== Path Analysis ===\n');
    pathCells = strsplit(path, pathsep);
    
    % Check for path conflicts (same function name in multiple directories)
    fprintf('Checking for path conflicts...\n');
    checkFiles = {'run_analysis.m', 'calc_shap_values.m', 'plot_shap_results.m'};
    
    for i = 1:length(checkFiles)
        fprintf('Checking for %s:\n', checkFiles{i});
        found = false;
        for j = 1:length(pathCells)
            testPath = fullfile(pathCells{j}, checkFiles{i});
            if exist(testPath, 'file')
                fprintf('  Found at: %s\n', testPath);
                found = true;
            end
        end
        if ~found
            fprintf('  Not found in path\n');
        end
    end
    
    % Check workspace memory - with platform compatibility check
    fprintf('\n=== Memory Analysis ===\n');
    try
        [userview, systemview] = memory;
        fprintf('Total available memory: %.2f GB\n', systemview.PhysicalMemory.Available/1e9);
        fprintf('MATLAB memory usage: %.2f GB\n', userview.MemUsedMATLAB/1e9);
    catch ME
        % Handle case where memory function is not available (like on ICC cluster)
        fprintf('Memory information not available on this platform.\n');
        fprintf('Platform: %s, MATLAB Version: %s\n', computer, version);
        
        % Try alternative system commands if on Linux
        if isunix
            try
                [status, cmdout] = system('free -g');
                if status == 0
                    fprintf('System memory information (from Linux free command):\n%s\n', cmdout);
                end
            catch
                % If even this fails, just skip memory diagnostics
                fprintf('Could not retrieve memory information using alternative methods.\n');
            end
        end
    end
    
    % Check script compatibility
    fprintf('\n=== Script Compatibility ===\n');
    % Check for deprecated functions
    fprintf('Looking for deprecated function calls...\n');
    
    % Add paths to ensure we can check the scripts
    addpath(genpath(fullfile(rootDir, 'src')));
    
    % Clean up and report
    fprintf('\n=== Diagnostic Complete ===\n');
    diary off;
    fprintf('Diagnostic completed. Results saved to matlab_diagnostic.log\n');
end

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
=======
function diagnostic()
    % This function diagnoses common issues with the model execution
    
    diary('matlab_diagnostic.log');
    fprintf('==== MATLAB Diagnostic ====\n');

    % Check current directory
    rootDir = pwd;
    fprintf('Root directory: %s\n', rootDir);
    
    % Check directory structure
    fprintf('\n=== Directory Structure ===\n');
    checkDir = @(dir) fprintf('Directory %s exists: %d\n', dir, exist(dir, 'dir'));
    
    checkDir(fullfile(rootDir, 'src', 'model'));
    checkDir(fullfile(rootDir, 'src', 'scripts'));
    checkDir(fullfile(rootDir, 'src', 'visualization'));
    checkDir(fullfile(rootDir, 'src', 'shap'));
    checkDir(fullfile(rootDir, 'results', 'analysis', 'full', 'data'));
    checkDir(fullfile(rootDir, 'results', 'analysis', 'full', 'figures'));
    
    % Check key files
    fprintf('\n=== Key Files ===\n');
    requiredFiles = {
        'src/scripts/run_analysis.m',
        'src/model/bpDNN4PyroProd.m',
        'data/processed/CopyrolysisFeedstock.mat',
        'data/raw/RawInputData.xlsx'
    };

    for i = 1:length(requiredFiles)
        filePath = fullfile(rootDir, requiredFiles{i});
        fprintf('File %s exists: %d\n', requiredFiles{i}, exist(filePath, 'file'));
    end
    
    % Check MATLAB paths and path conflicts
    fprintf('\n=== Path Analysis ===\n');
    pathCells = strsplit(path, pathsep);
    
    % Check for path conflicts (same function name in multiple directories)
    fprintf('Checking for path conflicts...\n');
    checkFiles = {'run_analysis.m', 'calc_shap_values.m', 'plot_shap_results.m'};
    
    for i = 1:length(checkFiles)
        fprintf('Checking for %s:\n', checkFiles{i});
        found = false;
        for j = 1:length(pathCells)
            testPath = fullfile(pathCells{j}, checkFiles{i});
            if exist(testPath, 'file')
                fprintf('  Found at: %s\n', testPath);
                found = true;
            end
        end
        if ~found
            fprintf('  Not found in path\n');
        end
    end
    
    % Check workspace memory - with platform compatibility check
    fprintf('\n=== Memory Analysis ===\n');
    try
        [userview, systemview] = memory;
        fprintf('Total available memory: %.2f GB\n', systemview.PhysicalMemory.Available/1e9);
        fprintf('MATLAB memory usage: %.2f GB\n', userview.MemUsedMATLAB/1e9);
    catch ME
        % Handle case where memory function is not available (like on ICC cluster)
        fprintf('Memory information not available on this platform.\n');
        fprintf('Platform: %s, MATLAB Version: %s\n', computer, version);
        
        % Try alternative system commands if on Linux
        if isunix
            try
                [status, cmdout] = system('free -g');
                if status == 0
                    fprintf('System memory information (from Linux free command):\n%s\n', cmdout);
                end
            catch
                % If even this fails, just skip memory diagnostics
                fprintf('Could not retrieve memory information using alternative methods.\n');
            end
        end
    end
    
    % Check script compatibility
    fprintf('\n=== Script Compatibility ===\n');
    % Check for deprecated functions
    fprintf('Looking for deprecated function calls...\n');
    
    % Add paths to ensure we can check the scripts
    addpath(genpath(fullfile(rootDir, 'src')));
    
    % Clean up and report
    fprintf('\n=== Diagnostic Complete ===\n');
    diary off;
    fprintf('Diagnostic completed. Results saved to matlab_diagnostic.log\n');
end

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
>>>>>>> 5532f15296ed6babaee64d74dd966e573480e3cb
end