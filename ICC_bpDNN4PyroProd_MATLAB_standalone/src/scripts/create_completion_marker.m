function create_completion_marker(markerType, filePath, metadata)
% CREATE_COMPLETION_MARKER Creates a completion marker for workflow steps
%
% Arguments:
%   markerType - String: Type of marker ('optimization', 'training', 'analysis', 'workflow')
%   filePath - String: Path to the file associated with this marker
%   metadata - Struct: Additional information to store in the marker
%
% This function creates or updates marker files to indicate completion
% of workflow steps and facilitate transitions between steps.

    % Create the marker directory if it doesn't exist
    markerDir = fullfile(fileparts(filePath), 'markers');
    if ~exist(markerDir, 'dir')
        mkdir(markerDir);
    end
    
    % Create marker filename based on type
    markerFile = fullfile(markerDir, [markerType '_complete.mat']);
    
    % Create marker data structure
    markerData = struct();
    markerData.type = markerType;
    markerData.sourceFile = filePath;
    markerData.completedAt = datestr(now);
    markerData.metadata = metadata;
    
    % Save the marker
    save(markerFile, 'markerData');
    
    fprintf('✓ Created completion marker for %s step at: %s\n', markerType, markerFile);
end