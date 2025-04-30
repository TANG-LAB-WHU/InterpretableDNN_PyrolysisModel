% filepath: d:\Users\TANG\Documents\GitHub\AGI_PyrolysisModel\ICC_bpDNN4PyroProd_MATLAB_standalone\src\scripts\workflow_config.m
% Global configuration for workflow transitions
function config = workflow_config()
    config.rootDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    config.resultsDir = fullfile(config.rootDir, 'results');
    config.dataDir = fullfile(config.rootDir, 'data');
    
    % File paths for transitions
    config.paths.optimization = fullfile(config.resultsDir, 'optimization', 'best_model.mat');
    config.paths.bestModel = fullfile(config.resultsDir, 'best_model', 'best_model.mat');
    config.paths.standardModel = fullfile(config.resultsDir, 'training', 'trained_model.mat');
    
    % Analysis configuration
    config.analysis.mode = 'full';
    
    % Workflow logging
    config.jobId = getenv('SLURM_JOB_ID');
    if isempty(config.jobId)
        config.jobId = datestr(now, 'yyyymmdd_HHMMSS');
    end
    
    % Add version info
    config.version = '3.0.0';
end