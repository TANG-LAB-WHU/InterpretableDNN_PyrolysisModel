function config = performanceConfig()
% Performance configuration for kinetic model optimization
% This file centralizes all performance-related parameters

config = struct();

%% Multi-mechanism configuration
config.multi = struct();
config.multi.enabled = true;          % Enable multi-mechanism mode
config.multi.modes = {'series', 'parallel', 'hybrid'};
config.multi.maxBranches = 4;         % Support maximal 4 branches for hybrid mode
config.multi.maxMechsPerBranch = 3;   % Support maximal 3 mechanisms per branch
config.multi.maxMechs = 6;            % Support maximal 6 mechanisms for series and parallel modes

%% Integration settings
config.integration = struct();
config.integration.nSteps = 500;
config.integration.tolerance = 1e-6;

%% Error calculation settings
config.error = struct();
config.error.temperatureWeighting = true;
config.error.shapePenalty = true;
config.error.rangePenalty = true;
config.error.endpointPenalty = true;
config.error.useAdaptiveWeights = true;  % Use DTG-driven adaptive weights

% Optional alignment penalties
config.error.enableT50Penalty = true;
config.error.t50Weight        = 0.15;    % per °C
config.error.enableDTGPenalty = true;
config.error.dtgWeight        = 100.0;   % High weight for rate matching

%% Logging settings
config.logging = struct();
config.logging.detailed = true;          % Enable detailed logging for multi-mode
config.logging.progressInterval = 2;     % Print progress every N iterations

%% Optional fast mode toggle and GA presets (used if present)
% These fields are optional; other modules will fall back to sensible defaults
config.fastMode = false;   % quick test run; kineticIntegration remains enabled

config.singleMode = false;      % Only single-mechanism models
config.multiOnlyMode = true;     % Only multi-mechanism models
% Mode logic:
%   singleMode=true,  multiOnlyMode=false  -> Single only
%   singleMode=false, multiOnlyMode=true   -> Multi only
%   singleMode=false, multiOnlyMode=false  -> Both

% Single-model GA presets (used by getGAOptions if present)
config.gaSingle = struct();
if config.fastMode
  config.gaSingle.populationSize   = 20;
  config.gaSingle.maxGenerations   = 8;
else
  config.gaSingle.populationSize   = 50;
  config.gaSingle.maxGenerations   = 20;
end
config.gaSingle.eliteCount         = 4;
config.gaSingle.crossoverFraction  = 0.8;
config.gaSingle.functionTolerance  = 1e-6;

%% Two-stage GA optimization settings (Coarse-to-Fine)
config.twoStageGA = struct();
config.twoStageGA.enabled = false;          % DISABLED for higher reliability in production
config.twoStageGA.topN = 3;                 % Number of top single-mechanism candidates for Stage 2

% Single-mechanism Stage 1: Coarse/Fast GA for screening
config.twoStageGA.coarse = struct();
config.twoStageGA.coarse.populationSize = 192;
config.twoStageGA.coarse.maxGenerations = 5;
config.twoStageGA.coarse.eliteCount = 10;
config.twoStageGA.coarse.crossoverFraction = 0.8;
config.twoStageGA.coarse.functionTolerance = 1e-4;

% Single-mechanism Stage 2: Fine/Precise GA for top candidates
config.twoStageGA.fine = struct();
config.twoStageGA.fine.populationSize = 192;
config.twoStageGA.fine.maxGenerations = 25;
config.twoStageGA.fine.eliteCount = 20;
config.twoStageGA.fine.crossoverFraction = 0.85;
config.twoStageGA.fine.functionTolerance = 1e-6;

% Multi-mechanism two-stage settings
config.twoStageGA.multiTopN = 5;            % Top multi-mechanism configs for Stage 2

config.twoStageGA.multiCoarse = struct();
config.twoStageGA.multiCoarse.populationSize = 192;
config.twoStageGA.multiCoarse.maxGenerations = 8;
config.twoStageGA.multiCoarse.eliteCount = 10;
config.twoStageGA.multiCoarse.crossoverFraction = 0.8;
config.twoStageGA.multiCoarse.functionTolerance = 1e-4;

config.twoStageGA.multiFine = struct();
config.twoStageGA.multiFine.populationSize = 192;
config.twoStageGA.multiFine.maxGenerations = 30;
config.twoStageGA.multiFine.eliteCount = 20;
config.twoStageGA.multiFine.crossoverFraction = 0.85;
config.twoStageGA.multiFine.functionTolerance = 1e-6;

% Multi-model GA presets (used by getGAOptions if present)
config.gaMulti = struct();
if config.fastMode
  config.gaMulti.populationSize    = 60;
  config.gaMulti.maxGenerations    = 15;
else
  % HIGH RESOLUTION PRODUCTION: Optimized for 192-core scaling
  config.gaMulti.populationSize    = 192;
  config.gaMulti.maxGenerations    = 50;
end
config.gaMulti.eliteCount          = 20;
config.gaMulti.crossoverFraction   = 0.85;
config.gaMulti.functionTolerance   = 1e-7;

% Error calculation fast path switch
% If false, modules will use the extremely fast and exact interpolation over legacy slow kinetic reintegration
config.error.useKineticIntegration = false;

end