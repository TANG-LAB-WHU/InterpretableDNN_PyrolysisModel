function result = helperFunctions()
% Helper function library - Utils module
% Provides various general helper functions
%
% Output parameters:
%   result - Function list

result = {
  'formatNumber', 'formatNumber(value, precision)', 'Format number display';
  'validateInput', 'validateInput(input, type, range)', 'Validate input parameters';
  'safeDivision', 'safeDivision(numerator, denominator)', 'Safe division operation';
  'interpolateData', 'interpolateData(x, y, x_new)', 'Data interpolation';
  'calculateStatistics', 'calculateStatistics(data)', 'Calculate statistics';
  'checkConvergence', 'checkConvergence(iterations, tolerance)', 'Check convergence';
  'normalizeVector', 'normalizeVector(vector)', 'Vector normalization';
  'logProgress', 'logProgress(message, level)', 'Log progress';
  'timeFunction', 'timeFunction(func, varargin)', 'Function execution time measurement';
  'memoryUsage', 'memoryUsage()', 'Memory usage information';
  };

fprintf('Helper functions available:\n');
for i = 1:size(result, 1)
  fprintf('  %s: %s\n', result{i,1}, result{i,2});
end
end

function formatted = formatNumber(value, precision)
% Format number display
%
% Input parameters:
%   value - Numeric value
%   precision - Precision digits
%
% Output parameters:
%   formatted - Formatted string

if nargin < 2
  precision = 4;
end

formatted = sprintf('%.*f', precision, value);
end

function isValid = validateInput(input, type, range)
% Validate input parameters
%
% Input parameters:
%   input - Input value
%   type - Type ('numeric', 'positive', 'range')
%   range - Range [min, max]
%
% Output parameters:
%   isValid - Whether valid

isValid = true;

switch type
  case 'numeric'
    if ~isnumeric(input)
      isValid = false;
    end

  case 'positive'
    if ~isnumeric(input) || any(input <= 0)
      isValid = false;
    end

  case 'range'
    if ~isnumeric(input) || any(input < range(1)) || any(input > range(2))
      isValid = false;
    end

  otherwise
    error('Unknown validation type: %s', type);
end
end

function result = safeDivision(numerator, denominator)
% Safe division operation
%
% Input parameters:
%   numerator - Numerator
%   denominator - Denominator
%
% Output parameters:
%   result - Division result

if denominator == 0
  result = Inf;
  warning('Division by zero detected');
else
  result = numerator / denominator;
end
end

function y_interp = interpolateData(x, y, x_new)
% Data interpolation
%
% Input parameters:
%   x - Original x coordinates
%   y - Original y values
%   x_new - New x coordinates for interpolation
%
% Output parameters:
%   y_interp - Interpolated y values

if length(x) ~= length(y)
  error('Input vectors x and y must have the same length');
end

y_interp = interp1(x, y, x_new, 'pchip', 'extrap');
end

function stats = calculateStatistics(data)
% Calculate statistics
%
% Input parameters:
%   data - Data vector
%
% Output parameters:
%   stats - Statistics structure

stats = struct();
stats.mean = mean(data);
stats.std = std(data);
stats.min = min(data);
stats.max = max(data);
stats.median = median(data);
stats.count = length(data);
end

function converged = checkConvergence(iterations, tolerance)
% Check convergence
%
% Input parameters:
%   iterations - Iteration history
%   tolerance - Convergence tolerance
%
% Output parameters:
%   converged - Whether converged

if length(iterations) < 2
  converged = false;
  return;
end

recent_diff = abs(iterations(end) - iterations(end-1));
converged = recent_diff < tolerance;
end

function normalized = normalizeVector(vector)
% Vector normalization
%
% Input parameters:
%   vector - Input vector
%
% Output parameters:
%   normalized - Normalized vector

if isempty(vector)
  normalized = [];
  return;
end

min_val = min(vector);
max_val = max(vector);

if max_val == min_val
  normalized = ones(size(vector));
else
  normalized = (vector - min_val) / (max_val - min_val);
end
end

function logProgress(message, level)
% Log progress
%
% Input parameters:
%   message - Log message
%   level - Log level ('info', 'warning', 'error')

if nargin < 2
  level = 'info';
end

timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
fprintf('[%s] [%s] %s\n', timestamp, upper(level), message);
end

function executionTime = timeFunction(func, varargin)
% Function execution time measurement
%
% Input parameters:
%   func - Function handle to measure
%   varargin - Function arguments
%
% Output parameters:
%   executionTime - Execution time in seconds

tic;
func(varargin{:});
executionTime = toc;
end

function memInfo = memoryUsage()
% Memory usage information
%
% Output parameters:
%   memInfo - Memory information structure

memInfo = struct();

try
  % Get memory information
  [~, systemview] = memory;

  memInfo.total = systemview.PhysicalMemory.Total;
  memInfo.available = systemview.PhysicalMemory.Available;
  memInfo.used = memInfo.total - memInfo.available;
  memInfo.usage_percent = (memInfo.used / memInfo.total) * 100;

catch
  % Fallback if memory function not available
  memInfo.total = NaN;
  memInfo.available = NaN;
  memInfo.used = NaN;
  memInfo.usage_percent = NaN;
end
end