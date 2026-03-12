function saveFigureForPublication(figHandle, outputPath, baseName, dpi)
% SAVEFIGUREFORPUBLICATION Saves figures in multiple formats suitable for publication
%   This function takes a figure handle and saves it in multiple formats:
%   - .fig (MATLAB figure)
%   - .png (Bitmap image) 
%   - .eps (Encapsulated PostScript for publication)
%
%   Parameters:
%       figHandle - Handle to the figure to be saved
%       outputPath - Directory where the figure should be saved
%       baseName - Base name for the figure files (without extension)
%       dpi - Resolution in dots per inch (default: 600)
%
%   Example:
%       fig = figure;
%       plot(1:10);
%       saveFigureForPublication(fig, 'figures', 'myPlot', 600);
%
%   This will save:
%       - figures/myPlot.fig
%       - figures/myPlot.png (600 dpi)
%       - figures/myPlot.eps (600 dpi)
%
%   Created by: GitHub Copilot
%   Date: May 22, 2025
%   For: Generalizable Pyrolysis Model Project

% Set default DPI if not specified
if nargin < 4
    dpi = 600;
end

% Create the output directory if it doesn't exist
if ~exist(outputPath, 'dir')
    mkdir(outputPath);
    fprintf('Created directory: %s\n', outputPath);
end

% Construct full path with filename
fullPathBase = fullfile(outputPath, baseName);

% Format the figure for publication (optional additional formatting)
set(figHandle, 'Color', 'white'); % White background
set(figHandle, 'PaperPositionMode', 'auto');

% Save in MATLAB .fig format for future editing
savefig(figHandle, [fullPathBase '.fig']);

% Save as PNG with specified DPI
print(figHandle, [fullPathBase '.png'], '-dpng', ['-r' num2str(dpi)]);

% Save as EPS for publication quality
print(figHandle, [fullPathBase '.eps'], '-depsc2', ['-r' num2str(dpi)]);

fprintf('Figure saved as:\n');
fprintf('  %s.fig\n', fullPathBase);
fprintf('  %s.png (%d dpi)\n', fullPathBase, dpi);
fprintf('  %s.eps (%d dpi)\n', fullPathBase, dpi);
end
