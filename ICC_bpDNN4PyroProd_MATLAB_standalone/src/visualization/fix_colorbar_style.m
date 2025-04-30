%% Fix Colorbar Style Script
% This script creates a colorbar legend file with a standardized style
% to ensure consistent appearance across all SHAP result plots

% Check if shapDir exists in the workspace
if ~exist('shapDir', 'var')
    % Get script directory
    scriptPath = mfilename('fullpath');
    [scriptDir, ~, ~] = fileparts(scriptPath);

    % Get project root directory
    rootDir = fileparts(fileparts(scriptDir));
    resultsDir = fullfile(rootDir, 'results');

    % Define SHAP directories to check
    shapDirs = {
        fullfile(resultsDir, 'analysis', 'full'),
        fullfile(resultsDir, 'analysis', 'debug')
    };

    % Process each SHAP directory
    for i = 1:length(shapDirs)
        shapDir = shapDirs{i};
        
        if exist(shapDir, 'dir')
            fprintf('Processing SHAP directory: %s\n', shapDir);
            
            % Get the figures directory
            figDir = fullfile(shapDir, 'figures');
            
            % Create figures subfolder if it doesn't exist
            if ~exist(figDir, 'dir')
                mkdir(figDir);
                fprintf('Created figures directory: %s\n', figDir);
            end
            
            % Create a standardized colorbar legend to match shap_summary.png
            fprintf('Creating standardized colorbar legend...\n');
            createStandardColorbarLegend(figDir);
        end
    end
else
    % Use the shapDir provided in the workspace
    fprintf('Using provided shapDir: %s\n', shapDir);
    
    % Get the figures directory
    figDir = fullfile(shapDir, 'figures');
    
    % Create figures subfolder if it doesn't exist
    if ~exist(figDir, 'dir')
        mkdir(figDir);
        fprintf('Created figures directory: %s\n', figDir);
    end
    
    % Create a standardized colorbar legend to match shap_summary.png
    fprintf('Creating standardized colorbar legend...\n');
    createStandardColorbarLegend(figDir);
end

fprintf('Done. Colorbar style updated in all SHAP result directories.\n');

%% Function to create standardized colorbar legend
function createStandardColorbarLegend(figDir)
    % Create a simple figure just for the colorbar legend
    figure('Position', [100, 100, 800, 200]);
    
    % Create a gradient image
    x = linspace(0, 1, 256);
    y = ones(size(x));
    [X, Y] = meshgrid(x, y);
    
    % Plot the gradient image
    imagesc(X);
    colormap(jet(256)); % Standard jet colormap (blue=low, red=high)
    
    % Remove axes but keep colorbar
    axis off;
    
    % Add colorbar with explanatory labels - match style in original SHAP plots
    cb = colorbar('Location', 'southoutside');
    cb.Label.String = 'Feature Value';
    cb.Label.FontSize = 14;
    cb.Label.FontWeight = 'normal';
    
    % Add explicit labels at min, mid, and max points
    cb.Ticks = [0, 0.5, 1];
    cb.TickLabels = {'Low', 'Medium', 'High'};
    
    % Add title explaining the color scheme
    title('Feature Value Color Scale', 'FontSize', 16);
    
    % Save the legend - avoid duplicate saves to prevent potential errors
    saveas(gcf, fullfile(figDir, 'color_scale_legend.png'));
    saveas(gcf, fullfile(figDir, 'color_scale_legend.fig'));

    % Save the legend with high resolution (600 dpi)
    print(gcf, fullfile(figDir, 'color_scale_legend.png'), '-dpng', '-r600');
    fprintf('Created standardized colorbar legend in: %s\n', figDir);
    
    % Close the figure
    close(gcf);
end 