%% Regenerate plots with improved formatting
% This script loads existing results and regenerates plots with better formatting
% Run this script to update plots without needing to rerun the entire optimization

clear;
clc;
fprintf('Regenerating plots with improved formatting...\n');

% Check if Results_kinetics.mat exists
if exist(fullfile('2_GA4Tsolution_PARA', 'Results_kinetics.mat'), 'file')
    % Load optimization results
    load(fullfile('2_GA4Tsolution_PARA', 'Results_kinetics.mat'));
    fprintf('Loaded optimization results successfully.\n');
else
    error('Cannot find Results_kinetics.mat. Please ensure optimization has been run successfully.');
end

% Create results directory for plots if it doesn't exist
if ~exist('model_evaluation', 'dir')
    mkdir('model_evaluation');
    fprintf('Created model_evaluation directory.\n');
end

% Calculate weight values using TG data
w_inf = alphaTG_exp(end, 3)/100;
w_t = (1 - alpha .* (1 - w_inf)) .* 100;

% Initialize arrays to store fit statistics
rmse_values = zeros(size(G_alpha, 2), 1);
r2_values = zeros(size(G_alpha, 2), 1);

%% Plot 1: T ~ alpha with improved formatting
fprintf('Generating improved TsolAlpha_comparison plot...\n');

figure('Position', [100, 100, 1200, 800], 'Color', 'white');
for j = 1 : size(G_alpha, 2)
    subplot(13, 10, j);
    
    % Plot model data with blue circles and line
    plot(alpha', Tsol(:, j), 'bo-', 'MarkerSize', 4, 'LineWidth', 1);
    hold on;
    
    % Plot experimental data with red line
    plot(alphaTG_exp(:, 2), alphaTG_exp(:, 1)+273, 'r-', 'LineWidth', 1.2);
    
    % Set title with consistent formatting
    title(['G(\alpha) ' num2str(j)], 'FontSize', 8, 'FontWeight', 'bold');
    
    % Set consistent axis limits
    xlim([0 1]);
    ylim([0 1000]); % Temperature in Kelvin for better range
    
    % Set axis ticks
    set(gca, 'XTick', [0 0.5 1]);
    set(gca, 'YTick', [0 500 1000]);
    
    % Set font size for tick labels
    set(gca, 'FontSize', 7);
    
    % Show labels only on leftmost and bottom plots
    if mod(j-1, 10) == 0 % First column
        ylabel('Temperature (K)', 'FontSize', 8);
    else
        set(gca, 'YTickLabel', []);
    end
    
    if j > (13-1)*10 % Last row
        xlabel('Conversion (\alpha)', 'FontSize', 8);
    else
        set(gca, 'XTickLabel', []);
    end
    
    % Add grid for better readability
    grid on;
    set(gca, 'GridLineStyle', ':');
    set(gca, 'GridAlpha', 0.3);
end

% Adjust spacing between subplots
set(gcf, 'PaperPositionMode', 'auto');
try
    if exist('tightfig', 'file')
        tightfig(gcf);
    else
        fprintf('Note: tightfig function not found. Subplot spacing might not be optimal.\n');
    end
catch err
    fprintf('Warning: Error adjusting subplot spacing: %s\n', err.message);
end

% Save with higher resolution (600dpi) for better visibility and add EPS format
saveas(gcf, fullfile('model_evaluation', 'TsolAlpha_comparison.fig'));
print(gcf, fullfile('model_evaluation', 'TsolAlpha_comparison.png'), '-dpng', '-r600');
print(gcf, fullfile('model_evaluation', 'TsolAlpha_comparison.eps'), '-depsc2', '-r600');

%% Plot 2: TG curve comparison with improved formatting
fprintf('Generating improved TG_simulated_comparison plot...\n');

figure('Position', [100, 100, 1200, 800], 'Color', 'white');
for m = 1 : size(G_alpha, 2)
    subplot(13, 10, m);
    
    % Plot simulated data
    plot(Tsol(:, m)-273, w_t', 'bo-', 'MarkerSize', 4, 'LineWidth', 1);
    hold on;
    
    % Plot experimental data
    plot(alphaTG_exp(:, 1), alphaTG_exp(:, 3), 'r-', 'LineWidth', 1.2);
    
    % Set title formatting
    title(['G(\alpha) ' num2str(m)], 'FontSize', 8, 'FontWeight', 'bold');
    
    % Calculate fit metrics
    [rmse, r2] = evaluateModelFit(w_t', alphaTG_exp(:, 3));
    rmse_values(m) = rmse;
    r2_values(m) = r2;
    
    % Set axis limits
    xlim([min(alphaTG_exp(:, 1))-10, max(alphaTG_exp(:, 1))+10]);
    ylim([min(alphaTG_exp(:, 3))-5, max(alphaTG_exp(:, 3))+5]);
    
    % Set font size
    set(gca, 'FontSize', 7);
    
    % Show labels only on leftmost and bottom plots
    if mod(m-1, 10) == 0 % First column
        ylabel('Weight (%)', 'FontSize', 8);
    else
        set(gca, 'YTickLabel', []);
    end
    
    if m > (13-1)*10 % Last row
        xlabel('Temperature (°C)', 'FontSize', 8);
    else
        set(gca, 'XTickLabel', []);
    end
    
    % Add grid
    grid on;
    set(gca, 'GridLineStyle', ':');
    set(gca, 'GridAlpha', 0.3);
end

% Adjust spacing
set(gcf, 'PaperPositionMode', 'auto');
try
    if exist('tightfig', 'file')
        tightfig(gcf);
    else
        fprintf('Note: tightfig function not found. Subplot spacing might not be optimal.\n');
    end
catch err
    fprintf('Warning: Error adjusting subplot spacing: %s\n', err.message);
end

% Save with higher resolution (600dpi) for better visibility and add EPS format
saveas(gcf, fullfile('model_evaluation', 'TG_simulated_comparison.fig'));
print(gcf, fullfile('model_evaluation', 'TG_simulated_comparison.png'), '-dpng', '-r600');
print(gcf, fullfile('model_evaluation', 'TG_simulated_comparison.eps'), '-depsc2', '-r600');

%% Plot 3: Best fitting model with improved appearance
fprintf('Finding best fitting model and generating plot...\n');

% Find the best fitting model based on RMSE
[best_rmse, best_idx] = min(rmse_values);
best_r2 = r2_values(best_idx);

% Plot the best model with improved styling
figure('Position', [100, 100, 600, 500], 'Color', 'white');

% Plot data
plot(Tsol(:, best_idx)-273, w_t', 'b-', 'LineWidth', 2.5, 'Color', [0 0.4470 0.7410]);
hold on;
plot(alphaTG_exp(:, 1), alphaTG_exp(:, 3), 'r--', 'LineWidth', 2, 'Color', [0.8500 0.3250 0.0980]);

% Enhance axis formatting
set(gca, 'LineWidth', 1.2, 'FontSize', 12, 'Box', 'on');
set(gca, 'XGrid', 'on', 'YGrid', 'on', 'GridLineStyle', '--', 'GridAlpha', 0.2);

% Add labels and title
xlabel('Temperature (°C)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Weight (%)', 'FontSize', 14, 'FontWeight', 'bold');
title(sprintf('Best Fitting Model: G(\\alpha) %d\nRMSE = %.2f, R^2 = %.4f', best_idx, best_rmse, best_r2), 'FontSize', 16, 'FontWeight', 'bold');

% Add legend
legend('Simulated', 'Experimental', 'Location', 'best', 'FontSize', 12, 'EdgeColor', [0.8 0.8 0.8]);

% Set axis limits
xlim([min(alphaTG_exp(:, 1))-10, max(alphaTG_exp(:, 1))+10]);
ylim([min(alphaTG_exp(:, 3))-5, max(alphaTG_exp(:, 3))+5]);

% Add text box with model info
annotation('textbox', [0.15, 0.15, 0.3, 0.1], 'String', ...
           {sprintf('Model: G(\\alpha) %d', best_idx), ...
            sprintf('RMSE: %.2f', best_rmse), ...
            sprintf('R^2: %.4f', best_r2)}, ...
           'EdgeColor', [0.7 0.7 0.7], ...
           'BackgroundColor', [0.97 0.97 0.97], ...
           'FaceAlpha', 0.8, ...
           'FontSize', 11);

% Save with higher resolution (600dpi) for publication quality and add EPS format
saveas(gcf, fullfile('model_evaluation', 'Best_model_fit.fig'));
print(gcf, fullfile('model_evaluation', 'Best_model_fit.png'), '-dpng', '-r600');
print(gcf, fullfile('model_evaluation', 'Best_model_fit.eps'), '-depsc2', '-r600');

% Save model evaluation results
model_stats = table((1:size(G_alpha, 2))', rmse_values, r2_values, ...
                   'VariableNames', {'ModelIndex', 'RMSE', 'R2'});
save(fullfile('model_evaluation', 'model_evaluation_results.mat'), ...
    'model_stats', 'best_idx', 'best_rmse', 'best_r2');

% Display results summary
fprintf('\n===== Model Evaluation Results =====\n');
fprintf('Best fitting model: G(alpha) %d\n', best_idx);
fprintf('RMSE: %.4f\n', best_rmse);
fprintf('R-squared: %.4f\n', best_r2);
fprintf('===================================\n');
fprintf('All plots have been regenerated with improved formatting.\n');
fprintf('Results saved to model_evaluation directory.\n\n');
