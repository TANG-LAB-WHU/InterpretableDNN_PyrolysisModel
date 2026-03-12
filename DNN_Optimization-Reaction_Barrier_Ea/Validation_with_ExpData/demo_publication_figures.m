%% Publication Quality Figure Example
% This script demonstrates how to create and save publication-quality figures
% using the saveFigureForPublication function.
%
% Created by: GitHub Copilot
% Date: May 22, 2025
% For: Generalizable Pyrolysis Model Project

% Clear workspace and figures
clear;
clc;
close all;

% Create a sample figure
figure('Position', [100, 100, 800, 600]);

% Create some sample data
x = linspace(0, 10, 100);
y1 = sin(x);
y2 = cos(x);

% Plot with proper formatting
plot(x, y1, 'b-', 'LineWidth', 2);
hold on;
plot(x, y2, 'r--', 'LineWidth', 2);

% Add labels and title with proper font sizes
xlabel('X-axis', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Y-axis', 'FontSize', 12, 'FontWeight', 'bold');
title('Sample Publication Quality Figure', 'FontSize', 14, 'FontWeight', 'bold');

% Add legend
legend('sin(x)', 'cos(x)', 'Location', 'best', 'FontSize', 10);

% Enhance appearance
grid on;
set(gca, 'LineWidth', 1.2, 'FontSize', 10);
set(gca, 'Box', 'on');

% Use the saveFigureForPublication function to save the figure
saveFigureForPublication(gcf, 'figure_examples', 'sample_figure', 600);

fprintf('\nThis example shows how to create a publication-quality figure and save it in multiple formats.\n');
fprintf('The figure has been saved in the "figure_examples" directory.\n\n');
fprintf('To use this functionality in your own scripts, simply call:\n');
fprintf('  saveFigureForPublication(figureHandle, outputDirectory, baseFileName, dpi);\n\n');
fprintf('For example:\n');
fprintf('  saveFigureForPublication(gcf, ''model_evaluation'', ''MyPlot'', 600);\n\n');
