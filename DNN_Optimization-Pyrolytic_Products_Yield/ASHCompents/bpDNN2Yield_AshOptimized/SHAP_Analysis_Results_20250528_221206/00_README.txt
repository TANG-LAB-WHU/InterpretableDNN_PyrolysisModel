SHAP Analysis Results (Using MATLAB Neural Network Model)
==================================================

Analysis Date: 2025-05-28 22:15:06
Note: Analysis performed using ALL sample data without train/test splitting
Only the original MATLAB model is used (no fallback models)

File Formats:
------------
All plots are provided in two formats:
- PNG: Raster format (pixel-based) good for screen viewing and web use
- EPS: Vector format good for publication-quality graphics that maintain
       quality when resized or printed at high resolution

Directory Structure:
-------------------
00_README.txt - This file
00_analysis_log.txt - Processing log with details and any errors
00_feature_names_used.txt - List of feature names used in the analysis
00_YIELD_CONVERSION_NOTE.txt - Information about yield value conversions and file formats

01_Char/ - Analysis results for Char yield
  |- 00_model_info.txt - Information about the model used
  |- 01_beeswarm_plot.png/eps - Shows feature importance and impact direction
  |- 02_feature_importance_plot.png/eps - Bar plot of feature importance
  |- 03_instance_explanation_plot.png/eps - Force plot for a single instance
  |- 04_dependence_plot.png/eps - Dependence plot for the most important feature
  |- 05_waterfall_plot_1.png/eps to 05_waterfall_plot_5.png/eps - Five waterfall plots showing feature contributions for different instances
  |- 06_top_features_bar_plot.png/eps - Top 20 important features
  |- 07_significant_features.txt - List of significant features
  |- shap_values.npy - Saved SHAP values for future use
  |- expected_value.npy - Saved expected value for future use
  |- shap_values_data.xlsx - Excel file with SHAP values and feature importance

01_Liquid/ - Analysis results for Liquid yield
  |- 00_model_info.txt - Information about the model used
  |- 01_beeswarm_plot.png/eps - Shows feature importance and impact direction
  |- 02_feature_importance_plot.png/eps - Bar plot of feature importance
  |- 03_instance_explanation_plot.png/eps - Force plot for a single instance
  |- 04_dependence_plot.png/eps - Dependence plot for the most important feature
  |- 05_waterfall_plot_1.png/eps to 05_waterfall_plot_5.png/eps - Five waterfall plots showing feature contributions for different instances
  |- 06_top_features_bar_plot.png/eps - Top 20 important features
  |- 07_significant_features.txt - List of significant features
  |- shap_values.npy - Saved SHAP values for future use
  |- expected_value.npy - Saved expected value for future use
  |- shap_values_data.xlsx - Excel file with SHAP values and feature importance

01_Gas/ - Analysis results for Gas yield
  |- 00_model_info.txt - Information about the model used
  |- 01_beeswarm_plot.png/eps - Shows feature importance and impact direction
  |- 02_feature_importance_plot.png/eps - Bar plot of feature importance
  |- 03_instance_explanation_plot.png/eps - Force plot for a single instance
  |- 04_dependence_plot.png/eps - Dependence plot for the most important feature
  |- 05_waterfall_plot_1.png/eps to 05_waterfall_plot_5.png/eps - Five waterfall plots showing feature contributions for different instances
  |- 06_top_features_bar_plot.png/eps - Top 20 important features
  |- 07_significant_features.txt - List of significant features
  |- shap_values.npy - Saved SHAP values for future use
  |- expected_value.npy - Saved expected value for future use
  |- shap_values_data.xlsx - Excel file with SHAP values and feature importance

Analysis completed.
