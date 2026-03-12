SHAP Analysis Results (Using MATLAB Neural Network Model)
==================================================

Analysis Date: 2025-05-13 12:51:33
Note: Analysis performed using ALL sample data without train/test splitting
Only the original MATLAB model is used (no fallback models)

Directory Structure:
-------------------
00_README.txt - This file
00_analysis_log.txt - Processing log with details and any errors
00_feature_names_used.txt - List of feature names used in the analysis

01_Char/ - Analysis results for Char yield
  |- 00_model_info.txt - Information about the model used
  |- 01_beeswarm_plot.png - Shows feature importance and impact direction
  |- 02_feature_importance_plot.png - Bar plot of feature importance
  |- 03_instance_explanation_plot.png - Force plot for a single instance
  |- 04_dependence_plot.png - Dependence plot for the most important feature
  |- 05_waterfall_plot_1.png to 05_waterfall_plot_5.png - Five waterfall plots showing feature contributions for different instances
  |- 06_top_features_bar_plot.png - Top 20 important features
  |- 07_significant_features.txt - List of significant features
  |- shap_values.npy - Saved SHAP values for future use
  |- expected_value.npy - Saved expected value for future use

01_Liquid/ - Analysis results for Liquid yield
  |- 00_model_info.txt - Information about the model used
  |- 01_beeswarm_plot.png - Shows feature importance and impact direction
  |- 02_feature_importance_plot.png - Bar plot of feature importance
  |- 03_instance_explanation_plot.png - Force plot for a single instance
  |- 04_dependence_plot.png - Dependence plot for the most important feature
  |- 05_waterfall_plot_1.png to 05_waterfall_plot_5.png - Five waterfall plots showing feature contributions for different instances
  |- 06_top_features_bar_plot.png - Top 20 important features
  |- 07_significant_features.txt - List of significant features
  |- shap_values.npy - Saved SHAP values for future use
  |- expected_value.npy - Saved expected value for future use

01_Gas/ - Analysis results for Gas yield
  |- 00_model_info.txt - Information about the model used
  |- 01_beeswarm_plot.png - Shows feature importance and impact direction
  |- 02_feature_importance_plot.png - Bar plot of feature importance
  |- 03_instance_explanation_plot.png - Force plot for a single instance
  |- 04_dependence_plot.png - Dependence plot for the most important feature
  |- 05_waterfall_plot_1.png to 05_waterfall_plot_5.png - Five waterfall plots showing feature contributions for different instances
  |- 06_top_features_bar_plot.png - Top 20 important features
  |- 07_significant_features.txt - List of significant features
  |- shap_values.npy - Saved SHAP values for future use
  |- expected_value.npy - Saved expected value for future use

Analysis completed.
