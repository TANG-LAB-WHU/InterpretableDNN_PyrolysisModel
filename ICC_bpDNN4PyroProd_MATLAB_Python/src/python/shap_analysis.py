#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import shap
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib as mpl

# Set chart style and font
plt.style.use('default')
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5

# Create a function to set figure style
def set_figure_style(fig, ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=11)
    return fig, ax

# Create output directory if it doesn't exist
os.makedirs('output/shap_analysis', exist_ok=True)
os.makedirs('output/figures_tracking_training_process', exist_ok=True)
os.makedirs('output/shap_analysis/significant_features', exist_ok=True)

print("Loading MATLAB trained model...")
# First try the new structure, then fall back to the old one if needed
matlab_file_paths = [
    'output/shap_analysis/trained_model.mat',
    'output/results/trained_model.mat',  # Newly added location
    'output_shap_analysis/trained_model.mat'  # Fallback to old location
]

matlab_data = None
for file_path in matlab_file_paths:
    if os.path.exists(file_path):
        print(f"Found model file at: {file_path}")
        matlab_data = sio.loadmat(file_path)
        break

if matlab_data is None:
    # Try to convert from Results_trained.mat
    results_paths = [
        'output/results/Results_trained.mat',
        'output/shap_analysis/Results_trained.mat',
        'Results_trained.mat'
    ]
    
    for path in results_paths:
        if os.path.exists(path):
            print(f"Found Results_trained.mat at: {path}")
            try:
                results_data = sio.loadmat(path)
                # Extract required variables
                if all(key in results_data for key in ['Xi', 'Yi', 'Xt', 'Yt']):
                    X = results_data['Xi']
                    Y = results_data['Yi']
                    X_test = results_data['Xt']
                    Y_test = results_data['Yt']
                    # Save as trained_model.mat for future use
                    matlab_save_data = {
                        'X': X, 'Y': Y, 'XTest': X_test, 'YTest': Y_test
                    }
                    save_path = 'output/shap_analysis/trained_model.mat'
                    sio.savemat(save_path, matlab_save_data)
                    print(f"Converted Results_trained.mat data and saved to {save_path}")
                    matlab_data = matlab_save_data
                    break
                else:
                    print(f"Found {path} but it doesn't contain required variables.")
            except Exception as e:
                print(f"Error loading {path}: {e}")
    
    if matlab_data is None:
        raise FileNotFoundError("Could not find trained_model.mat or Results_trained.mat in any expected location")

# Extract data
X = matlab_data['X']  # Training features
Y = matlab_data['Y']  # Training targets
X_test = matlab_data['XTest']  # Test features
Y_test = matlab_data['YTest']  # Test targets

# Try to create meaningful feature names - if available in the MATLAB data
# Otherwise generate default names
try:
    # Attempt to load feature names from MATLAB data if available
    if 'feature_names' in matlab_data:
        feature_names = [str(name[0]) for name in matlab_data['feature_names'].flatten()]
    else:
        # Generate default feature names
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    print(f"Using feature names: {feature_names}")
except Exception as e:
    print(f"Could not load feature names: {e}")
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

# Create a DataFrame with feature names for better visualization
X_df = pd.DataFrame(X, columns=feature_names)
X_test_df = pd.DataFrame(X_test, columns=feature_names)

# We'll need to convert the MATLAB model to a format compatible with SHAP
# For this example, we'll train a surrogate model using scikit-learn
print("Training surrogate model for SHAP analysis...")
# Create and train the surrogate model for multi-target regression
# Wrap the RandomForestRegressor in MultiOutputRegressor for handling multiple outputs
base_model = RandomForestRegressor(n_estimators=100, random_state=42)
surrogate_model = MultiOutputRegressor(base_model)
surrogate_model.fit(X, Y)

# Save the surrogate model
joblib.dump(surrogate_model, 'output/shap_analysis/surrogate_model.pkl')

print("Performing SHAP analysis...")
# Initialize the SHAP explainer for the first output variable (typically char yield)
# We'll use the first estimator from our multi-output model
primary_model = surrogate_model.estimators_[0]

# Use TreeExplainer for the surrogate RandomForest model
print("Using TreeExplainer for surrogate Random Forest model")
explainer = shap.TreeExplainer(primary_model)

# Calculate SHAP values for the first output
shap_values = explainer.shap_values(X_test)

# Create summary plot with violin visualization
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_df, plot_type="violin", show=False)
plt.title('SHAP Values Distribution (Violin Plot)')
plt.tight_layout()
plt.savefig('output/shap_analysis/shap_violin_plot.png', dpi=600, bbox_inches='tight')
plt.close()

# Create beeswarm plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_df, plot_type="dot", show=False)
plt.title('SHAP Values Distribution (Beeswarm Plot)')
plt.tight_layout()
plt.savefig('output/shap_analysis/shap_beeswarm_plot.png', dpi=600, bbox_inches='tight')
plt.close()

# Create dependence plots for the most important features
feature_importance = np.abs(shap_values).mean(0)
feature_indices = np.argsort(-feature_importance)

print("Creating dependence plots for top features...")
for i in range(min(5, X.shape[1])):  # Top 5 features or all if less than 5
    idx = feature_indices[i]
    feature_name = feature_names[idx]
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(idx, shap_values, X_test_df, show=False)
    plt.title(f'SHAP Dependence Plot for {feature_name}')
    plt.tight_layout()
    plt.savefig(f'output/shap_analysis/shap_dependence_plot_feature_{idx}_{feature_name}.png', dpi=600)
    plt.close()

# Create waterfall plot for a sample prediction
plt.figure(figsize=(12, 8))
sample_idx = 0  # First test sample
shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[sample_idx], 
                                      features=X_test_df.iloc[sample_idx], 
                                      feature_names=feature_names, 
                                      show=False)
plt.title('SHAP Waterfall Plot for Sample Prediction')
plt.tight_layout()
plt.savefig('output/shap_analysis/shap_waterfall_plot.png', dpi=600)
plt.close()

# Save SHAP values for further analysis
np.save('output/shap_analysis/shap_values.npy', shap_values)
np.save('output/shap_analysis/feature_names.npy', np.array(feature_names))

# Save SHAP values as CSV for easier viewing and processing
print("Saving SHAP values as CSV file...")
# Create a DataFrame containing SHAP values for each sample
shap_df = pd.DataFrame(data=shap_values, columns=feature_names)
# Add sample index as the first column
shap_df.insert(0, 'Sample_Index', range(len(shap_values)))
# Save as CSV
shap_df.to_csv('output/shap_analysis/shap_values.csv', index=False)
print(f"Saved SHAP values for {len(shap_values)} samples and {len(feature_names)} features to CSV.")

# Save feature importance analysis
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})
importance_df = importance_df.sort_values('Importance', ascending=False)
importance_df.to_csv('output/shap_analysis/feature_importance.csv', index=False)

# Significance analysis - added based on SHAP documentation example
print("Performing significance analysis...")

# Calculate mean and standard deviation of feature importance
mean_importance = np.mean(feature_importance)
std_importance = np.std(feature_importance)

# Define significance threshold - features with importance above average + 1 standard deviation are considered significant
significance_threshold = mean_importance + std_importance
print(f"Significance threshold: {significance_threshold:.6f}")

# Filter significant features
significant_indices = np.where(feature_importance > significance_threshold)[0]
significant_features = [feature_names[i] for i in significant_indices]
significant_importance = feature_importance[significant_indices]

print(f"Identified {len(significant_features)} significant features: {', '.join(significant_features)}")

# Create bar chart of significant features by importance
plt.figure(figsize=(12, 8))
y_pos = np.arange(len(significant_features))
plt.barh(y_pos, significant_importance, align='center')
plt.yticks(y_pos, significant_features)
plt.xlabel('Mean |SHAP Value|')
plt.title('Significant Features by Importance')
plt.tight_layout()
plt.savefig('output/shap_analysis/significant_features_importance.png', dpi=600, bbox_inches='tight')
plt.close()

# Recreate summary plot with only significant features
print("Creating plots with only significant features...")

# Select X data and SHAP values for significant features
X_test_significant = X_test_df.iloc[:, significant_indices]
shap_values_significant = shap_values[:, significant_indices]

# Create violin plot with only significant features
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_significant, X_test_significant, plot_type="violin", show=False)
plt.title('SHAP Values Distribution for Significant Features (Violin Plot)')
plt.tight_layout()
plt.savefig('output/shap_analysis/significant_features_violin_plot.png', dpi=600, bbox_inches='tight')
plt.close()

# Create beeswarm plot with only significant features
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_significant, X_test_significant, plot_type="dot", show=False)
plt.title('SHAP Values Distribution for Significant Features (Beeswarm Plot)')
plt.tight_layout()
plt.savefig('output/shap_analysis/significant_features_beeswarm_plot.png', dpi=600, bbox_inches='tight')
plt.close()

# Create dependence plots for significant features
for i, idx in enumerate(significant_indices):
    feature_name = feature_names[idx]
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(i, shap_values_significant, X_test_significant, show=False)
    plt.title(f'SHAP Dependence Plot for Significant Feature: {feature_name}')
    plt.tight_layout()
    plt.savefig(f'output/shap_analysis/significant_features/dependence_plot_{feature_name}.png', dpi=600)
    plt.close()

# Save SHAP values analysis for significant features
significant_importance_df = pd.DataFrame({
    'Feature': significant_features,
    'Importance': significant_importance,
    'Relative_Importance': significant_importance / np.sum(significant_importance) * 100  # Percentage importance
})
significant_importance_df = significant_importance_df.sort_values('Importance', ascending=False)
significant_importance_df.to_csv('output/shap_analysis/significant_features_importance.csv', index=False)

# Create simple pie chart for significant features, showing their relative contribution to model output
plt.figure(figsize=(10, 10))
plt.pie(significant_importance_df['Relative_Importance'], 
        labels=significant_importance_df['Feature'], 
        autopct='%1.1f%%', 
        startangle=90)
plt.axis('equal')
plt.title('Relative Importance of Significant Features')
plt.tight_layout()
plt.savefig('output/shap_analysis/significant_features_pie_chart.png', dpi=600, bbox_inches='tight')
plt.close()

# Heatmap showing SHAP interaction between significant features and target variable
if len(significant_indices) > 1:  # At least two features are needed for interaction analysis
    plt.figure(figsize=(12, 10))
    sig_interaction_values = np.zeros((len(significant_indices), len(significant_indices)))
    
    for i, idx_i in enumerate(significant_indices):
        for j, idx_j in enumerate(significant_indices):
            if i != j:
                # Calculate interaction strength between two features
                # This is a simplified method, actual interaction calculation is more complex
                correlation = np.corrcoef(X_test[:, idx_i], X_test[:, idx_j])[0, 1]
                sig_interaction_values[i, j] = abs(correlation)
    
    # Create heatmap
    plt.imshow(sig_interaction_values, cmap='viridis')
    plt.colorbar(label='Interaction Strength')
    plt.xticks(np.arange(len(significant_features)), significant_features, rotation=45, ha='right')
    plt.yticks(np.arange(len(significant_features)), significant_features)
    plt.title('Interaction Strength Between Significant Features')
    plt.tight_layout()
    plt.savefig('output/shap_analysis/significant_features_interaction_heatmap.png', dpi=600, bbox_inches='tight')
    plt.close()

print("SHAP analysis completed. Results saved in 'output/shap_analysis' directory.")
print(f"Significance analysis results saved in 'output/shap_analysis/significant_features' directory.")

# Optional: If you want to try analyzing the original neural network model directly
# This requires TensorFlow/Keras and might not work with your MATLAB model structure
try:
    print("Attempting to analyze original MATLAB neural network model (optional)...")
    # This is experimental and might require additional code to convert MATLAB NN to TensorFlow format
    # If this fails, the above analysis with the surrogate model will still provide valid results
    
    # Skip this experimental section if not needed
    """
    # Example of how you might use DeepExplainer if you convert the MATLAB model
    import tensorflow as tf
    # Code to convert MATLAB NN to TensorFlow would go here
    # ...
    
    # Using DeepExplainer
    deep_explainer = shap.DeepExplainer(converted_model, X[:100])  # Background samples
    deep_shap_values = deep_explainer.shap_values(X_test)
    
    # Save additional analysis
    np.save('output/shap_analysis/deep_shap_values.npy', deep_shap_values)
    """
    
except Exception as e:
    print(f"Neural network direct analysis skipped or failed: {e}")
    print("Using surrogate model results only, which is valid for interpretation.") 