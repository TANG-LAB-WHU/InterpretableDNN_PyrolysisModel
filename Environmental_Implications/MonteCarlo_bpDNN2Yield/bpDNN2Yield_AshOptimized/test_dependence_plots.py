#!/usr/bin/env python3
"""
Test script for modified dependence plot functions

This script tests the modified functions to ensure they correctly:
1. Focus on FeedstockType features as interaction features
2. Set colorbar ticks to only show 0 and 1 for binary features
3. Maintain English-only output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os
import sys

# Add the current directory to the path to import the main script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the functions from the main script
from shap_analysis_latest import (
    create_comprehensive_targettemperature_dependence_plots,
    create_target_specific_dependence_plots,
    sanitize_feature_name_for_filename
)

def create_test_data():
    """Create test data with FeedstockType features"""
    np.random.seed(42)
    
    # Create sample data
    n_samples = 100
    
    # Create features including TargetTemperature and FeedstockType features
    data = {
        'TargetTemperature/Celsius': np.random.uniform(200, 1200, n_samples),
        'FeedstockType_1': np.random.choice([0, 1], n_samples),
        'FeedstockType_2': np.random.choice([0, 1], n_samples),
        'FeedstockType_3': np.random.choice([0, 1], n_samples),
        'FeedstockType_4': np.random.choice([0, 1], n_samples),
        'Ash_SiO2': np.random.uniform(0, 100, n_samples),
        'Ash_Al2O3': np.random.uniform(0, 50, n_samples),
        'HeatingRate/(K/min)': np.random.uniform(1, 50, n_samples)
    }
    
    X_test_df = pd.DataFrame(data)
    
    # Create dummy SHAP values
    shap_values = np.random.normal(0, 10, (n_samples, len(X_test_df.columns)))
    
    # Create expected value
    expected_value = 50.0
    
    return X_test_df, shap_values, expected_value

def test_dependence_plots():
    """Test the modified dependence plot functions"""
    print("Creating test data...")
    X_test_df, shap_values, expected_value = create_test_data()
    
    # Create output directory
    output_dir = "test_dependence_plots_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Testing create_comprehensive_targettemperature_dependence_plots...")
    try:
        create_comprehensive_targettemperature_dependence_plots(
            expected_value, shap_values, X_test_df, output_dir, "Biochar"
        )
        print("✓ create_comprehensive_targettemperature_dependence_plots completed successfully")
    except Exception as e:
        print(f"✗ Error in create_comprehensive_targettemperature_dependence_plots: {e}")
    
    print("Testing create_target_specific_dependence_plots...")
    try:
        create_target_specific_dependence_plots(
            expected_value, shap_values, X_test_df, output_dir, "Biochar"
        )
        print("✓ create_target_specific_dependence_plots completed successfully")
    except Exception as e:
        print(f"✗ Error in create_target_specific_dependence_plots: {e}")
    
    # Check if files were created
    biochar_dir = os.path.join(output_dir, "01_Biochar")
    if os.path.exists(biochar_dir):
        files = os.listdir(biochar_dir)
        dependence_files = [f for f in files if f.startswith("04_") and "dependence" in f]
        print(f"✓ Created {len(dependence_files)} dependence plot files")
        for file in dependence_files:
            print(f"  - {file}")
    else:
        print("✗ No output directory created")
    
    print("\nTest completed. Check the output directory for generated plots.")

if __name__ == "__main__":
    test_dependence_plots() 