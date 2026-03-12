"""
SHAP Analysis for Activation Energy (Ea) Prediction Neural Network

This script loads a pre-trained neural network model from MATLAB .mat file
and performs SHAP (SHapley Additive exPlanations) analysis to interpret the model
that predicts activation energy for pyrolysis reactions.

Visualizations include:
- Beeswarm plot (summary plot)
- Instance explanation (force plot)
- Dependence plot
- Waterfall plot
- Feature importance rankings
- Top features bar plot

Output Formats:
- PNG: Raster format for web/screen viewing
- SVG: Vector format optimized for Adobe Illustrator editing (best compatibility)
- EPS: Vector format for publication with TrueType font embedding

Adobe Illustrator Compatibility:
- SVG files provide the best editing experience in Adobe Illustrator
- EPS files use TrueType fonts (ps_fonttype=42) to avoid font embedding issues
- Both vector formats maintain quality at any scale

References:
- https://shap.readthedocs.io/en/latest/index.html
- https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import shap
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib
import datetime
import sklearn
from sklearn.dummy import DummyRegressor

# -----------------------------------------------------------------------------
# Added logging, deterministic seeding, and helper utilities (borrowed from
# shap_analysis.py for consistency across scripts).
# -----------------------------------------------------------------------------

import logging
import random  # ensure reproducibility across Python's RNG as well

# Configure logging – INFO by default, DEBUG when the --debug flag is supplied
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# Helper to print only in debug mode (mimics the behaviour in shap_analysis.py)
def dprint(msg: str):
    """Debug-print replacement that obeys the global DEBUG_MODE flag."""
    if 'DEBUG_MODE' in globals() and DEBUG_MODE:
        logger.debug(msg)


# -----------------------------------------------------------------------------
# Deterministic seeding – ensures that all random operations (NumPy, Python,
# PyTorch, SHAP) are repeatable across script runs.  Critical for getting
# identical SHAP values, background sampling and beeswarm plots.
# -----------------------------------------------------------------------------

GLOBAL_SEED = 42

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# SHAP >=0.45 has its own RNG wrapper
try:
    shap.random.seed(GLOBAL_SEED)  # type: ignore[attr-defined]
except AttributeError:
    # Older SHAP versions fall back to NumPy's RNG (already seeded)
    pass

torch.manual_seed(GLOBAL_SEED)

# Base feature names from MATLAB script bpDNN4Ea.m
BASE_FEATURE_NAMES = [
    'Location',
    'VolatileMatters/%',
    'FixedCarbon/%',
    'Ash/%',
    'C/%',
    'H/%',
    'N/%',
    'O/%',
    'S/%',
    'Ash_SiO2',
    'Ash_Na2O',
    'Ash_MgO',
    'Ash_Al2O3',
    'Ash_K2O',
    'Ash_CaO',
    'Ash_P2O5',
    'Ash_CuO',
    'Ash_ZnO',
    'Ash_Fe2O3',
    'Degree_conversion'
]

def load_matlab_data(file_path):
    """
    Load data from a MATLAB .mat file
    
    Args:
        file_path: Path to the .mat file
        
    Returns:
        Dictionary containing the loaded data
    """
    print(f"Loading MATLAB file: {file_path}")
    try:
        mat_data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        # Remove built-in fields
        return {k: v for k, v in mat_data.items() if not k.startswith('__')}
    except Exception as e:
        print(f"Error loading MATLAB file: {e}")
        return {}

def extract_neural_network_data(mat_data):
    """
    Extract input data, target data, and network structure from MATLAB data
    
    Args:
        mat_data: Dictionary containing MATLAB data
        
    Returns:
        Tuple of (X, y, network_structure)
    """
    # Initialize variables
    X = None
    y = None
    network_structure = None
    
    # Try to find standard variable names for inputs/outputs
    possible_input_names = ['input', 'Variables', 'Variables0', 'Feedstock4training']
    possible_output_names = ['target', 'Ea', 'Target']
    
    # First try standard variable names
    for input_name in possible_input_names:
        if input_name in mat_data:
            X = mat_data[input_name]
            print(f"Found input data in variable '{input_name}' with shape {X.shape}")
            break
    
    for output_name in possible_output_names:
        if output_name in mat_data:
            y = mat_data[output_name]
            print(f"Found target data in variable '{output_name}' with shape {y.shape}")
            break
    
    # Extract network structure if 'net' variable exists
    if 'net' in mat_data:
        network_structure = mat_data['net']
        print("Found network structure in 'net' variable")
    
    # If we couldn't find standard names, try to infer based on variable shapes
    if X is None or y is None:
        print("Could not find standard input/output variable names, trying to infer from shapes...")
        
        for var_name, var in mat_data.items():
            if not var_name.startswith('__') and isinstance(var, np.ndarray):
                # Input data likely has more columns than rows for features
                if len(var.shape) == 2 and var.shape[1] > 20:
                    if X is None or var.shape[1] > X.shape[1]:  # Prefer the one with more features
                        X = var
                        print(f"Inferred input data from variable '{var_name}' with shape {X.shape}")
                
                # Output data likely has fewer columns than input (single Ea output)
                elif len(var.shape) == 2 and var.shape[1] == 1:
                    if y is None:
                        y = var
                        print(f"Inferred target data from variable '{var_name}' with shape {y.shape}")
                elif len(var.shape) == 1:  # Could be 1D array for single output
                    if y is None:
                        y = var
                        print(f"Inferred target data from variable '{var_name}' with shape {y.shape}")
    
    # If data is still not found, raise an error
    if X is None or y is None:
        raise ValueError("Could not find or infer input and target data from the MATLAB file")
    
    # Check if the data is transposed (more features than samples)
    # For the MATLAB format, input shape (256, 233) means 256 features and 233 samples
    # So we need to transpose: features are in rows, samples are in columns
    if len(X.shape) == 2 and X.shape[0] > X.shape[1]:
        print(f"Input data appears to be in MATLAB format (features, samples). Transposing to (samples, features)...")
        X = X.T
        print(f"Transposed input data shape: {X.shape}")
    
    # Ensure y is 1D for single output (Ea)
    if len(y.shape) == 2 and y.shape[1] == 1:
        y = y.ravel()
        print(f"Converted target data to 1D: {y.shape}")
    elif len(y.shape) == 2 and y.shape[0] == 1:
        y = y.ravel()
        print(f"Converted target data to 1D: {y.shape}")
    
    print(f"Final data shapes: X={X.shape}, y={y.shape}")
    
    return X, y, network_structure

def generate_feature_names(X, mat_data=None):
    """
    Generate complete feature names based on the input data
    
    Args:
        X: Input data array
        mat_data: MATLAB data dictionary (optional)
        
    Returns:
        List of feature names
    """
    total_features = X.shape[1]
    base_count = len(BASE_FEATURE_NAMES)
    
    print(f"Total features: {total_features}, Base features: {base_count}")
    
    # If we have exactly the right number of base features, return them
    if total_features == base_count:
        print(f"Using {base_count} base feature names")
        return BASE_FEATURE_NAMES
    
    # Calculate feedstock count based on remaining features
    remaining_features = total_features - base_count
    print(f"Remaining features after base: {remaining_features}")
    
    # For the Ea model, the remaining features seem to be feedstock-related
    # Let's assume they are organized as feedstock type and ratio pairs
    if remaining_features % 2 == 0:
        feedstock_count = remaining_features // 2  # Each feedstock type has both Type and Ratio features
        print(f"Detected {feedstock_count} feedstock types (assuming Type+Ratio pairs)")
        
        # Generate complete feature list
        feature_names = BASE_FEATURE_NAMES.copy()
        
        # Add FeedstockType features
        for i in range(feedstock_count):
            feature_names.append(f'FeedstockType_{i+1}')
        
        # Add MixingRatio features
        for i in range(feedstock_count):
            feature_names.append(f'MixingRatio_{i+1}')
            
    else:
        # If not even, just create generic feedstock features
        print(f"Creating {remaining_features} generic feedstock features")
        feature_names = BASE_FEATURE_NAMES.copy()
        
        for i in range(remaining_features):
            feature_names.append(f'Feedstock_Feature_{i+1}')
    
    print(f"Generated {len(feature_names)} total feature names")
    return feature_names

class MatlabNeuralNetworkWrapper:
    """
    Wrapper for MATLAB neural network to work with SHAP for Ea prediction
    
    This wrapper provides a predict method that uses the MATLAB neural network
    to make predictions for activation energy (single output).
    """
    def __init__(self, net_struct):
        """
        Initialize the wrapper with the MATLAB neural network structure
        
        Args:
            net_struct: MATLAB neural network structure
        """
        self.net_struct = net_struct
        self.model_type = "matlab_nn"
        
        # Debug the network structure to understand what we're working with
        self._debug_network_structure()
        
        # Check if a simple predict method is directly available
        if hasattr(net_struct, 'predict'):
            print(f"Found direct predict method in MATLAB neural network")
            self.predict_method = self._direct_predict
        else:
            # Try to extract weights and biases using different approaches
            print(f"Attempting to extract network architecture from MATLAB structure...")
            
            # Try BP Neural Network with cell arrays
            if self._extract_bp_neural_network():
                print("Successfully extracted network using BP neural network format with cell arrays")
                self.predict_method = self._forward_pass_predict
            
            # Try standard Neural Network Toolbox structure
            elif self._extract_standard_nn_structure():
                print("Successfully extracted network using standard Neural Network Toolbox structure")
                self.predict_method = self._forward_pass_predict
            
            # Try legacy MATLAB Neural Network structure
            elif self._extract_legacy_nn_structure():
                print("Successfully extracted network using legacy MATLAB NN structure")
                self.predict_method = self._matlab_legacy_predict
            
            # Try custom structure extraction
            elif self._extract_custom_structure():
                print("Successfully extracted network using custom structure pattern")
                self.predict_method = self._custom_predict
                
            else:
                print("CRITICAL ERROR: Could not extract network architecture using any known method")
                raise AttributeError("Could not extract network architecture using any known method")
        
        # Test the prediction method on a dummy input
        test_input = np.ones((1, self.get_input_size()))
        test_output = self.predict(test_input)
        print(f"Test prediction successful. Output shape: {test_output.shape}")
    
    def _debug_network_structure(self):
        """Debug the network structure to understand available attributes"""
        try:
            print("\nDEBUG: Exploring MATLAB network structure attributes:")
            
            # List top-level attributes
            attributes = [attr for attr in dir(self.net_struct) if not attr.startswith('__')]
            print(f"Top-level attributes: {attributes}")
            
            # Check for common network attributes
            for attr in ['layers', 'Layers', 'layer', 'Layer', 'weights', 'Weights', 'biases', 'Biases',
                        'IW', 'LW', 'b', 'inputWeights', 'layerWeights', 'inputs', 'outputs', 'weight', 'bias']:
                if hasattr(self.net_struct, attr):
                    obj = getattr(self.net_struct, attr)
                    if isinstance(obj, (list, tuple, np.ndarray)):
                        print(f"Found array attribute '{attr}' with length {len(obj)}")
                    else:
                        print(f"Found attribute '{attr}' of type {type(obj)}")
                        
        except Exception as e:
            print(f"Error during network structure debugging: {e}")
    
    def _extract_bp_neural_network(self):
        """Extract weights and biases from MATLAB BP neural network format"""
        try:
            if hasattr(self.net_struct, 'weight') and hasattr(self.net_struct, 'bias'):
                print("Attempting to extract MATLAB BP neural network with cell arrays...")
                
                # Get the weight and bias arrays and layer information
                weight_cell = self.net_struct.weight
                bias_cell = self.net_struct.bias
                
                # Check if these are cell arrays
                if hasattr(weight_cell, '__len__') and hasattr(bias_cell, '__len__'):
                    print(f"Found weight cell array with length {len(weight_cell)}")
                    print(f"Found bias cell array with length {len(bias_cell)}")
                    
                    # Initialize lists for weights and biases
                    self.weights = []
                    self.biases = []
                    
                    # Get layer information if available
                    layer_info = None
                    if hasattr(self.net_struct, 'layer'):
                        layer_info = self.net_struct.layer
                        print(f"Found layer information with {len(layer_info)} layers")
                    
                    # Extract activation functions if available
                    self.activation_funcs = []
                    
                    # Process each layer's weights and biases
                    for i in range(len(weight_cell)):
                        # Get the weight matrix and bias vector for this layer
                        w = weight_cell[i]
                        b = bias_cell[i]
                        
                        # Print debug information
                        print(f"Layer {i} - Weight type: {type(w)}, Bias type: {type(b)}")
                        if hasattr(w, 'shape'):
                            print(f"Layer {i} - Weight shape: {w.shape}")
                        if hasattr(b, 'shape'):
                            print(f"Layer {i} - Bias shape: {b.shape}")
                        
                        # Make sure they are numpy arrays
                        if not isinstance(w, np.ndarray):
                            w = np.array(w) if hasattr(w, '__array__') else None
                        
                        if not isinstance(b, np.ndarray):
                            # Handle the case where bias might be a scalar (especially for output layer)
                            if isinstance(b, (int, float)):
                                b = np.array([b])
                            else:
                                b = np.array(b) if hasattr(b, '__array__') else None
                        
                        # Special handling for output layer
                        if i == len(weight_cell) - 1:  # Last layer (output layer)
                            # If weight is 1D, reshape to (1, input_size) for output layer
                            if w is not None and len(w.shape) == 1:
                                w = w.reshape(1, -1)
                                print(f"Reshaped output layer weight to: {w.shape}")
                            # Ensure bias is 1D array
                            if b is not None and len(b.shape) == 0:
                                b = np.array([b])
                                print(f"Reshaped output layer bias to: {b.shape}")
                        
                        # Add to the lists
                        self.weights.append(w)
                        self.biases.append(b)
                        
                        # Get activation function for this layer
                        if layer_info is not None and i < len(layer_info):
                            if hasattr(layer_info[i], 'transferFcn'):
                                self.activation_funcs.append(layer_info[i].transferFcn)
                            else:
                                # Default activation functions
                                if i < len(weight_cell) - 1:
                                    self.activation_funcs.append('logsig')  # Hidden layers
                                else:
                                    self.activation_funcs.append('purelin')  # Output layer
                        else:
                            # Default activation functions
                            if i < len(weight_cell) - 1:
                                self.activation_funcs.append('logsig')  # Hidden layers
                            else:
                                self.activation_funcs.append('purelin')  # Output layer
                    
                    print(f"Extracted {len(self.weights)} weight matrices and {len(self.biases)} bias vectors")
                    print(f"Activation functions: {self.activation_funcs}")
                    
                    # Debug final shapes
                    for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                        if w is not None and b is not None:
                            print(f"Final Layer {i} - Weight: {w.shape}, Bias: {b.shape}")
                    
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error extracting BP neural network: {e}")
            return False
    
    def _extract_standard_nn_structure(self):
        """Extract weights and biases using standard Neural Network Toolbox structure"""
        # Implementation similar to original but simplified
        return False
    
    def _extract_legacy_nn_structure(self):
        """Extract using the traditional MATLAB neural network structure (IW, LW, b)"""
        try:
            if (hasattr(self.net_struct, 'IW') and hasattr(self.net_struct, 'LW') and 
                hasattr(self.net_struct, 'b')):
                
                print("Found IW, LW, b structure (traditional MATLAB NN format)")
                self.iw = self.net_struct.IW
                self.lw = self.net_struct.LW
                self.b = self.net_struct.b
                
                # Store transfer functions if available
                if hasattr(self.net_struct, 'layers') and hasattr(self.net_struct.layers[0], 'transferFcn'):
                    self.transfer_functions = [layer.transferFcn for layer in self.net_struct.layers]
                else:
                    # Default to logsig for hidden layers and purelin for output
                    if hasattr(self.net_struct.b, '__len__'):
                        self.transfer_functions = ['logsig'] * (len(self.net_struct.b) - 1) + ['purelin']
                    else:
                        self.transfer_functions = ['logsig', 'purelin']
                
                return True
            
            return False
            
        except Exception as e:
            print(f"Error extracting legacy NN structure: {e}")
            return False
    
    def _extract_custom_structure(self):
        """Extract weights using custom structure patterns"""
        # Simplified implementation for Ea model
        return False
    
    def get_input_size(self):
        """Determine the input size of the network from the structure"""
        try:
            if hasattr(self, 'weights') and len(self.weights) > 0 and self.weights[0] is not None:
                if isinstance(self.weights[0], np.ndarray):
                    if len(self.weights[0].shape) > 1:
                        # For first layer: weight shape is (neurons, features)
                        input_size = self.weights[0].shape[1]
                        print(f"Determined input size from first layer weights: {input_size}")
                        return input_size
                    else:
                        input_size = self.weights[0].shape[0]
                        print(f"Determined input size from first layer weights (1D): {input_size}")
                        return input_size
            elif hasattr(self, 'iw') and hasattr(self.iw, '__len__') and len(self.iw) > 0:
                input_size = self.iw[0][0].shape[1]
                print(f"Determined input size from IW structure: {input_size}")
                return input_size
        except Exception as e:
            print(f"Error determining input size: {e}")
        
        # Default fallback size based on what we know from the data
        print("Using default input size based on data structure")
        return 256  # Based on the MATLAB data structure we saw
    
    def _direct_predict(self, X):
        """Use the network's built-in predict method"""
        try:
            result = self.net_struct.predict(X)
            if isinstance(result, np.ndarray):
                return result.ravel()
            else:
                return np.array([result] * X.shape[0])
        except Exception as e:
            print(f"Error in direct prediction: {e}")
            return np.zeros(X.shape[0])
    
    def _forward_pass_predict(self, X):
        """Implement forward pass through the network using extracted weights and biases"""
        try:
            # Forward pass through the network
            a = X
            
            for i in range(len(self.weights)):
                if self.weights[i] is None or self.biases[i] is None:
                    continue
                
                try:
                    w = self.weights[i]
                    b = self.biases[i]
                    
                    # Linear transformation: a = a * W.T + b
                    # For MATLAB networks: W has shape (neurons_out, neurons_in)
                    # So for matrix multiplication: (batch, neurons_in) @ (neurons_in, neurons_out)
                    # We need to transpose W: (neurons_out, neurons_in) -> (neurons_in, neurons_out)
                    z = np.dot(a, w.T) + b
                    
                    # Apply activation function
                    activation = self.activation_funcs[i].lower() if isinstance(self.activation_funcs[i], str) else 'linear'
                    
                    # MATLAB activation functions
                    if activation == 'logsig':  # MATLAB sigmoid function
                        a = 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to prevent overflow
                    elif activation == 'tansig':  # MATLAB tanh function
                        a = np.tanh(z)
                    elif activation == 'purelin':  # MATLAB linear function
                        a = z
                    elif activation == 'relu':
                        a = np.maximum(0, z)
                    else:
                        # Default to linear activation
                        a = z
                
                except Exception as e:
                    print(f"Error in forward pass at layer {i}: {e}")
                    # Skip to next layer if this one fails
                    continue
            
            # Ensure output is 1D for single output (Ea)
            if len(a.shape) > 1:
                if a.shape[1] == 1:
                    a = a.ravel()
                else:
                    # If multiple outputs, just take the first one (shouldn't happen for Ea model)
                    a = a[:, 0]
            
            return a
            
        except Exception as e:
            print(f"Error in forward pass prediction: {e}")
            return np.zeros(X.shape[0])
    
    def _matlab_legacy_predict(self, X):
        """Use the traditional MATLAB neural network structure (IW, LW, b)"""
        try:
            # First layer
            a1 = np.tanh(np.dot(X, self.iw[0][0].T) + self.b[0])
            
            # Hidden layers
            for i in range(1, len(self.b)-1):
                a1 = np.tanh(np.dot(a1, self.lw[i][i-1].T) + self.b[i])
            
            # Output layer (linear activation)
            output = np.dot(a1, self.lw[len(self.b)-1][len(self.b)-2].T) + self.b[len(self.b)-1]
            
            return output.ravel()
        except Exception as e:
            print(f"Error in legacy MATLAB prediction: {e}")
            return np.zeros(X.shape[0])
    
    def _custom_predict(self, X):
        """Implement custom prediction for unique network structures"""
        # Fallback implementation
        return np.zeros(X.shape[0])
    
    def predict(self, X):
        """
        Predict using the selected method
        
        Args:
            X: Input data
            
        Returns:
            Predictions for activation energy (Ea)
        """
        # Convert input to numpy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Call the selected prediction method
        try:
            predictions = self.predict_method(X)
            # Ensure positive values for activation energy
            return np.abs(predictions)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return np.zeros(X.shape[0]) 

def save_plot_multi_format(file_path_without_ext, dpi=600, bbox_inches='tight', facecolor='white'):
    """
    Save the current matplotlib figure in multiple formats optimized for Adobe Illustrator
    
    Args:
        file_path_without_ext: File path without extension
        dpi: Resolution for raster format (PNG)
        bbox_inches: Bounding box setting
        facecolor: Background color
    """
    # Save as PNG (raster format - good for web/screen)
    png_path = f"{file_path_without_ext}.png"
    print(f"Saving plot to: {png_path}")
    try:
        plt.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches, facecolor=facecolor, format='png')
    except Exception as e:
        print(f"Error saving PNG plot: {e}, trying with minimal options")
        plt.savefig(png_path, dpi=dpi)
    
    # Save as SVG (vector format - best for Adobe Illustrator compatibility)
    svg_path = f"{file_path_without_ext}.svg"
    print(f"Saving plot to: {svg_path}")
    try:
        plt.savefig(svg_path, 
                   bbox_inches=bbox_inches, 
                   facecolor=facecolor, 
                   format='svg',
                   transparent=False,
                   edgecolor='none')
    except Exception as e:
        print(f"Error saving SVG plot: {e}, trying with minimal options")
        try:
            plt.savefig(svg_path, format='svg')
        except Exception as e2:
            print(f"Failed to save SVG plot: {e2}")
    
    # Save as EPS (vector format - optimized for Adobe Illustrator)
    eps_path = f"{file_path_without_ext}.eps"
    print(f"Saving plot to: {eps_path}")
    try:
        # Set matplotlib backend parameters for better EPS compatibility
        import matplotlib
        original_backend = matplotlib.get_backend()
        
        plt.savefig(eps_path, 
                   bbox_inches=bbox_inches, 
                   facecolor=facecolor, 
                   format='eps',
                   transparent=False,
                   edgecolor='none',
                   # EPS-specific parameters for Adobe Illustrator compatibility
                   ps_fonttype=42,  # Use TrueType fonts (Type 42) instead of Type 3
                   orientation='portrait',
                   papertype='letter')
    except Exception as e:
        print(f"Error saving EPS plot: {e}, trying with minimal options")
        try:
            plt.savefig(eps_path, format='eps', ps_fonttype=42)
        except Exception as e2:
            print(f"Failed to save EPS plot: {e2}")

def create_custom_waterfall_plot(explainer, shap_values, X_test_df, output_dir, instance_idx=0, plot_number=1):
    """
    Create a custom waterfall plot with feature values on the y-axis for Ea prediction
    
    Args:
        explainer: SHAP explainer object
        shap_values: SHAP values
        X_test_df: Test data as DataFrame with feature names
        output_dir: Directory to save plots
        instance_idx: Index of instance to analyze (default: 0)
        plot_number: Number to append to the filename (default: 1)
    """
    print(f"Creating waterfall plot {plot_number} for instance {instance_idx}...")
    plt.figure(figsize=(10, 8))
    
    # Check if expected value is an array
    expected_value = explainer.expected_value
    if isinstance(expected_value, np.ndarray) or isinstance(expected_value, list):
        expected_value = expected_value[0]  # Get the first element
    
    # Get the feature values for the specific instance
    instance_values = X_test_df.iloc[instance_idx]
    
    # Create the waterfall plot
    try:
        shap.plots._waterfall.waterfall_legacy(
            expected_value,
            shap_values[instance_idx],
            X_test_df.iloc[instance_idx],
            max_display=20,  # Display top 20 features
            show=False,
            pos_color="#FF0052",  # Bright red for positive values
            neg_color="#0088FF",  # Bright blue for negative values
            linewidth=0,
            alpha=0.8
        )
    except TypeError as e:
        print(f"Warning: Falling back to standard parameters due to error: {e}")
        shap.plots._waterfall.waterfall_legacy(
            expected_value,
            shap_values[instance_idx],
            X_test_df.iloc[instance_idx],
            max_display=20,
            show=False
        )
    
    # Get current y-axis labels
    ax = plt.gca()
    y_labels = [text.get_text() for text in ax.get_yticklabels()]
    
    # Create new labels with feature values
    new_labels = []
    for label in y_labels:
        if label == 'E[f(X)]' or label == 'f(x)' or 'other features' in label:
            new_labels.append(label)
        else:
            feature_name = label.strip()
            try:
                feature_value = instance_values[feature_name]
                if isinstance(feature_value, (int, float)):
                    if abs(feature_value) < 0.01:
                        value_str = f"{feature_value:.4f}"
                    elif abs(feature_value) < 1:
                        value_str = f"{feature_value:.3f}"
                    elif abs(feature_value) < 10:
                        value_str = f"{feature_value:.2f}"
                    elif abs(feature_value) < 100:
                        value_str = f"{feature_value:.1f}"
                    else:
                        value_str = f"{int(feature_value)}"
                    new_labels.append(f"{value_str} = {feature_name}")
                else:
                    new_labels.append(feature_name)
            except KeyError:
                new_labels.append(feature_name)
    
    # Set new y-axis labels
    ax.set_yticklabels(new_labels, fontsize=9)
    
    # Modify x-axis label
    ax.set_xlabel("Activation Energy Ea (kJ/mol)", fontsize=10)
    
    # Update text annotations to hide baseline and prediction values
    for text in ax.texts:
        if "f(x)" in text.get_text() or "E[f(X)]" in text.get_text():
            text.set_visible(False)
                
    plt.tight_layout()
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(left=0.2, right=0.9, top=0.95, bottom=0.1)
    
    # Save the plot
    output_file_base = f"{output_dir}/05_waterfall_plot_{plot_number}"
    save_plot_multi_format(output_file_base, dpi=600, bbox_inches='tight', facecolor='white')
    
    plt.close()

def create_shap_plots(explainer, shap_values, X_test_df, output_dir, instance_idx=0):
    """
    Create various SHAP plots for Ea model interpretation
    
    Args:
        explainer: SHAP explainer object
        shap_values: SHAP values
        X_test_df: Test data as DataFrame with feature names
        output_dir: Directory to save plots
        instance_idx: Index of instance to analyze (default: 0)
    """
    # Get the expected value
    expected_value = explainer.expected_value
    if isinstance(expected_value, np.ndarray) or isinstance(expected_value, list):
        expected_value = expected_value[0]
    
    # Get feature importance by mean absolute SHAP value
    feature_importance = np.abs(shap_values).mean(0)
    
    # 1. Create a summary plot (beeswarm plot)
    print("Creating beeswarm plot...")
    plt.figure(figsize=(12, 14))
    shap.summary_plot(shap_values, X_test_df, plot_type="dot", show=False)
    plt.xlabel("SHAP value (impact on Ea prediction)")
    plt.tight_layout()
    save_plot_multi_format(f"{output_dir}/01_beeswarm_plot", dpi=600, bbox_inches='tight')
    plt.close()
    
    # 2. Create a bar summary plot with feature importance
    print("Creating feature importance plot...")
    plt.figure(figsize=(12, 14))
    shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
    plt.title("Activation Energy (Ea) Feature Importance - Bar Plot", fontsize=16)
    plt.xlabel("Mean |SHAP value| (impact on Ea prediction)")
    plt.tight_layout()
    save_plot_multi_format(f"{output_dir}/02_feature_importance_plot", dpi=600, bbox_inches='tight')
    plt.close()
    
    # 3. Create a force plot for a single instance
    print("Creating force plot for a single instance...")
    plt.figure(figsize=(20, 5))
    
    force_plot = shap.force_plot(
        expected_value,
        shap_values[instance_idx],
        X_test_df.iloc[instance_idx],
        matplotlib=True,
        show=False
    )
    plt.title("Activation Energy (Ea) Single Instance Explanation", fontsize=16)
    plt.tight_layout()
    save_plot_multi_format(f"{output_dir}/03_instance_explanation_plot", dpi=600, bbox_inches='tight')
    plt.close()
    
    # 4. Create a dependence plot for the most important feature
    print("Creating dependence plot...")
    most_important_idx = np.argmax(feature_importance)
    most_important_feature = X_test_df.columns[most_important_idx]
    
    plt.figure(figsize=(12, 8))
    shap.dependence_plot(
        most_important_idx,
        shap_values,
        X_test_df,
        interaction_index="auto",
        show=False
    )
    plt.title(f"Activation Energy (Ea) Dependence Plot for {most_important_feature}", fontsize=16)
    plt.tight_layout()
    save_plot_multi_format(f"{output_dir}/04_dependence_plot", dpi=600, bbox_inches='tight')
    plt.close()
    
    # 5. Create 5 custom waterfall plots for different instances
    print("Creating 5 waterfall plots for Ea...")
    num_instances = len(X_test_df)
    if num_instances >= 5:
        step = num_instances // 5
        instance_indices = [i * step for i in range(5)]
    else:
        instance_indices = list(range(num_instances))
    
    for i, idx in enumerate(instance_indices):
        create_custom_waterfall_plot(explainer, shap_values, X_test_df, output_dir, 
                                     instance_idx=idx, plot_number=i+1)
    
    # 6. Generate a bar plot of the top significant features
    print("Creating top features bar plot...")
    top_n = 20
    indices = np.argsort(-feature_importance)[:top_n]
    significant_features = [X_test_df.columns[i] for i in indices]
    top_importance = [feature_importance[i] for i in indices]
    
    plt.figure(figsize=(12, 10))
    plt.barh(range(len(significant_features)), top_importance, align='center')
    plt.yticks(range(len(significant_features)), significant_features)
    plt.xlabel('Mean |SHAP value| (impact on Ea prediction)')
    plt.title(f"Activation Energy Top {top_n} Important Features", fontsize=16)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    save_plot_multi_format(f"{output_dir}/06_top_features_bar_plot", dpi=600, bbox_inches='tight')
    plt.close()
    
    # 7. Save the top significant features to a text file
    print("Saving significant features to text file...")
    with open(f"{output_dir}/07_significant_features.txt", "w", encoding='utf-8') as f:
        f.write("Top Significant Features for Activation Energy (Ea) Prediction (by SHAP value magnitude):\n")
        f.write("=" * 80 + "\n\n")
        for i, feature in enumerate(significant_features):
            importance = top_importance[i]
            f.write(f"{i+1}. {feature}: {importance:.6f}\n")

def save_feature_names(feature_names, output_dir):
    """
    Save feature names to a text file
    
    Args:
        feature_names: List of feature names
        output_dir: Directory to save the file
    """
    with open(f"{output_dir}/00_feature_names_used.txt", "w", encoding='utf-8') as f:
        f.write("Feature Names Used in SHAP Analysis for Activation Energy (Ea) Prediction:\n")
        f.write("=" * 70 + "\n\n")
        for i, name in enumerate(feature_names):
            f.write(f"{i+1}. {name}\n")

def save_shap_to_excel(shap_values, expected_value, feature_names, output_dir):
    """
    Convert SHAP values and expected value to Excel format for Ea prediction
    
    Args:
        shap_values: SHAP values array
        expected_value: Expected value (baseline)
        feature_names: List of feature names
        output_dir: Directory to save Excel files
    """
    print("Converting SHAP values for Ea to Excel format...")
    
    try:
        # Create DataFrame for SHAP values with feature names as columns
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        
        # Add expected value to a separate DataFrame
        expected_df = pd.DataFrame({'expected_value': [expected_value]})
        
        # Create an Excel writer object
        excel_path = f"{output_dir}/shap_values_data.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Write SHAP values to sheet
            shap_df.to_excel(writer, sheet_name="Ea_SHAP_values", index=True)
            
            # Write expected value to its own sheet
            expected_df.to_excel(writer, sheet_name="Ea_Expected_Value", index=False)
            
            # Add a summary sheet with feature importance
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Mean_Abs_SHAP': np.abs(shap_values).mean(0),
                'Max_Abs_SHAP': np.abs(shap_values).max(0)
            })
            feature_importance = feature_importance.sort_values('Mean_Abs_SHAP', ascending=False)
            feature_importance.to_excel(writer, sheet_name="Ea_Feature_Importance", index=False)
        
        print(f"Successfully saved SHAP values to Excel: {excel_path}")
    
    except Exception as e:
        print(f"Error saving SHAP values to Excel: {e}")

def run_shap_analysis(matlab_file, debug=False):
    """
    Run SHAP analysis for activation energy (Ea) prediction model
    
    Args:
        matlab_file: Path to MATLAB .mat file
        debug: Whether to print detailed debugging information
    """
    
    # Enable/disable debugging output
    global DEBUG_MODE
    DEBUG_MODE = debug
    if DEBUG_MODE:
        print("DEBUG MODE ENABLED - Will print detailed debugging information")
    
    # Create a single organized output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = f'SHAP_Analysis_Ea_Results_{timestamp}'
    os.makedirs(main_output_dir, exist_ok=True)
    print(f"Creating main output directory: {main_output_dir}")
    
    # Create a README file explaining the analysis
    with open(f"{main_output_dir}/00_EA_ANALYSIS_NOTE.txt", "w", encoding='utf-8') as f:
        f.write("SHAP Analysis for Activation Energy (Ea) Prediction\n")
        f.write("===============================================\n\n")
        f.write("This analysis interprets a neural network model that predicts activation energy\n")
        f.write("for pyrolysis reactions based on feedstock properties and process conditions.\n\n")
        f.write("The model predicts activation energy in kJ/mol units.\n")
        f.write("All visualizations show feature contributions to Ea prediction.\n\n")
        f.write("FILE FORMATS\n")
        f.write("===========\n\n")
        f.write("All plots are provided in three formats:\n")
        f.write("1. PNG - Raster format for screen viewing and web use\n")
        f.write("2. SVG - Vector format optimized for Adobe Illustrator editing\n")
        f.write("3. EPS - Vector format for publication-quality graphics\n\n")
        f.write("For Adobe Illustrator users:\n")
        f.write("- SVG files offer the best compatibility and editing capability\n")
        f.write("- EPS files are optimized with TrueType fonts for better compatibility\n")
        f.write("- Both vector formats maintain quality at any scale and are suitable for publication\n")
    
    # Create a log file to capture errors and warnings
    log_file = os.path.join(main_output_dir, "00_analysis_log.txt")
    with open(log_file, "w", encoding='utf-8') as log:
        log.write(f"SHAP Analysis Log for Ea Prediction - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write("="*90 + "\n\n")
        
        try:
            # Load MATLAB data
            log.write("Loading MATLAB data...\n")
            mat_data = load_matlab_data(matlab_file)
            if not mat_data:
                log.write("ERROR: Failed to load MATLAB data.\n")
                print("Failed to load MATLAB data.")
                return
            
            # If debug mode is enabled, inspect the structure of the loaded MATLAB data
            if DEBUG_MODE:
                print("\nDEBUG: MATLAB Data Structure:")
                for key, value in mat_data.items():
                    print(f"Key: {key}, Type: {type(value)}")
                    if key == 'net':
                        print("  Net attributes:", [attr for attr in dir(value) if not attr.startswith('__')])
            
            # Extract input/output data and network structure
            try:
                log.write("Extracting neural network data...\n")
                X, y, network_structure = extract_neural_network_data(mat_data)
                log.write(f"Success: X shape={X.shape}, y shape={y.shape}\n")
            except Exception as e:
                log.write(f"ERROR: Failed to extract data from MATLAB file: {e}\n")
                print(f"Error extracting data from MATLAB file: {e}")
                return
            
            # Generate feature names
            feature_names = generate_feature_names(X, mat_data)
            log.write(f"Generated {len(feature_names)} feature names\n")
            
            # Save feature names to main directory
            save_feature_names(feature_names, main_output_dir)
            
            # Use all data for SHAP analysis (no train/test split)
            log.write(f"Using all data for analysis: X={X.shape}, y={y.shape}\n")
            
            # Create DataFrame with feature names for all data
            X_all_df = pd.DataFrame(X, columns=feature_names)
            
            # Set the number of samples for SHAP analysis
            sample_size = min(1000, X.shape[0])  # Use up to 1000 samples for SHAP analysis
            X_analysis = X_all_df.iloc[:sample_size]
            
            log.write(f"\nProcessing Activation Energy (Ea) prediction...\n")
            print(f"\nProcessing Activation Energy (Ea) prediction...")
            
            try:
                # Create a wrapper for the MATLAB neural network model
                log.write("Creating MATLAB neural network wrapper...\n")
                model = MatlabNeuralNetworkWrapper(network_structure)
                log.write(f"Model type: {model.model_type}\n")
                
                # Use a subset of all data for background (deterministic sampling)
                num_background = min(100, X.shape[0])
                log.write(f"Using {num_background} background samples for KernelExplainer (deterministic)\n")
                # Deterministic sampling with global seed for reproducibility
                X_background = shap.sample(X, num_background, random_state=GLOBAL_SEED)
                X_background_df = pd.DataFrame(X_background, columns=feature_names)
                
                # ------------------------------------------------------------------
                # Use the newer, faster generic SHAP Explainer with an Independent
                # masker.  This mirrors the implementation in shap_analysis.py and
                # provides deterministic explanations when combined with GLOBAL_SEED.
                # ------------------------------------------------------------------

                log.write("Creating SHAP Explainer with Independent masker (permutation algorithm)...\n")

                masker = shap.maskers.Independent(X_background_df)

                explainer = shap.Explainer(
                    model.predict,
                    masker,
                    algorithm="permutation",
                    seed=GLOBAL_SEED,
                )

                # Calculate SHAP values for the data samples
                log.write(f"Calculating SHAP values for {sample_size} data samples...\n")

                explanation = explainer(X_analysis)
                shap_values = explanation.values  # ndarray of shape (n_samples, n_features)

                # ------------------------------------------------------------------
                # Robustly determine the model's expected value (baseline prediction)
                # PermutationExplainer doesn't define .expected_value, so we derive it
                # from explanation.base_values and then attach it back for downstream
                # compatibility with the existing plotting utilities.
                # ------------------------------------------------------------------

                try:
                    expected_value_local = explainer.expected_value  # type: ignore[attr-defined]
                except AttributeError:
                    base_vals = explanation.base_values
                    if isinstance(base_vals, (list, np.ndarray)):
                        base_arr = np.array(base_vals)
                        # Take the first element (scalar) for a single-output model
                        expected_value_local = float(base_arr.flatten()[0])
                    else:
                        expected_value_local = float(base_vals)

                    # Attach for compatibility so existing code can use explainer.expected_value
                    setattr(explainer, 'expected_value', expected_value_local)

                log.write(f"SHAP values shape: {np.array(shap_values).shape}\n")
                
                # Save the model type information
                with open(f"{main_output_dir}/00_model_info.txt", "w", encoding='utf-8') as f:
                    f.write(f"Model Type: {model.model_type}\n")
                    f.write("Target: Activation Energy (Ea) in kJ/mol\n")
                    f.write("Network Architecture: Based on bpDNN4Ea.m (42-42 hidden layers)\n")
                
                # Save SHAP values and expected value for later use
                np.save(f"{main_output_dir}/shap_values.npy", shap_values)
                np.save(f"{main_output_dir}/expected_value.npy", explainer.expected_value)
                
                # Save to Excel format
                save_shap_to_excel(shap_values, explainer.expected_value, feature_names, main_output_dir)
                log.write("Successfully saved SHAP values to Excel\n")
                
                # Create all SHAP plots
                try:
                    log.write("Creating SHAP plots...\n")
                    create_shap_plots(explainer, shap_values, X_analysis, main_output_dir)
                    log.write("Successfully created all plots\n")
                except Exception as e:
                    log.write(f"ERROR creating plots: {e}\n")
                    print(f"Error creating plots: {e}")
                
                log.write("SHAP analysis for Ea prediction completed successfully.\n")
            
            except Exception as e:
                log.write(f"ERROR during SHAP analysis for Ea: {e}\n")
                print(f"Error during SHAP analysis for Ea: {e}")
            
            # Create a comprehensive README file in the output directory
            with open(f"{main_output_dir}/00_README.txt", "w", encoding='utf-8') as f:
                f.write("SHAP Analysis Results for Activation Energy (Ea) Prediction\n")
                f.write("=========================================================\n\n")
                f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Target Variable: Activation Energy (Ea) in kJ/mol\n")
                f.write("Model: Neural Network trained with bpDNN4Ea.m\n")
                f.write("Analysis: Performed using ALL sample data without train/test splitting\n\n")
                f.write("File Formats:\n")
                f.write("------------\n")
                f.write("All plots are provided in three formats:\n")
                f.write("- PNG: Raster format (pixel-based) good for screen viewing and web use\n")
                f.write("- SVG: Vector format optimized for Adobe Illustrator editing\n")
                f.write("- EPS: Vector format for publication-quality graphics that maintain\n")
                f.write("       quality when resized or printed at high resolution\n\n")
                f.write("Directory Structure:\n")
                f.write("-------------------\n")
                f.write("00_README.txt - This file\n")
                f.write("00_analysis_log.txt - Processing log with details and any errors\n")
                f.write("00_feature_names_used.txt - List of feature names used in the analysis\n")
                f.write("00_EA_ANALYSIS_NOTE.txt - Information about Ea prediction analysis\n")
                f.write("00_model_info.txt - Information about the model used\n")
                f.write("01_beeswarm_plot.png/svg/eps - Shows feature importance and impact direction\n")
                f.write("02_feature_importance_plot.png/svg/eps - Bar plot of feature importance\n")
                f.write("03_instance_explanation_plot.png/svg/eps - Force plot for a single instance\n")
                f.write("04_dependence_plot.png/svg/eps - Dependence plot for the most important feature\n")
                f.write("05_waterfall_plot_1.png/svg/eps to 05_waterfall_plot_5.png/svg/eps - Five waterfall plots\n")
                f.write("06_top_features_bar_plot.png/svg/eps - Top 20 important features\n")
                f.write("07_significant_features.txt - List of significant features\n")
                f.write("shap_values.npy - Saved SHAP values for future use\n")
                f.write("expected_value.npy - Saved expected value for future use\n")
                f.write("shap_values_data.xlsx - Excel file with SHAP values and feature importance\n")
                f.write("\nAnalysis completed.\n")
            
            log.write("\nEa SHAP analysis completed.\n")
        except Exception as e:
            log.write(f"CRITICAL ERROR in SHAP analysis: {e}\n")
            print(f"Critical error in SHAP analysis: {e}")
    
    print(f"\nEa SHAP analysis completed. Results saved to '{main_output_dir}' directory.")
    print(f"Check the log file {log_file} for details on any errors or warnings.")

def main():
    """Main function to run SHAP analysis for Ea prediction"""
    # Default MATLAB file path
    matlab_file = 'Results_trained.mat'
    
    # Parse command line arguments
    debug = False  # Default: debug mode off
    
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.lower() in ['-d', '--debug', 'debug']:
                debug = True
            elif arg.lower() in ['-h', '--help', 'help']:
                print("Usage: python shap_analysis_ea.py [-d|--debug]")
                print("  -d, --debug: Enable debug mode with additional output")
                print("\nOutput:")
                print("  - All visualizations are saved in PNG, SVG, and EPS formats")
                print("  - PNG: Raster format good for screen viewing")
                print("  - SVG: Vector format optimized for Adobe Illustrator editing")
                print("  - EPS: Vector format good for publication-quality graphics")
                print("  - Excel file with SHAP values and feature importance")
                return
    
    # Run SHAP analysis
    run_shap_analysis(matlab_file, debug)

# Define a global debug flag
DEBUG_MODE = False

if __name__ == "__main__":
    main() 