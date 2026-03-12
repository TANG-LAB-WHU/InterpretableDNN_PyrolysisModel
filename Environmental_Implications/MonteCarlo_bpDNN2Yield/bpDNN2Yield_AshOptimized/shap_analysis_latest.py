"""
SHAP Analysis for Pyrolysis Neural Network

This script loads a pre-trained neural network model from MATLAB .mat file
and performs SHAP (SHapley Additive exPlanations) analysis to interpret the model.

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
import logging
import datetime
import random  # for deterministic seeding

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import shap
import matplotlib

# -----------------------------------------------------------------------------
# Global configuration & utilities
# -----------------------------------------------------------------------------

# Global debug flag - declared at the top for proper access
DEBUG_MODE = False

# Logging configuration – INFO by default, DEBUG when –-debug flag supplied
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# Helper to print only in debug mode (keeps existing behaviour with minimal edits)
def dprint(msg: str):
    """Debug-print replacement that obeys DEBUG_MODE flag."""
    if DEBUG_MODE:
        logger.debug(msg)


# -----------------------------------------------------------------------------
# Deterministic seeding – ensures that *all* random operations (NumPy, Python,
# and SHAP) are repeatable across script runs.  This is critical for
# getting identical SHAP values, background sampling and beeswarm plots.
# -----------------------------------------------------------------------------

GLOBAL_SEED = 42

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
# SHAP has its own RNG wrapper from v0.45 onwards
try:
    shap.random.seed(GLOBAL_SEED)  # type: ignore[attr-defined]
except AttributeError:
    # Older SHAP versions fall back to NumPy's RNG (already seeded)
    pass

# Base feature names from MATLAB script
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
    'TargetTemperature/Celsius',
    'ReactionTime/min',
    'HeatingRate/(K/min)',
    'ReactorType'
]

def load_matlab_data(file_path):
    """
    Load data from a MATLAB .mat file
    
    Args:
        file_path: Path to the .mat file
        
    Returns:
        Dictionary containing the loaded data
        
    Raises:
        FileNotFoundError: If the MATLAB file doesn't exist
        ValueError: If the file cannot be loaded or is empty
    """
    logger.info(f"Loading MATLAB file: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"MATLAB file not found: {file_path}")
    
    try:
        mat_data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        # Remove built-in fields
        cleaned_data = {k: v for k, v in mat_data.items() if not k.startswith('__')}
        
        if not cleaned_data:
            raise ValueError(f"MATLAB file appears to be empty or contains no valid data: {file_path}")
        
        logger.info(f"Successfully loaded MATLAB file with {len(cleaned_data)} variables")
        return cleaned_data
        
    except Exception as e:
        logger.error(f"Error loading MATLAB file: {e}")
        raise

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
    possible_input_names = ['X', 'Xnorm', 'x', 'input', 'inputs', 'nn_input', 'Variables', 'Variables0', 'Feedstock4training']
    possible_output_names = ['Y', 'y', 'Ynorm', 'output', 'outputs', 'target', 'targets', 'nn_output', 'ProductsYield']
    
    # First try standard variable names
    for input_name in possible_input_names:
        if input_name in mat_data:
            X = mat_data[input_name]
            dprint(f"Found input data in variable '{input_name}' with shape {X.shape}")
            break
    
    for output_name in possible_output_names:
        if output_name in mat_data:
            y = mat_data[output_name]
            dprint(f"Found target data in variable '{output_name}' with shape {y.shape}")
            break
    
    # Extract network structure if 'net' variable exists
    if 'net' in mat_data:
        network_structure = mat_data['net']
        dprint("Found network structure in 'net' variable")
    
    # If we couldn't find standard names, try to infer based on variable shapes
    if X is None or y is None:
        dprint("Could not find standard input/output variable names, trying to infer from shapes...")
        
        for var_name, var in mat_data.items():
            if not var_name.startswith('__') and isinstance(var, np.ndarray):
                # Input data likely has more columns than rows for features
                if len(var.shape) == 2 and var.shape[1] > 20:
                    if X is None or var.shape[1] > X.shape[1]:  # Prefer the one with more features
                        X = var
                        dprint(f"Inferred input data from variable '{var_name}' with shape {X.shape}")
                
                # Output data likely has fewer columns than input
                elif len(var.shape) == 2 and var.shape[1] <= 10:
                    if y is None:
                        y = var
                        dprint(f"Inferred target data from variable '{var_name}' with shape {y.shape}")
    
    # If data is still not found, raise an error
    if X is None or y is None:
        raise ValueError("Could not find or infer input and target data from the MATLAB file")
    
    # Check if the data is transposed (more features than samples)
    # Neural networks typically have more samples than features
    # In MATLAB, data may be stored with samples in columns and features in rows
    if X.shape[0] < X.shape[1]:
        dprint("Input data seems to be transposed (features in rows). Transposing to (samples, features)...")
        X = X.T
        dprint(f"Transposed input data shape: {X.shape}")
    
    if y.shape[0] < y.shape[1]:
        dprint("Target data seems to be transposed (outputs in rows). Transposing to (samples, outputs)...")
        y = y.T
        dprint(f"Transposed target data shape: {y.shape}")
    
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
    
    # If we have exactly the right number of base features, return them
    if total_features == base_count:
        dprint(f"Using {base_count} base feature names")
        return BASE_FEATURE_NAMES
    
    # Calculate feedstock count based on remaining features
    remaining_features = total_features - base_count
    if remaining_features % 2 != 0:
        dprint(f"Warning: Remaining feature count ({remaining_features}) is not even")
        # Generate generic names
        return [f"Feature_{i+1}" for i in range(total_features)]
    
    feedstock_count = remaining_features // 2  # Each feedstock type has both Type and Ratio features
    dprint(f"Detected {feedstock_count} feedstock types")
    
    # Generate complete feature list
    feature_names = BASE_FEATURE_NAMES.copy()
    
    # Add FeedstockType features
    for i in range(feedstock_count):
        feature_names.append(f'FeedstockType_{i+1}')
    
    # Add MixingRatio features
    for i in range(feedstock_count):
        feature_names.append(f'MixingRatio_{i+1}')
    
    return feature_names

class MatlabNeuralNetworkWrapper:
    """
    Wrapper for MATLAB neural network to work with SHAP
    
    This wrapper provides a predict method that uses the MATLAB neural network
    to make predictions. It handles different network structures and includes
    better error handling.
    """
    def __init__(self, net_struct, target_idx=0):
        """
        Initialize the wrapper with the MATLAB neural network structure
        
        Args:
            net_struct: MATLAB neural network structure
            target_idx: Index of the target variable to predict (default: 0 for Biochar yield)
        """
        self.net_struct = net_struct
        self.target_idx = target_idx
        self.model_type = "matlab_nn"  # Only use MATLAB neural network
        
        # Debug the network structure to understand what we're working with
        self._debug_network_structure()
        
        # Check if a simple predict method is directly available
        if hasattr(net_struct, 'predict'):
            dprint(f"Found direct predict method in MATLAB neural network")
            self.predict_method = self._direct_predict
        else:
            # Try to extract weights and biases using different approaches
            dprint(f"Attempting to extract network architecture from MATLAB structure...")
            
            # NEW APPROACH: BP Neural Network with cell arrays (matched to screenshot)
            if self._extract_bp_neural_network():
                dprint("Successfully extracted network using BP neural network format with cell arrays")
                self.predict_method = self._forward_pass_predict
            
            # APPROACH 1: Standard Neural Network Toolbox structure
            elif self._extract_standard_nn_structure():
                dprint("Successfully extracted network using standard Neural Network Toolbox structure")
                self.predict_method = self._forward_pass_predict
            
            # APPROACH 2: Deep Learning Toolbox structure
            elif self._extract_deep_learning_toolbox_structure():
                dprint("Successfully extracted network using Deep Learning Toolbox structure")
                self.predict_method = self._forward_pass_predict
            
            # APPROACH 3: Legacy MATLAB Neural Network structure
            elif self._extract_legacy_nn_structure():
                dprint("Successfully extracted network using legacy MATLAB NN structure")
                self.predict_method = self._matlab_legacy_predict
            
            # APPROACH 4: Direct weights extraction
            elif self._extract_direct_weights():
                dprint("Successfully extracted weights directly from network structure")
                self.predict_method = self._forward_pass_predict
                
            # APPROACH 5: Custom structure extraction for this specific network
            elif self._extract_custom_structure():
                dprint("Successfully extracted network using custom structure pattern")
                self.predict_method = self._custom_predict
                
            else:
                dprint("CRITICAL ERROR: Could not extract network architecture using any known method")
                dprint("Cannot proceed with SHAP analysis without a valid MATLAB model")
                raise AttributeError("Could not extract network architecture using any known method")
        
        # Test the prediction method on a dummy input
        test_input = np.ones((1, self.get_input_size()))
        test_output = self.predict(test_input)
        dprint(f"Test prediction successful. Output shape: {test_output.shape}")
    
    def _debug_network_structure(self):
        """Debug the network structure to understand available attributes"""
        try:
            dprint("\nDEBUG: Exploring MATLAB network structure attributes:")
            
            # List top-level attributes
            attributes = [attr for attr in dir(self.net_struct) if not attr.startswith('__')]
            dprint(f"Top-level attributes: {attributes}")
            
            # Check for common network attributes
            for attr in ['layers', 'Layers', 'layer', 'Layer', 'weights', 'Weights', 'biases', 'Biases',
                        'IW', 'LW', 'b', 'inputWeights', 'layerWeights', 'inputs', 'outputs']:
                if hasattr(self.net_struct, attr):
                    obj = getattr(self.net_struct, attr)
                    if isinstance(obj, (list, tuple, np.ndarray)):
                        dprint(f"Found array attribute '{attr}' with length {len(obj)}")
                    else:
                        dprint(f"Found attribute '{attr}' of type {type(obj)}")
            
            # If there's a 'layer' attribute that's an array, inspect the first layer
            if hasattr(self.net_struct, 'layer'):
                layers = self.net_struct.layer
                if hasattr(layers, '__len__') and len(layers) > 0:
                    first_layer = layers[0]
                    layer_attrs = [attr for attr in dir(first_layer) if not attr.startswith('__')]
                    dprint(f"Layer 0 attributes: {layer_attrs}")
                    
                    # Check for weight matrix attributes
                    for attr in layer_attrs:
                        if attr.lower() in ['weights', 'weight', 'w', 'kernel', 'bias', 'b']:
                            val = getattr(first_layer, attr)
                            if isinstance(val, np.ndarray):
                                dprint(f"  Layer 0 '{attr}' shape: {val.shape}")
            
            # Check for saved weights elsewhere
            if hasattr(self.net_struct, 'userdata'):
                dprint("Found 'userdata' attribute, might contain weights")
            
            if hasattr(self.net_struct, 'trainFcn'):
                dprint(f"Training function: {self.net_struct.trainFcn}")
                
        except Exception as e:
            dprint(f"Error during network structure debugging: {e}")
    
    def _extract_standard_nn_structure(self):
        """Extract weights and biases using standard Neural Network Toolbox structure"""
        try:
            # Common attributes for layers
            layer_attrs = ['layers', 'Layers', 'layer', 'Layer']
            self.layers = None
            
            # Find layers
            for attr in layer_attrs:
                if hasattr(self.net_struct, attr):
                    self.layers = getattr(self.net_struct, attr)
                    dprint(f"Found layers in attribute '{attr}'")
                    break
            
            if self.layers is None:
                return False
            
            # Initialize lists to store weights and biases
            self.weights = []
            self.biases = []
            self.activation_funcs = []
            
            # Common attribute names for weights, biases, and activation functions
            weight_attrs = ['Weights', 'weights', 'W', 'w', 'inputWeights', 'layerWeights']
            bias_attrs = ['Bias', 'bias', 'b', 'biases']
            activation_attrs = ['Activation', 'activation', 'TransferFcn', 'transferFcn', 'actFcn', 'act', 'activationFcn']
            
            # Extract weights, biases, and activation functions from each layer
            valid_layers = 0
            for i in range(len(self.layers)):
                layer = self.layers[i]
                
                # Extract weights
                weights_found = False
                for w_attr in weight_attrs:
                    if hasattr(layer, w_attr):
                        weight = getattr(layer, w_attr)
                        if isinstance(weight, np.ndarray):
                            self.weights.append(weight)
                            valid_layers += 1
                            weights_found = True
                            dprint(f"  Found weights for layer {i} in '{w_attr}' with shape {weight.shape}")
                            break
                
                if not weights_found:
                    dprint(f"  Warning: Could not find weights for layer {i}")
                    self.weights.append(None)
                
                # Extract biases
                bias_found = False
                for b_attr in bias_attrs:
                    if hasattr(layer, b_attr):
                        bias = getattr(layer, b_attr)
                        if isinstance(bias, np.ndarray):
                            self.biases.append(bias)
                            bias_found = True
                            dprint(f"  Found bias for layer {i} in '{b_attr}' with shape {bias.shape}")
                            break
                
                if not bias_found:
                    dprint(f"  Warning: Could not find bias for layer {i}")
                    self.biases.append(None)
                
                # Extract activation functions
                activation_found = False
                for act_attr in activation_attrs:
                    if hasattr(layer, act_attr):
                        act_func = getattr(layer, act_attr)
                        self.activation_funcs.append(act_func)
                        activation_found = True
                        dprint(f"  Found activation for layer {i}: {act_func}")
                        break
                
                if not activation_found:
                    # Default to linear activation
                    self.activation_funcs.append('linear')
                    dprint(f"  No activation found for layer {i}, defaulting to 'linear'")
            
            # Check if we found any valid layers
            return valid_layers > 0
            
        except Exception as e:
            dprint(f"Error extracting standard NN structure: {e}")
            return False
            
    def _extract_deep_learning_toolbox_structure(self):
        """Extract weights using MATLAB Deep Learning Toolbox structure"""
        try:
            # Check if we have the deep learning structure
            if hasattr(self.net_struct, 'Layers'):
                dl_layers = self.net_struct.Layers
                self.weights = []
                self.biases = []
                self.activation_funcs = []
                
                # Process each layer
                valid_layers = 0
                
                # Check if Layers is a list-like object
                if hasattr(dl_layers, '__len__'):
                    for i, layer in enumerate(dl_layers):
                        # Skip non-weighted layers like input or output layers
                        if hasattr(layer, 'Weights') and hasattr(layer, 'Bias'):
                            self.weights.append(layer.Weights)
                            self.biases.append(layer.Bias)
                            valid_layers += 1
                            dprint(f"  Found weights/bias for DL layer {i}")
                            
                            # Determine activation function
                            if hasattr(layer, 'ActivationFunction'):
                                self.activation_funcs.append(layer.ActivationFunction)
                            else:
                                # Try to infer from layer name
                                layer_name = layer.__class__.__name__ if hasattr(layer, '__class__') else ""
                                if 'relu' in layer_name.lower():
                                    self.activation_funcs.append('relu')
                                elif 'sigmoid' in layer_name.lower():
                                    self.activation_funcs.append('sigmoid')
                                elif 'tanh' in layer_name.lower():
                                    self.activation_funcs.append('tanh')
                                else:
                                    self.activation_funcs.append('linear')
                        else:
                            # Add placeholders for non-weighted layers
                            self.weights.append(None)
                            self.biases.append(None)
                            self.activation_funcs.append('linear')
                
                return valid_layers > 0
            
            return False
            
        except Exception as e:
            dprint(f"Error extracting Deep Learning Toolbox structure: {e}")
            return False
            
    def _extract_legacy_nn_structure(self):
        """Extract using the traditional MATLAB neural network structure (IW, LW, b)"""
        try:
            # Check if we have the legacy structure
            if (hasattr(self.net_struct, 'IW') and hasattr(self.net_struct, 'LW') and 
                hasattr(self.net_struct, 'b')):
                
                dprint("Found IW, LW, b structure (traditional MATLAB NN format)")
                self.iw = self.net_struct.IW
                self.lw = self.net_struct.LW
                self.b = self.net_struct.b
                
                # Additionally store transfer functions if available
                if hasattr(self.net_struct, 'transferFcn'):
                    self.transfer_functions = self.net_struct.transferFcn
                elif hasattr(self.net_struct, 'layers') and hasattr(self.net_struct.layers[0], 'transferFcn'):
                    self.transfer_functions = [layer.transferFcn for layer in self.net_struct.layers]
                else:
                    # Default to tansig for hidden layers and purelin for output
                    if hasattr(self.net_struct.b, '__len__'):
                        self.transfer_functions = ['tansig'] * (len(self.net_struct.b) - 1) + ['purelin']
                    else:
                        self.transfer_functions = ['tansig', 'purelin']
                
                return True
            
            return False
            
        except Exception as e:
            dprint(f"Error extracting legacy NN structure: {e}")
            return False
    
    def _extract_direct_weights(self):
        """Look for direct weights and biases without assuming specific structure"""
        try:
            # Look for arrays that might be weights and biases at the top level
            weights_candidates = []
            biases_candidates = []
            
            for attr in dir(self.net_struct):
                if attr.startswith('__'):
                    continue
                    
                val = getattr(self.net_struct, attr)
                
                # Check if it's an array
                if isinstance(val, np.ndarray):
                    # Weights are usually 2D, biases 1D
                    if len(val.shape) == 2 and val.shape[0] > 1 and val.shape[1] > 1:
                        weights_candidates.append((attr, val))
                        dprint(f"  Found potential weights in '{attr}' with shape {val.shape}")
                    elif len(val.shape) == 1 or (len(val.shape) == 2 and (val.shape[0] == 1 or val.shape[1] == 1)):
                        biases_candidates.append((attr, val))
                        dprint(f"  Found potential biases in '{attr}' with shape {val.shape}")
                        
                # Check if it's a list of arrays
                elif isinstance(val, (list, tuple)) and len(val) > 0:
                    for i, item in enumerate(val):
                        if isinstance(item, np.ndarray):
                            if len(item.shape) == 2 and item.shape[0] > 1 and item.shape[1] > 1:
                                weights_candidates.append((f"{attr}[{i}]", item))
                                dprint(f"  Found potential weights in '{attr}[{i}]' with shape {item.shape}")
                            elif len(item.shape) == 1 or (len(item.shape) == 2 and (item.shape[0] == 1 or item.shape[1] == 1)):
                                biases_candidates.append((f"{attr}[{i}]", item))
                                dprint(f"  Found potential biases in '{attr}[{i}]' with shape {item.shape}")
            
            # If we found potential weights and biases, organize them
            if len(weights_candidates) > 0 and len(biases_candidates) > 0:
                dprint(f"Found {len(weights_candidates)} potential weight arrays and {len(biases_candidates)} potential bias arrays")
                
                # For simplicity, just use the arrays we found in order
                self.weights = [w[1] for w in weights_candidates]
                self.biases = [b[1] for b in biases_candidates]
                
                # Assume ReLU activation for hidden layers and linear for output
                self.activation_funcs = ['relu'] * (len(self.weights) - 1) + ['linear']
                
                return True
            
            return False
            
        except Exception as e:
            dprint(f"Error during direct weight extraction: {e}")
            return False
    
    def _extract_custom_structure(self):
        """
        Extract weights using custom structure patterns specific to this MATLAB model
        
        This is a fallback method when standard approaches fail. It looks for network
        parameters in common locations specific to certain MATLAB neural network exports.
        """
        try:
            # Pattern 0: Check for MATLAB BP neural network format (weight and bias at top level)
            # This seems to be the format used in the current model
            if (hasattr(self.net_struct, 'weight') and hasattr(self.net_struct, 'bias') and 
                hasattr(self.net_struct, 'layer')):
                
                dprint("Found MATLAB BP neural network format with top-level weight and bias arrays")
                
                # Get weight and bias arrays
                weight_array = self.net_struct.weight
                bias_array = self.net_struct.bias
                layers = self.net_struct.layer
                
                # Check if they're properly shaped arrays
                dprint(f"  Weight array shape: {weight_array.shape if hasattr(weight_array, 'shape') else 'unknown'}")
                dprint(f"  Bias array shape: {bias_array.shape if hasattr(bias_array, 'shape') else 'unknown'}")
                
                # Get the number of layers
                num_layers = len(layers)
                dprint(f"  Network has {num_layers} layers")
                
                # Initialize lists for weights, biases, and activation functions
                self.weights = []
                self.biases = []
                self.activation_funcs = []
                
                # Get layer input/output sizes to determine weight matrices dimensions
                layer_sizes = []
                for i in range(num_layers):
                    if hasattr(layers[i], 'size'):
                        layer_sizes.append(layers[i].size)
                        dprint(f"  Layer {i} size: {layers[i].size}")
                
                # Add input size at the beginning
                if hasattr(self.net_struct, 'numInput'):
                    input_size = self.net_struct.numInput
                    layer_sizes.insert(0, input_size)
                    dprint(f"  Input size: {input_size}")
                
                # If we have layer sizes, we can reconstruct the weight matrices
                if len(layer_sizes) >= 2:
                    # Check if weight_array is a cell array (common in MATLAB neural networks)
                    # This seems to be the case in the current model
                    if hasattr(weight_array, '__len__') and len(weight_array) == num_layers:
                        dprint("  Weight array appears to be a cell array with one entry per layer")
                        
                        for i in range(num_layers):
                            try:
                                # In MATLAB cell arrays, each cell can contain a matrix
                                # Extract each cell's contents as a numpy array
                                if hasattr(weight_array[i], '__array__') or isinstance(weight_array[i], np.ndarray):
                                    w = weight_array[i]
                                    b = bias_array[i]
                                    
                                    dprint(f"  Direct access - Layer {i} weight shape: {w.shape if hasattr(w, 'shape') else 'unknown'}")
                                    dprint(f"  Direct access - Layer {i} bias shape: {b.shape if hasattr(b, 'shape') else 'unknown'}")
                                    
                                    self.weights.append(w)
                                    self.biases.append(b)
                                else:
                                    dprint(f"  Warning: Weight for layer {i} is not a numpy array")
                                    self.weights.append(None)
                                    self.biases.append(None)
                            except Exception as e:
                                dprint(f"  Error accessing weight/bias for layer {i}: {e}")
                                self.weights.append(None)
                                self.biases.append(None)
                    
                    # If not a cell array, check if it's a flattened vector that needs to be reshaped
                    elif len(weight_array.shape) == 1:
                        dprint("  Weights stored as flattened vector, reconstructing matrices...")
                        
                        # In this case, we need to calculate the indices for splitting
                        weight_indices = []
                        bias_indices = []
                        weight_idx = 0
                        bias_idx = 0
                        
                        for i in range(num_layers):
                            if i < len(layer_sizes) - 1:
                                # Calculate number of weights for this layer
                                num_weights = layer_sizes[i] * layer_sizes[i+1]
                                weight_indices.append((weight_idx, weight_idx + num_weights))
                                weight_idx += num_weights
                                
                                # Calculate number of biases for this layer
                                num_biases = layer_sizes[i+1]
                                bias_indices.append((bias_idx, bias_idx + num_biases))
                                bias_idx += num_biases
                        
                        # Now extract and reshape weights and biases for each layer
                        for i in range(num_layers):
                            if i < len(weight_indices):
                                start_w, end_w = weight_indices[i]
                                start_b, end_b = bias_indices[i]
                                
                                if end_w <= len(weight_array) and end_b <= len(bias_array):
                                    # Extract and reshape weight matrix
                                    w = weight_array[start_w:end_w]
                                    w = w.reshape(layer_sizes[i+1], layer_sizes[i])  # Shape as [output, input]
                                    self.weights.append(w)
                                    
                                    # Extract bias vector
                                    b = bias_array[start_b:end_b]
                                    self.biases.append(b)
                                    
                                    dprint(f"  Reconstructed weights for layer {i}: {w.shape}")
                                    dprint(f"  Reconstructed bias for layer {i}: {b.shape}")
                                else:
                                    dprint(f"  Warning: Index out of range for layer {i}")
                                    self.weights.append(None)
                                    self.biases.append(None)
                    else:
                        # If weight_array contains any other format
                        dprint("  Weight array format not recognized")
                        return False
                
                # Get activation functions for each layer
                for i in range(num_layers):
                    if hasattr(layers[i], 'transferFcn'):
                        act_func = layers[i].transferFcn
                        self.activation_funcs.append(act_func)
                        dprint(f"  Layer {i} activation: {act_func}")
                    else:
                        # Default to reasonable activation based on layer position
                        if i < num_layers - 1:
                            self.activation_funcs.append('logsig')  # Hidden layers
                        else:
                            self.activation_funcs.append('purelin')  # Output layer
                
                # Check if we have all necessary components
                valid_weights = [w for w in self.weights if w is not None]
                if valid_weights:
                    return True
            
            # Continue with original Pattern 1-4...
            # Pattern 1: Check for 'net' structure with embedded 'userdata' containing matrices
            if hasattr(self.net_struct, 'userdata'):
                userdata = self.net_struct.userdata
                
                # Look for arrays in userdata
                if isinstance(userdata, (dict, object)):
                    weights_list = []
                    biases_list = []
                    
                    # If it's a dictionary-like object
                    for key in dir(userdata):
                        if key.startswith('__'):
                            continue
                            
                        val = getattr(userdata, key)
                        
                        # Look for weight-like arrays
                        if isinstance(val, np.ndarray):
                            if len(val.shape) == 2 and min(val.shape) > 1:
                                weights_list.append(val)
                                dprint(f"  Found weights in userdata.{key} with shape {val.shape}")
                            elif len(val.shape) == 1 or (len(val.shape) == 2 and min(val.shape) == 1):
                                biases_list.append(val)
                                dprint(f"  Found biases in userdata.{key} with shape {val.shape}")
                    
                    if weights_list and biases_list:
                        self.weights = weights_list
                        self.biases = biases_list
                        self.activation_funcs = ['relu'] * (len(weights_list) - 1) + ['linear']
                        return True
            
            # Pattern 2: Look for NetworkWeights/NetworkBiases structure
            weight_containers = ['NetworkWeights', 'networkWeights', 'weights', 'Weights', 'NetWeights']
            bias_containers = ['NetworkBiases', 'networkBiases', 'biases', 'Biases', 'NetBiases']
            
            for wc in weight_containers:
                if hasattr(self.net_struct, wc):
                    weight_container = getattr(self.net_struct, wc)
                    if isinstance(weight_container, (list, tuple, np.ndarray)) and len(weight_container) > 0:
                        # Extract weights
                        self.weights = weight_container
                        dprint(f"  Found weights in {wc} with {len(weight_container)} layers")
                        
                        # Try to find matching biases
                        for bc in bias_containers:
                            if hasattr(self.net_struct, bc):
                                bias_container = getattr(self.net_struct, bc)
                                if isinstance(bias_container, (list, tuple, np.ndarray)) and len(bias_container) > 0:
                                    self.biases = bias_container
                                    dprint(f"  Found biases in {bc} with {len(bias_container)} layers")
                                    
                                    # Assume ReLU for hidden, linear for output
                                    self.activation_funcs = ['relu'] * (len(self.weights) - 1) + ['linear']
                                    return True
            
            # Pattern 3: Look for serialized weights in 'parameters', 'params' or similar
            param_attrs = ['parameters', 'params', 'Parameters', 'trainParam', 'trained', 'serialized']
            for param_attr in param_attrs:
                if hasattr(self.net_struct, param_attr):
                    params = getattr(self.net_struct, param_attr)
                    # Try to parse the parameters object
                    weights = []
                    biases = []
                    
                    # Check if it's a structured object
                    if hasattr(params, '__dict__') or hasattr(params, '__dir__'):
                        # Look for weight and bias arrays
                        for key in dir(params):
                            if key.startswith('__'):
                                continue
                                
                            val = getattr(params, key)
                            
                            # Check for weight-like arrays
                            if isinstance(val, np.ndarray):
                                if len(val.shape) == 2 and min(val.shape) > 1:
                                    weights.append(val)
                                    dprint(f"  Found weights in {param_attr}.{key} with shape {val.shape}")
                                elif len(val.shape) == 1 or (len(val.shape) == 2 and min(val.shape) == 1):
                                    biases.append(val)
                                    dprint(f"  Found biases in {param_attr}.{key} with shape {val.shape}")
                            
                            # Check for list/tuple of arrays
                            elif isinstance(val, (list, tuple)) and len(val) > 0 and all(isinstance(x, np.ndarray) for x in val):
                                # Check if these look like weights or biases
                                if all(len(x.shape) == 2 and min(x.shape) > 1 for x in val):
                                    weights = val
                                    dprint(f"  Found weight list in {param_attr}.{key} with {len(val)} layers")
                                elif all(len(x.shape) == 1 or (len(x.shape) == 2 and min(x.shape) == 1) for x in val):
                                    biases = val
                                    dprint(f"  Found bias list in {param_attr}.{key} with {len(val)} layers")
                        
                        if weights and biases:
                            self.weights = weights
                            self.biases = biases
                            self.activation_funcs = ['relu'] * (len(weights) - 1) + ['linear']
                            return True
            
            # Pattern 4: Look for getters/accessors for weights and biases
            methods = [m for m in dir(self.net_struct) if callable(getattr(self.net_struct, m)) and not m.startswith('__')]
            getter_methods = [m for m in methods if m.startswith('get') or 'weight' in m.lower() or 'bias' in m.lower()]
            
            if getter_methods:
                dprint(f"Found potential getter methods: {getter_methods}")
                # Try to call getters to extract weights/biases (careful with this)
                weights = []
                biases = []
                
                for method in getter_methods:
                    try:
                        result = getattr(self.net_struct, method)()
                        if isinstance(result, np.ndarray):
                            if len(result.shape) == 2 and min(result.shape) > 1:
                                weights.append(result)
                                dprint(f"  Found weights from method {method} with shape {result.shape}")
                            elif len(result.shape) == 1 or (len(result.shape) == 2 and min(result.shape) == 1):
                                biases.append(result)
                                dprint(f"  Found biases from method {method} with shape {result.shape}")
                    except:
                        pass  # Ignore errors when calling methods
                
                if weights and biases:
                    self.weights = weights
                    self.biases = biases
                    self.activation_funcs = ['relu'] * (len(weights) - 1) + ['linear']
                    return True
            
            return False
            
        except Exception as e:
            dprint(f"Error during custom structure extraction: {e}")
            return False
    
    def _custom_predict(self, X):
        """
        Implement custom prediction for unique network structures
        """
        try:
            # Implement prediction based on what we found in _extract_custom_structure
            # This will vary based on what was extracted
            if hasattr(self, 'weights') and hasattr(self, 'biases'):
                # Forward pass through the network
                a = X
                for i in range(len(self.weights)):
                    # Linear transformation: a = a * W.T + b
                    z = np.dot(a, self.weights[i].T) + self.biases[i]
                    
                    # Apply activation function
                    if i < len(self.weights) - 1:  # Hidden layer
                        if self.activation_funcs[i] == 'relu':
                            a = np.maximum(0, z)
                        elif self.activation_funcs[i] == 'sigmoid' or self.activation_funcs[i] == 'logsig':
                            a = 1 / (1 + np.exp(-z))
                        elif self.activation_funcs[i] == 'tanh' or self.activation_funcs[i] == 'tansig':
                            a = np.tanh(z)
                        else:
                            a = z  # Linear
                    else:  # Output layer
                        a = z  # Usually linear activation for regression
                
                # If we have multiple outputs, select the target
                if len(a.shape) > 1 and a.shape[1] > self.target_idx:
                    return a[:, self.target_idx]
                else:
                    return a.ravel()
            
            # Fallback to zeros if we don't have weights/biases
            return np.zeros(X.shape[0])
            
        except Exception as e:
            dprint(f"Error in custom prediction: {e}")
            return np.zeros(X.shape[0])
    
    def get_input_size(self):
        """
        Determine the input size of the network from the structure
        
        Returns:
            Integer representing the input size
        """
        try:
            # Try common patterns
            if hasattr(self.net_struct, 'inputs') and hasattr(self.net_struct.inputs, 'size'):
                return self.net_struct.inputs.size
            elif hasattr(self, 'weights') and len(self.weights) > 0 and self.weights[0] is not None:
                if isinstance(self.weights[0], np.ndarray):
                    if len(self.weights[0].shape) > 1:
                        return self.weights[0].shape[1]  # Assuming weights has shape [neurons, features]
                    else:
                        return self.weights[0].shape[0]
                elif hasattr(self.weights[0], 'shape'):
                    return self.weights[0].shape[1] if len(self.weights[0].shape) > 1 else self.weights[0].shape[0]
            elif hasattr(self, 'iw') and hasattr(self.iw, '__len__') and len(self.iw) > 0:
                return self.iw[0][0].shape[1]
            elif hasattr(self.net_struct, 'inputSize'):
                return self.net_struct.inputSize
            elif hasattr(self.net_struct, 'input') and hasattr(self.net_struct.input, 'size'):
                return self.net_struct.input.size
            
            # Check layers for input size
            if hasattr(self.net_struct, 'layer') and len(self.net_struct.layer) > 0:
                first_layer = self.net_struct.layer[0]
                if hasattr(first_layer, 'inputSize'):
                    return first_layer.inputSize
                elif hasattr(first_layer, 'dimensions') and hasattr(first_layer.dimensions, 'input'):
                    return first_layer.dimensions.input
            
            # Last resort - check for any attribute that might indicate input size
            for attr in dir(self.net_struct):
                if attr.lower().endswith('inputsize') or attr.lower().endswith('inputdim'):
                    val = getattr(self.net_struct, attr)
                    if isinstance(val, (int, float)):
                        return int(val)
                
        except Exception as e:
            dprint(f"Error determining input size: {e}")
        
        # Default fallback size
        dprint("Using default input size of 259")
        return 259  # Based on the example data

    def _direct_predict(self, X):
        """
        Use the network's built-in predict method
        """
        try:
            result = self.net_struct.predict(X)
            if isinstance(result, np.ndarray) and len(result.shape) > 1 and result.shape[1] > self.target_idx:
                return result[:, self.target_idx]
            elif isinstance(result, np.ndarray):
                return result.ravel()
            else:
                return np.array([result] * X.shape[0])
        except Exception as e:
            dprint(f"Error in direct prediction: {e}")
            return np.zeros(X.shape[0])
    
    def _forward_pass_predict(self, X):
        """
        Implement forward pass through the network using extracted weights and biases
        """
        try:
            # Forward pass through the network
            a = X
            for i in range(len(self.weights)):
                if self.weights[i] is None or self.biases[i] is None:
                    continue
                
                try:
                    # Linear transformation: a = a * W.T + b
                    # For MATLAB networks, dimensions are often [outputs, inputs] for weights
                    # So we need to transpose weights for numpy's dot product
                    z = np.dot(a, self.weights[i].T) + self.biases[i]
                    
                    # Apply activation function
                    activation = self.activation_funcs[i].lower() if isinstance(self.activation_funcs[i], str) else 'linear'
                    
                    # MATLAB activation functions
                    if activation == 'logsig':  # MATLAB sigmoid function
                        a = 1 / (1 + np.exp(-z))
                    elif activation == 'tansig':  # MATLAB tanh function
                        a = np.tanh(z)
                    elif activation == 'purelin':  # MATLAB linear function
                        a = z
                    # Python/common activation functions
                    elif activation == 'relu' or activation == 'rectifiedlinear':
                        a = np.maximum(0, z)
                    elif activation == 'sigmoid':
                        a = 1 / (1 + np.exp(-z))
                    elif activation == 'tanh':
                        a = np.tanh(z)
                    else:
                        # Default to linear activation
                        a = z
                    
                    dprint(f"Layer {i} forward pass shape: {a.shape} using activation: {activation}")
                
                except Exception as e:
                    dprint(f"Error in forward pass at layer {i}: {e}")
                    # Skip to next layer if this one fails
                    continue
            
            # If we have multiple outputs, select the target
            if len(a.shape) > 1 and a.shape[1] > self.target_idx:
                return a[:, self.target_idx]
            else:
                return a.ravel()
        except Exception as e:
            dprint(f"Error in forward pass prediction: {e}")
            return np.zeros(X.shape[0])
    
    def _matlab_legacy_predict(self, X):
        """
        Use the traditional MATLAB neural network structure (IW, LW, b)
        """
        try:
            # Implement prediction using IW (input weights), LW (layer weights), and b (biases)
            net = self.net_struct
            
            # First layer
            a1 = np.tanh(np.dot(X, net.IW[0][0].T) + net.b[0])
            
            # Hidden layers
            for i in range(1, len(net.b)-1):
                a1 = np.tanh(np.dot(a1, net.LW[i][i-1].T) + net.b[i])
            
            # Output layer (linear activation)
            output = np.dot(a1, net.LW[len(net.b)-1][len(net.b)-2].T) + net.b[len(net.b)-1]
            
            # If we have multiple outputs, select the target
            if len(output.shape) > 1 and output.shape[1] > self.target_idx:
                return output[:, self.target_idx]
            else:
                return output.ravel()
        except Exception as e:
            dprint(f"Error in legacy MATLAB prediction: {e}")
            return np.zeros(X.shape[0])
    
    # RF fallback methods have been removed to ensure only MATLAB model is used
    
    def predict(self, X):
        """
        Predict using the selected method and ensure positive yield values
        
        Args:
            X: Input data
            
        Returns:
            Predictions for the target variable as positive yield percentages
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
            
            # Return raw predictions so that SHAP explains the true model output.
            # Any post-processing (e.g., scaling to percentage) should be applied
            # only for visualisation, _after_ SHAP values have been computed.
            return predictions
        except Exception as e:
            dprint(f"Error during prediction: {e}")
            return np.zeros(X.shape[0])

    def _extract_bp_neural_network(self):
        """
        Extract weights and biases from MATLAB BP neural network format
        
        This format has weights and biases as cell arrays with each element containing
        a weight matrix or bias vector for each layer.
        """
        try:
            if hasattr(self.net_struct, 'weight') and hasattr(self.net_struct, 'bias'):
                dprint("Attempting to extract MATLAB BP neural network with cell arrays...")
                
                # Get the weight and bias arrays and layer information
                weight_cell = self.net_struct.weight
                bias_cell = self.net_struct.bias
                
                # Check if these are cell arrays
                if hasattr(weight_cell, '__len__') and hasattr(bias_cell, '__len__'):
                    dprint(f"Found weight cell array with length {len(weight_cell)}")
                    dprint(f"Found bias cell array with length {len(bias_cell)}")
                    
                    # Initialize lists for weights and biases
                    self.weights = []
                    self.biases = []
                    
                    # Get layer information if available
                    layer_info = None
                    if hasattr(self.net_struct, 'layer'):
                        layer_info = self.net_struct.layer
                        dprint(f"Found layer information with {len(layer_info)} layers")
                    
                    # Extract activation functions if available
                    self.activation_funcs = []
                    
                    # Process each layer's weights and biases
                    for i in range(len(weight_cell)):
                        # Get the weight matrix and bias vector for this layer
                        w = weight_cell[i]
                        b = bias_cell[i]
                        
                        # Print debug information
                        dprint(f"Layer {i} - Weight type: {type(w)}, Bias type: {type(b)}")
                        if hasattr(w, 'shape'):
                            dprint(f"Layer {i} - Weight shape: {w.shape}")
                        if hasattr(b, 'shape'):
                            dprint(f"Layer {i} - Bias shape: {b.shape}")
                        
                        # Make sure they are numpy arrays
                        if not isinstance(w, np.ndarray):
                            dprint(f"Warning: Weight for layer {i} is not a numpy array")
                            w = np.array(w) if hasattr(w, '__array__') else None
                        
                        if not isinstance(b, np.ndarray):
                            dprint(f"Warning: Bias for layer {i} is not a numpy array")
                            b = np.array(b) if hasattr(b, '__array__') else None
                        
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
                    
                    # Print summary
                    dprint(f"Extracted {len(self.weights)} weight matrices and {len(self.biases)} bias vectors")
                    dprint(f"Activation functions: {self.activation_funcs}")
                    
                    return True
            
            return False
            
        except Exception as e:
            dprint(f"Error extracting BP neural network: {e}")
            return False

def create_custom_waterfall_plot(expected_value, shap_values, X_test_df, output_dir, target_name, instance_idx=0, plot_number=1):
    """
    Create a custom waterfall plot with feature values on the y-axis
    
    Args:
        expected_value: Expected value (baseline prediction)
        shap_values: SHAP values
        X_test_df: Test data as DataFrame with feature names
        output_dir: Directory to save plots
        target_name: Name of the target variable (e.g., 'Biochar')
        instance_idx: Index of instance to analyze (default: 0)
        plot_number: Number to append to the filename (default: 1)
    """
    dprint(f"Creating waterfall plot {plot_number} for instance {instance_idx}...")
    # Adjust figure size to match the reference image style
    plt.figure(figsize=(10, 8))
    
    # Ensure expected_value is scalar (handle lists / arrays from SHAP)
    if isinstance(expected_value, (np.ndarray, list)):
        expected_value = np.array(expected_value).flatten()[0]
    
    # Scale expected value to percentage if in decimal form
    expected_value_abs = abs(expected_value)
    if expected_value_abs <= 1.0 and expected_value_abs > 0.01:
        expected_value_pct = expected_value_abs * 100.0
    else:
        expected_value_pct = expected_value_abs
    
    # Scale SHAP values to match the scale of the expected value
    # If expected value was scaled, scale SHAP values too
    shap_values_adj = shap_values.copy()
    if expected_value_abs <= 1.0 and expected_value_abs > 0.01:
        scale_factor = 100.0
    else:
        scale_factor = 1.0
        
    # Ensure we preserve the sign of SHAP values (positive/negative contributions)
    # This allows for the intercrossing red/blue bars in the waterfall plot
    
    # Get the feature values for the specific instance
    instance_values = X_test_df.iloc[instance_idx]
    
    # Create the waterfall plot without title
    try:
        # Try with all custom parameters
        shap.plots._waterfall.waterfall_legacy(
            expected_value_pct,  # Use properly scaled expected value
            shap_values_adj[instance_idx] * scale_factor,  # Scale SHAP values to match
            X_test_df.iloc[instance_idx],
            max_display=20,  # Display top 20 features as requested
            show=False,
            pos_color="#FF0052",  # Bright red for positive values
            neg_color="#0088FF",  # Bright blue for negative values
            linewidth=0,  # Remove bar borders
            alpha=0.8  # Slight transparency for better look
        )
    except TypeError as e:
        dprint(f"Warning: Falling back to standard parameters due to error: {e}")
        # Fall back to standard parameters if custom ones cause errors
        shap.plots._waterfall.waterfall_legacy(
            expected_value_pct,  # Use properly scaled expected value
            shap_values_adj[instance_idx] * scale_factor,  # Scale SHAP values to match
            X_test_df.iloc[instance_idx],
            max_display=20,  # Display top 20 features as requested
            show=False
        )
    
    # Get current y-axis labels
    ax = plt.gca()
    y_labels = [text.get_text() for text in ax.get_yticklabels()]
    
    # Create new labels with feature values - formatted to match reference image
    new_labels = []
    for label in y_labels:
        if label == 'E[f(X)]' or label == 'f(x)' or 'other features' in label:
            # Keep original labels for the base value, final prediction, and collapsed features
            new_labels.append(label)
        else:
            feature_name = label.strip()
            try:
                feature_value = instance_values[feature_name]
                # Format the value based on its magnitude - simpler format to match reference image
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
                    # Format label in the style: "value = Feature" to match reference image
                    new_labels.append(f"{value_str} = {feature_name}")
                else:
                    new_labels.append(feature_name)
            except KeyError:
                # If feature not found in instance values, keep original label
                new_labels.append(feature_name)
    
    # Set new y-axis labels with smaller font to match reference image
    ax.set_yticklabels(new_labels, fontsize=9)
    
    # Modify x-axis label to indicate yield percentage
    ax.set_xlabel(f"{target_name} Yield (%)", fontsize=10)
    
    # Do not set title to remove the f(x) label from the top right corner
    
    # Update text annotations to match reference image style
    # Get the prediction value and baseline text elements
    prediction_value = None
    baseline_value = None
    
    for text in ax.texts:
        if "f(x)" in text.get_text():
            pred_text = text
            # Extract the current value
            value_str = pred_text.get_text().split('=')[1].strip()
            try:
                value = float(value_str.replace('%', ''))
                # Check if value needs scaling (likely in decimal form)
                if value <= 1.0 and value > 0.01:
                    value *= 100.0
                # Store the value for title
                prediction_value = value
                # Completely hide the f(x) text
                pred_text.set_visible(False)  # Hide the text completely instead of styling it
            except:
                # Keep original if parsing fails
                pass
        elif "E[f(X)]" in text.get_text():
            base_text = text
            # Extract the current value
            value_str = base_text.get_text().split('=')[1].strip() 
            try:
                value = float(value_str.replace('%', ''))
                # Check if value needs scaling (likely in decimal form)
                if value <= 1.0 and value > 0.01:
                    value *= 100.0
                # Store the value
                baseline_value = value
                # Completely hide the baseline value text
                base_text.set_visible(False)  # Hide the text completely instead of styling it
            except:
                # Keep original if parsing fails
                pass
                
    # Remove the f(x) title from the top right corner
    
    # Format the plot to match reference image
    plt.tight_layout()
    
    # Add grid lines for better readability
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add more white space around the plot for cleaner appearance
    plt.subplots_adjust(left=0.2, right=0.9, top=0.95, bottom=0.1)
    
    # Make sure target directory exists
    target_subdir = os.path.join(output_dir, f"01_{target_name}")
    os.makedirs(target_subdir, exist_ok=True)
    
    # Save the plot with numbered filename in multiple formats (PNG and EPS)
    output_file_base = f"{target_subdir}/05_waterfall_plot_{plot_number}"
    save_plot_multi_format(output_file_base, dpi=600, bbox_inches='tight', facecolor='white')
    
    plt.close()

def convert_yield_to_percentage(value):
    """Convert pyrolysis yield values to positive percentages.

    Improvements compared to the original implementation:
    1. If *all* absolute values are within the 0–1 range, the array is treated as fractional yields and multiplied by 100.
    2. If *all* absolute values are >1, the array is assumed to be already expressed as percentages.
    3. In mixed cases (some values ≤1, some >1), only the ≤1 entries are scaled.
    4. Returned values are always non-negative and the original shape is preserved.
    """

    import numpy as _np  # Local import to avoid polluting the global namespace

    arr = _np.asarray(value, dtype=float)
    abs_arr = _np.abs(arr)

    # Case A: all values ≤1 — treat as fractional yields and scale all by 100
    if _np.all(abs_arr <= 1.0 + 1e-12):
        abs_arr *= 100.0

    # Case B: all values >1 — already percentages; keep as-is (absolute values already taken)
    elif _np.all(abs_arr > 1.0 + 1e-12):
        pass  # no scaling needed

    # Case C: mixed — scale only those entries ≤1
    else:
        scale_mask = abs_arr <= 1.0 + 1e-12
        abs_arr[scale_mask] *= 100.0

    # Preserve scalar vs. ndarray shape of the input
    if _np.isscalar(value):
        return float(abs_arr)
    return abs_arr

def save_plot_multi_format(file_path_without_ext, dpi=600, bbox_inches='tight', facecolor='white'):
    """
    Save the current matplotlib figure in multiple formats optimized for Adobe Illustrator
    
    This function handles EPS format errors gracefully by trying multiple approaches:
    1. Modern EPS parameters (no ps_fonttype)
    2. Minimal EPS parameters
    3. Basic EPS save
    4. PDF fallback if EPS completely fails
    
    Args:
        file_path_without_ext: File path without extension
        dpi: Resolution for raster format (PNG)
        bbox_inches: Bounding box setting
        facecolor: Background color
    """
    # Save as PNG (raster format - good for web/screen)
    png_path = f"{file_path_without_ext}.png"
    logger.info(f"Saving plot to: {png_path}")
    try:
        plt.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches, facecolor=facecolor, format='png')
    except Exception as e:
        logger.error(f"Error saving PNG plot: {e}, trying with minimal options")
        plt.savefig(png_path, dpi=dpi)
    
    # Save as SVG (vector format - best for Adobe Illustrator compatibility)
    svg_path = f"{file_path_without_ext}.svg"
    logger.info(f"Saving plot to: {svg_path}")
    try:
        plt.savefig(svg_path, 
                   bbox_inches=bbox_inches, 
                   facecolor=facecolor, 
                   format='svg',
                   transparent=False,
                   edgecolor='none')
    except Exception as e:
        logger.error(f"Error saving SVG plot: {e}, trying with minimal options")
        try:
            plt.savefig(svg_path, format='svg')
        except Exception as e2:
            logger.error(f"Failed to save SVG plot: {e2}")
    
    # Save as EPS (vector format - optimized for Adobe Illustrator)
    eps_path = f"{file_path_without_ext}.eps"
    logger.info(f"Saving plot to: {eps_path}")
    try:
        # Try with modern EPS parameters first
        plt.savefig(eps_path, 
                   bbox_inches=bbox_inches, 
                   facecolor=facecolor, 
                   format='eps',
                   transparent=False,
                   edgecolor='none')
    except Exception as e:
        logger.error(f"Error saving EPS plot: {e}, trying with minimal options")
        try:
            # Try with minimal parameters (no ps_fonttype)
            plt.savefig(eps_path, format='eps', bbox_inches=bbox_inches)
        except Exception as e2:
            logger.error(f"Failed to save EPS plot with minimal options: {e2}")
            try:
                # Last resort: basic EPS save
                plt.savefig(eps_path, format='eps')
            except Exception as e3:
                logger.error(f"Failed to save EPS plot completely: {e3}")
                # Try PDF as fallback for vector format
                try:
                    pdf_path = f"{file_path_without_ext}.pdf"
                    logger.info(f"Trying PDF as fallback: {pdf_path}")
                    plt.savefig(pdf_path, format='pdf', bbox_inches=bbox_inches, facecolor=facecolor)
                    logger.info(f"Successfully saved PDF as fallback for EPS")
                except Exception as e4:
                    logger.error(f"Failed to save PDF fallback: {e4}")

def create_shap_plots(expected_value, shap_values, X_test_df, output_dir, target_name, instance_idx=0):
    """
    Create various SHAP plots for model interpretation
    
    Args:
        expected_value: Expected value (baseline prediction)
        shap_values: SHAP values
        X_test_df: Test data as DataFrame with feature names
        output_dir: Directory to save plots
        target_name: Name of the target variable (e.g., 'Biochar')
        instance_idx: Index of instance to analyze (default: 0)
    """
    # Determine scaling based on the provided expected value
    if isinstance(expected_value, (np.ndarray, list)):
        expected_value = np.array(expected_value).flatten()[0]
    
    # Scale expected value to percentage if in decimal form
    expected_value_abs = abs(expected_value)
    needs_scaling = expected_value_abs <= 1.0 and expected_value_abs > 0.01
    scaling_factor = 100.0 if needs_scaling else 1.0
    
    # Scale SHAP values for consistent visualization
    shap_values_scaled = shap_values * scaling_factor
    
    # Scale expected value for consistent visualization
    expected_value_scaled = abs(expected_value) * scaling_factor
    
    # Get feature importance by mean absolute SHAP value
    feature_importance = np.abs(shap_values_scaled).mean(0)
    
    # Create target-specific subfolder
    target_subdir = os.path.join(output_dir, f"01_{target_name}")
    os.makedirs(target_subdir, exist_ok=True)
    
    # 1. Create a summary plot (beeswarm plot) without title
    logger.info("Creating beeswarm plot...")
    plt.figure(figsize=(12, 14))
    # shap.summary_plot mutates the passed Explanation/numpy array (#1970,#2207)
    shap.summary_plot(shap_values_scaled.copy(), X_test_df, plot_type="dot", show=False)
    plt.xlabel(f"SHAP value (impact on {target_name} yield %)")
    plt.tight_layout()
    save_plot_multi_format(f"{target_subdir}/01_beeswarm_plot", dpi=600, bbox_inches='tight')
    plt.close()
    
    # 2. Create a bar summary plot with feature importance
    logger.info("Creating feature importance plot...")
    plt.figure(figsize=(12, 14))
    # shap.summary_plot mutates the passed Explanation/numpy array (#1970,#2207)
    shap.summary_plot(shap_values_scaled.copy(), X_test_df, plot_type="bar", show=False)
    plt.title(f"{target_name} Yield Feature Importance - Bar Plot", fontsize=16)
    plt.xlabel(f"Mean |SHAP value| (impact on {target_name} yield %)")
    plt.tight_layout()
    save_plot_multi_format(f"{target_subdir}/02_feature_importance_plot", dpi=600, bbox_inches='tight')
    plt.close()
    
    # 3. Create a force plot for a single instance
    logger.info("Creating force plot for a single instance...")
    plt.figure(figsize=(20, 5))
    
    force_plot = shap.force_plot(
        expected_value_scaled,
        shap_values_scaled[instance_idx],
        X_test_df.iloc[instance_idx],
        matplotlib=True,
        show=False
    )
    plt.title(f"{target_name} Yield (%) Single Instance Explanation", fontsize=16)
    plt.tight_layout()
    save_plot_multi_format(f"{target_subdir}/03_instance_explanation_plot", dpi=600, bbox_inches='tight')
    plt.close()
    
    # 4. Dependence plots are now handled by create_target_specific_dependence_plots function
    # which generates more detailed and target-specific dependence plots
    
    # 5. Create 5 custom waterfall plots with feature values for different instances
    logger.info(f"Creating 5 waterfall plots for {target_name}...")
    # Calculate the number of available instances
    num_instances = len(X_test_df)
    # Choose 5 different instances evenly distributed across the dataset
    if num_instances >= 5:
        # Calculate step size to get even distribution
        step = num_instances // 5
        # Generate indices with even spacing
        instance_indices = [i * step for i in range(5)]
    else:
        # If we have fewer than 5 instances, use what we have
        instance_indices = list(range(num_instances))
    
    # Create waterfall plot for each selected instance
    for i, idx in enumerate(instance_indices):
        create_custom_waterfall_plot(expected_value, shap_values, X_test_df, output_dir, target_name, 
                                     instance_idx=idx, plot_number=i+1)
    
    # 6. Generate a bar plot of the top significant features
    logger.info("Creating top features bar plot...")
    top_n = 20
    indices = np.argsort(-feature_importance)[:top_n]
    significant_features = [X_test_df.columns[i] for i in indices]
    top_importance = [feature_importance[i] for i in indices]
    
    plt.figure(figsize=(12, 10))
    plt.barh(range(len(significant_features)), top_importance, align='center')
    plt.yticks(range(len(significant_features)), significant_features)
    plt.xlabel(f'Mean |SHAP value| (impact on {target_name} yield %)')
    plt.title(f"{target_name} Yield Top {top_n} Important Features", fontsize=16)
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at top
    plt.tight_layout()
    save_plot_multi_format(f"{target_subdir}/06_top_features_bar_plot", dpi=600, bbox_inches='tight')
    plt.close()
    
    # 7. Save the top significant features to a text file
    logger.info("Saving significant features to text file...")
    with open(f"{target_subdir}/07_significant_features.txt", "w", encoding='utf-8') as f:
        f.write(f"Top Significant Features for {target_name} Yield (by SHAP value magnitude):\n")
        f.write("=" * 70 + "\n\n")
        for i, feature in enumerate(significant_features):
            importance = top_importance[i]
            f.write(f"{i+1}. {feature}: {importance:.6f}\n")
    
    # Save SHAP values and expected value for later use
    np.save(f"{target_subdir}/shap_values.npy", shap_values)
    np.save(f"{target_subdir}/expected_value.npy", expected_value)
    
def create_target_specific_dependence_plots(expected_value, shap_values, X_test_df, output_dir, target_name, instance_idx=0):
    """
    Create target-specific dependence plots based on the interaction patterns for each pyrolysis product
    
    This function focuses on creating dependence plots where TargetTemperature is the main feature
    and FeedstockType features are used as interaction features, with proper colorbar scaling.
    
    Args:
        expected_value: Expected value (baseline prediction)
        shap_values: SHAP values
        X_test_df: Test data as DataFrame with feature names
        output_dir: Directory to save plots
        target_name: Name of the target variable (Biochar, Bioliquid, Biogas)
        instance_idx: Index of instance to analyze (default: 0)
    """
    # Scale values for visualization
    if isinstance(expected_value, (np.ndarray, list)):
        expected_value = np.array(expected_value).flatten()[0]
    
    expected_value_abs = abs(expected_value)
    needs_scaling = expected_value_abs <= 1.0 and expected_value_abs > 0.01
    scaling_factor = 100.0 if needs_scaling else 1.0
    shap_values_scaled = shap_values * scaling_factor
    
    # Get feature importance
    feature_importance = np.abs(shap_values_scaled).mean(0)
    most_important_idx = np.argmax(feature_importance)
    most_important_feature = X_test_df.columns[most_important_idx]
    
    # Create target-specific subfolder
    target_subdir = os.path.join(output_dir, f"01_{target_name}")
    os.makedirs(target_subdir, exist_ok=True)
    
    # Find FeedstockType features for interaction
    feedstock_features = []
    for feature_name in X_test_df.columns:
        if 'FeedstockType' in feature_name:
            feedstock_features.append(feature_name)
    
    if not feedstock_features:
        logger.warning(f"No FeedstockType features found in dataset for {target_name}")
        return
    
    logger.info(f"Creating {len(feedstock_features)} target-specific dependence plots for {target_name}...")
    
    # Create dependence plots for each FeedstockType feature
    for interaction_feature in feedstock_features:
        try:
            if interaction_feature in X_test_df.columns:
                interaction_idx = X_test_df.columns.get_loc(interaction_feature)
                
                plt.figure(figsize=(12, 8))
                shap.dependence_plot(
                    most_important_idx,
                    shap_values_scaled.copy(),
                    X_test_df,
                    interaction_index=interaction_idx,
                    show=False
                )
                
                # Get feature importance ranking
                importance_rank = np.where(np.argsort(feature_importance)[::-1] == interaction_idx)[0][0] + 1
                
                # Create descriptive title with target-specific information
                title = f"{target_name} Yield: {most_important_feature} vs {interaction_feature}\nFeature Importance Rank: {importance_rank} (out of {len(X_test_df.columns)})"
                plt.title(title, fontsize=14, fontweight='bold')
                
                # Customize the colorbar for binary features (0 and 1 only)
                ax = plt.gca()
                if hasattr(ax, 'collections') and len(ax.collections) > 0:
                    # Find the scatter plot collection
                    scatter = None
                    for collection in ax.collections:
                        if hasattr(collection, 'get_offsets') and len(collection.get_offsets()) > 0:
                            scatter = collection
                            break
                    
                    if scatter is not None:
                        # Get the colorbar
                        cbar = plt.colorbar(scatter, ax=ax)
                        cbar.set_label(interaction_feature, fontsize=12)
                        
                        # For binary features, set ticks to only show 0 and 1
                        feature_values = X_test_df[interaction_feature].values
                        unique_values = np.unique(feature_values)
                        if len(unique_values) <= 2:  # Binary feature
                            cbar.set_ticks([0, 1])
                            cbar.set_ticklabels(['0', '1'])
                
                # Customize axis labels
                plt.xlabel(f"{most_important_feature}", fontsize=12, fontweight='bold')
                plt.ylabel(f"SHAP value for {most_important_feature}", fontsize=12, fontweight='bold')
                
                plt.tight_layout()
                
                # Save with target-specific filename using sanitized feature names
                sanitized_most_important = sanitize_feature_name_for_filename(most_important_feature)
                sanitized_interaction = sanitize_feature_name_for_filename(interaction_feature)
                filename = f"04_{target_name}_dependence_{sanitized_most_important}_vs_{sanitized_interaction}"
                save_plot_multi_format(f"{target_subdir}/{filename}", dpi=600, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Created {target_name} dependence plot: {most_important_feature} vs {interaction_feature}")
            else:
                logger.warning(f"Interaction feature {interaction_feature} not found in dataset for {target_name}")
                
        except Exception as e:
            logger.error(f"Error creating {target_name} dependence plot with {interaction_feature}: {e}")
            plt.close()

def create_comparative_dependence_analysis(expected_values, shap_values_dict, X_test_df, output_dir):
    """
    Create comprehensive comparative analysis of temperature interactions across all three targets
    
    This function creates comparative dependence plots for ALL input features with TargetTemperature,
    showing how temperature interactions differ across Biochar, Bioliquid, and Biogas products.
    
    Args:
        expected_values: Dictionary of expected values for each target
        shap_values_dict: Dictionary of SHAP values for each target
        X_test_df: Test data as DataFrame with feature names
        output_dir: Directory to save plots
    """
    logger.info("Creating comprehensive comparative dependence analysis across all targets...")
    
    # Create comparative subfolder
    comparative_subdir = os.path.join(output_dir, "02_Comparative_Analysis")
    try:
        os.makedirs(comparative_subdir, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating comparative analysis directory: {e}")
        return
    
    # Find TargetTemperature index
    target_temp_idx = None
    for i, feature_name in enumerate(X_test_df.columns):
        if 'TargetTemperature' in feature_name or 'Temperature' in feature_name:
            target_temp_idx = i
            break
    
    if target_temp_idx is None:
        logger.warning("TargetTemperature feature not found in dataset for comparative analysis")
        return
    
    target_temp_feature = X_test_df.columns[target_temp_idx]
    
    # Get all features except TargetTemperature itself
    all_features = list(X_test_df.columns)
    features_to_analyze = [f for f in all_features if f != target_temp_feature]
    
    logger.info(f"Creating {len(features_to_analyze)} comparative dependence plots with {target_temp_feature}...")
    
    # Create comparative plots for each feature
    plots_created = 0
    for i, feature_name in enumerate(features_to_analyze, 1):
        try:
            if feature_name in X_test_df.columns:
                feature_idx = X_test_df.columns.get_loc(feature_name)
                
                # Create subplot for each target
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                for j, target_name in enumerate(['Biochar', 'Bioliquid', 'Biogas']):
                    if target_name in shap_values_dict:
                        # Scale values for this target
                        expected_value = expected_values.get(target_name, 0)
                        if isinstance(expected_value, (np.ndarray, list)):
                            expected_value = np.array(expected_value).flatten()[0]
                        
                        expected_value_abs = abs(expected_value)
                        needs_scaling = expected_value_abs <= 1.0 and expected_value_abs > 0.01
                        scaling_factor = 100.0 if needs_scaling else 1.0
                        shap_values_scaled = shap_values_dict[target_name] * scaling_factor
                        
                        # Create dependence plot with TargetTemperature as main feature
                        shap.dependence_plot(
                            target_temp_idx,  # Use TargetTemperature as main feature
                            shap_values_scaled.copy(),
                            X_test_df,
                            interaction_index=feature_idx,  # Use other feature as interaction
                            show=False,
                            ax=axes[j]
                        )
                        
                        axes[j].set_title(f"{target_name}: {target_temp_feature} vs {feature_name}", fontsize=12)
                
                plt.tight_layout()
                # Use sanitized feature name for filename
                sanitized_feature = sanitize_feature_name_for_filename(feature_name)
                save_plot_multi_format(f"{comparative_subdir}/comparative_dependence_{sanitized_feature}", dpi=600, bbox_inches='tight')
                plt.close()
                
                plots_created += 1
                logger.info(f"Created comparative dependence plot {i}/{len(features_to_analyze)}: {feature_name}")
        except Exception as e:
            logger.error(f"Error creating comparative dependence plot for {feature_name}: {e}")
            plt.close()  # Ensure plot is closed even if error occurs
    
    # Create comprehensive summary table of interaction patterns
    try:
        with open(f"{comparative_subdir}/interaction_patterns_summary.txt", "w", encoding='utf-8') as f:
            f.write("Comprehensive Temperature Interaction Patterns Across Pyrolysis Products\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Features Analyzed: {len(features_to_analyze)}\n")
            f.write(f"Target Temperature Feature: {target_temp_feature}\n\n")
            
            f.write("Analysis Summary:\n")
            f.write("-" * 20 + "\n")
            f.write("This comparative analysis shows how TargetTemperature interacts with each input feature\n")
            f.write("across all three pyrolysis products (Biochar, Bioliquid, Biogas).\n\n")
            
            f.write("Key Insights:\n")
            f.write("- Biochar: Primarily influenced by ash composition and feedstock properties\n")
            f.write("- Bioliquid: Balanced influence from ash components and process parameters\n")
            f.write("- Biogas: Similar to Bioliquid but with different interaction intensities\n")
            f.write("- Temperature: Critical parameter affecting all three product yields\n\n")
            
            f.write("Features Analyzed:\n")
            f.write("-" * 20 + "\n")
            for i, feature_name in enumerate(features_to_analyze, 1):
                f.write(f"{i:2d}. {feature_name}\n")
            
            f.write(f"\nTotal Comparative Plots Created: {plots_created}\n")
            f.write("Each plot shows TargetTemperature vs one feature across all three products.\n")
        
        logger.info("Created comprehensive interaction patterns summary")
        
    except Exception as e:
        logger.error(f"Error creating interaction patterns summary: {e}")

def save_feature_names(feature_names, output_dir):
    """
    Save feature names to a text file
    
    Args:
        feature_names: List of feature names
        output_dir: Directory to save the file
    """
    with open(f"{output_dir}/00_feature_names_used.txt", "w", encoding='utf-8') as f:
        f.write("Feature Names Used in SHAP Analysis:\n")
        f.write("=" * 50 + "\n\n")
        for i, name in enumerate(feature_names):
            f.write(f"{i+1}. {name}\n")

def save_shap_to_excel(shap_values, expected_value, feature_names, target_name, target_subdir):
    """
    Convert SHAP values and expected value from .npy files to Excel format
    
    Args:
        shap_values: SHAP values array
        expected_value: Expected value (baseline)
        feature_names: List of feature names
        target_name: Name of the target variable (e.g., 'Biochar')
        target_subdir: Directory to save Excel files
    """
    logger.info(f"Converting SHAP values for {target_name} to Excel format…")
    
    try:
        # Create DataFrame for SHAP values with feature names as columns
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        
        # Add expected value to a separate DataFrame
        expected_df = pd.DataFrame({'expected_value': [expected_value]})
        
        # Lazy import openpyxl / fallback
        try:
            import openpyxl  # noqa: F401
            excel_engine = 'openpyxl'
        except ImportError:
            logger.warning("openpyxl is not installed; falling back to xlsxwriter. Some formatting may be limited.")
            excel_engine = 'xlsxwriter'

        excel_path = f"{target_subdir}/shap_values_data.xlsx"
        with pd.ExcelWriter(excel_path, engine=excel_engine) as writer:
            # Write SHAP values to sheet with target name
            shap_df.to_excel(writer, sheet_name=f"{target_name}_SHAP_values", index=True)
            
            # Write expected value to its own sheet
            expected_df.to_excel(writer, sheet_name=f"{target_name}_Expected_Value", index=False)
            
            # Add a summary sheet with feature importance
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Mean_Abs_SHAP': np.abs(shap_values).mean(0),
                'Max_Abs_SHAP': np.abs(shap_values).max(0)
            })
            feature_importance = feature_importance.sort_values('Mean_Abs_SHAP', ascending=False)
            feature_importance.to_excel(writer, sheet_name=f"{target_name}_Feature_Importance", index=False)
        
        logger.info(f"Successfully saved SHAP values to Excel: {excel_path}")
    
    except Exception as e:
        logger.error(f"Error saving SHAP values to Excel: {e}")

def convert_saved_npy_to_excel(output_dir, feature_names):
    """
    Convert already saved .npy files to Excel format
    
    Args:
        output_dir: Directory containing the saved .npy files
        feature_names: List of feature names
    """
    target_names = ['Biochar', 'Bioliquid', 'Biogas']
    
    for target_name in target_names:
        target_subdir = os.path.join(output_dir, f"01_{target_name}")
        
        # Check if the directory and files exist
        if not os.path.exists(target_subdir):
            logger.warning(f"Directory not found: {target_subdir}")
            continue
            
        shap_file = os.path.join(target_subdir, "shap_values.npy")
        expected_file = os.path.join(target_subdir, "expected_value.npy")
        
        if not os.path.exists(shap_file) or not os.path.exists(expected_file):
            logger.warning(f"SHAP files not found in {target_subdir}")
            continue
            
        try:
            # Load saved numpy arrays
            shap_values = np.load(shap_file)
            expected_value = np.load(expected_file)
            
            # Convert to Excel
            save_shap_to_excel(shap_values, expected_value, feature_names, target_name, target_subdir)
            
        except Exception as e:
            logger.error(f"Error converting saved .npy files for {target_name}: {e}")

def run_shap_analysis(matlab_file, target_idx=None, debug=False):
    """
    Run SHAP analysis for specific target index or all targets
    
    Args:
        matlab_file: Path to MATLAB .mat file
        target_idx: Target index (0=Biochar, 1=Bioliquid, 2=Biogas, None=all)
        debug: Whether to print detailed debugging information
    """
    target_names = ['Biochar', 'Bioliquid', 'Biogas']
    
    # Enable/disable debugging output
    global DEBUG_MODE
    DEBUG_MODE = debug
    if DEBUG_MODE:
        logger.info("DEBUG MODE ENABLED - Will print detailed debugging information")
    
    # Create a single organized output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = f'SHAP_Analysis_Results_{timestamp}'
    os.makedirs(main_output_dir, exist_ok=True)
    logger.info(f"Creating main output directory: {main_output_dir}")
    
    # Check memory usage and warn if high
    try:
        import psutil
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > 80:
            logger.warning(f"High memory usage detected: {memory_usage:.1f}%. Consider reducing sample size if analysis fails.")
    except ImportError:
        logger.info("psutil not available - skipping memory usage check")
    
    # Create a README file explaining the positive yield conversion
    with open(f"{main_output_dir}/00_YIELD_CONVERSION_NOTE.txt", "w", encoding='utf-8') as f:
        f.write("IMPORTANT NOTE ABOUT YIELD VALUES\n")
        f.write("==============================\n\n")
        f.write("All yield values in this analysis are presented as absolute percentages (0-100%).\n")
        f.write("The original model might output negative values or values in decimal form (0-1 range).\n")
        f.write("For consistency and clarity, all yield values have been:\n")
        f.write("1. Converted to absolute values (removing negative signs)\n")
        f.write("2. Scaled to percentage form if in decimal format\n\n")
        f.write("This conversion ensures all visualizations use a consistent scale and are easier to interpret.\n\n")
        f.write("FILE FORMATS\n")
        f.write("===========\n\n")
        f.write("All plots are provided in multiple formats:\n")
        f.write("1. PNG - Raster format for screen viewing and web use\n")
        f.write("2. SVG - Vector format optimized for Adobe Illustrator editing\n")
        f.write("3. EPS - Vector format for publication-quality graphics\n")
        f.write("4. PDF - Vector format (fallback if EPS fails)\n\n")
        f.write("For Adobe Illustrator users:\n")
        f.write("- SVG files offer the best compatibility and editing capability\n")
        f.write("- EPS files are optimized for publication-quality graphics\n")
        f.write("- PDF files are created as fallback if EPS format fails\n")
        f.write("- All vector formats maintain quality at any scale and are suitable for publication\n")
    
    # Create a log file to capture errors and warnings
    log_file = os.path.join(main_output_dir, "00_analysis_log.txt")
    with open(log_file, "w", encoding='utf-8') as log:
        log.write(f"SHAP Analysis Log - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write("="*80 + "\n\n")
        
        try:
            # Load MATLAB data
            log.write("Loading MATLAB data...\n")
            mat_data = load_matlab_data(matlab_file)
            if not mat_data:
                log.write("ERROR: Failed to load MATLAB data.\n")
                logger.error("Failed to load MATLAB data.")
                return
            
            # If debug mode is enabled, inspect the structure of the loaded MATLAB data
            if DEBUG_MODE:
                logger.info("\nDEBUG: MATLAB Data Structure:")
                for key, value in mat_data.items():
                    logger.info(f"Key: {key}, Type: {type(value)}")
                    if key == 'net':
                        logger.info("  Net attributes:", [attr for attr in dir(value) if not attr.startswith('__')])
                        
                        # Inspect weight and bias arrays if they exist
                        if hasattr(value, 'weight'):
                            weight = value.weight
                            logger.info(f"  Weight type: {type(weight)}")
                            if hasattr(weight, '__len__'):
                                logger.info(f"  Weight length: {len(weight)}")
                                for i in range(min(len(weight), 3)):  # Show first 3 entries
                                    logger.info(f"    Weight[{i}] type: {type(weight[i])}")
                                    if hasattr(weight[i], 'shape'):
                                        logger.info(f"    Weight[{i}] shape: {weight[i].shape}")
                        
                        if hasattr(value, 'bias'):
                            bias = value.bias
                            logger.info(f"  Bias type: {type(bias)}")
                            if hasattr(bias, '__len__'):
                                logger.info(f"  Bias length: {len(bias)}")
                                for i in range(min(len(bias), 3)):  # Show first 3 entries
                                    logger.info(f"    Bias[{i}] type: {type(bias[i])}")
                                    if hasattr(bias[i], 'shape'):
                                        logger.info(f"    Bias[{i}] shape: {bias[i].shape}")
            
            # Extract input/output data and network structure
            try:
                log.write("Extracting neural network data...\n")
                X, y, network_structure = extract_neural_network_data(mat_data)
                log.write(f"Success: X shape={X.shape}, y shape={y.shape}\n")
            except Exception as e:
                log.write(f"ERROR: Failed to extract data from MATLAB file: {e}\n")
                logger.error(f"Error extracting data from MATLAB file: {e}")
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
            
            # Process each target index
            targets_to_process = range(3) if target_idx is None else [target_idx]
            
            for idx in targets_to_process:
                if idx < 0 or idx > 2:
                    log.write(f"WARNING: Invalid target index: {idx}. Skipping.\n")
                    continue
                
                log.write(f"\nProcessing {target_names[idx]} yield (index {idx})...\n")
                logger.info(f"\nProcessing {target_names[idx]} yield...")
                
                try:
                    # Create a wrapper for the MATLAB neural network model
                    log.write("Creating MATLAB neural network wrapper...\n")
                    model = MatlabNeuralNetworkWrapper(network_structure, target_idx=idx)
                    log.write(f"Model type: {model.model_type}\n")
                    
                    # Build a compact background set (Independent masker is efficient for tabular data)
                    num_background = min(100, X.shape[0])
                    log.write(f"Using {num_background} background samples for SHAP masker\n")
                    # Deterministic background sampling
                    X_background = shap.sample(X, num_background, random_state=GLOBAL_SEED)
                    X_background_df = pd.DataFrame(X_background, columns=feature_names)

                    # Create model-agnostic explainer; this auto-selects Deep / Gradient explainer
                    log.write("Creating SHAP Explainer (auto algorithm)\n")
                    masker = shap.maskers.Independent(X_background_df)
                    explainer = shap.Explainer(
                        model.predict,
                        masker,
                        algorithm="permutation",
                        seed=GLOBAL_SEED  # deterministic explanations
                    )
                    log.write("Created SHAP PermutationExplainer with fixed GLOBAL_SEED\n")

                    # Calculate SHAP values for the data samples
                    log.write(f"Calculating SHAP values for {sample_size} data samples...\n")

                    explanation = explainer(X_analysis)
                    shap_values = explanation.values
                    
                    # ------------------------------------------------------------------
                    # Robustly determine the model's expected value (baseline prediction)
                    # ------------------------------------------------------------------
                    try:
                        expected_value_local = explainer.expected_value  # Tree / Kernel / Linear explainers
                    except AttributeError:
                        # Fallback for PermutationExplainer and others
                        base_vals = explanation.base_values
                        if isinstance(base_vals, (list, np.ndarray)):
                            base_arr = np.array(base_vals)
                            if base_arr.ndim == 0:
                                expected_value_local = float(base_arr)
                            elif base_arr.ndim == 1:
                                expected_value_local = float(base_arr[0])
                            else:  # shape (n_samples, n_outputs)
                                expected_value_local = float(base_arr[0, idx])
                        else:
                            expected_value_local = float(base_vals)

                    log.write(f"SHAP values shape: {np.array(shap_values).shape}\n")
                    
                    # Create target-specific subfolder
                    target_subdir = os.path.join(main_output_dir, f"01_{target_names[idx]}")
                    os.makedirs(target_subdir, exist_ok=True)
                    
                    # Save the model type information
                    with open(f"{target_subdir}/00_model_info.txt", "w", encoding='utf-8') as f:
                        f.write(f"Model Type: {model.model_type}\n")
                        f.write(f"Input Size: {model.get_input_size()}\n")
                        f.write(f"Target Index: {idx}\n")
                        f.write(f"Target Name: {target_names[idx]}\n")
                        f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write("Note: Using MATLAB neural network model for SHAP analysis\n")
                    
                    # Save SHAP values and expected value for later use
                    np.save(f"{target_subdir}/shap_values.npy", shap_values)
                    np.save(f"{target_subdir}/expected_value.npy", expected_value_local)
                    
                    # Save to Excel format
                    save_shap_to_excel(shap_values, expected_value_local, feature_names, 
                                      target_names[idx], target_subdir)
                    log.write("Successfully saved SHAP values to Excel\n")
                    
                    # Create all SHAP plots
                    try:
                        log.write("Creating SHAP plots...\n")
                        create_shap_plots(expected_value_local, shap_values, X_analysis, main_output_dir, target_names[idx])
                        create_comprehensive_targettemperature_dependence_plots(expected_value_local, shap_values, X_analysis, main_output_dir, target_names[idx])
                        create_target_specific_dependence_plots(expected_value_local, shap_values, X_analysis, main_output_dir, target_names[idx])
                        log.write("Successfully created all plots\n")
                    except Exception as e:
                        log.write(f"ERROR creating plots: {e}\n")
                        logger.error(f"Error creating plots: {e}")
                    
                    log.write(f"SHAP analysis for {target_names[idx]} yield completed successfully.\n")
                
                except Exception as e:
                    log.write(f"ERROR during SHAP analysis for {target_names[idx]}: {e}\n")
                    logger.error(f"Error during SHAP analysis for {target_names[idx]}: {e}")
            
            # Create a README file in the output directory
            with open(f"{main_output_dir}/00_README.txt", "w", encoding='utf-8') as f:
                f.write("SHAP Analysis Results (Using MATLAB Neural Network Model)\n")
                f.write("==================================================\n\n")
                f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Note: Analysis performed using ALL sample data without train/test splitting\n")
                f.write("Only the original MATLAB model is used (no fallback models)\n\n")
                f.write("File Formats:\n")
                f.write("------------\n")
                f.write("All plots are provided in multiple formats:\n")
                f.write("1. PNG - Raster format for screen viewing and web use\n")
                f.write("2. SVG - Vector format optimized for Adobe Illustrator editing\n")
                f.write("3. EPS - Vector format for publication-quality graphics\n")
                f.write("4. PDF - Vector format (fallback if EPS fails)\n\n")
                f.write("For Adobe Illustrator users:\n")
                f.write("- SVG files offer the best compatibility and editing capability\n")
                f.write("- EPS files are optimized for publication-quality graphics\n")
                f.write("- PDF files are created as fallback if EPS format fails\n")
                f.write("- All vector formats maintain quality at any scale and are suitable for publication\n")
                f.write("Directory Structure:\n")
                f.write("-------------------\n")
                f.write("00_README.txt - This file\n")
                f.write("00_analysis_log.txt - Processing log with details and any errors\n")
                f.write("00_feature_names_used.txt - List of feature names used in the analysis\n")
                f.write("00_YIELD_CONVERSION_NOTE.txt - Information about yield value conversions and file formats\n")
                f.write("\nNEW: Comprehensive Dependence Analysis\n")
                f.write("The script now creates dependence plots for TargetTemperature with FeedstockType interaction features,\n")
                f.write("automatically identifying FeedstockType features from the dataset.\n")
                f.write("Colorbar scaling is optimized for binary features (0 and 1 values only).\n")
                
                for idx, name in enumerate(target_names):
                    if target_idx is None or target_idx == idx:
                        f.write(f"\n01_{name}/ - Analysis results for {name} yield\n")
                        f.write(f"  |- 00_model_info.txt - Information about the model used\n")
                        f.write(f"  |- 01_beeswarm_plot.png/svg/eps - Shows feature importance and impact direction\n")
                        f.write(f"  |- 02_feature_importance_plot.png/svg/eps - Bar plot of feature importance\n")
                        f.write(f"  |- 03_instance_explanation_plot.png/svg/eps - Force plot for a single instance\n")
                        f.write(f"  |- 04_{name}_dependence_TargetTemperature_vs_*.png/svg/eps - Comprehensive dependence plots with TargetTemperature\n")
                        f.write(f"  |- 04_{name}_dependence_plots_summary.txt - Summary of all dependence plots created\n")
                        f.write(f"  |- 05_waterfall_plot_1.png/svg/eps to 05_waterfall_plot_5.png/svg/eps - Five waterfall plots showing feature contributions for different instances\n")
                        f.write(f"  |- 06_top_features_bar_plot.png/svg/eps - Top 20 important features\n")
                        f.write(f"  |- 07_significant_features.txt - List of significant features\n")
                        f.write(f"  |- shap_values.npy - Saved SHAP values for future use\n")
                        f.write(f"  |- expected_value.npy - Saved expected value for future use\n")
                        f.write(f"  |- shap_values_data.xlsx - Excel file with SHAP values and feature importance\n")
                        f.write(f"  |- 02_Comparative_Analysis/ - Comparative analysis across all targets\n")
                        f.write(f"    |- comparative_dependence_*.png/svg/eps - Side-by-side comparison plots\n")
                        f.write(f"    |- interaction_patterns_summary.txt - Summary of interaction patterns\n")
                
                f.write("\nAnalysis completed.\n")
            
            log.write("\nAll SHAP analyses completed.\n")
            
            # Create comparative analysis across all targets
            if target_idx is None:  # Only when analyzing all targets
                try:
                    log.write("Creating comparative analysis across all targets...\n")
                    
                    # Collect all SHAP values and expected values
                    expected_values_dict = {}
                    shap_values_dict = {}
                    
                    for idx in range(3):
                        target_name = target_names[idx]
                        target_subdir = os.path.join(main_output_dir, f"01_{target_name}")
                        
                        # Load saved values
                        shap_file = os.path.join(target_subdir, "shap_values.npy")
                        expected_file = os.path.join(target_subdir, "expected_value.npy")
                        
                        if os.path.exists(shap_file) and os.path.exists(expected_file):
                            shap_values_dict[target_name] = np.load(shap_file)
                            expected_values_dict[target_name] = np.load(expected_file)
                    
                    if len(shap_values_dict) == 3:
                        create_comparative_dependence_analysis(expected_values_dict, shap_values_dict, X_analysis, main_output_dir)
                        log.write("Successfully created comparative analysis\n")
                    else:
                        log.write("WARNING: Could not create comparative analysis - missing data for some targets\n")
            
                except Exception as e:
                    log.write(f"ERROR creating comparative analysis: {e}\n")
                    logger.error(f"Error creating comparative analysis: {e}")
        except Exception as e:
            log.write(f"CRITICAL ERROR in SHAP analysis: {e}\n")
            logger.error(f"Critical error in SHAP analysis: {e}")
    
    logger.info(f"\nAll SHAP analyses completed. Results saved to '{main_output_dir}' directory.")
    logger.info(f"Check the log file {log_file} for details on any errors or warnings.")

def main():
    """Main function to run SHAP analysis"""
    # Default MATLAB file path
    matlab_file = 'GPM_SHAP_matlab/Results/Training/Results_trained.mat'
    
    # Parse command line arguments
    target_idx = None  # Default: analyze all targets
    debug = False  # Default: debug mode off
    convert_only = False  # Default: run full analysis
    
    if len(sys.argv) > 1:
        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i].lower()
            
            # Check if argument is a MATLAB file path
            if arg.endswith('.mat'):
                matlab_file = sys.argv[i]
                logger.info(f"Using MATLAB file: {matlab_file}")
            elif arg in ['biochar', 'char', '0']:
                target_idx = 0
            elif arg in ['bioliquid', 'liquid', '1']:
                target_idx = 1
            elif arg in ['biogas', 'gas', '2']:
                target_idx = 2
            elif arg in ['all']:
                target_idx = None
            elif arg in ['-d', '--debug', 'debug']:
                debug = True
            elif arg in ['-c', '--convert', 'convert']:
                convert_only = True
            elif arg in ['-h', '--help', 'help']:
                print("Usage: python shap_analysis_latest.py [matlab_file.mat] [biochar|bioliquid|biogas|all] [-d|--debug] [-c|--convert]")
                print("  matlab_file.mat: Path to MATLAB .mat file (default: GPM_SHAP_matlab/Results/Training/Results_trained.mat)")
                print("  biochar, bioliquid, biogas, all: Target to analyze (default: all)")
                print("  -d, --debug: Enable debug mode with additional output")
                print("  -c, --convert: Only convert existing .npy files to Excel without running analysis")
                print("  -h, --help: Show this help message")
                print("\nOutput:")
                print("  - All visualizations are saved in PNG, SVG, and EPS formats")
                print("  - PNG: Raster format good for screen viewing")
                print("  - SVG: Vector format optimized for Adobe Illustrator editing")
                print("  - EPS: Vector format good for publication-quality graphics")
                return
            else:
                logger.warning(f"Unknown argument: {sys.argv[i]}")
            
            i += 1
    
    # If convert_only is set, just convert existing .npy files to Excel
    if convert_only:
        # Find the most recent results directory 
        result_dirs = [d for d in os.listdir() if d.startswith('SHAP_Analysis_Results_')]
        if result_dirs:
            latest_dir = max(result_dirs)
            logger.info(f"Converting files in the most recent results directory: {latest_dir}")
            
            # Load feature names
            feature_names_file = os.path.join(latest_dir, "00_feature_names_used.txt")
            if os.path.exists(feature_names_file):
                with open(feature_names_file, 'r') as f:
                    lines = f.readlines()
                
                # Skip header lines and extract feature names
                feature_names = []
                for line in lines:
                    if line.strip() and '.' in line:
                        parts = line.strip().split('. ', 1)
                        if len(parts) > 1:
                            feature_names.append(parts[1])
                
                if feature_names:
                    convert_saved_npy_to_excel(latest_dir, feature_names)
                else:
                    logger.warning("Could not extract feature names from the file.")
            else:
                logger.warning(f"Feature names file not found: {feature_names_file}")
        else:
            logger.warning("No SHAP analysis results directories found.")
        return
    
    # Run SHAP analysis
    run_shap_analysis(matlab_file, target_idx, debug)

def sanitize_feature_name_for_filename(feature_name):
    """
    Sanitize feature names for use in filenames by replacing invalid characters
    
    Args:
        feature_name: Original feature name that may contain invalid filename characters
        
    Returns:
        Sanitized feature name safe for use in filenames
    """
    # Replace invalid filename characters with underscores
    # Invalid characters: / \ : * ? " < > |
    sanitized = feature_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    sanitized = sanitized.replace('*', '_').replace('?', '_').replace('"', '_')
    sanitized = sanitized.replace('<', '_').replace('>', '_').replace('|', '_')
    
    # Replace other potentially problematic characters
    sanitized = sanitized.replace('(', '_').replace(')', '_').replace('%', 'pct')
    sanitized = sanitized.replace(' ', '_').replace('-', '_')
    
    # Remove any leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    # Ensure the filename is not empty
    if not sanitized:
        sanitized = "feature"
    
    return sanitized

def create_comprehensive_targettemperature_dependence_plots(expected_value, shap_values, X_test_df, output_dir, target_name, instance_idx=0):
    """
    Create comprehensive dependence plots for TargetTemperature with FeedstockType interaction features
    
    This function specifically focuses on creating dependence plots where TargetTemperature is the main feature
    and FeedstockType features are used as interaction features. The plots are designed to match the style
    of the reference image with proper colorbar scaling for binary features.
    
    Output files follow the format: 04_{target_name}_dependence_TargetTemperature_vs_{feature_name}
    
    Args:
        expected_value: Expected value (baseline prediction)
        shap_values: SHAP values
        X_test_df: Test data as DataFrame with feature names
        output_dir: Directory to save plots
        target_name: Name of the target variable (Biochar, Bioliquid, Biogas)
        instance_idx: Index of instance to analyze (default: 0)
    """
    # Scale values for visualization
    if isinstance(expected_value, (np.ndarray, list)):
        expected_value = np.array(expected_value).flatten()[0]
    
    expected_value_abs = abs(expected_value)
    needs_scaling = expected_value_abs <= 1.0 and expected_value_abs > 0.01
    scaling_factor = 100.0 if needs_scaling else 1.0
    shap_values_scaled = shap_values * scaling_factor
    
    # Get feature importance to identify important features from beeswarm plot
    feature_importance = np.abs(shap_values_scaled).mean(0)
    
    # Find TargetTemperature index
    target_temp_idx = None
    target_temp_feature = None
    for i, feature_name in enumerate(X_test_df.columns):
        if 'TargetTemperature' in feature_name or 'Temperature' in feature_name:
            target_temp_idx = i
            target_temp_feature = feature_name
            break
    
    if target_temp_idx is None:
        logger.warning(f"TargetTemperature feature not found in dataset for {target_name}")
        return
    
    # Create target-specific subfolder
    target_subdir = os.path.join(output_dir, f"01_{target_name}")
    os.makedirs(target_subdir, exist_ok=True)
    
    # Find FeedstockType features for interaction
    feedstock_features = []
    for feature_name in X_test_df.columns:
        if 'FeedstockType' in feature_name and feature_name != target_temp_feature:
            feedstock_features.append(feature_name)
    
    if not feedstock_features:
        logger.warning(f"No FeedstockType features found in dataset for {target_name}")
        return
    
    logger.info(f"Creating {len(feedstock_features)} dependence plots with {target_temp_feature} for {target_name}...")
    
    # Create dependence plots for each FeedstockType feature with TargetTemperature
    plots_created = 0
    for i, feature_name in enumerate(feedstock_features, 1):
        logger.info(f"Processing {target_name} dependence plot {i}/{len(feedstock_features)}: {feature_name}")
        try:
            feature_idx = X_test_df.columns.get_loc(feature_name)
            
            # Create figure with specific size to match reference style
            plt.figure(figsize=(12, 8))
            
            # Create the dependence plot
            shap.dependence_plot(
                target_temp_idx,  # Use TargetTemperature as main feature
                shap_values_scaled.copy(),
                X_test_df,
                interaction_index=feature_idx,  # Use FeedstockType feature as interaction
                show=False
            )
            
            # Get feature importance ranking
            importance_rank = np.where(np.argsort(feature_importance)[::-1] == feature_idx)[0][0] + 1
            
            # Create title matching the reference style
            title = f"{target_name} Yield: {target_temp_feature} vs {feature_name}\nFeature Importance Rank: {importance_rank} (out of {len(X_test_df.columns)})"
            plt.title(title, fontsize=14, fontweight='bold')
            
            # Customize the colorbar for binary features (0 and 1 only)
            ax = plt.gca()
            if hasattr(ax, 'collections') and len(ax.collections) > 0:
                # Find the scatter plot collection
                scatter = None
                for collection in ax.collections:
                    if hasattr(collection, 'get_offsets') and len(collection.get_offsets()) > 0:
                        scatter = collection
                        break
                
                if scatter is not None:
                    # Get the colorbar
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label(feature_name, fontsize=12)
                    
                    # For binary features, set ticks to only show 0 and 1
                    feature_values = X_test_df[feature_name].values
                    unique_values = np.unique(feature_values)
                    if len(unique_values) <= 2:  # Binary feature
                        cbar.set_ticks([0, 1])
                        cbar.set_ticklabels(['0', '1'])
            
            # Customize axis labels to match reference style
            plt.xlabel(f"{target_temp_feature}", fontsize=12, fontweight='bold')
            plt.ylabel(f"SHAP value for {target_temp_feature}", fontsize=12, fontweight='bold')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save with sanitized feature names - TargetTemperature first
            sanitized_temp = sanitize_feature_name_for_filename(target_temp_feature)
            sanitized_feature = sanitize_feature_name_for_filename(feature_name)
            filename = f"04_{target_name}_dependence_{sanitized_temp}_vs_{sanitized_feature}"
            save_plot_multi_format(f"{target_subdir}/{filename}", dpi=600, bbox_inches='tight')
            plt.close()
            
            plots_created += 1
            logger.info(f"Created {target_name} dependence plot: {target_temp_feature} vs {feature_name}")
            
        except Exception as e:
            logger.error(f"Error creating {target_name} dependence plot with {feature_name}: {e}")
            plt.close()
    
    logger.info(f"Successfully created {plots_created} dependence plots for {target_name}")
    
    # Save summary of created plots with feature importance ranking
    summary_file = os.path.join(target_subdir, f"04_{target_name}_dependence_plots_summary.txt")
    with open(summary_file, "w", encoding='utf-8') as f:
        f.write(f"Dependence Plots Summary for {target_name}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total plots created: {plots_created}\n")
        f.write(f"Target temperature feature: {target_temp_feature}\n")
        f.write(f"Analysis date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Plots created (with feature importance ranking):\n")
        f.write("-" * 50 + "\n")
        for i, feature_name in enumerate(feedstock_features):
            feature_idx = X_test_df.columns.get_loc(feature_name)
            importance_rank = np.where(np.argsort(feature_importance)[::-1] == feature_idx)[0][0] + 1
            sanitized_temp = sanitize_feature_name_for_filename(target_temp_feature)
            sanitized_feature = sanitize_feature_name_for_filename(feature_name)
            filename = f"04_{target_name}_dependence_{sanitized_temp}_vs_{sanitized_feature}"
            f.write(f"Rank {importance_rank:2d}: {filename}.png/svg/eps ({feature_name})\n")

if __name__ == "__main__":
    main()