"""
PyroBot: Neural Network Surrogate Models Loader & Forward Pass
=============================================================
This module provides high-performance, unified wrapper classes to load and
run inference on pre-trained MATLAB neural networks (bpDNN2Ea and bpDNN2Yield)
without requiring a MATLAB runtime installation, utilizing SciPy and NumPy.
"""

from __future__ import annotations
import os
import sys
import warnings
from pathlib import Path
from typing import Tuple, Optional

# Suppress duplicate variable name warnings from SciPy's mat loading
warnings.filterwarnings("ignore", message="Duplicate variable name")

import numpy as np
import scipy.io as sio

# Canonical feature list mapped to the raw column headers
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

# Mapping of alternative column names to canonical model feature names
CANONICAL_NAME_MAP = {
    "volatilematter": "VolatileMatters/%",
    "vm": "VolatileMatters/%",
    "fixedcarbon": "FixedCarbon/%",
    "fc": "FixedCarbon/%",
    "ash": "Ash/%",
    "c": "C/%",
    "h": "H/%",
    "o": "O/%",
    "n": "N/%",
    "s": "S/%",
    "sio2": "Ash_SiO2",
    "na2o": "Ash_Na2O",
    "mgo": "Ash_MgO",
    "al2o3": "Ash_Al2O3",
    "k2o": "Ash_K2O",
    "cao": "Ash_CaO",
    "p2o5": "Ash_P2O5",
    "cuo": "Ash_CuO",
    "zno": "Ash_ZnO",
    "fe2o3": "Ash_Fe2O3",
    "targettemperature": "TargetTemperature/Celsius",
    "heatingrate": "HeatingRate/(K/min)",
    "reactiontime": "ReactionTime/min",
    "reactortype": "ReactorType"
}


def clean_name_key(name: str) -> str:
    """Standardize column names to lowercase alphanumerics."""
    import re
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def canonicalize_feature_name(name: str) -> str:
    """Return canonical model feature name given a raw column header."""
    if not isinstance(name, str):
        return name
    key = clean_name_key(name)
    return CANONICAL_NAME_MAP.get(key, name)


class MatlabNeuralNetworkWrapper:
    """
    High-performance NumPy-based forward inference engine that replicates the
    activation and layer transformations of pre-trained MATLAB neural networks.
    """
    def __init__(self, net_struct, target_idx: Optional[int] = None):
        """
        Args:
            net_struct: MATLAB neural network structure loaded via scipy.io.loadmat.
            target_idx: Target index to select for multi-output predictions (e.g. 0=Biochar, 1=Bioliquid, 2=Biogas).
                        If None, return all outputs.
        """
        self.net_struct = net_struct
        self.target_idx = target_idx
        self.weights = []
        self.biases = []
        self.activation_funcs = []
        
        self._extract_architecture()
        
    def _extract_architecture(self):
        """Autonomously inspects and extracts network parameters from standard structures."""
        # Approach 1: Check standard BP neural network format with cell arrays
        if hasattr(self.net_struct, 'weight') and hasattr(self.net_struct, 'bias'):
            w_cell = self.net_struct.weight
            b_cell = self.net_struct.bias
            layer_info = getattr(self.net_struct, 'layer', None)
            
            for i in range(len(w_cell)):
                w = np.array(w_cell[i])
                b = np.array(b_cell[i])
                
                # Enforce output dimension sanity
                if i == len(w_cell) - 1:
                    if len(w.shape) == 1:
                        w = w.reshape(1, -1)
                    if len(b.shape) == 0 or (len(b.shape) == 1 and b.shape[0] == 0):
                        b = np.array([b]) if isinstance(b, (int, float)) else b.reshape(1)
                
                self.weights.append(w)
                self.biases.append(b)
                
                # Fetch activation functions
                if layer_info is not None and i < len(layer_info) and hasattr(layer_info[i], 'transferFcn'):
                    self.activation_funcs.append(str(layer_info[i].transferFcn).lower())
                else:
                    self.activation_funcs.append('logsig' if i < len(w_cell) - 1 else 'purelin')
            return
            
        # Approach 2: Check traditional legacy IW/LW/b structure
        if hasattr(self.net_struct, 'IW') and hasattr(self.net_struct, 'LW') and hasattr(self.net_struct, 'b'):
            iw = self.net_struct.IW
            lw = self.net_struct.LW
            b = self.net_struct.b
            
            # Layer 1
            self.weights.append(np.array(iw[0][0]))
            self.biases.append(np.array(b[0]))
            self.activation_funcs.append('tansig')
            
            # Subsequent layers
            for i in range(1, len(b)):
                self.weights.append(np.array(lw[i][i - 1]))
                self.biases.append(np.array(b[i]))
                self.activation_funcs.append('purelin' if i == len(b) - 1 else 'tansig')
            return
            
        raise AttributeError("Could not resolve or extract neural network parameters from the MATLAB structure.")

    def get_input_size(self) -> int:
        """Return expected input feature dimension."""
        if self.weights:
            return self.weights[0].shape[1]
        return len(BASE_FEATURE_NAMES)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Execute vectorized forward propagation through the network.
        
        Args:
            X: Input matrix of shape (samples, features) normalized to [0, 1].
        """
        a = np.asarray(X, dtype=float)
        
        for i in range(len(self.weights)):
            # Linear combination: z = a * W.T + b
            z = np.dot(a, self.weights[i].T) + self.biases[i]
            
            # Activation functions
            act = self.activation_funcs[i]
            if act == 'logsig':
                a = 1.0 / (1.0 + np.exp(-np.clip(z, -100, 100)))
            elif act == 'tansig':
                a = np.tanh(z)
            elif act == 'purelin':
                a = z
            else:
                a = np.maximum(0, z)  # Relu fallback
                
        # Select target index if multi-output
        if len(a.shape) > 1 and a.shape[1] > 1:
            if self.target_idx is not None and self.target_idx < a.shape[1]:
                return a[:, self.target_idx]
            return a
        return a.ravel()


def load_matlab_data(file_path: str) -> dict:
    """Load and clean mat file structure."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"MATLAB data file not found: {file_path}")
    
    mat_data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    return {k: v for k, v in mat_data.items() if not k.startswith('__')}


def extract_neural_network_data(mat_data: dict) -> Tuple[np.ndarray, np.ndarray, any]:
    """Extract input training limits, outputs, and model network structure."""
    X = None
    y = None
    net_struct = None
    
    # Locate inputs/outputs variables
    for k in ['X', 'Xnorm', 'x', 'nn_input', 'Variables0', 'Feedstock4training']:
        if k in mat_data:
            X = mat_data[k]
            break
    for k in ['Y', 'y', 'Ynorm', 'nn_output', 'ProductsYield']:
        if k in mat_data:
            y = mat_data[k]
            break
    if 'net' in mat_data:
        net_struct = mat_data['net']
        
    if X is None or y is None or net_struct is None:
        raise ValueError("MATLAB data is missing critical neural network or training matrices variables.")
        
    # Enforce standard (samples, features) layout
    if X.shape[0] < X.shape[1]:
        X = X.T
    if len(y.shape) > 1 and y.shape[0] < y.shape[1]:
        y = y.T
        
    return X, y, net_struct


def generate_feature_names(X: np.ndarray) -> list[str]:
    """Dynamically reconstruct complete input feature list (118 co-pyrolysis blends support)."""
    total_features = X.shape[1]
    base_count = len(BASE_FEATURE_NAMES)
    
    if total_features == base_count:
        return BASE_FEATURE_NAMES
        
    remaining = total_features - base_count
    feedstock_count = remaining // 2
    
    feature_names = BASE_FEATURE_NAMES.copy()
    for i in range(feedstock_count):
        feature_names.append(f'FeedstockType_{i+1}')
    for i in range(feedstock_count):
        feature_names.append(f'MixingRatio_{i+1}')
        
    return feature_names


def scale_features(X_raw: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """Scale raw features exactly to MATLAB [0, 1] bounds."""
    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)
    denom = train_max - train_min
    denom[denom == 0] = 1.0
    return np.clip((X_raw - train_min) / denom, 0.0, 1.0)


def unscale_outputs(y_pred: np.ndarray, y_train: np.ndarray, target_idx: Optional[int] = None) -> np.ndarray:
    """Unscale output predictions from network bounds to absolute values."""
    y_train_arr = np.asarray(y_train, dtype=float)
    if target_idx is not None:
        y_train_arr = y_train_arr[:, target_idx] if len(y_train_arr.shape) > 1 else y_train_arr
        
    tgt_min = y_train_arr.min(axis=0)
    tgt_max = y_train_arr.max(axis=0)
    tgt_range = tgt_max - tgt_min
    tgt_range[tgt_range == 0] = 1.0
    
    # MATLAB network targets normalization is typically [0, 1] or [-1, 1]
    if np.all(tgt_min >= -1e-6) and np.all(tgt_max <= 1.0 + 1e-6):
        return y_pred * tgt_range + tgt_min
    else:
        return ((y_pred + 1.0) / 2.0) * tgt_range + tgt_min


def load_trained_model(mat_path: Path) -> Tuple[
    MatlabNeuralNetworkWrapper,
    list[str],
    np.ndarray,
    np.ndarray,
]:
    """
    Zero-Configuration model loader.
    
    Returns:
        wrapper: Neural Network inference engine
        feature_names: List of variables aligned with network order
        X_train: Original training features matrix (used for [0, 1] scaling bounds)
        y_train: Original training outputs matrix (used for target bounds unscaling)
    """
    mat_data = load_matlab_data(str(mat_path))
    X_train, y_train, net_struct = extract_neural_network_data(mat_data)
    feature_names = generate_feature_names(X_train)
    wrapper = MatlabNeuralNetworkWrapper(net_struct)
    
    return wrapper, feature_names, X_train, y_train


def main():
    print("Surrogate models library loaded successfully.")


if __name__ == "__main__":
    main()

