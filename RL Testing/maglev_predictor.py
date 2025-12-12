"""
Magnetic Levitation Force and Torque Predictor
Optimized Inference using Pre-Trained Scikit-Learn Model

This module loads a saved .pkl model (PolynomialFeatures + LinearRegression)
and executes predictions using optimized NumPy vectorization for high-speed simulation.

Usage:
    predictor = MaglevPredictor("maglev_model.pkl")
    force, torque = predictor.predict(currL=-15, currR=-15, roll=1.0, gap_height=10.0)
"""

import numpy as np
import joblib
import os

class MaglevPredictor:
    def __init__(self, model_path='maglev_model.pkl'):
        """
        Initialize predictor by loading the pickle file and extracting
        raw matrices for fast inference.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found. Please train and save the model first.")

        print(f"Loading maglev model from {model_path}...")
        data = joblib.load(model_path)
        
        # 1. Extract Scikit-Learn Objects
        poly_transformer = data['poly_features']
        linear_model = data['model']
        
        # 2. Extract Raw Matrices for Speed (Bypasses sklearn overhead)
        # powers_: Matrix of shape (n_output_features, n_input_features)
        # Represents the exponents for each term. e.g. x1^2 * x2^1
        self.powers = poly_transformer.powers_.T # Transpose for broadcasting
        
        # coef_: Shape (n_targets, n_output_features) -> (2, n_poly_terms)
        self.coef = linear_model.coef_
        
        # intercept_: Shape (n_targets,) -> (2,)
        self.intercept = linear_model.intercept_
        
        print(f"Model loaded. Degree: {data['degree']}")
        print(f"Force R2: {data['performance']['force_r2']:.4f}")
        print(f"Torque R2: {data['performance']['torque_r2']:.4f}")

    def predict(self, currL, currR, roll, gap_height):
        """
        Fast single-sample prediction using pure NumPy.
        
        Args:
            currL, currR: Currents [A]
            roll: Roll angle [deg]
            gap_height: Gap [mm]
            
        Returns:
            (force [N], torque [mN·m])
        """
        # 1. Pre-process Input (Must match training order: currL, currR, roll, invGap)
        # Clamp gap to avoid division by zero
        safe_gap = max(gap_height, 1e-6)
        invGap = 1.0 / safe_gap
        
        # Input Vector: shape (4,)
        x = np.array([currL, currR, roll, invGap])
        
        # 2. Polynomial Expansion (Vectorized)
        # Compute x^p for every term. 
        # x is (4,), self.powers is (4, n_terms)
        # Broadcasting: x[:, None] is (4,1). Result is (4, n_terms).
        # Product along axis 0 reduces it to (n_terms,)
        
        # Note: This is equivalent to PolynomialFeatures.transform but 100x faster for single samples
        poly_features = np.prod(x[:, None] ** self.powers, axis=0)
        
        # 3. Linear Prediction
        # (2, n_terms) dot (n_terms,) -> (2,)
        result = np.dot(self.coef, poly_features) + self.intercept
        
        return result[0], result[1]

    def predict_batch(self, currL, currR, roll, gap_height):
        """
        Fast batch prediction for array inputs.
        """
        # 1. Pre-process Inputs
        gap_height = np.asarray(gap_height)
        invGap = 1.0 / np.maximum(gap_height, 1e-6)
        
        # Stack inputs: shape (N, 4)
        X = np.column_stack((currL, currR, roll, invGap))
        
        # 2. Polynomial Expansion
        # X is (N, 4). Powers is (4, n_terms).
        # We want (N, n_terms).
        # Method: X[:, :, None] -> (N, 4, 1)
        # Powers[None, :, :] -> (1, 4, n_terms)
        # Power: (N, 4, n_terms)
        # Prod axis 1: (N, n_terms)
        poly_features = np.prod(X[:, :, None] ** self.powers[None, :, :], axis=1)
        
        # 3. Linear Prediction
        # (N, n_terms) @ (n_terms, 2) + (2,)
        result = np.dot(poly_features, self.coef.T) + self.intercept
        
        return result[:, 0], result[:, 1]

if __name__ == "__main__":
    # Test
    try:
        p = MaglevPredictor()
        f, t = p.predict(-15, -15, 1.0, 10.0)
        print(f"Force: {f:.3f}, Torque: {t:.3f}")
    except FileNotFoundError as e:
        print(e)