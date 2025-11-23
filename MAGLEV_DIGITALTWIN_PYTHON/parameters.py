"""
Quad parameters and constants for maglev simulation
Ported from quadParamsScript.m and constantsScript.m
"""

import numpy as np


# Global parameter variations - initialized once per simulation run
_param_variations = None


def initialize_parameter_variations(noise_level=0.05, seed=None):
    """
    Initialize mechanical and electrical parameter variations.
    Call this once at the start of a simulation to set random variations.
    
    Parameters
    ----------
    noise_level : float, optional
        Standard deviation of multiplicative noise (default: 0.05 = 5%)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Dictionary with parameter variation factors
    """
    global _param_variations
    
    if seed is not None:
        np.random.seed(seed)
    
    # Generate multiplicative variation factors for each parameter
    _param_variations = {
        'mass': 1 + np.random.normal(0, noise_level),
        'Jq_components': np.array([1 + np.random.normal(0, noise_level) for _ in range(3)]),  # Individual noise for each inertia component
        'frame_l': 1 + np.random.normal(0, noise_level * 0.5),  # Smaller variation for dimensions
        'frame_w': 1 + np.random.normal(0, noise_level * 0.5),
        'yh': 1 + np.random.normal(0, noise_level * 0.5),
        # Individual yoke variations (4 yokes)
        'yoke_R': np.array([1 + np.random.normal(0, noise_level * 0.3) for _ in range(4)]),
        'yoke_L': np.array([1 + np.random.normal(0, noise_level * 0.3) for _ in range(4)]),
        # Position noise for yoke/rotor locations (4 locations, 3D)
        'rotor_pos_noise': np.random.normal(0, noise_level * 0.2, (3, 4)),
        # Position noise for sensor locations (4 sensors, 3D)
        'sensor_pos_noise': np.random.normal(0, noise_level * 0.2, (3, 4))
    }
    
    return _param_variations.copy()


def get_parameter_variations():
    """
    Get current parameter variations. If not initialized, use nominal (no variation).
    
    Returns
    -------
    dict
        Dictionary with parameter variation factors
    """
    global _param_variations
    
    if _param_variations is None:
        # Use nominal values (no variation)
        _param_variations = {
            'mass': 1.0,
            'Jq_components': np.ones(3),
            'frame_l': 1.0,
            'frame_w': 1.0,
            'yh': 1.0,
            'yoke_R': np.ones(4),
            'yoke_L': np.ones(4),
            'rotor_pos_noise': np.zeros((3, 4)),
            'sensor_pos_noise': np.zeros((3, 4))
        }
    
    return _param_variations


def reset_parameter_variations():
    """Reset parameter variations to force reinitialization"""
    global _param_variations
    _param_variations = None


class QuadParams:
    """Quadrotor/maglev pod parameters"""
    
    def __init__(self):
        # Get parameter variations
        pv = get_parameter_variations()
        
        # Pod mechanical characteristics (with variations)
        frame_l = 0.61 * pv['frame_l']
        frame_w = 0.149 * pv['frame_w']
        self.yh = 3 * 0.0254 * pv['yh']  # yoke height
        yh = self.yh
        
        # Store dimensions for reference
        self.frame_l = frame_l
        self.frame_w = frame_w
        
        # Yoke/rotor locations (at corners) with position noise
        nominal_rotor_loc = np.array([
            [frame_l/2, frame_l/2, -frame_l/2, -frame_l/2],
            [-frame_w/2, frame_w/2, frame_w/2, -frame_w/2],
            [yh, yh, yh, yh]
        ])
        self.rotor_loc = nominal_rotor_loc * (1 + pv['rotor_pos_noise'])
        
        # Sensor locations (independent from yoke/rotor locations, at edge centers) with position noise
        nominal_sensor_loc = np.array([
            [frame_l/2, 0, -frame_l/2, 0],
            [0, frame_w/2, 0, -frame_w/2],
            [yh, yh, yh, yh]
        ])
        self.sensor_loc = nominal_sensor_loc * (1 + pv['sensor_pos_noise'])
        
        self.gap_sigma = 0.5e-3  # usually on micron scale
        
        # Mass of the quad, in kg (with variation)
        self.m = 6 * pv['mass']
        
        # The quad's moment of inertia, expressed in the body frame, in kg-m^2 (with individual component variations)
        nominal_Jq = np.diag([0.017086, 0.125965, 0.131940])
        self.Jq = nominal_Jq * np.diag(pv['Jq_components'])  # Apply different noise to each diagonal component
        self.invJq = np.linalg.inv(self.Jq)
        
        # Quad electrical characteristics (with variations)
        maxcurrent = 30
        
        # Individual yoke resistances and inductances (4 yokes with individual variations)
        self.yokeR_individual = 2.2 * pv['yoke_R']  # 4-element array
        self.yokeL_individual = 5e-3 * pv['yoke_L']  # 4-element array
        
        self.maxVoltage = maxcurrent * self.yokeR_individual  # max magnitude voltage supplied to each yoke


class Constants:
    """Physical constants"""
    
    def __init__(self):
        # Acceleration due to gravity, in m/s^2
        self.g = 9.81
        
        # Mass density of moist air, in kg/m^3
        self.rho = 1.225
