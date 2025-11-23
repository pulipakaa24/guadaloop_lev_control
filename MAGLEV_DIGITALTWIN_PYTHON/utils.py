"""
Utility functions for maglev simulation
Ported from MATLAB to Python
"""

import numpy as np


# Global magnetic characteristics with random variation
# These are initialized once and persist across function calls
_mag_characteristics = None


def initialize_magnetic_characteristics(noise_level=0.05, seed=None):
    """
    Initialize magnetic force characteristics with random variation.
    Call this once at the start of a simulation to set random parameters.
    
    Parameters
    ----------
    noise_level : float, optional
        Standard deviation of multiplicative noise (default: 0.05 = 5%)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Dictionary with perturbed magnetic coefficients
    """
    global _mag_characteristics
    
    if seed is not None:
        np.random.seed(seed)
    
    # Nominal coefficients from fmag2
    nominal = {
        'N': 250,
        'const1': 0.2223394555e5,
        'const2': 0.2466906550e10,
        'const3': 0.6886569338e8,
        'const4': 0.6167266375e9,
        'const5': 0.3042813963e19
    }
    
    # Add multiplicative noise to each coefficient
    _mag_characteristics = {
        key: value * (1 + np.random.normal(0, noise_level))
        for key, value in nominal.items()
    }
    
    return _mag_characteristics.copy()


def get_magnetic_characteristics():
    """
    Get current magnetic characteristics. If not initialized, use nominal values.
    
    Returns
    -------
    dict
        Dictionary with magnetic coefficients
    """
    global _mag_characteristics
    
    if _mag_characteristics is None:
        # Use nominal values if not initialized
        _mag_characteristics = {
            'N': 250,
            'const1': 0.2223394555e5,
            'const2': 0.2466906550e10,
            'const3': 0.6886569338e8,
            'const4': 0.6167266375e9,
            'const5': 0.3042813963e19
        }
    
    return _mag_characteristics


def reset_magnetic_characteristics():
    """Reset magnetic characteristics to force reinitialization"""
    global _mag_characteristics
    _mag_characteristics = None


def cross_product_equivalent(u):
    """
    Outputs the cross-product-equivalent matrix uCross such that for arbitrary 
    3-by-1 vectors u and v, cross(u,v) = uCross @ v.
    
    Parameters
    ----------
    u : array_like, shape (3,)
        3-element vector
    
    Returns
    -------
    uCross : ndarray, shape (3, 3)
        Skew-symmetric cross-product equivalent matrix
    """
    u1, u2, u3 = u[0], u[1], u[2]
    
    uCross = np.array([
        [0, -u3, u2],
        [u3, 0, -u1],
        [-u2, u1, 0]
    ])
    
    return uCross


def rotation_matrix(aHat, phi):
    """
    Generates the rotation matrix R corresponding to a rotation through an 
    angle phi about the axis defined by the unit vector aHat. This is a 
    straightforward implementation of Euler's formula for a rotation matrix.
    
    Parameters
    ----------
    aHat : array_like, shape (3,)
        3-by-1 unit vector constituting the axis of rotation
    phi : float
        Angle of rotation, in radians
    
    Returns
    -------
    R : ndarray, shape (3, 3)
        Rotation matrix
    """
    aHat = np.array(aHat).reshape(3, 1)
    R = (np.cos(phi) * np.eye(3) + 
         (1 - np.cos(phi)) * (aHat @ aHat.T) - 
         np.sin(phi) * cross_product_equivalent(aHat.flatten()))
    
    return R


def euler2dcm(e):
    """
    Converts Euler angles phi = e[0], theta = e[1], and psi = e[2]
    (in radians) into a direction cosine matrix for a 3-1-2 rotation.
    
    Let the world (W) and body (B) reference frames be initially aligned. In a
    3-1-2 order, rotate B away from W by angles psi (yaw, about the body Z
    axis), phi (roll, about the body X axis), and theta (pitch, about the body Y
    axis). R_BW can then be used to cast a vector expressed in W coordinates as
    a vector in B coordinates: vB = R_BW @ vW
    
    Parameters
    ----------
    e : array_like, shape (3,)
        Vector containing the Euler angles in radians: phi = e[0], 
        theta = e[1], and psi = e[2]
    
    Returns
    -------
    R_BW : ndarray, shape (3, 3)
        Direction cosine matrix
    """
    a1 = np.array([0, 0, 1])
    a2 = np.array([1, 0, 0])
    a3 = np.array([0, 1, 0])
    
    phi = e[0]
    theta = e[1]
    psi = e[2]
    
    R_BW = rotation_matrix(a3, theta) @ rotation_matrix(a2, phi) @ rotation_matrix(a1, psi)
    
    return R_BW


def dcm2euler(R_BW):
    """
    Converts a direction cosine matrix R_BW to Euler angles phi = e[0], 
    theta = e[1], and psi = e[2] (in radians) for a 3-1-2 rotation. 
    If the conversion to Euler angles is singular (not unique), then this 
    function raises an error.
    
    Parameters
    ----------
    R_BW : ndarray, shape (3, 3)
        Direction cosine matrix
    
    Returns
    -------
    e : ndarray, shape (3,)
        Vector containing the Euler angles in radians: phi = e[0], 
        theta = e[1], and psi = e[2]
    
    Raises
    ------
    ValueError
        If gimbal lock occurs (|phi| = pi/2)
    """
    R_BW = np.real(R_BW)
    
    if R_BW[1, 2] == 1:
        raise ValueError("Error: Gimbal lock since |phi| = pi/2")
    
    theta = np.arctan2(-R_BW[0, 2], R_BW[2, 2])
    phi = np.arcsin(R_BW[1, 2])
    psi = np.arctan2(-R_BW[1, 0], R_BW[1, 1])
    
    e = np.array([phi, theta, psi])
    
    return e


def fmag2(i, z):
    """
    Converts a given gap distance, z, and current, i, to yield the attraction
    force Fm to a steel plate.
    
    i > 0 runs current in direction to weaken Fm
    i < 0 runs current to strengthen Fm
    
    z is positive for valid conditions
    
    Parameters
    ----------
    i : float or ndarray
        Current in amperes
    z : float or ndarray
        Gap distance in meters (must be positive)
    
    Returns
    -------
    Fm : float or ndarray
        Magnetic force in Newtons (-1 if z < 0, indicating error)
    """
    # Get current magnetic characteristics (nominal or perturbed)
    mc = get_magnetic_characteristics()
    
    # Handle scalar and array inputs
    z_scalar = np.isscalar(z)
    i_scalar = np.isscalar(i)
    
    if z_scalar and i_scalar:
        if z < 0:
            return -1
        
        term1 = (-2 * i * mc['N'] + mc['const1'])
        denominator = (mc['const2'] * z + mc['const3'])
        
        Fm = (mc['const4'] * term1**2 / denominator**2 - 
              mc['const5'] * z * term1**2 / denominator**3)
        
        return Fm
    else:
        # Handle array inputs
        z = np.asarray(z)
        i = np.asarray(i)
        
        # Check for negative z values
        if np.any(z < 0):
            # Create output array
            Fm = np.zeros_like(z, dtype=float)
            valid_mask = z >= 0
            
            if np.any(valid_mask):
                z_valid = z[valid_mask]
                i_valid = i if i_scalar else i[valid_mask]
                
                term1 = (-2 * i_valid * mc['N'] + mc['const1'])
                denominator = (mc['const2'] * z_valid + mc['const3'])
                
                Fm[valid_mask] = (mc['const4'] * term1**2 / denominator**2 - 
                                   mc['const5'] * z_valid * term1**2 / denominator**3)
            
            Fm[~valid_mask] = -1
            return Fm
        else:
            term1 = (-2 * i * mc['N'] + mc['const1'])
            denominator = (mc['const2'] * z + mc['const3'])
            
            Fm = (mc['const4'] * term1**2 / denominator**2 - 
                  mc['const5'] * z * term1**2 / denominator**3)
            
            return Fm
