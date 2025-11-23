"""
ODE function for maglev quadrotor dynamics
Ported from quadOdeFunctionHF.m
"""

import numpy as np
from utils import cross_product_equivalent, fmag2


def quad_ode_function_hf(t, X, eaVec, distVec, P):
    """
    Ordinary differential equation function that models quadrotor dynamics 
    -- high-fidelity version. For use with scipy's ODE solvers (e.g., solve_ivp).
    
    Parameters
    ----------
    t : float
        Scalar time input, as required by ODE solver format
    
    X : ndarray, shape (22,)
        Quad state, arranged as:
        X = [rI, vI, RBI(flattened), omegaB, currents]
        
        rI = position vector in I in meters (3 elements)
        vI = velocity vector wrt I and in I, in meters/sec (3 elements)
        RBI = attitude matrix from I to B frame, flattened (9 elements)
        omegaB = angular rate vector of body wrt I, expressed in B, rad/sec (3 elements)
        currents = vector of yoke currents, in amperes (4 elements)
    
    eaVec : ndarray, shape (4,)
        4-element vector of voltages applied to yokes, in volts
    
    distVec : ndarray, shape (3,)
        3-element vector of constant disturbance forces acting on quad's
        center of mass, expressed in Newtons in I
    
    P : dict
        Structure with the following elements:
        - quadParams : QuadParams object
        - constants : Constants object
    
    Returns
    -------
    Xdot : ndarray, shape (22,)
        Time derivative of the input vector X
    """
    yokeR = P['quadParams'].yokeR  # in ohms
    yokeL = P['quadParams'].yokeL  # in henries
    
    # Extract state variables
    currents = X[18:22]  # indices 19:22 in MATLAB (1-indexed) = 18:22 in Python
    currentsdot = np.zeros(4)
    
    R = X[6:15].reshape(3, 3)  # indices 7:15 in MATLAB = 6:15 in Python
    
    w = X[15:18]  # indices 16:18 in MATLAB = 15:18 in Python
    wx = cross_product_equivalent(w)
    rdot = X[3:6]  # indices 4:6 in MATLAB = 3:6 in Python
    
    z = X[2]  # index 3 in MATLAB = 2 in Python (zI of cg)
    
    rl = P['quadParams'].sensor_loc
    gaps = np.abs(z) - np.array([0, 0, 1]) @ R.T @ rl
    gaps = gaps.flatten()
    
    # Calculate magnetic forces
    Fm = fmag2(currents, gaps)
    
    # Handle collision with track
    if np.any(Fm == -1):
        rdot = rdot.copy()
        rdot[2] = 0  # rdot(end) in MATLAB
        Fm = np.zeros(4)
    
    sumF = np.sum(Fm)
    
    # Calculate linear acceleration
    vdot = (np.array([0, 0, -P['quadParams'].m * P['constants'].g]) + 
            np.array([0, 0, sumF]) + 
            distVec) / P['quadParams'].m
    
    # Calculate rotation rate
    Rdot = -wx @ R
    
    # Calculate torques on body
    Nb = np.zeros(3)
    for i in range(4):  # loop through each yoke
        # Voltage-motor modeling
        currentsdot[i] = (eaVec[i] - currents[i] * yokeR) / yokeL
        
        NiB = np.zeros(3)  # since yokes can't cause moment by themselves
        FiB = np.array([0, 0, Fm[i]])
        
        # Accumulate torque (rotate FiB to inertial frame since fmag is always towards track)
        Nb = Nb + (NiB + cross_product_equivalent(P['quadParams'].rotor_loc[:, i]) @ R.T @ FiB)
    
    # Calculate angular acceleration
    wdot = P['quadParams'].invJq @ (Nb - wx @ P['quadParams'].Jq @ w)
    
    # Enforce constraint if pod is against track
    if np.any(Fm == -1):
        vdot[2] = 0
    
    # Assemble state derivative
    Xdot = np.concatenate([
        rdot,              # 3 elements
        vdot,              # 3 elements
        Rdot.flatten(),    # 9 elements
        wdot,              # 3 elements
        currentsdot        # 4 elements
    ])
    
    return Xdot
