"""
Simulate maglev control
Ported from simulateMaglevControl.m
"""

import numpy as np
from scipy.integrate import solve_ivp
from utils import euler2dcm, dcm2euler
from dynamics import quad_ode_function_hf
from controller import DecentralizedPIDController


def simulate_maglev_control(R, S, P):
    """
    Simulates closed-loop control of a maglev quadrotor.
    
    Parameters
    ----------
    R : dict
        Reference structure with the following elements:
        - tVec : (N,) array of uniformly-sampled time offsets from initial time (seconds)
        - rIstar : (N, 3) array of desired CM positions in I frame (meters)
        - vIstar : (N, 3) array of desired CM velocities (meters/sec)
        - aIstar : (N, 3) array of desired CM accelerations (meters/sec^2)
        - xIstar : (N, 3) array of desired body x-axis direction (unit vectors)
    
    S : dict
        Simulation structure with the following elements:
        - oversampFact : Oversampling factor (must be >= 1)
        - state0 : Initial state dict with:
            - r : (3,) position in world frame (meters)
            - e : (3,) Euler angles (radians)
            - v : (3,) velocity (meters/sec)
            - omegaB : (3,) angular rate vector (rad/sec)
        - distMat : (N-1, 3) array of disturbance forces (Newtons)
    
    P : dict
        Parameters structure with:
        - quadParams : QuadParams object
        - constants : Constants object
    
    Returns
    -------
    Q : dict
        Output structure with:
        - tVec : (M,) array of output sample time points
        - state : dict with:
            - rMat : (M, 3) position matrix
            - eMat : (M, 3) Euler angles matrix
            - vMat : (M, 3) velocity matrix
            - omegaBMat : (M, 3) angular rate matrix
            - gaps : (M, 4) gap measurements
            - currents : (M, 4) yoke currents
    """
    # Extract initial state
    s0 = S['state0']
    r0 = s0['r']
    e0 = s0['e']
    v0 = s0['v']
    w0 = s0['omegaB']
    currents0 = np.zeros(4)
    
    R0 = euler2dcm(e0).flatten()
    
    x0 = np.concatenate([r0, v0, R0, w0, currents0])
    
    # Setup simulation parameters
    N = len(R['tVec'])
    dtIn = R['tVec'][1] - R['tVec'][0]
    dt = dtIn / S['oversampFact']
    
    p = {
        'quadParams': P['quadParams'],
        'constants': P['constants']
    }
    
    rl = P['quadParams'].rotor_loc
    
    # Initialize outputs
    tOut = []
    xOut = []
    gapsOut = []
    
    xk = x0
    
    # Create controller instance to maintain state
    controller = DecentralizedPIDController()
    
    for k in range(N - 1):  # loop through each time step
        tspan = np.arange(S['tVec'][k], S['tVec'][k+1] + dt/2, dt)
        
        # Setup reference for this time step
        Rk = {
            'rIstark': R['rIstar'][k, :],
            'vIstark': R['vIstar'][k, :],
            'aIstark': R['aIstar'][k, :],
            'xIstark': R['xIstar'][k, :]
        }
        
        # Setup state for this time step
        Sk = {
            'statek': {
                'rI': xk[0:3],
                'vI': xk[3:6],
                'RBI': xk[6:15],
                'omegaB': xk[15:18],
                'currents': xk[18:22]
            }
        }
        
        # Compute control voltage with noise
        eak = controller.control(Rk, Sk, P) + np.random.normal(0, 0.01, 4)
        
        # Disturbance for this time step
        dk = S['distMat'][k, :]
        
        # Integrate ODE
        sol = solve_ivp(
            lambda t, X: quad_ode_function_hf(t, X, eak, dk, p),
            [tspan[0], tspan[-1]],
            xk,
            t_eval=tspan,
            method='RK45',
            max_step=dt
        )
        
        tk = sol.t
        xk_traj = sol.y.T  # Transpose to get (time, state) shape
        
        # Calculate gaps for all time points in this segment
        NTK = len(tk)
        gapsk = np.zeros((NTK, 4))
        for q in range(NTK):
            R_q = xk_traj[q, 6:15].reshape(3, 3)
            gapskq = np.abs(xk_traj[q, 2]) - np.array([0, 0, 1]) @ R_q.T @ rl
            gapsk[q, :] = gapskq.flatten()
        
        # Store results (excluding last point to avoid duplication)
        tOut.append(tk[:-1])
        xOut.append(xk_traj[:-1, :])
        gapsOut.append(gapsk[:-1, :])
        
        # Prepare for next iteration
        xk = xk_traj[-1, :]
    
    # Concatenate all segments
    tOut = np.concatenate(tOut)
    xOut = np.vstack(xOut)
    gapsOut = np.vstack(gapsOut)
    
    # Add final point
    tOut = np.append(tOut, tk[-1])
    xOut = np.vstack([xOut, xk_traj[-1, :]])
    gapsOut = np.vstack([gapsOut, gapsk[-1, :]])
    
    M = len(tOut)
    
    # Form Euler angles from rotation matrices
    eMat = np.zeros((M, 3))
    for k in range(M):
        Rk = xOut[k, 6:15].reshape(3, 3)
        ek = dcm2euler(Rk)
        eMat[k, :] = np.real(ek)
    
    # Assemble output structure
    Q = {
        'tVec': tOut,
        'state': {
            'rMat': xOut[:, 0:3],
            'vMat': xOut[:, 3:6],
            'eMat': eMat,
            'omegaBMat': xOut[:, 15:18],
            'gaps': gapsOut,
            'currents': xOut[:, 18:22]
        }
    }
    
    return Q
