"""
Decentralized PID controller for maglev system
Ported from decentralizedPIDcontroller.m
"""

import numpy as np


class DecentralizedPIDController:
    """
    Decentralized PID controller for quadrotor/maglev control.
    Controls altitude, roll, and pitch using gap sensor feedback.
    """
    
    def __init__(self):
        # Persistent variables (maintain state between calls)
        self.preverror = np.zeros(4)
        self.cumerror = np.zeros(4)
    
    def reset(self):
        """Reset persistent variables"""
        self.preverror = np.zeros(4)
        self.cumerror = np.zeros(4)
    
    def control(self, R, S, P):
        """
        Compute control voltages for each yoke.
        
        Parameters
        ----------
        R : dict
            Reference structure with elements:
            - rIstark : 3-element array of desired CM position at time tk in I frame (meters)
            - vIstark : 3-element array of desired CM velocity (meters/sec)
            - aIstark : 3-element array of desired CM acceleration (meters/sec^2)
        
        S : dict
            State structure with element:
            - statek : dict containing:
                - rI : 3-element position in I frame (meters)
                - RBI : 3x3 or 9-element direction cosine matrix
                - vI : 3-element velocity (meters/sec)
                - omegaB : 3-element angular rate vector in body frame (rad/sec)
        
        P : dict
            Parameters structure with elements:
            - quadParams : QuadParams object
            - constants : Constants object
        
        Returns
        -------
        ea : ndarray, shape (4,)
            4-element vector with voltages applied to each yoke
        """
        # Extract current state
        zcg = S['statek']['rI'][2]  # z-component of CG position
        rl = P['quadParams'].sensor_loc
        
        # Reshape RBI if needed
        RBI = S['statek']['RBI']
        if RBI.shape == (9,):
            RBI = RBI.reshape(3, 3)
        
        # Calculate gaps at sensor locations
        gaps = np.abs(zcg) - np.array([0, 0, 1]) @ RBI.T @ rl
        gaps = gaps.flatten()
        
        # Controller gains
        kp = 10000
        ki = 0
        kd = 50000
        
        # Reference z position
        refz = R['rIstark'][2]
        meangap = np.mean(gaps)
        
        # Transform gaps using corner-based sensing
        # gaps indices: 0=front, 1=right, 2=back, 3=left
        gaps_transformed = np.array([
            gaps[0] + gaps[3] - meangap,  # gaps(1)+gaps(4)-meangap in MATLAB
            gaps[0] + gaps[1] - meangap,  # gaps(1)+gaps(2)-meangap
            gaps[2] + gaps[1] - meangap,  # gaps(3)+gaps(2)-meangap
            gaps[2] + gaps[3] - meangap   # gaps(3)+gaps(4)-meangap
        ])
        
        # Calculate error for each yoke
        err = -refz - gaps_transformed
        derr = err - self.preverror
        
        self.preverror = err
        self.cumerror = self.cumerror + err
        
        # Calculate desired voltages
        eadesired = kp * err + derr * kd + ki * self.cumerror
        
        # Apply saturation
        s = np.sign(eadesired)
        maxea = P['quadParams'].maxVoltage * np.ones(4)
        
        ea = s * np.minimum(np.abs(eadesired), maxea)
        
        return ea


def decentralized_pid_controller(R, S, P, controller=None):
    """
    Wrapper function to maintain compatibility with MATLAB-style function calls.
    
    Parameters
    ----------
    R, S, P : dict
        See DecentralizedPIDController.control() for details
    controller : DecentralizedPIDController, optional
        Controller instance to use. If None, creates a new one (loses state)
    
    Returns
    -------
    ea : ndarray, shape (4,)
        4-element vector with voltages applied to each yoke
    """
    if controller is None:
        controller = DecentralizedPIDController()
    
    return controller.control(R, S, P)
