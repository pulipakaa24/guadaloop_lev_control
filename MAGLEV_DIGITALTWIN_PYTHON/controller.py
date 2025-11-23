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
        self.preverror = 0
        self.cumerror = 0
        self.prevErrLR = 0
        self.cumErrLR = 0
        self.prevErrFB = 0
        self.cumErrFB = 0
    
    def reset(self):
        """Reset persistent variables"""
        self.preverror = 0
        self.cumerror = 0
        self.prevErrLR = 0
        self.cumErrLR = 0
        self.prevErrFB = 0
        self.cumErrFB = 0
    
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
        
        # Controller gains - average gap control
        kp = 14000
        ki = 0
        kd = 80000
        
        # Left-Right differential gains
        kpLR = 6000
        kiLR = 0
        kdLR = 12000
        
        # Front-Back differential gains
        kpFB = 6000
        kiFB = 0
        kdFB = 12000
        
        # Reference z position
        refz = R['rIstark'][2]
        
        # Calculate average gap and scalar error
        avg_gap = np.mean(gaps)
        err = -refz - avg_gap
        derr = err - self.preverror
        self.preverror = err
        self.cumerror += err
        
        # Difference between long-side sensors (left - right)
        # gaps indices: 0=front, 1=right, 2=back, 3=left
        long_side_err = gaps[3] - gaps[1]  # left - right
        long_side_derr = long_side_err - self.prevErrLR
        self.cumErrLR += long_side_err
        self.prevErrLR = long_side_err
        
        # Difference between short-side sensors (front - back)
        short_side_err = gaps[0] - gaps[2]  # front - back
        short_side_derr = short_side_err - self.prevErrFB
        self.cumErrFB += short_side_err
        self.prevErrFB = short_side_err
        
        # Apply same control to all yokes based on average error
        eadesired = (kp * err + derr * kd + ki * self.cumerror) * np.ones(4)
        
        # Negate since we're trying to counteract whatever error is happening
        eaLR = -(kpLR * long_side_err + kdLR * long_side_derr + kiLR * self.cumErrLR)
        eaFB = -(kpFB * short_side_err + kdFB * short_side_derr + kiFB * self.cumErrFB)
        
        # Apply differential corrections
        # Pattern: [FL, FR, BL, BR] = [front-left, front-right, back-left, back-right]
        # LR: [1, -1, -1, 1] means left side gets +, right side gets -
        # FB: [1, 1, -1, -1] means front gets +, back gets -
        eadesired += eaLR * np.array([1, -1, -1, 1])
        eadesired += eaFB * np.array([1, 1, -1, -1])
        
        # Apply saturation
        s = np.sign(eadesired)
        maxea = P['quadParams'].maxVoltage
        
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
