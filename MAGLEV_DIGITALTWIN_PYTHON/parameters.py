"""
Quad parameters and constants for maglev simulation
Ported from quadParamsScript.m and constantsScript.m
"""

import numpy as np


class QuadParams:
    """Quadrotor/maglev pod parameters"""
    
    def __init__(self):
        # Pod mechanical characteristics
        frame_l = 0.61
        frame_w = 0.149  # in meters
        self.yh = 3 * 0.0254  # yoke height
        yh = self.yh
        
        # Yoke/rotor locations (at corners)
        self.rotor_loc = np.array([
            [frame_l/2, frame_l/2, -frame_l/2, -frame_l/2],
            [-frame_w/2, frame_w/2, frame_w/2, -frame_w/2],
            [yh, yh, yh, yh]
        ])
        
        # Sensor locations (independent from yoke/rotor locations, at edge centers)
        self.sensor_loc = np.array([
            [frame_l/2, 0, -frame_l/2, 0],
            [0, frame_w/2, 0, -frame_w/2],
            [yh, yh, yh, yh]
        ])
        
        self.gap_sigma = 0.5e-3  # usually on micron scale
        
        # Mass of the quad, in kg
        self.m = 6
        
        # The quad's moment of inertia, expressed in the body frame, in kg-m^2
        self.Jq = np.diag([0.017086, 0.125965, 0.131940])
        self.invJq = np.linalg.inv(self.Jq)
        
        # Quad electrical characteristics
        maxcurrent = 30
        self.yokeR = 2.2  # in ohms
        self.yokeL = 5e-3  # in henries (2.5mH per yoke)
        self.maxVoltage = maxcurrent * self.yokeR  # max magnitude voltage supplied to each yoke


class Constants:
    """Physical constants"""
    
    def __init__(self):
        # Acceleration due to gravity, in m/s^2
        self.g = 9.81
        
        # Mass density of moist air, in kg/m^3
        self.rho = 1.225
