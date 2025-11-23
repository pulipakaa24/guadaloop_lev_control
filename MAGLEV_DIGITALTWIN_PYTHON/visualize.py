"""
Visualization function for quadrotor/maglev pod
Ported from visualizeQuad.m
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import interp1d
from utils import euler2dcm
from parameters import QuadParams


def visualize_quad(S):
    """
    Takes in an input structure S and visualizes the resulting 3D motion 
    in approximately real-time. Outputs the data used to form the plot.
    
    Parameters
    ----------
    S : dict
        Structure with the following elements:
        - rMat : (M, 3) matrix of quad positions, in meters
        - eMat : (M, 3) matrix of quad attitudes, in radians
        - tVec : (M,) vector of times corresponding to each measurement
        - plotFrequency : scalar number of frames per second (Hz)
        - bounds : 6-element list [xmin, xmax, ymin, ymax, zmin, zmax]
        - makeGifFlag : boolean (if True, export plot to .gif)
        - gifFileName : string with the file name of the .gif
    
    Returns
    -------
    P : dict
        Structure with the following elements:
        - tPlot : (N,) vector of time points used in the plot
        - rPlot : (N, 3) vector of positions used to generate the plot
        - ePlot : (N, 3) vector of attitudes used to generate the plot
    """
    # UT colors
    burntOrangeUT = np.array([191, 87, 0]) / 255
    darkGrayUT = np.array([51, 63, 72]) / 255
    
    # Get quad parameters
    quad_params = QuadParams()
    
    frame_l = 1.2192
    frame_w = 0.711  # in meters
    yh = quad_params.yh
    
    # Parameters for the rotors
    rotorLocations = np.array([
        [frame_l/2, frame_l/2, -frame_l/2, -frame_l/2],
        [frame_w/2, -frame_w/2, frame_w/2, -frame_w/2],
        [yh, yh, yh, yh]
    ])
    r_rotor = 0.13
    
    # Determine the location of the corners of the body box in the body frame
    bpts = np.array([
        [frame_l/2, frame_l/2, -frame_l/2, -frame_l/2, frame_l/2, frame_l/2, -frame_l/2, -frame_l/2],
        [frame_w/2, -frame_w/2, frame_w/2, -frame_w/2, frame_w/2, -frame_w/2, frame_w/2, -frame_w/2],
        [0.03, 0.03, 0.03, 0.03, -0.030, -0.030, -0.030, -0.030]
    ])
    
    # Rectangles representing each side of the body box
    b1 = bpts[:, [0, 4, 5, 1]]
    b2 = bpts[:, [0, 4, 6, 2]]
    b3 = bpts[:, [2, 6, 7, 3]]
    b4 = bpts[:, [0, 2, 3, 1]]
    b5 = bpts[:, [4, 6, 7, 5]]
    b6 = bpts[:, [1, 5, 7, 3]]
    
    # Create a circle for each rotor
    t_circ = np.linspace(0, 2*np.pi, 20)
    circpts = np.zeros((3, 20))
    for i in range(20):
        circpts[:, i] = r_rotor * np.array([np.cos(t_circ[i]), np.sin(t_circ[i]), 0])
    
    m = len(S['tVec'])
    
    # Single epoch plot
    if m == 1:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract params
        RIB = euler2dcm(S['eMat'][0, :]).T
        r = S['rMat'][0, :]
        
        # Plot rotors
        for i in range(4):
            rotor_circle = r.reshape(3, 1) + RIB @ (circpts + rotorLocations[:, i].reshape(3, 1))
            ax.plot(rotor_circle[0, :], rotor_circle[1, :], rotor_circle[2, :], 
                   color=darkGrayUT, linewidth=2)
        
        # Plot body
        plot_body(ax, r, RIB, [b1, b2, b3, b4, b5, b6])
        
        # Plot body axes
        plot_axes(ax, r, RIB)
        
        # Add steel plate
        plot_steel_plate(ax, S['bounds'])
        
        ax.set_xlim([S['bounds'][0], S['bounds'][1]])
        ax.set_ylim([S['bounds'][2], S['bounds'][3]])
        ax.set_zlim([S['bounds'][4], S['bounds'][5]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True)
        
        plt.show()
        
        P = {
            'tPlot': S['tVec'],
            'rPlot': S['rMat'],
            'ePlot': S['eMat']
        }
        
    else:  # Animation
        # Create time vectors
        tf = 1 / S['plotFrequency']
        tmax = S['tVec'][-1]
        tmin = S['tVec'][0]
        tPlot = np.arange(tmin, tmax + tf/2, tf)
        tPlotLen = len(tPlot)
        
        # Interpolate to regularize times
        t2unique, indUnique = np.unique(S['tVec'], return_index=True)
        
        rPlot_interp = interp1d(t2unique, S['rMat'][indUnique, :], axis=0, 
                                kind='linear', fill_value='extrapolate')
        ePlot_interp = interp1d(t2unique, S['eMat'][indUnique, :], axis=0, 
                                kind='linear', fill_value='extrapolate')
        
        rPlot = rPlot_interp(tPlot)
        ePlot = ePlot_interp(tPlot)
        
        # Create figure and axis
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Storage for plot elements
        plot_elements = []
        
        def init():
            ax.set_xlim([S['bounds'][0], S['bounds'][1]])
            ax.set_ylim([S['bounds'][2], S['bounds'][3]])
            ax.set_zlim([S['bounds'][4], S['bounds'][5]])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=0, azim=0)
            ax.grid(True)
            
            # Add steel plate
            plot_steel_plate(ax, S['bounds'])
            
            return []
        
        def update(frame):
            nonlocal plot_elements
            
            # Clear previous frame
            for elem in plot_elements:
                elem.remove()
            plot_elements = []
            
            # Extract data
            RIB = euler2dcm(ePlot[frame, :]).T
            r = rPlot[frame, :]
            
            # Plot rotors
            for i in range(4):
                rotor_circle = r.reshape(3, 1) + RIB @ (circpts + rotorLocations[:, i].reshape(3, 1))
                line, = ax.plot(rotor_circle[0, :], rotor_circle[1, :], rotor_circle[2, :], 
                               color=darkGrayUT, linewidth=2)
                plot_elements.append(line)
            
            # Plot body
            body_elem = plot_body(ax, r, RIB, [b1, b2, b3, b4, b5, b6])
            plot_elements.append(body_elem)
            
            # Plot axes
            axis_elems = plot_axes(ax, r, RIB)
            plot_elements.extend(axis_elems)
            
            ax.set_title(f'Time: {tPlot[frame]:.3f} s')
            
            return plot_elements
        
        anim = FuncAnimation(fig, update, init_func=init, frames=tPlotLen,
                           interval=tf*1000, blit=False, repeat=True)
        
        # Save as GIF if requested
        if S.get('makeGifFlag', False):
            writer = PillowWriter(fps=S['plotFrequency'])
            anim.save(S['gifFileName'], writer=writer)
            print(f"Animation saved to {S['gifFileName']}")
            plt.close(fig)  # Close without displaying
        else:
            plt.show()
        
        P = {
            'tPlot': tPlot,
            'rPlot': rPlot,
            'ePlot': ePlot
        }
    
    return P


def plot_body(ax, r, RIB, body_faces):
    """Plot the body box"""
    vertices = []
    for face in body_faces:
        face_r = r.reshape(3, 1) + RIB @ face
        vertices.append(face_r.T)
    
    # Create 3D polygon collection
    poly = Poly3DCollection(vertices, alpha=0.5, facecolor=[0.5, 0.5, 0.5], 
                           edgecolor='black', linewidths=1)
    ax.add_collection3d(poly)
    return poly


def plot_steel_plate(ax, bounds):
    """
    Plot a steel plate at z=0 with 0.25 inch thickness
    
    Parameters
    ----------
    ax : Axes3D
        The 3D axis to plot on
    bounds : list
        6-element list [xmin, xmax, ymin, ymax, zmin, zmax]
    
    Returns
    -------
    plate : Poly3DCollection
        The plate collection object
    """
    plate_thickness = 0.25 * 0.0254  # 0.25 inches to meters
    x_bounds = [bounds[0], bounds[1]]
    y_bounds = [bounds[2], bounds[3]]
    
    # Create plate vertices (top at z=0, bottom at z=-thickness)
    plate_vertices = [
        # Top surface
        [[x_bounds[0], y_bounds[0], plate_thickness],
         [x_bounds[1], y_bounds[0], plate_thickness],
         [x_bounds[1], y_bounds[1], plate_thickness],
         [x_bounds[0], y_bounds[1], plate_thickness]],
        # Bottom surface
        [[x_bounds[0], y_bounds[0], 0],
         [x_bounds[1], y_bounds[0], 0],
         [x_bounds[1], y_bounds[1], 0],
         [x_bounds[0], y_bounds[1], 0]],
        # Side 1
        [[x_bounds[0], y_bounds[0], plate_thickness],
         [x_bounds[1], y_bounds[0], plate_thickness],
         [x_bounds[1], y_bounds[0], 0],
         [x_bounds[0], y_bounds[0], 0]],
        # Side 2
        [[x_bounds[1], y_bounds[0], plate_thickness],
         [x_bounds[1], y_bounds[1], plate_thickness],
         [x_bounds[1], y_bounds[1], 0],
         [x_bounds[1], y_bounds[0], 0]],
        # Side 3
        [[x_bounds[1], y_bounds[1], plate_thickness],
         [x_bounds[0], y_bounds[1], plate_thickness],
         [x_bounds[0], y_bounds[1], 0],
         [x_bounds[1], y_bounds[1], 0]],
        # Side 4
        [[x_bounds[0], y_bounds[1], plate_thickness],
         [x_bounds[0], y_bounds[0], plate_thickness],
         [x_bounds[0], y_bounds[0], 0],
         [x_bounds[0], y_bounds[1], 0]]
    ]
    
    # Steel gray color
    steel_color = [0.7, 0.7, 0.75]
    plate = Poly3DCollection(plate_vertices, alpha=0.3, facecolor=steel_color, 
                            edgecolor='darkgray', linewidths=0.5)
    ax.add_collection3d(plate)
    return plate


def plot_axes(ax, r, RIB):
    """Plot body axes"""
    bodyX = 0.5 * RIB @ np.array([1, 0, 0])
    bodyY = 0.5 * RIB @ np.array([0, 1, 0])
    bodyZ = 0.5 * RIB @ np.array([0, 0, 1])
    
    q1 = ax.quiver(r[0], r[1], r[2], bodyX[0], bodyX[1], bodyX[2], 
                   color='red', arrow_length_ratio=0.3)
    q2 = ax.quiver(r[0], r[1], r[2], bodyY[0], bodyY[1], bodyY[2], 
                   color='blue', arrow_length_ratio=0.3)
    q3 = ax.quiver(r[0], r[1], r[2], bodyZ[0], bodyZ[1], bodyZ[2], 
                   color='green', arrow_length_ratio=0.3)
    
    return [q1, q2, q3]
