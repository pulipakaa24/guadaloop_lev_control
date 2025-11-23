"""
Top-level script for calling simulate_maglev_control
Ported from topSimulate.m to Python
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from parameters import QuadParams, Constants
from utils import euler2dcm, fmag2
from simulate import simulate_maglev_control
from visualize import visualize_quad


def main():
    """Main simulation script"""
    
    # SET REFERENCE HERE
    # Total simulation time, in seconds
    Tsim = 2
    
    # Update interval, in seconds. This value should be small relative to the
    # shortest time constant of your system.
    delt = 0.005  # sampling interval
    
    # Maglev parameters and constants
    quad_params = QuadParams()
    constants = Constants()
    
    m = quad_params.m
    g = constants.g
    J = quad_params.Jq
    
    # Time vector, in seconds
    N = int(np.floor(Tsim / delt))
    tVec = np.arange(N) * delt
    
    # Matrix of disturbance forces acting on the body, in Newtons, expressed in I
    distMat = np.random.normal(0, 10, (N-1, 3))
    
    # Oversampling factor
    oversampFact = 10
    
    # Check nominal gap
    print(f"Force check: {4*fmag2(0, 11.239e-3) - m*g}")
    
    ref_gap = 10.830e-3  # from python code
    z0 = ref_gap + 2e-3
    
    # Create reference trajectories
    rIstar = np.zeros((N, 3))
    vIstar = np.zeros((N, 3))
    aIstar = np.zeros((N, 3))
    xIstar = np.zeros((N, 3))
    
    for k in range(N):
        rIstar[k, :] = [0, 0, -ref_gap]
        vIstar[k, :] = [0, 0, 0]
        aIstar[k, :] = [0, 0, 0]
        xIstar[k, :] = [0, 1, 0]
    
    # Setup reference structure
    R = {
        'tVec': tVec,
        'rIstar': rIstar,
        'vIstar': vIstar,
        'aIstar': aIstar,
        'xIstar': xIstar
    }
    
    # Initial state
    state0 = {
        'r': np.array([0, 0, -(z0 + quad_params.yh)]),
        'v': np.array([0, 0, 0]),
        'e': np.array([0.01, 0.01, np.pi/2]),  # xyz euler angles
        'omegaB': np.array([0.00, 0.00, 0])
    }
    
    # Setup simulation structure
    S = {
        'tVec': tVec,
        'distMat': distMat,
        'oversampFact': oversampFact,
        'state0': state0
    }
    
    # Setup parameters structure
    P = {
        'quadParams': quad_params,
        'constants': constants
    }
    
    # Run simulation
    print("Running simulation...")
    P0 = simulate_maglev_control(R, S, P)
    print("Simulation complete!")
    
    # Extract results
    tVec_out = P0['tVec']
    state = P0['state']
    rMat = state['rMat']
    eMat = state['eMat']
    vMat = state['vMat']
    gaps = state['gaps']
    currents = state['currents']
    omegaBMat = state['omegaBMat']
    
    # Calculate forces
    Fm = fmag2(currents[:, 0], gaps[:, 0])
    
    # Calculate ex and ey for visualization
    N_out = len(eMat)
    ex = np.zeros(N_out)
    ey = np.zeros(N_out)
    
    for k in range(N_out):
        Rk = euler2dcm(eMat[k, :])
        x = Rk @ np.array([1, 0, 0])
        ex[k] = x[0]
        ey[k] = x[1]
    
    # Visualize the quad motion
    print("Generating 3D visualization...")
    S2 = {
        'tVec': tVec_out,
        'rMat': rMat,
        'eMat': eMat,
        'plotFrequency': 20,
        'makeGifFlag': True,
        'gifFileName': 'testGif.gif',
        'bounds': [-1, 1, -1, 1, -300e-3, 0.000]
    }
    visualize_quad(S2)
    
    # Create plots
    fig = plt.figure(figsize=(12, 8))
    
    # Plot 1: Gaps
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(tVec_out, gaps * 1e3)
    plt.axhline(y=ref_gap * 1e3, color='k', linestyle='-', linewidth=1)
    plt.ylabel('Vertical (mm)')
    plt.title('Gaps')
    plt.grid(True)
    plt.xticks([])
    
    # Plot 2: Currents
    ax2 = plt.subplot(3, 1, 2)
    plt.plot(tVec_out, currents)
    plt.ylabel('Current (Amps)')
    plt.title('Power')
    plt.grid(True)
    plt.xticks([])
    
    # Plot 3: Forces
    ax3 = plt.subplot(3, 1, 3)
    plt.plot(tVec_out, Fm)
    plt.xlabel('Time (sec)')
    plt.ylabel('Fm (N)')
    plt.title('Forcing')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=150)
    print("Saved plot to simulation_results.png")
    
    # Commented out horizontal position plot (as in MATLAB)
    # fig_horizontal = plt.figure()
    # plt.plot(rMat[:, 0], rMat[:, 1])
    # spacing = 300
    # plt.quiver(rMat[::spacing, 0], rMat[::spacing, 1], 
    #           ex[::spacing], -ey[::spacing])
    # plt.axis('equal')
    # plt.grid(True)
    # plt.xlabel('X (m)')
    # plt.ylabel('Y (m)')
    # plt.title('Horizontal position of CM')
    
    # Frequency analysis (mech freq)
    Fs = 1/delt * oversampFact  # Sampling frequency
    T = 1/Fs  # Sampling period
    L = len(tVec_out)  # Length of signal
    
    Y = np.fft.fft(Fm)
    frequencies = Fs / L * np.arange(L)
    
    fig2 = plt.figure(figsize=(10, 6))
    plt.semilogx(frequencies, np.abs(Y), linewidth=3)
    plt.title("Complex Magnitude of fft Spectrum")
    plt.xlabel("f (Hz)")
    plt.ylabel("|fft(X)|")
    plt.grid(True)
    plt.savefig('fft_spectrum.png', dpi=150)
    print("Saved FFT plot to fft_spectrum.png")
    
    # Show all plots
    plt.show()
    
    return P0


if __name__ == '__main__':
    results = main()
    print("\nSimulation completed successfully!")
    print(f"Final time: {results['tVec'][-1]:.3f} seconds")
    print(f"Number of time points: {len(results['tVec'])}")
