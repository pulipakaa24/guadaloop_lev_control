"""
Multiple trial simulation with randomized noise for maglev system
Runs multiple simulations with different parameter variations
"""

import numpy as np
import matplotlib.pyplot as plt
from parameters import QuadParams, Constants, initialize_parameter_variations, reset_parameter_variations
from utils import euler2dcm, fmag2, initialize_magnetic_characteristics, reset_magnetic_characteristics, get_magnetic_characteristics
from simulate import simulate_maglev_control
from visualize import visualize_quad
import os

# ===== SIMULATION CONFIGURATION =====
NUM_TRIALS = 5  # Number of trials to run
NOISE_LEVEL = 0.1  # Standard deviation of noise (5%)
# =====================================


def generate_parameter_report(trial_num, quad_params, output_dir):
    """Generate a text report of all parameters for this trial"""
    
    # Get magnetic characteristics
    mag_chars = get_magnetic_characteristics()
    
    report_filename = f'{output_dir}/trial_{trial_num:02d}_parameters.txt'
    
    with open(report_filename, 'w') as f:
        f.write(f"="*70 + "\n")
        f.write(f"Parameter Report for Trial {trial_num}\n")
        f.write(f"Noise Level: {NOISE_LEVEL*100:.1f}%\n")
        f.write(f"="*70 + "\n\n")
        
        # MECHANICAL PARAMETERS
        f.write(f"MECHANICAL PARAMETERS\n")
        f.write(f"-" * 70 + "\n")
        f.write(f"Mass (m):                  {quad_params.m:.6f} kg\n")
        f.write(f"\nMoment of Inertia (Jq):    (kg⋅m²)\n")
        f.write(f"  Jxx: {quad_params.Jq[0,0]:.9f}\n")
        f.write(f"  Jyy: {quad_params.Jq[1,1]:.9f}\n")
        f.write(f"  Jzz: {quad_params.Jq[2,2]:.9f}\n")
        
        f.write(f"\nFrame Dimensions:\n")
        f.write(f"  Length (frame_l):        {quad_params.frame_l:.6f} m ({quad_params.frame_l*1000:.3f} mm)\n")
        f.write(f"  Width (frame_w):         {quad_params.frame_w:.6f} m ({quad_params.frame_w*1000:.3f} mm)\n")
        f.write(f"  Yoke Height (yh):        {quad_params.yh:.6f} m ({quad_params.yh*1000:.3f} mm)\n")
        
        f.write(f"\nRotor/Yoke Locations (m): [x, y, z] for each corner\n")
        for i in range(4):
            f.write(f"  Yoke {i+1}: [{quad_params.rotor_loc[0,i]:8.6f}, "
                   f"{quad_params.rotor_loc[1,i]:8.6f}, "
                   f"{quad_params.rotor_loc[2,i]:8.6f}]\n")
        
        f.write(f"\nSensor Locations (m): [x, y, z] for each edge center\n")
        for i in range(4):
            f.write(f"  Sensor {i+1}: [{quad_params.sensor_loc[0,i]:8.6f}, "
                   f"{quad_params.sensor_loc[1,i]:8.6f}, "
                   f"{quad_params.sensor_loc[2,i]:8.6f}]\n")
        
        f.write(f"\nSensor Noise:              {quad_params.gap_sigma*1e6:.3f} μm (std dev)\n")
        
        # ELECTROMAGNETIC PARAMETERS
        f.write(f"\n\nELECTROMAGNETIC PARAMETERS\n")
        f.write(f"-" * 70 + "\n")
        
        f.write(f"Yoke Electrical Characteristics (per yoke):\n")
        for i in range(4):
            f.write(f"  Yoke {i+1}:\n")
            f.write(f"    Resistance (R):        {quad_params.yokeR_individual[i]:.6f} Ω\n")
            f.write(f"    Inductance (L):        {quad_params.yokeL_individual[i]:.9f} H ({quad_params.yokeL_individual[i]*1e3:.6f} mH)\n")
            f.write(f"    Max Voltage:           {quad_params.maxVoltage[i]:.3f} V\n")
        
        f.write(f"\nMagnetic Force Model Coefficients:\n")
        f.write(f"  N (turns):               {mag_chars['N']:.3f}\n")
        f.write(f"  const1:                  {mag_chars['const1']:.6e}\n")
        f.write(f"  const2:                  {mag_chars['const2']:.6e}\n")
        f.write(f"  const3:                  {mag_chars['const3']:.6e}\n")
        f.write(f"  const4:                  {mag_chars['const4']:.6e}\n")
        f.write(f"  const5:                  {mag_chars['const5']:.6e}\n")
        
        f.write(f"\n" + "="*70 + "\n")
        f.write(f"End of Report\n")
        f.write(f"="*70 + "\n")
    
    print(f"  Saved parameter report: {report_filename}")


def run_single_trial(trial_num, Tsim, delt, ref_gap, z0, output_dir):
    """Run a single simulation trial with randomized parameters"""
    
    # Reset and reinitialize noise for this trial
    reset_parameter_variations()
    reset_magnetic_characteristics()
    initialize_parameter_variations(noise_level=NOISE_LEVEL)
    initialize_magnetic_characteristics(noise_level=NOISE_LEVEL)
    
    # Maglev parameters and constants (with new noise)
    quad_params = QuadParams()
    constants = Constants()
    
    m = quad_params.m
    g = constants.g
    J = quad_params.Jq
    
    # Time vector, in seconds
    N = int(np.floor(Tsim / delt))
    tVec = np.arange(N) * delt
    
    # Matrix of disturbance forces acting on the body, in Newtons, expressed in I
    distMat = np.random.normal(0, 0, (N-1, 3))
    
    # Oversampling factor
    oversampFact = 10
    
    # Check nominal gap
    print(f"Force check: {4*fmag2(0, 10.830e-3) - m*g}")
    
    # SET REFERENCE HERE
    ref_gap = 10.830e-3  # from python code
    z0 = ref_gap - 2e-3
    
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
    
    # Generate parameter report for this trial
    generate_parameter_report(trial_num, quad_params, output_dir)
    
    # Run simulation
    print(f"  Running simulation for trial {trial_num}...")
    P0 = simulate_maglev_control(R, S, P)
    print(f"  Trial {trial_num} simulation complete!")
    
    # Extract results
    tVec_out = P0['tVec']
    state = P0['state']
    rMat = state['rMat']
    eMat = state['eMat']
    gaps = state['gaps']
    currents = state['currents']
    
    # Generate 3D visualization (GIF) without displaying
    print(f"  Generating 3D visualization for trial {trial_num}...")
    S2 = {
        'tVec': tVec_out,
        'rMat': rMat,
        'eMat': eMat,
        'plotFrequency': 20,
        'makeGifFlag': True,
        'gifFileName': f'{output_dir}/trial_{trial_num:02d}_animation.gif',
        'bounds': [-1, 1, -1, 1, -300e-3, 0.000]
    }
    visualize_quad(S2)
    plt.close('all')  # Close all figures to prevent display
    
    # Calculate forces
    Fm = fmag2(currents[:, 0], gaps[:, 0])
    
    return {
        'tVec': tVec_out,
        'gaps': gaps,
        'currents': currents,
        'Fm': Fm,
        'quad_params': quad_params
    }


def main():
    """Main simulation script - runs multiple trials"""
    
    # Create output directory if it doesn't exist
    output_dir = 'sim_results_multi'
    os.makedirs(output_dir, exist_ok=True)
    
    # Total simulation time, in seconds
    Tsim = 2
    
    # Update interval, in seconds
    delt = 0.005  # sampling interval
    
    # Reference gap and initial condition
    ref_gap = 10.830e-3
    z0 = ref_gap - 2e-3
    
    print(f"\n{'='*60}")
    print(f"Running {NUM_TRIALS} trials with noise level {NOISE_LEVEL*100:.1f}%")
    print(f"{'='*60}\n")
    
    # Run all trials
    trial_results = []
    for trial in range(1, NUM_TRIALS + 1):
        print(f"Trial {trial}/{NUM_TRIALS}")
        result = run_single_trial(trial, Tsim, delt, ref_gap, z0, output_dir)
        trial_results.append(result)
        print()
    
    # Create individual plots for each trial
    print("Generating plots...")
    for i, result in enumerate(trial_results, 1):
        tVec_out = result['tVec']
        gaps = result['gaps']
        currents = result['currents']
        Fm = result['Fm']
        
        # Create plots for this trial
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle(f'Trial {i} - Noise Level {NOISE_LEVEL*100:.1f}%', fontsize=14, fontweight='bold')
        
        # Plot 1: Gaps
        ax1 = plt.subplot(3, 1, 1)
        plt.plot(tVec_out, gaps * 1e3)
        plt.axhline(y=ref_gap * 1e3, color='k', linestyle='--', linewidth=1, label='Reference')
        plt.ylabel('Gap (mm)')
        plt.title('Sensor Gaps')
        plt.legend(['Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4', 'Reference'], 
                  loc='upper right', fontsize=8)
        plt.grid(True)
        plt.xticks([])
        
        # Plot 2: Currents
        ax2 = plt.subplot(3, 1, 2)
        plt.plot(tVec_out, currents)
        plt.ylabel('Current (A)')
        plt.title('Yoke Currents')
        plt.legend(['Yoke 1', 'Yoke 2', 'Yoke 3', 'Yoke 4'], 
                  loc='upper right', fontsize=8)
        plt.grid(True)
        plt.xticks([])
        
        # Plot 3: Forces
        ax3 = plt.subplot(3, 1, 3)
        plt.plot(tVec_out, Fm)
        plt.xlabel('Time (sec)')
        plt.ylabel('Force (N)')
        plt.title('Magnetic Force (Yoke 1)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/trial_{i:02d}_results.png', dpi=150)
        print(f"  Saved trial {i} plot")
        plt.close()
        
        # FFT for this trial
        oversampFact = 10
        Fs = 1/delt * oversampFact
        L = len(tVec_out)
        
        Y = np.fft.fft(Fm)
        frequencies = Fs / L * np.arange(L)
        
        fig2 = plt.figure(figsize=(10, 6))
        plt.semilogx(frequencies, np.abs(Y), linewidth=2)
        plt.title(f"FFT Spectrum - Trial {i}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.ylim([0, np.max(np.abs(Y[1:])) * 1.05])
        plt.grid(True)
        plt.savefig(f'{output_dir}/trial_{i:02d}_fft.png', dpi=150)
        print(f"  Saved trial {i} FFT")
        plt.close()
    
    # Create overlay comparison plots
    print("\nGenerating comparison plots...")
    
    # Gaps comparison
    fig_comp = plt.figure(figsize=(14, 10))
    fig_comp.suptitle(f'All Trials Comparison ({NUM_TRIALS} trials, {NOISE_LEVEL*100:.1f}% noise)', 
                      fontsize=14, fontweight='bold')
    
    ax1 = plt.subplot(3, 1, 1)
    for i, result in enumerate(trial_results, 1):
        avg_gap = np.mean(result['gaps'], axis=1)
        plt.plot(result['tVec'], avg_gap * 1e3, alpha=0.7, label=f'Trial {i}')
    plt.axhline(y=ref_gap * 1e3, color='k', linestyle='--', linewidth=2, label='Reference')
    plt.ylabel('Average Gap (mm)')
    plt.title('Average Gap Across All Trials')
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True)
    plt.xticks([])
    
    # Currents comparison
    ax2 = plt.subplot(3, 1, 2)
    for i, result in enumerate(trial_results, 1):
        avg_current = np.mean(result['currents'], axis=1)
        plt.plot(result['tVec'], avg_current, alpha=0.7, label=f'Trial {i}')
    plt.ylabel('Average Current (A)')
    plt.title('Average Current Across All Trials')
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True)
    plt.xticks([])
    
    # Forces comparison
    ax3 = plt.subplot(3, 1, 3)
    for i, result in enumerate(trial_results, 1):
        plt.plot(result['tVec'], result['Fm'], alpha=0.7, label=f'Trial {i}')
    plt.xlabel('Time (sec)')
    plt.ylabel('Force (N)')
    plt.title('Magnetic Force Across All Trials')
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_all_trials.png', dpi=150)
    print(f"Saved comparison plot")
    plt.close()  # Close without displaying
    
    print(f"\n{'='*60}")
    print(f"All trials completed!")
    print(f"Results saved to: {output_dir}/")
    print(f"  - Individual trial plots: trial_XX_results.png")
    print(f"  - Individual trial animations: trial_XX_animation.gif")
    print(f"  - Individual FFT plots: trial_XX_fft.png")
    print(f"  - Individual parameter reports: trial_XX_parameters.txt")
    print(f"  - Comparison plot: comparison_all_trials.png")
    print(f"{'='*60}\n")
    
    return trial_results


if __name__ == '__main__':
    results = main()
    print(f"\nCompleted {len(results)} trials successfully!")
