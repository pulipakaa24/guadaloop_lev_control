"""
Find equilibrium gap height for magnetic levitation system.

Given:
- Pod mass: 5.8 kg
- Required force: 5.8 * 9.81 = 56.898 N
- All currents: 0 A
- Roll angle: 0 degrees

Find: Gap height (mm) that produces this force
"""

import numpy as np
from maglev_predictor import MaglevPredictor

# Initialize predictor
predictor = MaglevPredictor()

# Target force
target_force = 5.8 * 9.81  # 56.898 N

print("=" * 70)
print("EQUILIBRIUM GAP HEIGHT FINDER")
print("=" * 70)
print(f"Target Force: {target_force:.3f} N (for 5.8 kg pod)")
print(f"Parameters: currL = 0 A, currR = 0 A, roll = 0°")
print()

# Define objective function
def force_error(gap_height):
    """Calculate difference between predicted force and target force."""
    force, _ = predictor.predict(currL=0, currR=0, roll=0, gap_height=gap_height)
    return force - target_force

# First, let's scan across gap heights to understand the behavior
print("Scanning gap heights from 5 to 30 mm...")
print("-" * 70)
gap_range = np.linspace(5, 30, 26)
forces = []

for gap in gap_range:
    force, torque = predictor.predict(currL=0, currR=0, roll=0, gap_height=gap)
    forces.append(force)
    if abs(force - target_force) < 5:  # Close to target
        print(f"Gap: {gap:5.1f} mm  →  Force: {force:7.3f} N  ←  CLOSE!")
    else:
        print(f"Gap: {gap:5.1f} mm  →  Force: {force:7.3f} N")

forces = np.array(forces)

print()
print("-" * 70)

# Check if target force is within the range
min_force = forces.min()
max_force = forces.max()

if target_force < min_force or target_force > max_force:
    print(f"⚠ WARNING: Target force {target_force:.3f} N is outside the range!")
    print(f"  Force range: {min_force:.3f} N to {max_force:.3f} N")
    print(f"  Cannot achieve equilibrium with 0A currents.")
else:
    # Find the gap height using root finding
    # Use brentq for robust bracketing (requires sign change)
    
    # Find bracketing interval
    idx_above = np.where(forces > target_force)[0]
    idx_below = np.where(forces < target_force)[0]
    
    if len(idx_above) > 0 and len(idx_below) > 0:
        # Find the transition point
        gap_low = gap_range[idx_below[-1]]
        gap_high = gap_range[idx_above[0]]
        
        print(f"✓ Target force is achievable!")
        print(f"  Bracketing interval: {gap_low:.1f} mm to {gap_high:.1f} mm")
        print()
        
        # Use simple bisection method for accurate root finding
        tol = 1e-6
        while (gap_high - gap_low) > tol:
            gap_mid = (gap_low + gap_high) / 2
            force_mid, _ = predictor.predict(currL=0, currR=0, roll=0, gap_height=gap_mid)
            
            if force_mid > target_force:
                gap_low = gap_mid
            else:
                gap_high = gap_mid
        
        equilibrium_gap = (gap_low + gap_high) / 2
        
        # Verify the result
        final_force, final_torque = predictor.predict(
            currL=0, currR=0, roll=0, gap_height=equilibrium_gap
        )
        
        print("=" * 70)
        print("EQUILIBRIUM FOUND!")
        print("=" * 70)
        print(f"Equilibrium Gap Height: {equilibrium_gap:.6f} mm")
        print(f"Predicted Force:        {final_force:.6f} N")
        print(f"Target Force:           {target_force:.6f} N")
        print(f"Error:                  {abs(final_force - target_force):.9f} N")
        print(f"Torque at equilibrium:  {final_torque:.6f} mN·m")
        print()
        print(f"✓ Pod will levitate at {equilibrium_gap:.3f} mm gap height")
        print(f"  with no current applied (permanent magnets only)")
        print("=" * 70)
    else:
        print("⚠ Could not find bracketing interval for bisection.")
        print("  Target force may not be achievable in the scanned range.")
