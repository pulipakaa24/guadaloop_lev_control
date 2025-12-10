# LevPodEnv Integration Summary

## Overview
`LevPodEnv` now fully interfaces with PyBullet simulation and uses the `maglev_predictor` to apply electromagnetic forces based on real-time gap heights and coil currents.

## Architecture

### System Configuration (From pod.xml and visualize_urdf.py)
- **Inverted Maglev System**: Pod hangs BELOW track (like a monorail)
- **Track**: Bottom surface at Z=0, 2m × 0.4m × 0.02m
- **Pod Mass**: 5.8 kg (from pod.xml inertial)
- **Yoke Positions** (local coordinates):
  - Front Right: (+0.1259m, +0.0508m, +0.08585m)
  - Front Left: (+0.1259m, -0.0508m, +0.08585m)
  - Back Right: (-0.1259m, +0.0508m, +0.08585m)
  - Back Left: (-0.1259m, -0.0508m, +0.08585m)
- **Y-axis distance between left/right**: 0.1016m

### Coil Configuration
- **Two Coils**: 
  - `coilL`: Left side (+Y), controls all +Y yokes
  - `coilR`: Right side (-Y), controls all -Y yokes
- **Parameters** (preserved from original):
  - Resistance: 1.1Ω
  - Inductance: 0.0025H (2.5mH)
  - Source Voltage: 12V
  - Max Current: 10.2A

## Action Space
- **Type**: Box(2)
- **Range**: [-1, 1] for each coil
- **Mapping**:
  - `action[0]`: PWM duty cycle for left coil (+Y side)
  - `action[1]`: PWM duty cycle for right coil (-Y side)

## Observation Space
- **Type**: Box(5)
- **Components**:
  1. `avg_gap_front` (m): Average gap height of front left & right yokes
  2. `avg_gap_back` (m): Average gap height of back left & right yokes
  3. `roll` (rad): Roll angle about X-axis (calculated from yoke Z positions)
  4. `roll_rate` (rad/s): Angular velocity about X-axis
  5. `z_velocity` (m/s): Vertical velocity

## Physics Pipeline (per timestep)

### 1. Coil Current Update
```python
currL = self.coilL.update(pwm_L, dt)  # First-order RL circuit model
currR = self.coilR.update(pwm_R, dt)
```

### 2. Gap Height Calculation
For each of 4 yokes:
- Transform local position to world coordinates using rotation matrix
- Add 5mm (half-height of 10mm yoke box) to get top surface
- Gap height = -yoke_top_z (track at Z=0, yoke below)
- Separate into front and back averages

### 3. Roll Angle Calculation
```python
roll = arctan2((right_z_avg - left_z_avg) / y_distance)
```
- Uses Z-position difference between left (+Y) and right (-Y) yokes
- Y-distance = 0.1016m (distance between yoke centerlines)

### 4. Force/Torque Prediction
```python
# Convert to Ansys convention (negative currents)
currL_ansys = -abs(currL)
currR_ansys = -abs(currR)

# Predict for front and back independently
force_front, torque_front = predictor.predict(currL_ansys, currR_ansys, roll_deg, gap_front_mm)
force_back, torque_back = predictor.predict(currL_ansys, currR_ansys, roll_deg, gap_back_mm)
```

### 5. Force Application
- **Front Force**: Applied at [+0.1259, 0, 0.08585] in local frame
- **Back Force**: Applied at [-0.1259, 0, 0.08585] in local frame
- **Roll Torque**: Average of front/back torques, applied about X-axis
  - Converted from mN·m to N·m: `torque_Nm = avg_torque / 1000`

### 6. Simulation Step
```python
p.stepSimulation()  # 240 Hz (dt = 1/240s)
```

## Reward Function
```python
reward = 1.0
reward -= gap_error * 100      # Target: 10mm gap
reward -= roll_error * 50      # Keep level
reward -= z_vel_penalty * 10   # Minimize oscillation
reward -= power * 0.01         # Efficiency
```

## Termination Conditions
- Gap outside [2mm, 30mm] range
- Roll angle exceeds ±10°

## Info Dictionary
Each step returns:
```python
{
    'currL': float,          # Left coil current (A)
    'currR': float,          # Right coil current (A)
    'gap_front': float,      # Front average gap (m)
    'gap_back': float,       # Back average gap (m)
    'roll': float,           # Roll angle (rad)
    'force_front': float,    # Front force prediction (N)
    'force_back': float,     # Back force prediction (N)
    'torque': float          # Average torque (mN·m)
}
```

## Key Design Decisions

### Why Two Coils Instead of Four?
- Physical system has one coil per side (left/right)
- Each coil's magnetic field affects both front and back yokes on that side
- Simplifies control: differential current creates roll torque

### Why Separate Front/Back Predictions?
- Gap heights can differ due to pitch angle
- More accurate force modeling
- Allows pitch control if needed in future

### Roll Angle from Yoke Positions
As requested: `roll = arctan((right_z - left_z) / y_distance)`
- Uses actual yoke Z positions in world frame
- More accurate than quaternion-based roll (accounts for deformation)
- Matches physical sensor measurements

### Current Sign Convention
- Coils produce positive current (0 to +10.2A)
- Ansys model expects negative currents (-15A to 0A)
- Conversion: `currL_ansys = -abs(currL)`

## Usage Example

```python
from lev_pod_env import LevPodEnv

# Create environment
env = LevPodEnv(use_gui=True)  # Set False for training

# Reset
obs, info = env.reset()
# obs = [gap_front, gap_back, roll, roll_rate, z_vel]

# Step
action = [0.5, 0.5]  # 50% PWM on both coils
obs, reward, terminated, truncated, info = env.step(action)

# Check results
print(f"Gaps: {info['gap_front']*1000:.2f}mm, {info['gap_back']*1000:.2f}mm")
print(f"Forces: {info['force_front']:.2f}N, {info['force_back']:.2f}N")
print(f"Currents: {info['currL']:.2f}A, {info['currR']:.2f}A")

env.close()
```

## Testing
Run `test_env.py` to verify integration:
```bash
cd "/Users/adipu/Documents/lev_control_4pt_small/RL Testing"
/opt/miniconda3/envs/RLenv/bin/python test_env.py
```

## Next Steps for RL Training
1. Test environment with random actions (test_env.py)
2. Verify force magnitudes are reasonable (should see ~50-100N upward)
3. Check that roll control works (differential currents produce torque)
4. Train RL agent (PPO, SAC, or TD3 recommended)
5. Tune reward function weights based on training results
