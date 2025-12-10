# Updated LevPodEnv - Physical System Clarification

## System Architecture

### Physical Configuration

**Two U-Shaped Magnetic Yokes:**
- **Front Yoke**: Located at X = +0.1259m
  - Has two ends: Left (+Y = +0.0508m) and Right (-Y = -0.0508m)
  - Force is applied at center: X = +0.1259m, Y = 0m
  
- **Back Yoke**: Located at X = -0.1259m
  - Has two ends: Left (+Y = +0.0508m) and Right (-Y = -0.0508m)
  - Force is applied at center: X = -0.1259m, Y = 0m

**Four Independent Coil Currents:**
1. `curr_front_L`: Current around front yoke's left (+Y) end
2. `curr_front_R`: Current around front yoke's right (-Y) end
3. `curr_back_L`: Current around back yoke's left (+Y) end
4. `curr_back_R`: Current around back yoke's right (-Y) end

**Current Range:** -15A to +15A (from Ansys CSV data)
- Negative current: Strengthens permanent magnet field → stronger attraction
- Positive current: Weakens permanent magnet field → weaker attraction

### Collision Geometry in URDF

**Yoke Ends (4 boxes):** Represent the tips of the U-yokes where gap is measured
- Front Left: (+0.1259m, +0.0508m, +0.08585m)
- Front Right: (+0.1259m, -0.0508m, +0.08585m)
- Back Left: (-0.1259m, +0.0508m, +0.08585m)
- Back Right: (-0.1259m, -0.0508m, +0.08585m)

**Sensors (4 cylinders):** Physical gap sensors at different locations
- Center Right: (0m, +0.0508m, +0.08585m)
- Center Left: (0m, -0.0508m, +0.08585m)
- Front: (+0.2366m, 0m, +0.08585m)
- Back: (-0.2366m, 0m, +0.08585m)

## RL Environment Interface

### Action Space
**Type:** `Box(4)`, Range: [-1, 1]

**Actions:** `[pwm_front_L, pwm_front_R, pwm_back_L, pwm_back_R]`
- PWM duty cycles for the 4 independent coils
- Converted to currents via RL circuit model: `di/dt = (V_pwm - I*R) / L`

### Observation Space
**Type:** `Box(4)`, Range: [-inf, inf]

**Observations:** `[sensor_center_right, sensor_center_left, sensor_front, sensor_back]`
- **Noisy sensor readings** (not direct yoke measurements)
- Noise: Gaussian with σ = 0.1mm (0.0001m)
- Agent must learn system dynamics from sensor data alone
- Velocities not directly provided - agent can learn from temporal sequence if needed

### Force Application Physics

For each timestep:

1. **Measure yoke end gap heights** (from 4 yoke collision boxes)
2. **Average left/right ends** for each U-yoke:
   - `avg_gap_front = (gap_front_L + gap_front_R) / 2`
   - `avg_gap_back = (gap_back_L + gap_back_R) / 2`

3. **Calculate roll angle** from yoke end positions:
   ```python
   roll_front = arctan((gap_right - gap_left) / y_distance)
   roll_back = arctan((gap_right - gap_left) / y_distance)
   roll = (roll_front + roll_back) / 2
   ```

4. **Predict forces** using maglev_predictor:
   ```python
   force_front, torque_front = predictor.predict(
       curr_front_L, curr_front_R, roll_deg, gap_front_mm
   )
   force_back, torque_back = predictor.predict(
       curr_back_L, curr_back_R, roll_deg, gap_back_mm
   )
   ```

5. **Apply forces at Y=0** (center of each U-yoke):
   - Front force at: `[+0.1259, 0, 0.08585]`
   - Back force at: `[-0.1259, 0, 0.08585]`

6. **Apply roll torques** from each yoke independently

### Key Design Decisions

**Why 4 actions instead of 2?**
- Physical system has 4 independent electromagnets (one per yoke end)
- Allows fine control of roll torque
- Left/right current imbalance on each yoke creates torque

**Why sensor observations instead of yoke measurements?**
- Realistic: sensors are at different positions than yokes
- Adds partial observability challenge
- Agent must learn system dynamics to infer unmeasured states
- Sensor noise simulates real measurement uncertainty

**Why not include velocities in observation?**
- Agent can learn velocities from temporal sequence (frame stacking)
- Reduces observation dimensionality
- Tests if agent can learn dynamic behavior from gap measurements alone

**Current sign convention:**
- No conversion needed - currents fed directly to predictor
- Range: -15A to +15A (from Ansys model)
- Coil RL circuit naturally produces currents in this range

### Comparison with Original Design

| Feature | Original | Updated |
|---------|----------|---------|
| **Actions** | 2 (left/right coils) | 4 (front_L, front_R, back_L, back_R) |
| **Observations** | 5 (gaps, roll, velocities) | 4 (noisy sensor gaps) |
| **Gap Measurement** | Direct yoke positions | Noisy sensor positions |
| **Force Application** | Front & back yoke centers | Front & back yoke centers ✓ |
| **Current Range** | Assumed negative only | -15A to +15A |
| **Roll Calculation** | From yoke end heights | From yoke end heights ✓ |

## Physics Pipeline (Per Timestep)

1. **Action → Currents**
   ```
   PWM[4] → RL Circuit Model → Currents[4]
   ```

2. **State Measurement**
   ```
   Yoke End Positions[4] → Gap Heights[4] → Average per Yoke[2]
   ```

3. **Roll Calculation**
   ```
   (Gap_Right - Gap_Left) / Y_distance → Roll Angle
   ```

4. **Force Prediction**
   ```
   (currL, currR, roll, gap) → Maglev Predictor → (force, torque)
   Applied separately for front and back yokes
   ```

5. **Force Application**
   ```
   Forces at Y=0 for each yoke + Roll torques
   ```

6. **Observation Generation**
   ```
   Sensor Positions[4] → Gap Heights[4] → Add Noise → Observation[4]
   ```

## Info Dictionary

Each `env.step()` returns comprehensive diagnostics:

```python
{
    'curr_front_L': float,      # Front left coil current (A)
    'curr_front_R': float,      # Front right coil current (A)
    'curr_back_L': float,       # Back left coil current (A)
    'curr_back_R': float,       # Back right coil current (A)
    'gap_front_yoke': float,    # Front yoke average gap (m)
    'gap_back_yoke': float,     # Back yoke average gap (m)
    'roll': float,              # Roll angle (rad)
    'force_front': float,       # Front yoke force (N)
    'force_back': float,        # Back yoke force (N)
    'torque_front': float,      # Front yoke torque (mN·m)
    'torque_back': float        # Back yoke torque (mN·m)
}
```

## Testing

Run the updated test script:
```bash
cd "/Users/adipu/Documents/lev_control_4pt_small/RL Testing"
/opt/miniconda3/envs/RLenv/bin/python test_env.py
```

Expected behavior:
- 4 sensors report gap heights with small noise variations
- Yoke gaps (in info) match sensor gaps approximately
- All 4 coils build up current over time (RL circuit dynamics)
- Forces should be ~50-100N upward at 10mm gap with moderate currents
- Pod should begin to levitate if forces overcome gravity (5.8kg × 9.81 = 56.898 N needed)

## Next Steps for RL Training

1. **Frame Stacking**: Use 3-5 consecutive observations to give agent velocity information
   ```python
   from stable_baselines3.common.vec_env import VecFrameStack
   env = VecFrameStack(env, n_stack=4)
   ```

2. **Algorithm Selection**: PPO or SAC recommended
   - PPO: Good for continuous control, stable training
   - SAC: Better sample efficiency, handles stochastic dynamics

3. **Reward Tuning**: Current reward weights may need adjustment based on training performance

4. **Curriculum Learning**: Start with smaller gap errors, gradually increase difficulty

5. **Domain Randomization**: Vary sensor noise, mass, etc. for robust policy
