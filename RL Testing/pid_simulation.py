"""
Feedforward + PID simulation runner for the maglev pod.
Uses LevPodEnv, MaglevPredictor (feedforward), and PIDController with configurable gains.
"""

import numpy as np
from lev_pod_env import LevPodEnv, TARGET_GAP
from maglev_predictor import MaglevPredictor
from pid_controller import PIDController, DEFAULT_GAINS

# Feedforward LUT (built on first use)
_FF_GAP_LUT = None
_FF_PWM_LUT = None


def build_feedforward_lut(
    pod_mass: float = 9.4,
    coil_r: float = 1.1,
    v_supply: float = 12.0,
    gap_min: float = 3.0,
    gap_max: float = 35.0,
    n_points: int = 500,
):
    """Build gap [mm] -> equilibrium PWM lookup table. Returns (gap_lut, pwm_lut) for np.interp."""
    global _FF_GAP_LUT, _FF_PWM_LUT
    target_per_yoke = pod_mass * 9.81 / 2.0
    predictor = MaglevPredictor()

    def _find_eq_current(gap_mm):
        a, b = -10.2, 10.2
        fa, _ = predictor.predict(a, a, 0.0, gap_mm)
        for _ in range(80):
            mid = (a + b) / 2.0
            fm, _ = predictor.predict(mid, mid, 0.0, gap_mm)
            if (fa > target_per_yoke) == (fm > target_per_yoke):
                a, fa = mid, fm
            else:
                b = mid
        return (a + b) / 2.0

    _FF_GAP_LUT = np.linspace(gap_min, gap_max, n_points)
    curr_lut = np.array([_find_eq_current(g) for g in _FF_GAP_LUT])
    _FF_PWM_LUT = np.clip(curr_lut * coil_r / v_supply, -1.0, 1.0)
    return _FF_GAP_LUT, _FF_PWM_LUT


def feedforward_pwm(gap_mm: float) -> float:
    """Interpolate equilibrium PWM for gap height [mm]. Builds LUT on first call."""
    global _FF_GAP_LUT, _FF_PWM_LUT
    if _FF_GAP_LUT is None:
        build_feedforward_lut()
    return float(np.interp(gap_mm, _FF_GAP_LUT, _FF_PWM_LUT))


def run_pid_simulation(
    initial_gap_mm: float = 14.0,
    max_steps: int = 2000,
    use_gui: bool = False,
    disturbance_step: int = None,
    disturbance_force: float = 0.0,
    disturbance_force_std: float = 0.0,
    use_feedforward: bool = True,
    record_video: bool = False,
    record_telemetry: bool = False,
    record_dir: str = "recordings",
    gains: dict = None,
    verbose: bool = True,
) -> dict:
    """
    Run one feedforward + PID simulation. Gains can be passed for tuning (e.g. Optuna).

    Args:
        initial_gap_mm: Starting gap height (mm).
        max_steps: Max simulation steps.
        use_gui: PyBullet GUI (avoid in notebooks).
        disturbance_step: Step for impulse (None = none).
        disturbance_force: Impulse force (N).
        disturbance_force_std: Noise std (N).
        use_feedforward: Use MaglevPredictor feedforward.
        record_video: Save MP4.
        record_telemetry: Save telemetry PNG.
        record_dir: Output directory.
        gains: Dict with keys height_kp, height_ki, height_kd, roll_kp, roll_ki, roll_kd,
               pitch_kp, pitch_ki, pitch_kd. If None, uses DEFAULT_GAINS.
        verbose: Print progress.

    Returns:
        data: dict of arrays (time, gap_avg, roll_deg, pitch_deg, currents, pwms, ...).
    """
    g = gains if gains is not None else DEFAULT_GAINS
    env = LevPodEnv(
        use_gui=use_gui,
        initial_gap_mm=initial_gap_mm,
        max_steps=max_steps,
        disturbance_force_std=disturbance_force_std,
        record_video=record_video,
        record_telemetry=record_telemetry,
        record_dir=record_dir,
    )
    dt = env.dt

    height_pid = PIDController(
        g["height_kp"], g["height_ki"], g["height_kd"],
        output_min=-1.0, output_max=1.0,
    )
    roll_pid = PIDController(
        g["roll_kp"], g["roll_ki"], g["roll_kd"],
        output_min=-0.5, output_max=0.5,
    )
    pitch_pid = PIDController(
        g["pitch_kp"], g["pitch_ki"], g["pitch_kd"],
        output_min=-0.5, output_max=0.5,
    )

    data = {
        "time": [], "gap_front": [], "gap_back": [], "gap_avg": [],
        "roll_deg": [], "pitch_deg": [],
        "current_FL": [], "current_FR": [], "current_BL": [], "current_BR": [],
        "current_total": [], "pwm_FL": [], "pwm_FR": [], "pwm_BL": [], "pwm_BR": [],
        "ff_pwm": [],
    }

    obs, _ = env.reset()
    target_gap = TARGET_GAP
    y_distance = 0.1016
    x_distance = 0.2518

    if verbose:
        print(f"Starting simulation: initial_gap={initial_gap_mm}mm, target={target_gap*1000:.2f}mm")
        if disturbance_step is not None:
            print(f"  Impulse disturbance: {disturbance_force}N at step {disturbance_step}")
        if disturbance_force_std > 0:
            print(f"  Stochastic noise: std={disturbance_force_std}N")
        print(f"  Feedforward: {'enabled' if use_feedforward else 'disabled'}")
        if record_video or record_telemetry:
            print(f"  Recording: video={record_video}, telemetry={record_telemetry} → {record_dir}/")

    for step in range(max_steps):
        t = step * dt
        gaps_normalized = obs[:4]
        gaps = gaps_normalized * env.gap_scale + TARGET_GAP
        gap_front = gaps[2]
        gap_back = gaps[3]
        gap_left = gaps[1]
        gap_right = gaps[0]
        gap_avg = (gap_front + gap_back + gap_left + gap_right) / 4

        roll_angle = np.arcsin(np.clip((gap_left - gap_right) / y_distance, -1, 1))
        pitch_angle = np.arcsin(np.clip((gap_back - gap_front) / x_distance, -1, 1))

        ff_base = feedforward_pwm(gap_avg * 1000) if use_feedforward else 0.0

        height_error = target_gap - gap_avg
        roll_error = -roll_angle
        pitch_error = -pitch_angle

        height_correction = height_pid.update(height_error, dt)
        roll_correction = roll_pid.update(roll_error, dt)
        pitch_correction = pitch_pid.update(pitch_error, dt)

        pwm_FL = np.clip(ff_base + height_correction - roll_correction - pitch_correction, -1, 1)
        pwm_FR = np.clip(ff_base + height_correction + roll_correction - pitch_correction, -1, 1)
        pwm_BL = np.clip(ff_base + height_correction - roll_correction + pitch_correction, -1, 1)
        pwm_BR = np.clip(ff_base + height_correction + roll_correction + pitch_correction, -1, 1)

        action = np.array([pwm_FL, pwm_FR, pwm_BL, pwm_BR], dtype=np.float32)

        if disturbance_step is not None and step == disturbance_step:
            env.apply_impulse(disturbance_force)
            # Coupled torque: random direction, magnitude = 0.5 * |impulse force| (N·m)
            torque_mag = 0.5 * abs(disturbance_force)
            direction = np.random.randn(3)
            direction = direction / (np.linalg.norm(direction) + 1e-12)
            torque_nm = (torque_mag * direction).tolist()
            env.apply_torque_impulse(torque_nm)
            if verbose:
                print(f"  Applied {disturbance_force}N impulse and {torque_mag:.2f} N·m torque at step {step}")

        obs, _, terminated, truncated, info = env.step(action)

        data["time"].append(t)
        data["gap_front"].append(info["gap_front_yoke"] * 1000)
        data["gap_back"].append(info["gap_back_yoke"] * 1000)
        data["gap_avg"].append(info["gap_avg"] * 1000)
        data["roll_deg"].append(np.degrees(info["roll"]))
        data["pitch_deg"].append(np.degrees(info["pitch"]))
        data["current_FL"].append(info["curr_front_L"])
        data["current_FR"].append(info["curr_front_R"])
        data["current_BL"].append(info["curr_back_L"])
        data["current_BR"].append(info["curr_back_R"])
        data["current_total"].append(
            abs(info["curr_front_L"]) + abs(info["curr_front_R"])
            + abs(info["curr_back_L"]) + abs(info["curr_back_R"])
        )
        data["ff_pwm"].append(ff_base)
        data["pwm_FL"].append(pwm_FL)
        data["pwm_FR"].append(pwm_FR)
        data["pwm_BL"].append(pwm_BL)
        data["pwm_BR"].append(pwm_BR)

        if terminated or truncated:
            if verbose:
                print(f"  Simulation ended at step {step} (terminated={terminated})")
            break

    env.close()

    for key in data:
        data[key] = np.array(data[key])

    if verbose:
        print(f"Simulation complete: {len(data['time'])} steps, {data['time'][-1]:.2f}s")
        print(f"  Final gap: {data['gap_avg'][-1]:.2f}mm (target: {target_gap*1000:.2f}mm)")
        print(f"  Final roll: {data['roll_deg'][-1]:.3f}°, pitch: {data['pitch_deg'][-1]:.3f}°")

    return data
