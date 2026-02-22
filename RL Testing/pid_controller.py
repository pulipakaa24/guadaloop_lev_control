"""
PID controller and default gains for the maglev three-loop (height, roll, pitch) control.
Used by lev_PID.ipynb and optuna_pid_tune.py.
"""

import numpy as np


class PIDController:
    """Simple PID controller with anti-windup."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_min: float = -1.0,
        output_max: float = 1.0,
        integral_limit: float = None,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_limit = (
            integral_limit if integral_limit is not None else abs(output_max) * 2
        )

        self.integral = 0.0
        self.prev_error = 0.0
        self.first_update = True

    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_update = True

    def update(self, error: float, dt: float) -> float:
        """Compute PID output.

        Args:
            error: Current error (setpoint - measurement)
            dt: Time step in seconds

        Returns:
            Control output (clamped to output limits)
        """
        p_term = self.kp * error

        self.integral += error * dt
        self.integral = np.clip(
            self.integral, -self.integral_limit, self.integral_limit
        )
        i_term = self.ki * self.integral

        if self.first_update:
            d_term = 0.0
            self.first_update = False
        else:
            d_term = self.kd * (error - self.prev_error) / dt

        self.prev_error = error
        output = p_term + i_term + d_term
        return np.clip(output, self.output_min, self.output_max)


# Default gains: height (main), roll, pitch.
# Optimizer and notebook can override via gains dict passed to run_pid_simulation.
DEFAULT_GAINS = {
    "height_kp": 50.0,
    "height_ki": 5.0,
    "height_kd": 10.0,
    "roll_kp": 2.0,
    "roll_ki": 0.5,
    "roll_kd": 0.5,
    "pitch_kp": 2.0,
    "pitch_ki": 0.5,
    "pitch_kd": 0.5,
}

# Backward-compat names for notebook (optional)
HEIGHT_KP = DEFAULT_GAINS["height_kp"]
HEIGHT_KI = DEFAULT_GAINS["height_ki"]
HEIGHT_KD = DEFAULT_GAINS["height_kd"]
ROLL_KP = DEFAULT_GAINS["roll_kp"]
ROLL_KI = DEFAULT_GAINS["roll_ki"]
ROLL_KD = DEFAULT_GAINS["roll_kd"]
PITCH_KP = DEFAULT_GAINS["pitch_kp"]
PITCH_KI = DEFAULT_GAINS["pitch_ki"]
PITCH_KD = DEFAULT_GAINS["pitch_kd"]
