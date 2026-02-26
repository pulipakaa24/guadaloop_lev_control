"""
Optuna (TPE / Bayesian-style) optimization for the three-PID maglev controller.
Tunes all nine gains (height, roll, pitch × Kp, Ki, Kd) jointly so coupling is respected.

Noisy optimization: set disturbance_force_std > 0 so the env applies random force/torque
each step (system changes slightly each run). To keep Bayesian optimization effective,
use n_objective_repeats > 1: each candidate is evaluated multiple times and the mean cost
is returned, reducing variance so TPE can compare trials reliably.

Run from the command line or import and call run_optimization() from a notebook.
"""

import json
import os
import sys

import numpy as np
import optuna
from optuna.samplers import TPESampler

from lev_pod_env import TARGET_GAP
from pid_controller import DEFAULT_GAINS
from pid_simulation import run_pid_simulation, build_feedforward_lut

# Save pid_best_params.json next to this script so notebook and CLI find it regardless of cwd
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default optimization config: long enough to see late instability (~8s+), not so long that optimizer goes ultra-conservative
DEFAULT_MAX_STEPS = 1500
DEFAULT_INITIAL_GAPS_MM = [8.0, 15.0]  # Two conditions for robustness (bracket 11.86mm target)
DEFAULT_N_TRIALS = 200
DEFAULT_TIMEOUT_S = 3600
TARGET_GAP_MM = TARGET_GAP * 1000


def _cost_from_data(
    data: dict,
    target_gap_mm: float,
    penalty_early: float = 500.0,
    weight_late_osc: float = 3.0,
    weight_steady_state: float = 2.0,
) -> float:
    """
    Single scalar cost from one run. Lower is better.

    - ITAE for gap error and |roll|/|pitch| (tracking over time).
    - Late-oscillation penalty: std(roll) and std(pitch) over the *last 50%* of the run,
      plus max |roll|/|pitch| in that window, so gains that go unstable after ~half the run are penalized.
    - Steady-state term: mean absolute gap error and mean |roll|/|pitch| over the *last 20%*,
      so we reward settling at target with small angles.
    - Penalty if episode ended early (crash/instability), regardless of run length.
    - Small penalty on mean total current (efficiency).
    """
    t = np.asarray(data["time"])
    dt = np.diff(t, prepend=0.0)
    gap_err_mm = np.abs(np.asarray(data["gap_avg"]) - target_gap_mm)
    itae_gap = np.sum(t * gap_err_mm * dt)

    roll_deg = np.asarray(data["roll_deg"])
    pitch_deg = np.asarray(data["pitch_deg"])
    roll_abs = np.abs(roll_deg)
    pitch_abs = np.abs(pitch_deg)
    itae_roll = np.sum(t * roll_abs * dt)
    itae_pitch = np.sum(t * pitch_abs * dt)

    n = len(t)
    early_penalty = penalty_early if data.get("terminated_early", False) else 0.0

    # Late half: penalize oscillation (std) and large angles (max) so "good for 6s then violent roll" is expensive
    half = max(1, n // 2)
    roll_last = roll_deg[-half:]
    pitch_last = pitch_deg[-half:]
    late_osc_roll = np.std(roll_last) + np.max(np.abs(roll_last))
    late_osc_pitch = np.std(pitch_last) + np.max(np.abs(pitch_last))
    late_osc_penalty = weight_late_osc * (late_osc_roll + late_osc_pitch)

    # Last 20%: steady-state error — want small gap error and small roll/pitch at end
    tail = max(1, n // 5)
    ss_gap = np.mean(np.abs(np.asarray(data["gap_avg"])[-tail:] - target_gap_mm))
    ss_roll = np.mean(roll_abs[-tail:])
    ss_pitch = np.mean(pitch_abs[-tail:])
    steady_state_penalty = weight_steady_state * (ss_gap + ss_roll + ss_pitch)

    mean_current = np.mean(data["current_total"])
    current_penalty = 0.01 * mean_current

    return (
        itae_gap
        + 2.0 * (itae_roll + itae_pitch)
        + early_penalty
        + late_osc_penalty
        + steady_state_penalty
        + current_penalty
    )


def objective(
    trial: optuna.Trial,
    initial_gaps_mm: list,
    max_steps: int,
    use_feedforward: bool,
    penalty_early: float,
    disturbance_force_std: float = 0.0,
    n_objective_repeats: int = 1,
) -> float:
    """
    Optuna objective: suggest 9 gains, run simulation(s), return cost.

    With disturbance_force_std > 0, the env applies random force/torque disturbances each step,
    so the same gains yield different costs across runs. For Bayesian optimization under noise,
    set n_objective_repeats > 1 to evaluate each candidate multiple times and return the mean cost,
    reducing variance so TPE can compare trials reliably.
    """
    # All three loops tuned together (recommended: they interact)
    height_kp = trial.suggest_float("height_kp", 5.0, 200.0, log=True)
    height_ki = trial.suggest_float("height_ki", 0.5, 50.0, log=True)
    height_kd = trial.suggest_float("height_kd", 1.0, 50.0, log=True)
    roll_kp = trial.suggest_float("roll_kp", 0.1, 20.0, log=True)
    roll_ki = trial.suggest_float("roll_ki", 0.01, 5.0, log=True)
    roll_kd = trial.suggest_float("roll_kd", 0.01, 5.0, log=True)
    pitch_kp = trial.suggest_float("pitch_kp", 0.1, 20.0, log=True)
    pitch_ki = trial.suggest_float("pitch_ki", 0.01, 5.0, log=True)
    pitch_kd = trial.suggest_float("pitch_kd", 0.01, 5.0, log=True)

    gains = {
        "height_kp": height_kp, "height_ki": height_ki, "height_kd": height_kd,
        "roll_kp": roll_kp, "roll_ki": roll_ki, "roll_kd": roll_kd,
        "pitch_kp": pitch_kp, "pitch_ki": pitch_ki, "pitch_kd": pitch_kd,
    }

    cost_sum = 0.0
    n_evals = 0
    for _ in range(n_objective_repeats):
        for initial_gap in initial_gaps_mm:
            data = run_pid_simulation(
                initial_gap_mm=initial_gap,
                max_steps=max_steps,
                use_gui=False,
                disturbance_force_std=disturbance_force_std,
                use_feedforward=use_feedforward,
                record_video=False,
                record_telemetry=False,
                gains=gains,
                verbose=False,
            )
            n = len(data["time"])
            data["terminated_early"] = n < max_steps - 10
            cost_sum += _cost_from_data(data, TARGET_GAP_MM, penalty_early=penalty_early)
            n_evals += 1
    return cost_sum / n_evals


def run_optimization(
    n_trials: int = DEFAULT_N_TRIALS,
    timeout: int = DEFAULT_TIMEOUT_S,
    initial_gaps_mm: list = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    use_feedforward: bool = True,
    penalty_early: float = 500.0,
    disturbance_force_std: float = 0.0,
    n_objective_repeats: int = 1,
    out_dir: str = None,
    study_name: str = "pid_maglev",
) -> optuna.Study:
    """
    Run Optuna study (TPE sampler) and save best params to JSON.

    disturbance_force_std: passed to env so each step gets random force/torque noise (N).
    n_objective_repeats: when > 1, each trial is evaluated this many times (different noise)
        and the mean cost is returned, so Bayesian optimization sees a less noisy objective.

    Returns:
        study: Optuna Study (study.best_params, study.best_value).
    """
    if initial_gaps_mm is None:
        initial_gaps_mm = DEFAULT_INITIAL_GAPS_MM
    if out_dir is None:
        out_dir = _SCRIPT_DIR

    # Build feedforward LUT once so first trial doesn't do it inside run
    build_feedforward_lut()

    sampler = TPESampler(
        n_startup_trials=min(20, n_trials // 4),
        n_ei_candidates=24,
        seed=42,
    )
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        sampler=sampler,
    )

    study.optimize(
        lambda t: objective(
            t,
            initial_gaps_mm=initial_gaps_mm,
            max_steps=max_steps,
            use_feedforward=use_feedforward,
            penalty_early=penalty_early,
            disturbance_force_std=disturbance_force_std,
            n_objective_repeats=n_objective_repeats,
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    out_path = os.path.join(out_dir, "pid_best_params.json")
    with open(out_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Best cost: {study.best_value:.4f}")
    print(f"Best params saved to {out_path}")
    return study


def run_staged_optimization(
    stage_steps: list = None,
    n_trials_per_stage: int = 50,
    timeout_per_stage: int = None,
    initial_gaps_mm: list = None,
    use_feedforward: bool = True,
    penalty_early: float = 500.0,
    disturbance_force_std: float = 0.0,
    n_objective_repeats: int = 1,
    out_dir: str = None,
) -> list:
    """
    Run optimization in stages with increasing max_steps, warm-starting each stage from the previous best.

    disturbance_force_std: passed to env for stochastic force/torque noise (N).
    n_objective_repeats: mean over this many evaluations per trial for a less noisy objective.

    Example: stage_steps=[1500, 3000, 6000]
    - Stage 1: optimize with 1500 steps (finds good gap/initial roll), save best.
    - Stage 2: optimize with 3000 steps; first trial is the 1500-step best (evaluated at 3000 steps), then TPE suggests improvements.
    - Stage 3: same with 6000 steps starting from stage 2's best.

    This keeps good lift-off and gap control from the short-horizon run while refining for late-horizon roll stability.
    """
    if stage_steps is None:
        stage_steps = [1500, 3000, 6000]
    if initial_gaps_mm is None:
        initial_gaps_mm = DEFAULT_INITIAL_GAPS_MM
    if out_dir is None:
        out_dir = _SCRIPT_DIR
    if timeout_per_stage is None:
        timeout_per_stage = DEFAULT_TIMEOUT_S

    build_feedforward_lut()
    best_params_path = os.path.join(out_dir, "pid_best_params.json")
    studies = []

    for stage_idx, max_steps in enumerate(stage_steps):
        print(f"\n--- Stage {stage_idx + 1}/{len(stage_steps)}: max_steps={max_steps} ---")
        study = optuna.create_study(
            direction="minimize",
            study_name=f"pid_maglev_stage_{max_steps}",
            sampler=TPESampler(
                n_startup_trials=min(20, n_trials_per_stage // 4),
                n_ei_candidates=24,
                seed=42 + stage_idx,
            ),
        )

        # Warm start: enqueue previous stage's best so we refine from it (stage 0 has no previous)
        if stage_idx > 0 and os.path.isfile(best_params_path):
            with open(best_params_path) as f:
                prev_best = json.load(f)
            study.enqueue_trial(prev_best)
            print(f"Enqueued previous best params (from stage {stage_steps[stage_idx - 1]} steps) as first trial.")

        study.optimize(
            lambda t: objective(
                t,
                initial_gaps_mm=initial_gaps_mm,
                max_steps=max_steps,
                use_feedforward=use_feedforward,
                penalty_early=penalty_early,
                disturbance_force_std=disturbance_force_std,
                n_objective_repeats=n_objective_repeats,
            ),
            n_trials=n_trials_per_stage,
            timeout=timeout_per_stage,
            show_progress_bar=True,
        )

        with open(best_params_path, "w") as f:
            json.dump(study.best_params, f, indent=2)
        stage_path = os.path.join(out_dir, f"pid_best_params_{max_steps}.json")
        with open(stage_path, "w") as f:
            json.dump(study.best_params, f, indent=2)
        print(f"Stage best cost: {study.best_value:.4f} -> saved to {best_params_path} and {stage_path}")
        studies.append(study)

    print(f"\nStaged optimization done. Final best params in {best_params_path}")
    return studies


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optuna PID tuning for maglev three-loop controller")
    parser.add_argument("--n_trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--gaps", type=float, nargs="+", default=DEFAULT_INITIAL_GAPS_MM)
    parser.add_argument("--out_dir", type=str, default=_SCRIPT_DIR, help="Directory for pid_best_params.json (default: same as script)")
    parser.add_argument("--no_feedforward", action="store_true")
    parser.add_argument("--staged", action="store_true", help="Staged optimization: 1500 -> 3000 -> 6000 steps, each stage warm-starts from previous best")
    parser.add_argument("--stage_steps", type=int, nargs="+", default=[1500, 3000, 6000], help="Steps per stage when using --staged (default: 1500 3000 6000)")
    parser.add_argument("--n_trials_per_stage", type=int, default=DEFAULT_N_TRIALS, help="Trials per stage when using --staged")
    parser.add_argument("--disturbance_force_std", type=float, default=0.0, help="Env disturbance force std (N); roll/pitch torque scale with this. Use >0 for noisy optimization.")
    parser.add_argument("--n_objective_repeats", type=int, default=1, help="Evaluate each trial this many times and report mean cost (reduces noise for Bayesian optimization)")
    args = parser.parse_args()

    if args.staged:
        run_staged_optimization(
            stage_steps=args.stage_steps,
            n_trials_per_stage=args.n_trials_per_stage,
            timeout_per_stage=args.timeout,
            initial_gaps_mm=args.gaps,
            use_feedforward=not args.no_feedforward,
            disturbance_force_std=args.disturbance_force_std,
            n_objective_repeats=args.n_objective_repeats,
            out_dir=args.out_dir,
        )
    else:
        run_optimization(
            n_trials=args.n_trials,
            timeout=args.timeout,
            initial_gaps_mm=args.gaps,
            max_steps=args.max_steps,
            use_feedforward=not args.no_feedforward,
            disturbance_force_std=args.disturbance_force_std,
            n_objective_repeats=args.n_objective_repeats,
            out_dir=args.out_dir,
        )
