import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, lsim

# ============================================================
# Circuit Parameters
# ============================================================
R = 1.5            # Resistance  [Ω]
L = 0.0025         # Inductance  [H]  (2.5 mH)
V_SUPPLY = 12.0    # Supply rail [V]
tau = L / R        # RL time constant ≈ 1.667 ms

# ============================================================
# PWM Parameters
# ============================================================
F_PWM = 16e3                    # PWM frequency  [Hz]
T_PWM = 1.0 / F_PWM            # PWM period     [s]  (62.5 µs)

# ============================================================
# Simulation Window
# ============================================================
T_END = 1e-3                    # 1 ms  →  16 full PWM cycles
DT    = 1e-7                    # 100 ns resolution (625 pts / PWM cycle)
t     = np.arange(0, T_END + DT / 2, DT)

# ============================================================
# Duty-Cycle Command  D(t)
# ============================================================
# Ramp from 20 % → 80 % over the window so every PWM cycle
# has a visibly different pulse width.
def duty_command(t_val):
    """Continuous duty-cycle setpoint (from a controller)."""
    return np.clip(0.2 + 0.6 * (np.asarray(t_val) / T_END), 0.0, 1.0)

D_continuous = duty_command(t)

# ============================================================
# MODEL 1 — Abstracted (Average-Voltage) Approximation
# ============================================================
# Treats the coil voltage as the smooth signal  D(t)·V.
# Transfer function:  I(s)/D(s) = V / (Ls + R)
G = TransferFunction([V_SUPPLY], [L, R])
_, i_avg, _ = lsim(G, U=D_continuous, T=t)

# ============================================================
# MODEL 2 — True PWM Waveform  (exact analytical solution)
# ============================================================
# Between every switching edge the RL circuit obeys:
#
#   di/dt = (V_seg − R·i) / L          (V_seg = V_SUPPLY or 0)
#
# Closed-form from initial condition i₀ at time t₀:
#
#   i(t) = V_seg/R  +  (i₀ − V_seg/R) · exp(−R·(t − t₀) / L)
#
# We propagate i analytically from edge to edge — zero
# numerical-integration error.  The only error source is
# IEEE-754 floating-point arithmetic (~1e-15 relative).

# --- Step 1: build segment table and propagate boundary currents ---
seg_t_start = []          # start time of each constant-V segment
seg_V       = []          # voltage applied during segment
seg_i0      = []          # current at segment start

i_boundary = 0.0          # coil starts de-energised

cycle = 0
while cycle * T_PWM < T_END:
    t_cycle = cycle * T_PWM
    D = float(duty_command(t_cycle))

    # ---- ON phase (V_SUPPLY applied) ----
    t_on_start = t_cycle
    t_on_end   = min(t_cycle + D * T_PWM, T_END)
    if t_on_end > t_on_start:
        seg_t_start.append(t_on_start)
        seg_V.append(V_SUPPLY)
        seg_i0.append(i_boundary)
        dt_on = t_on_end - t_on_start
        i_boundary = (V_SUPPLY / R) + (i_boundary - V_SUPPLY / R) * np.exp(-R * dt_on / L)

    # ---- OFF phase (0 V applied, free-wheeling through diode) ----
    t_off_start = t_on_end
    t_off_end   = min((cycle + 1) * T_PWM, T_END)
    if t_off_end > t_off_start:
        seg_t_start.append(t_off_start)
        seg_V.append(0.0)
        seg_i0.append(i_boundary)
        dt_off = t_off_end - t_off_start
        i_boundary = i_boundary * np.exp(-R * dt_off / L)

    cycle += 1

seg_t_start = np.array(seg_t_start)
seg_V       = np.array(seg_V)
seg_i0      = np.array(seg_i0)

# --- Step 2: evaluate on the dense time array (vectorised) ---
idx = np.searchsorted(seg_t_start, t, side='right') - 1
idx = np.clip(idx, 0, len(seg_t_start) - 1)

dt_in_seg = t - seg_t_start[idx]
V_at_t    = seg_V[idx]
i0_at_t   = seg_i0[idx]

i_pwm = (V_at_t / R) + (i0_at_t - V_at_t / R) * np.exp(-R * dt_in_seg / L)
v_pwm = V_at_t                       # switching waveform for plotting
v_avg = D_continuous * V_SUPPLY       # average-model voltage

# ============================================================
# Console Output — sanity-check steady-state values
# ============================================================
print(f"RL time constant  τ = L/R = {tau*1e3:.3f} ms")
print(f"PWM period        T = 1/f = {T_PWM*1e6:.1f} µs")
print(f"Sim resolution    Δt      = {DT*1e9:.0f} ns  ({int(T_PWM/DT)} pts/cycle)")
print()
print("Expected steady-state currents  i_ss = (V/R)·D :")
for d in [0.2, 0.4, 0.6, 0.8]:
    print(f"  D = {d:.1f}  →  i_ss = {V_SUPPLY / R * d:.3f} A")

# ============================================================
# Plotting — 4-panel comparison
# ============================================================
t_us = t * 1e6                        # time axis in µs

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
fig.suptitle("PWM RL-Circuit Model Comparison  (16 kHz, 1 ms window)",
             fontsize=13, fontweight='bold')

# --- 1. Duty-cycle command ---
ax = axes[0]
ax.plot(t_us, D_continuous * 100, color='tab:purple', linewidth=1.2)
ax.set_ylabel("Duty Cycle [%]")
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)

# --- 2. Voltage waveforms ---
ax = axes[1]
ax.plot(t_us, v_pwm, color='tab:orange', linewidth=0.6, label="True PWM voltage")
ax.plot(t_us, v_avg, color='tab:blue',   linewidth=1.4, label="Average voltage D·V",
        linestyle='--')
ax.set_ylabel("Voltage [V]")
ax.set_ylim(-0.5, V_SUPPLY + 1)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

# --- 3. Current comparison ---
ax = axes[2]
ax.plot(t_us, i_pwm, color='tab:red',  linewidth=0.7, label="True PWM current (exact)")
ax.plot(t_us, i_avg, color='tab:blue',  linewidth=1.4, label="Averaged-model current",
        linestyle='--')
ax.set_ylabel("Current [A]")
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

# --- 4. Difference / ripple ---
ax = axes[3]
ax.plot(t_us, (i_pwm - i_avg) * 1e3, color='tab:green', linewidth=0.7)
ax.set_ylabel("Δi  (PWM − avg) [mA]")
ax.set_xlabel("Time [µs]")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
