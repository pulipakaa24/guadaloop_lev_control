import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, lsim

# ---- Parameters (edit these) ----
R = 1.5          # ohms
L = 0.0025       # henries
V = 12.0         # volts

# Time base
t_end = 0.2
dt = 1e-4
t = np.arange(0, t_end, dt)

# ---- Define a duty command D(t) ----
# Example: start at 0, step to 0.2 at 20 ms, then to 0.6 at 80 ms, then to 1.0 at 140 ms (DC full on)
D = np.zeros_like(t)
D[t >= 0.020] = 0.2
D[t >= 0.080] = 0.6
D[t >= 0.140] = 1.0  # "straight DC" case (100% duty)

# Clamp just in case
D = np.clip(D, 0.0, 1.0)

# ---- Transfer function I(s)/D(s) = V / (L s + R) ----
# In scipy.signal.TransferFunction, numerator/denominator are polynomials in s
G = TransferFunction([V], [L, R])

# Simulate i(t) response to input D(t)
tout, i, _ = lsim(G, U=D, T=t)

# ---- Print steady-state expectations ----
# For constant duty D0, steady-state i_ss = (V/R)*D0
print("Expected steady-state currents (V/R * D):")
for D0 in [0.0, 0.2, 0.6, 1.0]:
    print(f"  D={D0:.1f} -> i_ss ~ {(V/R)*D0:.3f} A")


# ---- Plot ----
plt.figure()
plt.plot(tout, D, label="Duty D(t)")
plt.plot(tout, i, label="Current i(t) [A]")
plt.xlabel("Time [s]")
plt.grid(True)
plt.legend()
plt.show()

