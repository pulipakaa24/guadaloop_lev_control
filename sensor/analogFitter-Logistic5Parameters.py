import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Load CSV ---
data = pd.read_csv(r'C:\Users\k28ad\OneDrive\Documents\sensor\Sensor3DataNew.csv')
x = data["x"].values
# Stack them into a 2D array (shape: 5 × 100)
# stacked = np.column_stack((data["y"].values, data["y2"].values, data["y3"].values, data["y4"].values, data["y5"].values))

# Take the average across the 5 columns for each row
# y = np.mean(data["y"], axis=1)
# print(y)
y = data["y"].values
# --- Define 5-parameter logistic function ---
def logistic_5pl(x, A, K, B, C, nu):
    # Clip to avoid overflow in exp for large values
    z = -B * (x - C)
    z = np.clip(z, -500, 500)
    return A + (K - A) / (1.0 + np.exp(z))**(1/nu)

# --- Initial parameter guesses ---
A0 = np.min(y)          # lower asymptote
K0 = np.max(y)          # upper asymptote
B0 = 1.0                # slope
C0 = np.median(x)       # inflection point
nu0 = 1.0               # asymmetry (1 = symmetric, same as 4PL)
p0 = [A0, K0, B0, C0, nu0]

# --- Fit the curve ---
params, covariance = curve_fit(
    logistic_5pl, x, y, p0=p0, maxfev=20000,
    bounds=([-np.inf, -np.inf, 0, -np.inf, 0.1], [np.inf, np.inf, np.inf, np.inf, 10])
)

A_fit, K_fit, B_fit, C_fit, nu_fit = params
print("Fitted parameters:")
print(f"A  = {A_fit}")
print(f"K  = {K_fit}")
print(f"B  = {B_fit}")
print(f"C  = {C_fit}")
print(f"nu = {nu_fit}")

# --- Predicted values ---
y_pred = logistic_5pl(x, *params)

# --- Goodness of fit (R²) ---
residuals = y - y_pred
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)
print(f"\nGoodness of fit: R² = {r_squared:.6f}")

# --- Plot ---
x_fit = np.linspace(min(x), max(x), 400)
y_fit = logistic_5pl(x_fit, *params)
y_pred = logistic_5pl(x, *params)


plt.figure(figsize=(8, 5))
plt.scatter(x, y, label="Data", color="blue")
plt.plot(x_fit, y_fit, 'r-', label="5PL Fit")
plt.title("5-Parameter Logistic Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
