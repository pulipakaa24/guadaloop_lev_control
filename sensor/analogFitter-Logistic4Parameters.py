import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Load your CSV ---
data = pd.read_csv(r'C:\Users\k28ad\OneDrive\Documents\sensor\dataPrecise.csv')
x = data["x"].values
y = data["y"].values

# --- Define 4-parameter logistic function (A, K, B, C) ---
def logistic_4pl(x, A, K, B, C):
    return A + (K - A) / (1.0 + np.exp(-B * (x - C)))

# --- Initial guesses ---
A0 = np.min(y)
K0 = np.max(y)
B0 = 1.0
C0 = np.median(x)
p0 = [A0, K0, B0, C0]

# --- Fit the curve ---
params, covariance = curve_fit(logistic_4pl, x, y, p0=p0, maxfev=10000)
A_fit, K_fit, B_fit, C_fit = params

print(f"Fitted parameters:")
print(f"A (lower asymptote) = {A_fit:.4f}")
print(f"K (upper asymptote) = {K_fit:.4f}")
print(f"B (slope)          = {B_fit:.4f}")
print(f"C (midpoint)       = {C_fit:.4f}")

# --- Fitted curve ---
x_fit = np.linspace(min(x), max(x), 400)
y_fit = logistic_4pl(x_fit, *params)

# --- Plot data vs fitted curve ---
plt.figure(figsize=(8, 5))
plt.scatter(x, y, label="Data")
plt.plot(x_fit, y_fit, 'r-', label="Fitted Logistic Curve")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("4-Parameter Logistic Fit")
plt.show()

# --- Goodness of fit ---
y_pred = logistic_4pl(x, *params)
offsets = y_pred - y
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
rmse = np.sqrt(np.mean((y - y_pred) ** 2))
mae = np.mean(np.abs(y - y_pred))

n = len(y)
p = len(params)
r2_adj = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

print("\nGoodness of fit:")
# for value in y_pred: 
#     print(value)
# for value in y:
#     print (value)

print(f"R²          = {r_squared:.4f}")
print(f"Adjusted R² = {r2_adj:.4f}")
print(f"RMSE        = {rmse:.4f}")
print(f"MAE         = {mae:.4f}")

# --- Residual plot ---
residuals = y - y_pred
plt.figure(figsize=(8, 4))
plt.scatter(x, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('Residual (y - y_fit)')
plt.title('Residual Plot')
plt.show()
