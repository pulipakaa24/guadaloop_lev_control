import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Load your CSV ---
# If you exported from Google Sheets with headers "x,y":
data = pd.read_csv(r'C:\Users\k28ad\OneDrive\Documents\sensor\dataPrecise.csv')
x = data["x"].values
y = data["y"].values
print(x, y)
# --- Define logistic function with 3 parameters (a, b, c) ---
def logistic(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))
    # return c - ((1-b) * np.log((a - x)/x))

# --- Fit the curve ---
# p0 are initial guesses for (a, b, c) — tweak depending on your data
params, covariance = curve_fit(logistic, x, y, p0=[max(y), 1, np.median(x)])

a_fit, b_fit, c_fit = params
print(f"Fitted parameters:\na = {a_fit}\nb = {b_fit}\nc = {c_fit}")

# --- Plot data vs fitted curve ---
x_fit = np.linspace(min(x), max(x), 200)  # smooth x range
y_fit = logistic(x_fit, *params)

plt.scatter(x, y, label="Data")
plt.plot(x_fit, y_fit, 'r-', label="Fitted Logistic Curve")
plt.legend()
plt.show()
