import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Load your CSV ---
# If you exported from Google Sheets with headers "x,y":
data = pd.read_csv(r'C:\Users\k28ad\OneDrive\Documents\sensor\oldData\data.csv')
x = data["x"].values
y = data["y"].values
coeffs = np.polyfit(x, y, deg=5)  # 5th degree polynomial
poly = np.poly1d(coeffs)
print(poly)
x_smooth = np.linspace(0, 20, 200)
y_smooth = poly(x_smooth)

plt.scatter(x, y, label = "data")
plt.plot(x_smooth, y_smooth, 'r-', label="polynomial fit")
plt.show()
