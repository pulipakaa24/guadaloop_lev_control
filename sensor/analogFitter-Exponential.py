import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# Data from the chart
data = pd.read_csv(r'C:\Users\k28ad\OneDrive\Documents\sensor\oldData\data.csv')
x = data["x"].values
y = data["y"].values

# Define a possible fitting model (exponential fit in this case)
def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

# Perform the curve fitting
params, covariance = curve_fit(exponential, x, y)

# Plot the data and the fitting curve
x_fit = np.linspace(2, 20, 100)
y_fit = exponential(x_fit, *params)

plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x_fit, y_fit, color='red', label='Exponential fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Exponential Fit')
plt.grid(True)
plt.show()

# Output the fitted parameters
print("Fitted parameters:")
print(f"a = {params[0]}, b = {params[1]}, c = {params[2]}")