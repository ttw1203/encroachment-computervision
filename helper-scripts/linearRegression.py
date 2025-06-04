import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

# Step 1: Define data
ground_truth = np.array([
    6.522151899, 24.98181818, 24.60895522, 24.31858407, 26.33865815,
    25.52321981, 18.73636364, 11.8618705, 5.67377839, 27.94567271,
    18.69387755, 19.62857143, 31.22727273, 26.08860759, 16.68825911,
    17.76724138, 33.64897959, 26.94117647, 25.21100917, 14.46315789,
    36.31718062
])

predicted = np.array([
    7.76, 20.77, 29.16, 26.17, 28.07,
    30.52, 18.36, 13.43, 5.65, 28,
    22, 18.45, 35.07, 27.61, 16,
    17.42, 40, 32.89, 31, 16.1,
    32.7
])

# Before calibration metrics
mae_raw = mean_absolute_error(ground_truth, predicted)
rmse_raw = math.sqrt(mean_squared_error(ground_truth, predicted))
r2_raw = r2_score(ground_truth, predicted)

print("\nBefore Calibration:")
print(f"MAE:  {mae_raw:.4f}")
print(f"RMSE: {rmse_raw:.4f}")
print(f"R²:   {r2_raw:.4f}")


# Step 2: Fit Linear Regression
X = predicted.reshape(-1, 1)
y = ground_truth

model = LinearRegression()
model.fit(X, y)

# Step 3: Predict and calculate metrics
y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
rmse = math.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

# Step 4: Output results
print(f"Regression Equation: calibrated_speed = {model.coef_[0]:.4f} * predicted_speed + {model.intercept_:.4f}")
print(f"MAE:  {mae:.4f} km/h")
print(f"RMSE: {rmse:.4f} km/h")
print(f"R²:   {r2:.4f}")

import matplotlib.pyplot as plt
import numpy as np

# Plot actual data
plt.scatter(predicted, ground_truth, color='blue', label='Data points')

# Plot linear regression line
x_line = np.linspace(min(predicted), max(predicted), 100)
y_line = model.coef_[0] * x_line + model.intercept_
plt.plot(x_line, y_line, color='red', label='Linear Regression')

# Labels
plt.xlabel('Predicted Speed (km/h)')
plt.ylabel('Ground Truth Speed (km/h)')
plt.title('Speed Calibration: Predicted vs Ground Truth')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

