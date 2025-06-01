import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class SpeedRegressionAnalyzer:
    def __init__(self, csv_file_path=None, predicted_col='predicted_speeds', ground_truth_col='ground_truth_speeds'):
        """
        Initialize the analyzer with CSV file path and column names
        """
        self.csv_file_path = csv_file_path
        self.predicted_col = predicted_col
        self.ground_truth_col = ground_truth_col
        self.models = {}
        self.results = {}
        self.equations = {}  # To store model equations

    def load_data(self, predicted_speeds=None, ground_truth_speeds=None):
        """
        Load data either from CSV file or from provided lists
        """
        if self.csv_file_path:
            # Load from CSV file
            data = pd.read_csv(self.csv_file_path)
            self.X = data[self.predicted_col].values.reshape(-1, 1)
            self.y = data[self.ground_truth_col].values
        elif predicted_speeds is not None and ground_truth_speeds is not None:
            # Load from provided lists
            self.X = np.array(predicted_speeds).reshape(-1, 1)
            self.y = np.array(ground_truth_speeds)
        else:
            raise ValueError("Either provide CSV file path or both speed lists")

        print(f"Loaded {len(self.X)} data points")
        print(f"Predicted speeds range: {self.X.min():.2f} - {self.X.max():.2f} km/h")
        print(f"Ground truth range: {self.y.min():.2f} - {self.y.max():.2f} km/h")

    def fit_models(self):
        """
        Fit all regression models and extract their equations.
        """
        # 1. Linear Regression
        self.models['Linear'] = LinearRegression()
        self.models['Linear'].fit(self.X, self.y)

        # 2. Polynomial Regression (degree 2)
        self.models['Polynomial_2'] = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),  # Set include_bias=False
            ('linear', LinearRegression())
        ])
        self.models['Polynomial_2'].fit(self.X, self.y)

        # 3. Polynomial Regression (degree 3)
        self.models['Polynomial_3'] = Pipeline([
            ('poly', PolynomialFeatures(degree=3, include_bias=False)),  # Set include_bias=False
            ('linear', LinearRegression())
        ])
        self.models['Polynomial_3'].fit(self.X, self.y)

        # 4. RANSAC Regression
        self.models['RANSAC'] = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=int(0.3 * len(self.X)),
            residual_threshold=2.0,  # Adjust based on expected noise level
            random_state=42
        )
        self.models['RANSAC'].fit(self.X, self.y)

        print("All models fitted successfully.")
        self._extract_model_equations()  # Extract equations after fitting

    def _extract_model_equations(self):
        """
        Extracts and stores the regression equation string for each fitted model.
        """
        # Linear Regression
        lin_model = self.models['Linear']
        coef_lin = lin_model.coef_[0]
        intercept_lin = lin_model.intercept_
        self.equations['Linear'] = f"y = {coef_lin:.4f}x + {intercept_lin:.4f}"

        # Polynomial Regression (degree 2)
        # The 'linear' step in the pipeline contains the coefficients and intercept
        poly2_linear_model = self.models['Polynomial_2'].named_steps['linear']
        intercept_poly2 = poly2_linear_model.intercept_
        coeffs_poly2 = poly2_linear_model.coef_  # coeffs_poly2[0] for x, coeffs_poly2[1] for x^2
        self.equations['Polynomial_2'] = f"y = {intercept_poly2:.4f} + {coeffs_poly2[0]:.4f}x + {coeffs_poly2[1]:.4f}x²"

        # Polynomial Regression (degree 3)
        poly3_linear_model = self.models['Polynomial_3'].named_steps['linear']
        intercept_poly3 = poly3_linear_model.intercept_
        coeffs_poly3 = poly3_linear_model.coef_  # coeffs_poly3[0] for x, coeffs_poly3[1] for x^2, etc.
        self.equations['Polynomial_3'] = (f"y = {intercept_poly3:.4f} + {coeffs_poly3[0]:.4f}x + "
                                          f"{coeffs_poly3[1]:.4f}x² + {coeffs_poly3[2]:.4f}x³")

        # RANSAC Regression
        # The fitted estimator (LinearRegression) holds the coefficients for inliers
        ransac_estimator = self.models['RANSAC'].estimator_
        coef_ransac = ransac_estimator.coef_[0]
        intercept_ransac = ransac_estimator.intercept_
        self.equations['RANSAC'] = f"y = {coef_ransac:.4f}x + {intercept_ransac:.4f} (for inliers)"

        print("Model equations extracted.")

    def calculate_metrics(self):
        """
        Calculate performance metrics for all models
        """
        self.results = {}

        for name, model in self.models.items():
            y_pred = model.predict(self.X)

            mae = mean_absolute_error(self.y, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y, y_pred))
            r2 = r2_score(self.y, y_pred)

            self.results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'predictions': y_pred
            }
        print("Performance metrics calculated.")

    def plot_results(self):
        """
        Create visualization comparing all models
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Speed Regression Model Comparison', fontsize=16, fontweight='bold')

        # Generate smooth line for plotting
        X_range = np.linspace(self.X.min(), self.X.max(), 100).reshape(-1, 1)

        model_names = ['Linear', 'Polynomial_2', 'Polynomial_3', 'RANSAC']
        titles = ['Linear Regression', 'Polynomial Regression (Degree 2)',
                  'Polynomial Regression (Degree 3)', 'RANSAC Regression']

        for idx, (name, title) in enumerate(zip(model_names, titles)):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]

            # Scatter plot of actual data
            ax.scatter(self.X, self.y, alpha=0.6, color='blue', s=30, label='Actual Data')

            # Plot regression line
            if name == 'RANSAC':
                # For RANSAC, highlight inliers and outliers
                inlier_mask = self.models[name].inlier_mask_
                ax.scatter(self.X[inlier_mask], self.y[inlier_mask],
                           color='blue', s=30, alpha=0.6,
                           label='Inliers')  # Changed color to distinguish in RANSAC plot
                ax.scatter(self.X[~inlier_mask], self.y[~inlier_mask],
                           color='orange', s=30, alpha=0.8, label='Outliers')  # Changed color

            y_range_pred = self.models[name].predict(X_range)
            ax.plot(X_range, y_range_pred, color='red', linewidth=2, label='Regression Line')

            # Add perfect correlation line
            perfect_line_min = min(self.X.min(), self.y.min())
            perfect_line_max = max(self.X.max(), self.y.max())
            ax.plot([perfect_line_min, perfect_line_max], [perfect_line_min, perfect_line_max],
                    '--', color='green', alpha=0.7, label='Perfect Correlation')

            # Formatting
            ax.set_xlabel('Predicted Speeds (km/h)')
            ax.set_ylabel('Ground Truth Speeds (km/h)')
            ax.set_title(f'{title}\nR² = {self.results[name]["R²"]:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for suptitle
        plt.show()

    def display_results_table(self):
        """
        Display performance metrics and model equations in a formatted table
        """
        print("\n" + "=" * 130)  # Adjusted table width
        print("REGRESSION MODEL PERFORMANCE COMPARISON")
        print("=" * 130)
        # Adjusted column widths
        print(f"{'Model':<25} {'MAE':<12} {'RMSE':<12} {'R² Score':<12} {'Equation':<60}")
        print("-" * 130)

        best_mae_val = min(self.results.values(), key=lambda x: x['MAE'])['MAE']
        best_rmse_val = min(self.results.values(), key=lambda x: x['RMSE'])['RMSE']
        best_r2_val = max(self.results.values(), key=lambda x: x['R²'])['R²']

        for name, metrics in self.results.items():
            model_display = name.replace('_', ' (deg ') + (')' if '_' in name else '')
            mae_str = f"{metrics['MAE']:.4f}" + (" *" if metrics['MAE'] == best_mae_val else "")
            rmse_str = f"{metrics['RMSE']:.4f}" + (" *" if metrics['RMSE'] == best_rmse_val else "")
            r2_str = f"{metrics['R²']:.4f}" + (" *" if metrics['R²'] == best_r2_val else "")
            equation_str = self.equations.get(name, "N/A")  # Retrieve the equation

            print(f"{model_display:<25} {mae_str:<12} {rmse_str:<12} {r2_str:<12} {equation_str:<60}")

        print("-" * 130)
        print("* indicates best performance in that metric")

    def identify_best_model(self):
        """
        Identify the best performing model based on combined metrics
        """
        print("\n" + "=" * 60)
        print("BEST MODEL ANALYSIS")
        print("=" * 60)

        if not self.results:
            print("No results to analyze. Please run calculate_metrics() first.")
            return None

        mae_scores = [result['MAE'] for result in self.results.values()]
        rmse_scores = [result['RMSE'] for result in self.results.values()]
        r2_scores = [result['R²'] for result in self.results.values()]

        # Handle cases where all scores are the same to avoid division by zero
        mae_range = (max(mae_scores) - min(mae_scores))
        rmse_range = (max(rmse_scores) - min(rmse_scores))
        r2_range = (max(r2_scores) - min(r2_scores))

        composite_scores = {}
        for name, metrics in self.results.items():
            mae_norm = 0.5 if mae_range == 0 else 1 - (metrics['MAE'] - min(mae_scores)) / mae_range
            rmse_norm = 0.5 if rmse_range == 0 else 1 - (metrics['RMSE'] - min(rmse_scores)) / rmse_range
            r2_norm = 0.5 if r2_range == 0 else (metrics['R²'] - min(r2_scores)) / r2_range

            composite_scores[name] = (mae_norm + rmse_norm + r2_norm) / 3

        best_model_name = max(composite_scores, key=composite_scores.get)

        display_name = best_model_name.replace('_', ' (degree ') + (')' if '_' in best_model_name else '')
        print(f"Best Overall Model: {display_name}")
        print(f"Composite Score: {composite_scores[best_model_name]:.4f}")
        print(f"MAE: {self.results[best_model_name]['MAE']:.4f} km/h")
        print(f"RMSE: {self.results[best_model_name]['RMSE']:.4f} km/h")
        print(f"R² Score: {self.results[best_model_name]['R²']:.4f}")

        return best_model_name

    def run_complete_analysis(self, predicted_speeds=None, ground_truth_speeds=None):
        """
        Run the complete analysis pipeline
        """
        self.load_data(predicted_speeds, ground_truth_speeds)
        self.fit_models()  # This will also call _extract_model_equations
        self.calculate_metrics()
        self.display_results_table()
        self.plot_results()
        best_model = self.identify_best_model()

        return best_model


# Example usage and demonstration
if __name__ == "__main__":
    # Example data (replace with your actual data)
    np.random.seed(42)
    n_samples = 100

    # Generate synthetic speed data with some noise and outliers
    true_speeds = np.random.uniform(8, 25, n_samples)  # km/h
    # Introduce a slight non-linear relationship for polynomial models to potentially capture
    predicted_speeds = true_speeds * (
                1 + np.random.normal(0, 0.05, n_samples)) - 0.02 * true_speeds ** 2 + 5 + np.random.normal(0, 1.5,
                                                                                                           n_samples)
    predicted_speeds = np.clip(predicted_speeds, 0, 50)  # Ensure predicted speeds are non-negative and reasonable

    # Add some outliers
    outlier_indices = np.random.choice(n_samples, 5, replace=False)
    predicted_speeds[outlier_indices] += np.random.normal(0, 8, 5)  # Made outliers more significant

    # Create a dummy CSV for testing the CSV loading path, if needed
    # df_test = pd.DataFrame({'predicted_speeds': predicted_speeds, 'ground_truth_speeds': true_speeds})
    # df_test.to_csv('speed_data.csv', index=False)

    # Initialize analyzer (passing None for csv_file_path as we provide data directly)
    analyzer = SpeedRegressionAnalyzer(csv_file_path='speed_data.csv', predicted_col='predicted_speeds',ground_truth_col='ground_truth_speeds')

    # Run complete analysis using provided lists
    print("SPEED REGRESSION ANALYSIS")
    print("=" * 50)
    best_model = analyzer.run_complete_analysis(predicted_speeds=list(predicted_speeds),
                                                ground_truth_speeds=list(true_speeds))

    # Example of how to use with CSV file (if 'speed_data.csv' exists):
    # print("\n\nANALYSIS FROM CSV (Example - ensure 'speed_data.csv' exists)")
    # analyzer_csv = SpeedRegressionAnalyzer(csv_file_path='speed_data.csv',
    #                                      predicted_col='predicted_speeds',
    #                                      ground_truth_col='ground_truth_speeds')
    # best_model_csv = analyzer_csv.run_complete_analysis()