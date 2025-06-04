"""Diagnostic script to analyze speed stability from CSV output files."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import argparse


def analyze_speed_stability(csv_path: str, output_dir: str = None):
    """Analyze speed stability metrics from vehicle metrics CSV."""

    # Load data
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    if output_dir is None:
        output_dir = Path(csv_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

    # Group by vehicle ID
    vehicle_groups = df.groupby('vehicle_id')

    # Calculate stability metrics
    stability_metrics = []

    for vehicle_id, vehicle_data in vehicle_groups:
        if len(vehicle_data) < 5:  # Skip vehicles with too few frames
            continue

        speeds = vehicle_data['speed_km_h'].values
        frames = vehicle_data['frame'].values

        # Calculate metrics
        speed_std = np.std(speeds)
        speed_mean = np.mean(speeds)
        cv = speed_std / speed_mean if speed_mean > 0 else 0  # Coefficient of variation

        # Calculate acceleration between consecutive frames
        if len(speeds) > 1:
            # Convert km/h to m/s and calculate acceleration
            speeds_ms = speeds * (1 / 3.6)
            accelerations = np.diff(speeds_ms) * 30  # Assuming 30 fps
            max_accel = np.max(np.abs(accelerations))
            mean_accel = np.mean(np.abs(accelerations))

            # Detect direction reversals (speed sign changes)
            sign_changes = np.sum(np.diff(np.sign(speeds[speeds != 0])) != 0)
        else:
            max_accel = 0
            mean_accel = 0
            sign_changes = 0

        stability_metrics.append({
            'vehicle_id': vehicle_id,
            'num_frames': len(vehicle_data),
            'mean_speed_kmh': speed_mean,
            'std_speed_kmh': speed_std,
            'cv': cv,
            'max_acceleration_ms2': max_accel,
            'mean_acceleration_ms2': mean_accel,
            'direction_reversals': sign_changes
        })

    # Create metrics dataframe
    metrics_df = pd.DataFrame(stability_metrics)

    # Save metrics
    metrics_path = output_dir / 'speed_stability_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved stability metrics to: {metrics_path}")

    # Print summary statistics
    print("\n=== SPEED STABILITY ANALYSIS ===")
    print(f"Total vehicles analyzed: {len(metrics_df)}")
    print(f"\nAverage metrics across all vehicles:")
    print(f"  Mean CV (coefficient of variation): {metrics_df['cv'].mean():.3f}")
    print(f"  Mean max acceleration: {metrics_df['max_acceleration_ms2'].mean():.2f} m/s²")
    print(f"  Vehicles with direction reversals: {(metrics_df['direction_reversals'] > 0).sum()}")

    # Identify problematic vehicles
    print("\n=== PROBLEMATIC VEHICLES ===")

    # High variability
    high_cv = metrics_df[metrics_df['cv'] > 0.3].sort_values('cv', ascending=False)
    if not high_cv.empty:
        print(f"\nHigh speed variability (CV > 0.3):")
        for _, row in high_cv.head(5).iterrows():
            print(f"  Vehicle {int(row['vehicle_id'])}: CV={row['cv']:.3f}, "
                  f"mean={row['mean_speed_kmh']:.1f}±{row['std_speed_kmh']:.1f} km/h")

    # High acceleration
    high_accel = metrics_df[metrics_df['max_acceleration_ms2'] > 10].sort_values('max_acceleration_ms2',
                                                                                 ascending=False)
    if not high_accel.empty:
        print(f"\nUnrealistic accelerations (>10 m/s²):")
        for _, row in high_accel.head(5).iterrows():
            print(f"  Vehicle {int(row['vehicle_id'])}: max accel={row['max_acceleration_ms2']:.1f} m/s²")

    # Direction reversals
    reversals = metrics_df[metrics_df['direction_reversals'] > 0].sort_values('direction_reversals', ascending=False)
    if not reversals.empty:
        print(f"\nVehicles with direction reversals:")
        for _, row in reversals.head(5).iterrows():
            print(f"  Vehicle {int(row['vehicle_id'])}: {int(row['direction_reversals'])} reversals")

    # Plot examples of problematic vehicles
    plot_problematic_vehicles(df, metrics_df, output_dir)

    return metrics_df


def plot_problematic_vehicles(df, metrics_df, output_dir):
    """Plot speed profiles of problematic vehicles."""

    # Select top 3 most problematic vehicles
    problematic = metrics_df.nlargest(3, 'cv')

    if problematic.empty:
        return

    fig, axes = plt.subplots(len(problematic), 1, figsize=(10, 4 * len(problematic)))
    if len(problematic) == 1:
        axes = [axes]

    for idx, (_, vehicle_info) in enumerate(problematic.iterrows()):
        vehicle_id = vehicle_info['vehicle_id']
        vehicle_data = df[df['vehicle_id'] == vehicle_id].sort_values('frame')

        ax = axes[idx]

        # Plot speed over time
        frames = vehicle_data['frame'].values
        speeds = vehicle_data['speed_km_h'].values

        ax.plot(frames, speeds, 'b-', linewidth=2, label='Measured Speed')
        ax.scatter(frames, speeds, c='red', s=20, alpha=0.5)

        # Add rolling average
        if len(speeds) > 5:
            window = min(5, len(speeds))
            rolling_mean = pd.Series(speeds).rolling(window=window, center=True).mean()
            ax.plot(frames, rolling_mean, 'g--', linewidth=2, label=f'{window}-frame Average')

        ax.set_xlabel('Frame')
        ax.set_ylabel('Speed (km/h)')
        ax.set_title(f'Vehicle {int(vehicle_id)} - CV: {vehicle_info["cv"]:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'problematic_vehicles_speed_profiles.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved problematic vehicle plots to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze speed stability from vehicle metrics CSV')
    parser.add_argument('csv_path', help='Path to vehicle_metrics CSV file')
    parser.add_argument('--output_dir', help='Output directory for analysis results', default=None)

    args = parser.parse_args()

    if not Path(args.csv_path).exists():
        print(f"Error: File not found: {args.csv_path}")
        sys.exit(1)

    analyze_speed_stability(args.csv_path, args.output_dir)


if __name__ == "__main__":
    main()