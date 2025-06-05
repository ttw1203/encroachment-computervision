#!/usr/bin/env python3
"""
TTC-Encroachment Event Correlator

This script correlates Time-to-Collision (TTC) events with vehicle encroachment events
using vectorized operations for efficient processing of large datasets.
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path

# Configuration Parameters
CONFIG = {
    'encroachment_csv_path': 'D:\\thesisVideo\\mirpur\\results\\enc_events_06062025_022336.csv',
    'ttc_csv_path': 'D:\\thesisVideo\\mirpur\\results\\enhanced_ttc_events_06062025_022336.csv',
    'output_csv_path': 'D:\\thesisVideo\\mirpur\\results\\correlated_ttc_events.csv',
    'video_fps': 30.0,
    'time_correlation_window_before_entry': 5,  # seconds before encroachment entry
    'time_correlation_window_after_flag': 20.0  # seconds after encroachment flag
}


def load_and_validate_data(enc_path, ttc_path):
    """Load and validate input CSV files."""
    # Required columns for validation
    required_enc_cols = ['tracker_id', 't_entry_s', 't_flag_s', 'zone']
    required_ttc_cols = ['frame', 'follower_id', 'leader_id', 'ttc_s']

    try:
        # Load encroachment data
        enc_df = pd.read_csv(enc_path)
        missing_enc = [col for col in required_enc_cols if col not in enc_df.columns]
        if missing_enc:
            raise ValueError(f"Encroachment CSV missing columns: {missing_enc}")

        # Load TTC data
        ttc_df = pd.read_csv(ttc_path)
        missing_ttc = [col for col in required_ttc_cols if col not in ttc_df.columns]
        if missing_ttc:
            raise ValueError(f"TTC CSV missing columns: {missing_ttc}")

        print(f"Loaded {len(enc_df)} encroachment events and {len(ttc_df)} TTC events")
        return enc_df, ttc_df

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def prepare_encroachment_intervals(enc_df, window_before, window_after):
    """
    Prepare encroachment data with extended time intervals for efficient correlation.
    Creates start and end times for each encroachment event's active period.
    """
    enc_prepared = enc_df.copy()

    # Calculate extended time windows
    enc_prepared['correlation_start'] = enc_prepared['t_entry_s'] - window_before
    enc_prepared['correlation_end'] = enc_prepared['t_flag_s'] + window_after

    # Sort by tracker_id and correlation_start for efficient processing
    enc_prepared = enc_prepared.sort_values(['tracker_id', 'correlation_start'])

    return enc_prepared


def correlate_vehicle_encroachment(ttc_times, vehicle_ids, enc_prepared, vehicle_type):
    """
    Vectorized correlation of vehicle IDs with encroachment events.
    Returns correlation flags and details for the specified vehicle type.
    """
    n_ttc = len(ttc_times)
    is_encroaching = np.zeros(n_ttc, dtype=bool)
    enc_details = [''] * n_ttc

    # Get unique vehicle IDs to process
    unique_vehicles = pd.Series(vehicle_ids).dropna().unique()

    for vehicle_id in unique_vehicles:
        # Get TTC events for this vehicle
        vehicle_mask = (vehicle_ids == vehicle_id)
        if not vehicle_mask.any():
            continue

        vehicle_times = ttc_times[vehicle_mask]

        # Get encroachment events for this vehicle
        vehicle_enc = enc_prepared[enc_prepared['tracker_id'] == vehicle_id]
        if vehicle_enc.empty:
            continue

        # Vectorized interval overlap check
        for _, enc_event in vehicle_enc.iterrows():
            start_time = enc_event['correlation_start']
            end_time = enc_event['correlation_end']

            # Find TTC events within this encroachment's time window
            time_mask = (vehicle_times >= start_time) & (vehicle_times <= end_time)

            if time_mask.any():
                # Map back to original indices
                original_indices = np.where(vehicle_mask)[0][time_mask]

                # Update flags and details for matching events
                is_encroaching[original_indices] = True

                # Create detail string (use first encroachment if multiple)
                detail_str = f"zone:{enc_event['zone']},entry:{enc_event['t_entry_s']:.2f}s,flag:{enc_event['t_flag_s']:.2f}s"
                for idx in original_indices:
                    if enc_details[idx] == '':  # Only set if not already set
                        enc_details[idx] = detail_str

                break  # Use first matching encroachment per vehicle

    return is_encroaching, enc_details


def correlate_ttc_with_encroachment(ttc_df, enc_df, video_fps, window_before, window_after):
    """
    Main correlation function using vectorized operations.
    """
    # Convert frame numbers to time
    ttc_df = ttc_df.copy()
    ttc_df['ttc_time_s'] = ttc_df['frame'] / video_fps

    # Prepare encroachment data with extended time windows
    enc_prepared = prepare_encroachment_intervals(enc_df, window_before, window_after)

    # Extract time and ID arrays for vectorized processing
    ttc_times = ttc_df['ttc_time_s'].values
    follower_ids = ttc_df['follower_id'].values
    leader_ids = ttc_df['leader_id'].values

    print("Correlating follower vehicles with encroachment events...")
    follower_is_enc, follower_details = correlate_vehicle_encroachment(
        ttc_times, follower_ids, enc_prepared, 'follower'
    )

    print("Correlating leader vehicles with encroachment events...")
    leader_is_enc, leader_details = correlate_vehicle_encroachment(
        ttc_times, leader_ids, enc_prepared, 'leader'
    )

    # Add correlation results to TTC dataframe
    ttc_df['follower_is_encroaching'] = follower_is_enc
    ttc_df['leader_is_encroaching'] = leader_is_enc
    ttc_df['follower_enc_details'] = follower_details
    ttc_df['leader_enc_details'] = leader_details

    # Add summary flags
    ttc_df['any_vehicle_encroaching'] = follower_is_enc | leader_is_enc
    ttc_df['both_vehicles_encroaching'] = follower_is_enc & leader_is_enc

    return ttc_df


def generate_sample_output():
    """Generate sample output rows to demonstrate expected schema."""
    sample_data = {
        'frame': [1500, 2100],
        'follower_id': [101, 203],
        'leader_id': [87, 156],
        'ttc_s': [2.1, 1.8],
        'follower_class': ['car', 'truck'],
        'leader_class': ['car', 'car'],
        'closing_distance_m': [45.2, 38.7],
        'ttc_time_s': [50.0, 70.0],
        'follower_is_encroaching': [True, False],
        'leader_is_encroaching': [False, True],
        'follower_enc_details': ['zone:left,entry:48.50s,flag:49.20s', ''],
        'leader_enc_details': ['', 'zone:right,entry:68.10s,flag:69.80s'],
        'any_vehicle_encroaching': [True, True],
        'both_vehicles_encroaching': [False, False]
    }

    return pd.DataFrame(sample_data)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Correlate TTC events with encroachment events')
    parser.add_argument('--enc-csv', default=CONFIG['encroachment_csv_path'],
                        help='Path to encroachment events CSV')
    parser.add_argument('--ttc-csv', default=CONFIG['ttc_csv_path'],
                        help='Path to TTC events CSV')
    parser.add_argument('--output', default=CONFIG['output_csv_path'],
                        help='Output CSV path')
    parser.add_argument('--fps', type=float, default=CONFIG['video_fps'],
                        help='Video frames per second')
    parser.add_argument('--window-before', type=float,
                        default=CONFIG['time_correlation_window_before_entry'],
                        help='Time window before encroachment entry (seconds)')
    parser.add_argument('--window-after', type=float,
                        default=CONFIG['time_correlation_window_after_flag'],
                        help='Time window after encroachment flag (seconds)')
    parser.add_argument('--show-sample', action='store_true',
                        help='Show sample output schema and exit')

    args = parser.parse_args()

    if args.show_sample:
        print("Sample Expected Output Schema:")
        print("=" * 50)
        sample_df = generate_sample_output()
        print(sample_df.to_string(index=False))
        print("\nColumn Descriptions:")
        print("- ttc_time_s: TTC event time in seconds (converted from frame)")
        print("- follower_is_encroaching: Boolean flag for follower encroachment")
        print("- leader_is_encroaching: Boolean flag for leader encroachment")
        print("- *_enc_details: Encroachment zone and timing details")
        print("- any_vehicle_encroaching: Either follower or leader encroaching")
        print("- both_vehicles_encroaching: Both follower and leader encroaching")
        return

    print(f"TTC-Encroachment Correlation Analysis")
    print(f"Time window: -{args.window_before}s to +{args.window_after}s from encroachment period")
    print(f"Video FPS: {args.fps}")

    # Load and validate data
    enc_df, ttc_df = load_and_validate_data(args.enc_csv, args.ttc_csv)

    # Perform correlation
    print("Starting correlation analysis...")
    correlated_df = correlate_ttc_with_encroachment(
        ttc_df, enc_df, args.fps, args.window_before, args.window_after
    )

    # Generate summary statistics
    total_ttc = len(correlated_df)
    follower_enc_count = correlated_df['follower_is_encroaching'].sum()
    leader_enc_count = correlated_df['leader_is_encroaching'].sum()
    any_enc_count = correlated_df['any_vehicle_encroaching'].sum()
    both_enc_count = correlated_df['both_vehicles_encroaching'].sum()

    print(f"\nCorrelation Results:")
    print(f"Total TTC events: {total_ttc}")
    print(f"TTC events with encroaching follower: {follower_enc_count} ({follower_enc_count / total_ttc * 100:.1f}%)")
    print(f"TTC events with encroaching leader: {leader_enc_count} ({leader_enc_count / total_ttc * 100:.1f}%)")
    print(f"TTC events with any encroaching vehicle: {any_enc_count} ({any_enc_count / total_ttc * 100:.1f}%)")
    print(f"TTC events with both vehicles encroaching: {both_enc_count} ({both_enc_count / total_ttc * 100:.1f}%)")

    # Save results
    correlated_df.to_csv(args.output, index=False)
    print(f"\nCorrelated data saved to: {args.output}")


if __name__ == "__main__":
    main()