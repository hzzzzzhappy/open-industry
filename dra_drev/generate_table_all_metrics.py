#!/usr/bin/env python3
"""
Generate evaluation statistics tables for all metrics
Supports Overall ROC/PR, Seen-only ROC/PR, Unseen-only ROC/PR, Point-level ROC/PR
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# ========== Configuration ==========
DEFAULT_EVAL_DIR = "./experiments/eval"
DEFAULT_OUTPUT_DIR = "./experiments/metrics"

CLASSNAMES = [
    "airplane", "candybar", "car", "chicken", "diamond", "duck",
    "fish", "gemstone", "seahorse", "shell", "starfish", "toffees"
    ]
BACKBONES = ["Point_MAE", "Point_BERT"]
NANOMALIES = [2, 4]
SEEDS = [1, 2, 3, 4, 5]

# All supported metrics
METRICS_CONFIG = {
    'overall_roc': {'name': 'Overall ROC', 'pattern': r'Overall.*?ROC:\s*([\d.]+)'},
    'overall_pr': {'name': 'Overall PR', 'pattern': r'Overall.*?PR:\s*([\d.]+)'},
    'seen_roc': {'name': 'Seen-only ROC', 'pattern': r'Seen-only.*?ROC:\s*([\d.]+)'},
    'seen_pr': {'name': 'Seen-only PR', 'pattern': r'Seen-only.*?PR:\s*([\d.]+)'},
    'unseen_roc': {'name': 'Unseen-only ROC', 'pattern': r'Unseen-only.*?ROC:\s*([\d.]+)'},
    'unseen_pr': {'name': 'Unseen-only PR', 'pattern': r'Unseen-only.*?PR:\s*([\d.]+)'},
    'point_roc': {'name': 'Point-level ROC', 'pattern': r'Dataset-level Point ROC:\s*([\d.]+)'},
    'point_pr': {'name': 'Point-level PR', 'pattern': r'Dataset-level Point ROC:.*?PR:\s*([\d.]+)'},
    'point_seen_roc': {'name': 'Point Seen ROC', 'pattern': r'Point Seen ROC:\s*([\d.]+)'},
    'point_seen_pr': {'name': 'Point Seen PR', 'pattern': r'Point Seen ROC:.*?PR:\s*([\d.]+)'},
    'point_unseen_roc': {'name': 'Point Unseen ROC', 'pattern': r'Point Unseen ROC:\s*([\d.]+)'},
    'point_unseen_pr': {'name': 'Point Unseen PR', 'pattern': r'Point Unseen ROC:.*?PR:\s*([\d.]+)'},
    }


def get_method_dir_name(method):
    """Get the directory name corresponding to the method (handle case sensitivity)"""
    if method.upper() == 'DEVNET':
        return 'DevNet'
    return method.upper()


def extract_metric_from_log(log_file, metric_key):
    """Extract specified metric from eval.log"""
    if not os.path.exists(log_file):
        return None

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            pattern = METRICS_CONFIG[metric_key]['pattern']
            match = re.search(pattern, content)
            if match:
                return float(match.group(1))
    except Exception as e:
        print(f"[Warning] Failed to read log {log_file}: {e}")

    return None


def get_seed_metric(method, classname, backbone, n_anomaly, seed, eval_dir, metric_key):
    """Get metric value for a specific seed"""
    method_dir = get_method_dir_name(method)
    eval_path = Path(eval_dir) / method_dir / classname / f"{backbone}_nAnomaly{n_anomaly}_seed{seed}"
    log_file = eval_path / "eval.log"

    if log_file.exists():
        return extract_metric_from_log(str(log_file), metric_key)
    return None


def calculate_stats(method, classname, backbone, n_anomaly, eval_dir, metric_key):
    """Compute mean and std over 5 seeds"""
    values = []

    for seed in SEEDS:
        val = get_seed_metric(method, classname, backbone, n_anomaly, seed, eval_dir, metric_key)
        if val is not None:
            values.append(val)

    if not values:
        return None, None

    mean = np.mean(values)
    std = np.std(values)

    return mean, std


def format_value(mean, std):
    """Format value: XX.XXXX±XX.XXXX"""
    if mean is None or std is None:
        return "N/A"
    return f"{mean:.4f}±{std:.4f}"


def generate_table_for_metric(eval_dir, metric_key, method='BOTH'):
    """Generate table for a single metric"""
    metric_name = METRICS_CONFIG[metric_key]['name']

    print(f"\n{'='*80}")
    print(f"Generating {metric_name} table")
    print(f"{'='*80}")

    # Build column names
    columns = ['Class']
    if method in ['DRA', 'BOTH']:
        for backbone in BACKBONES:
            for n_anomaly in NANOMALIES:
                columns.append(f'DRA_{backbone.replace("Point_", "")}_nAnomaly{n_anomaly}')

    if method in ['DevNet', 'BOTH']:
        for backbone in BACKBONES:
            for n_anomaly in NANOMALIES:
                columns.append(f'DevNet_{backbone.replace("Point_", "")}_nAnomaly{n_anomaly}')

    # Collect data
    results = []

    for classname in CLASSNAMES:
        row = {'Class': classname}

        # DRA data
        if method in ['DRA', 'BOTH']:
            for backbone in BACKBONES:
                for n_anomaly in NANOMALIES:
                    col_name = f'DRA_{backbone.replace("Point_", "")}_nAnomaly{n_anomaly}'
                    mean, std = calculate_stats('DRA', classname, backbone, n_anomaly, eval_dir, metric_key)
                    row[col_name] = format_value(mean, std)

        # DevNet data
        if method in ['DevNet', 'BOTH']:
            for backbone in BACKBONES:
                for n_anomaly in NANOMALIES:
                    col_name = f'DevNet_{backbone.replace("Point_", "")}_nAnomaly{n_anomaly}'
                    mean, std = calculate_stats('DevNet', classname, backbone, n_anomaly, eval_dir, metric_key)
                    row[col_name] = format_value(mean, std)

        results.append(row)

    # Compute average row
        print("Computing overall averages...")
    avg_row = {'Class': 'Average'}
    for col in columns:
        if col == 'Class':
            continue

        # Collect all valid values for this column
        values = []
        for row in results:
            val_str = row.get(col, 'N/A')
            if val_str != 'N/A':
                # Parse "XX.XXXX±XX.XXXX"
                parts = val_str.split('±')
                if len(parts) == 2:
                    try:
                        values.append(float(parts[0]))
                    except:
                        pass

        if values:
            mean = np.mean(values)
            std = np.std(values)
            avg_row[col] = format_value(mean, std)
        else:
            avg_row[col] = "N/A"

    results.append(avg_row)

    return pd.DataFrame(results, columns=columns)


def main():
    parser = argparse.ArgumentParser(
        description='Generate evaluation statistics tables for all metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate tables for all metrics
  python generate_table_all_metrics.py

  # Generate only Overall ROC table
  python generate_table_all_metrics.py --metric overall_roc

  # Generate all metrics for DRA method only
  python generate_table_all_metrics.py --method DRA
        """
    )

    parser.add_argument('--metric', type=str, default='ALL',
                        choices=['ALL'] + list(METRICS_CONFIG.keys()),
                        help='Choose metric (default: ALL generates all metrics)')
    parser.add_argument('--method', type=str, default='BOTH',
                        choices=['DRA', 'DevNet', 'BOTH'],
                        help='Choose method (default: BOTH)')
    parser.add_argument('--eval_dir', type=str, default=DEFAULT_EVAL_DIR,
                        help=f'Evaluation results directory (default: {DEFAULT_EVAL_DIR})')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("   Evaluation Result Table Generator - All Metrics")
    print("="*80)
    print(f"Eval directory: {args.eval_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Method: {args.method}")
    print(f"Metric: {args.metric}")

    # Decide which metrics to generate
    if args.metric == 'ALL':
        metrics_to_generate = list(METRICS_CONFIG.keys())
    else:
        metrics_to_generate = [args.metric]

    # Generate table for each metric
    for metric_key in metrics_to_generate:
        df = generate_table_for_metric(args.eval_dir, metric_key, args.method)

        # Save CSV
        filename = f"{args.method}_{metric_key}.csv"
        output_path = os.path.join(args.output_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"✅ Saved: {output_path}")

        # Print preview
        print(f"\n{METRICS_CONFIG[metric_key]['name']} Preview:")
        print(df.to_string(index=False))

    print("\n" + "="*80)
    print("   All done!")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print(f"Files generated: {len(metrics_to_generate)}")


if __name__ == '__main__':
    main()
