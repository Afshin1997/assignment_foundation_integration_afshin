#!/usr/bin/env python3
"""
Compare All Three Methods: Synchronous, Threading, Server-Client
Plots all 10 joints showing Original vs all three inference methods
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def load_data(base_dir):
    """
    Load CSV files from all three method directories
    
    Args:
        base_dir: Base directory containing inference/, thread/, client_server/ subdirs
    
    Returns:
        Dictionary with dataframes for each method
    """
    data = {}
    
    # Define directories and their labels
    methods = {
        'sync': 'inference_2',
        'thread': 'thread_2', 
        'client_server': 'client_server_2'
    }
    
    print("=" * 60)
    print("Loading data from directories...")
    print("=" * 60)
    
    # Load original data (should be same for all)
    for method_name, dir_name in methods.items():
        org_path = os.path.join(base_dir, dir_name, 'org.csv')
        if os.path.exists(org_path):
            data['original'] = pd.read_csv(org_path)
            print(f"✓ Loaded original data: {len(data['original'])} samples")
            break
    
    if 'original' not in data:
        print("⚠️  Warning: Could not find org.csv in any directory")
        print("   Trying to load from base directory...")
        org_path = os.path.join(base_dir, 'org.csv')
        if os.path.exists(org_path):
            data['original'] = pd.read_csv(org_path)
            print(f"✓ Loaded original data: {len(data['original'])} samples")
    
    # Load inference data for each method
    for method_name, dir_name in methods.items():
        infer_path = os.path.join(base_dir, dir_name, 'infer.csv')
        if os.path.exists(infer_path):
            data[method_name] = pd.read_csv(infer_path)
            print(f"✓ Loaded {method_name}: {len(data[method_name])} samples")
        else:
            print(f"✗ Not found: {infer_path}")
    
    print("=" * 60)
    
    return data


def calculate_metrics(data):
    """Calculate performance metrics for each method"""
    
    metrics = {}
    
    joint_names = [
        'L1XP_JOINT', 'L2ZR_JOINT', 'L3ZP_JOINT',
        'A1ZR_JOINT', 'A2ZR_JOINT', 'A3XR_JOINT',
        'A4YR_JOINT', 'A5ZR_JOINT', 'H1ZR_JOINT',
        'EE_GRIPPER_JOINT'
    ]
    
    # Get minimum length across all methods
    min_len = min([len(df) for df in data.values() if 'original' not in str(df)])
    
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)
    
    for method in ['sync', 'thread', 'client_server']:
        if method not in data:
            continue
            
        method_metrics = {
            'mae_per_joint': [],
            'rmse_per_joint': [],
            'joint_names': joint_names
        }
        
        print(f"\n{method.upper()} vs ORIGINAL:")
        print("-" * 60)
        
        for i, joint_name in enumerate(joint_names):
            pos_col = f'field.position{i}'
            
            original_vals = data['original'][pos_col][:min_len].values
            method_vals = data[method][pos_col][:min_len].values
            
            mae = np.mean(np.abs(method_vals - original_vals))
            rmse = np.sqrt(np.mean((method_vals - original_vals)**2))
            
            method_metrics['mae_per_joint'].append(mae)
            method_metrics['rmse_per_joint'].append(rmse)
            
            print(f"{joint_name:20s} - MAE: {mae:.6f}, RMSE: {rmse:.6f}")
        
        avg_mae = np.mean(method_metrics['mae_per_joint'])
        avg_rmse = np.mean(method_metrics['rmse_per_joint'])
        
        print(f"\nAverage MAE:  {avg_mae:.6f}")
        print(f"Average RMSE: {avg_rmse:.6f}")
        
        metrics[method] = method_metrics
    
    print("=" * 60)
    
    return metrics


def calculate_timing_stats(data):
    """Calculate timing statistics for each method"""
    
    print("\n" + "=" * 60)
    print("TIMING STATISTICS")
    print("=" * 60)
    
    timing_stats = {}
    
    for method in ['sync', 'thread', 'client_server']:
        if method not in data:
            continue
        
        df = data[method]
        
        # Calculate time intervals
        times = df['%time'].values / 1e9  # Convert to seconds
        intervals = np.diff(times)
        
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        frequency = 1 / mean_interval
        
        timing_stats[method] = {
            'mean_interval_ms': mean_interval * 1000,
            'std_interval_ms': std_interval * 1000,
            'frequency_hz': frequency,
            'intervals': intervals
        }
        
        print(f"\n{method.upper()}:")
        print(f"  Samples: {len(df)}")
        print(f"  Mean interval: {mean_interval*1000:.2f}ms")
        print(f"  Std interval: {std_interval*1000:.2f}ms")
        print(f"  Frequency: {frequency:.2f} Hz")
    
    print("=" * 60)
    
    return timing_stats


def plot_joint_comparisons(data, output_file='all_methods_comparison.png'):
    """
    Create a comprehensive plot comparing all methods for all joints
    """
    
    joint_names = [
        'L1XP_JOINT', 'L2ZR_JOINT', 'L3ZP_JOINT',
        'A1ZR_JOINT', 'A2ZR_JOINT', 'A3XR_JOINT',
        'A4YR_JOINT', 'A5ZR_JOINT', 'H1ZR_JOINT',
        'EE_GRIPPER_JOINT'
    ]
    
    # Create figure with subplots (5 rows x 2 columns)
    fig, axes = plt.subplots(5, 2, figsize=(20, 20))
    fig.suptitle('Joint Position Comparison: Original vs All Methods', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Color scheme
    colors = {
        'original': '#2E86AB',      # Blue
        'sync': '#E63946',          # Red
        'thread': '#06A77D',        # Green
        'client_server': '#F77F00'  # Orange
    }
    
    labels = {
        'original': 'Original (Ground Truth)',
        'sync': 'Synchronous',
        'thread': 'Threading',
        'client_server': 'Server-Client'
    }
    
    # Get minimum length for fair comparison
    min_len = min([len(df) for df in data.values()])
    
    # Plot each joint
    for i, joint_name in enumerate(joint_names):
        ax = axes[i]
        pos_col = f'field.position{i}'
        
        # Plot original (ground truth)
        if 'original' in data:
            ax.plot(data['original'][pos_col][:min_len], 
                   label=labels['original'],
                   color=colors['original'],
                   linewidth=2.5,
                   alpha=0.8)
        
        # Plot each method
        for method in ['sync', 'thread', 'client_server']:
            if method in data:
                ax.plot(data[method][pos_col][:min_len],
                       label=labels[method],
                       color=colors[method],
                       linewidth=1.5,
                       alpha=0.7,
                       linestyle='--' if method != 'sync' else '-.')
        
        ax.set_title(f'{joint_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Sample', fontsize=10)
        ax.set_ylabel('Position (rad)', fontsize=10)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add MAE text for each method
        text_y = 0.98
        for method in ['sync', 'thread', 'client_server']:
            if method in data:
                original_vals = data['original'][pos_col][:min_len].values
                method_vals = data[method][pos_col][:min_len].values
                mae = np.mean(np.abs(method_vals - original_vals))
                
                ax.text(0.02, text_y, f'{labels[method]}: MAE={mae:.4f}',
                       transform=ax.transAxes,
                       fontsize=7,
                       verticalalignment='top',
                       color=colors[method],
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                text_y -= 0.08
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Joint comparison plot saved: {output_file}")


def plot_error_comparison(data, metrics, output_file='error_comparison.png'):
    """
    Plot error comparison across methods
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Error Analysis: All Methods Comparison', fontsize=16, fontweight='bold')
    
    colors = {
        'sync': '#E63946',
        'thread': '#06A77D',
        'client_server': '#F77F00'
    }
    
    # Plot 1: MAE per joint
    ax = axes[0, 0]
    x = np.arange(10)
    width = 0.25
    
    for idx, method in enumerate(['sync', 'thread', 'client_server']):
        if method in metrics:
            offset = (idx - 1) * width
            ax.bar(x + offset, metrics[method]['mae_per_joint'], 
                  width, label=method.replace('_', '-').title(),
                  color=colors[method], alpha=0.8)
    
    ax.set_xlabel('Joint Index', fontsize=11)
    ax.set_ylabel('Mean Absolute Error (rad)', fontsize=11)
    ax.set_title('MAE per Joint', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'J{i}' for i in range(10)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Average MAE comparison
    ax = axes[0, 1]
    methods = []
    avg_maes = []
    
    for method in ['sync', 'thread', 'client_server']:
        if method in metrics:
            methods.append(method.replace('_', '-').title())
            avg_maes.append(np.mean(metrics[method]['mae_per_joint']))
    
    bars = ax.bar(methods, avg_maes, color=[colors[m] for m in ['sync', 'thread', 'client_server'] if m in metrics], alpha=0.8)
    ax.set_ylabel('Average MAE (rad)', fontsize=11)
    ax.set_title('Average MAE Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.6f}',
               ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Error distribution (box plot)
    ax = axes[1, 0]
    
    min_len = min([len(df) for df in data.values()])
    error_data = []
    labels_list = []
    
    for method in ['sync', 'thread', 'client_server']:
        if method in data:
            # Calculate error for all joints combined
            errors = []
            for i in range(10):
                pos_col = f'field.position{i}'
                original_vals = data['original'][pos_col][:min_len].values
                method_vals = data[method][pos_col][:min_len].values
                errors.extend(np.abs(method_vals - original_vals))
            
            error_data.append(errors)
            labels_list.append(method.replace('_', '-').title())
    
    bp = ax.boxplot(error_data, labels=labels_list, patch_artist=True)
    
    for patch, method in zip(bp['boxes'], ['sync', 'thread', 'client_server']):
        if method in colors:
            patch.set_facecolor(colors[method])
            patch.set_alpha(0.8)
    
    ax.set_ylabel('Absolute Error (rad)', fontsize=11)
    ax.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: RMSE comparison
    ax = axes[1, 1]
    
    for idx, method in enumerate(['sync', 'thread', 'client_server']):
        if method in metrics:
            offset = (idx - 1) * width
            ax.bar(x + offset, metrics[method]['rmse_per_joint'],
                  width, label=method.replace('_', '-').title(),
                  color=colors[method], alpha=0.8)
    
    ax.set_xlabel('Joint Index', fontsize=11)
    ax.set_ylabel('Root Mean Square Error (rad)', fontsize=11)
    ax.set_title('RMSE per Joint', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'J{i}' for i in range(10)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Error comparison plot saved: {output_file}")


def plot_timing_comparison(timing_stats, output_file='timing_comparison.png'):
    """
    Plot timing comparison across methods
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Timing Performance: All Methods Comparison', fontsize=16, fontweight='bold')
    
    colors = {
        'sync': '#E63946',
        'thread': '#06A77D',
        'client_server': '#F77F00'
    }
    
    # Plot 1: Cycle time distribution
    ax = axes[0, 0]
    for method in ['sync', 'thread', 'client_server']:
        if method in timing_stats:
            intervals = timing_stats[method]['intervals'] * 1000
            ax.hist(intervals, bins=50, alpha=0.6, 
                   label=method.replace('_', '-').title(),
                   color=colors[method])
    
    ax.set_xlabel('Cycle Time (ms)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Cycle Time Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Cycle time over time
    ax = axes[0, 1]
    for method in ['sync', 'thread', 'client_server']:
        if method in timing_stats:
            intervals = timing_stats[method]['intervals'] * 1000
            ax.plot(intervals, alpha=0.7, label=method.replace('_', '-').title(),
                   color=colors[method], linewidth=1.5)
    
    ax.axhline(y=20, color='r', linestyle='--', linewidth=2, label='Target (50Hz = 20ms)')
    ax.set_xlabel('Sample', fontsize=11)
    ax.set_ylabel('Cycle Time (ms)', fontsize=11)
    ax.set_title('Cycle Time Over Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Average metrics comparison
    ax = axes[1, 0]
    
    methods = []
    mean_times = []
    std_times = []
    frequencies = []
    
    for method in ['sync', 'thread', 'client_server']:
        if method in timing_stats:
            methods.append(method.replace('_', '-').title())
            mean_times.append(timing_stats[method]['mean_interval_ms'])
            std_times.append(timing_stats[method]['std_interval_ms'])
            frequencies.append(timing_stats[method]['frequency_hz'])
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, mean_times, width, label='Mean Cycle Time (ms)',
                   color=[colors[m] for m in ['sync', 'thread', 'client_server'] if m in timing_stats],
                   alpha=0.8)
    
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, frequencies, width, label='Frequency (Hz)',
                    color='gray', alpha=0.5)
    
    ax.set_xlabel('Method', fontsize=11)
    ax.set_ylabel('Cycle Time (ms)', fontsize=11)
    ax2.set_ylabel('Frequency (Hz)', fontsize=11)
    ax.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}ms',
               ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}Hz',
                ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Speedup comparison
    ax = axes[1, 1]
    
    if 'sync' in timing_stats:
        sync_time = timing_stats['sync']['mean_interval_ms']
        speedups = []
        method_names = []
        
        for method in ['thread', 'client_server']:
            if method in timing_stats:
                speedup = sync_time / timing_stats[method]['mean_interval_ms']
                speedups.append(speedup)
                method_names.append(method.replace('_', '-').title())
        
        bars = ax.bar(method_names, speedups,
                     color=[colors[m] for m in ['thread', 'client_server'] if m in timing_stats],
                     alpha=0.8)
        
        ax.axhline(y=1, color='r', linestyle='--', linewidth=2, label='Baseline (Sync)')
        ax.set_ylabel('Speedup (x times faster)', fontsize=11)
        ax.set_title('Speedup vs Synchronous Baseline', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}x',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Timing comparison plot saved: {output_file}")


def generate_report(data, metrics, timing_stats, output_file='comparison_report.txt'):
    """Generate a text report with all statistics"""
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE COMPARISON REPORT\n")
        f.write("Synchronous vs Threading vs Server-Client\n")
        f.write("=" * 80 + "\n\n")
        
        # Sample counts
        f.write("SAMPLE COUNTS:\n")
        f.write("-" * 80 + "\n")
        for method in ['original', 'sync', 'thread', 'client_server']:
            if method in data:
                f.write(f"{method.upper():20s}: {len(data[method])} samples\n")
        f.write("\n")
        
        # Timing statistics
        f.write("TIMING PERFORMANCE:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Method':<20s} {'Mean (ms)':<12s} {'Std (ms)':<12s} {'Freq (Hz)':<12s}\n")
        f.write("-" * 80 + "\n")
        
        for method in ['sync', 'thread', 'client_server']:
            if method in timing_stats:
                stats = timing_stats[method]
                f.write(f"{method:<20s} {stats['mean_interval_ms']:>10.2f}   "
                       f"{stats['std_interval_ms']:>10.2f}   "
                       f"{stats['frequency_hz']:>10.2f}\n")
        f.write("\n")
        
        # Speedup
        if 'sync' in timing_stats:
            f.write("SPEEDUP vs SYNCHRONOUS:\n")
            f.write("-" * 80 + "\n")
            sync_time = timing_stats['sync']['mean_interval_ms']
            
            for method in ['thread', 'client_server']:
                if method in timing_stats:
                    method_time = timing_stats[method]['mean_interval_ms']
                    speedup = sync_time / method_time
                    improvement = ((sync_time - method_time) / sync_time) * 100
                    f.write(f"{method:<20s}: {speedup:.2f}x faster ({improvement:.1f}% improvement)\n")
            f.write("\n")
        
        # Tracking accuracy
        f.write("TRACKING ACCURACY (Average across all joints):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Method':<20s} {'Avg MAE':<15s} {'Avg RMSE':<15s}\n")
        f.write("-" * 80 + "\n")
        
        for method in ['sync', 'thread', 'client_server']:
            if method in metrics:
                avg_mae = np.mean(metrics[method]['mae_per_joint'])
                avg_rmse = np.mean(metrics[method]['rmse_per_joint'])
                f.write(f"{method:<20s} {avg_mae:>13.6f}   {avg_rmse:>13.6f}\n")
        f.write("\n")
        
        # Best/worst joints
        f.write("BEST & WORST TRACKING JOINTS:\n")
        f.write("-" * 80 + "\n")
        
        joint_names = [
            'L1XP', 'L2ZR', 'L3ZP', 'A1ZR', 'A2ZR',
            'A3XR', 'A4YR', 'A5ZR', 'H1ZR', 'EE_GRIPPER'
        ]
        
        for method in ['sync', 'thread', 'client_server']:
            if method in metrics:
                maes = metrics[method]['mae_per_joint']
                best_idx = np.argmin(maes)
                worst_idx = np.argmax(maes)
                
                f.write(f"\n{method.upper()}:\n")
                f.write(f"  Best:  {joint_names[best_idx]:<15s} MAE = {maes[best_idx]:.6f}\n")
                f.write(f"  Worst: {joint_names[worst_idx]:<15s} MAE = {maes[worst_idx]:.6f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("CONCLUSION:\n")
        f.write("=" * 80 + "\n")
        
        # Determine best method
        if len(timing_stats) >= 2:
            fastest = min(timing_stats.keys(), 
                         key=lambda m: timing_stats[m]['mean_interval_ms'])
            most_accurate = min(metrics.keys(),
                               key=lambda m: np.mean(metrics[m]['mae_per_joint']))
            
            f.write(f"Fastest method: {fastest.upper()}\n")
            f.write(f"Most accurate method: {most_accurate.upper()}\n\n")
            
            if fastest == most_accurate:
                f.write(f"✅ {fastest.upper()} achieves both best speed AND best accuracy!\n")
            else:
                f.write(f"⚖️  Trade-off: {fastest.upper()} is fastest, but {most_accurate.upper()} is most accurate.\n")
    
    print(f"✓ Comparison report saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare all three inference methods")
    parser.add_argument('--base_dir', type=str, default='.',
                       help='Base directory containing inference/, thread/, client_server/ subdirectories')
    parser.add_argument('--output_prefix', type=str, default='comparison',
                       help='Prefix for output files')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("COMPREHENSIVE METHOD COMPARISON")
    print("=" * 60)
    
    # Load data
    data = load_data(args.base_dir)
    
    if len(data) < 2:
        print("\n❌ Error: Need at least original + one method's data")
        print(f"   Found: {list(data.keys())}")
        return
    
    # Calculate metrics
    metrics = calculate_metrics(data)
    
    # Calculate timing stats
    timing_stats = calculate_timing_stats(data)
    
    # Generate plots
    print("\nGenerating comparison plots...")
    plot_joint_comparisons(data, f'{args.output_prefix}_joints.png')
    plot_error_comparison(data, metrics, f'{args.output_prefix}_errors.png')
    plot_timing_comparison(timing_stats, f'{args.output_prefix}_timing.png')
    
    # Generate report
    generate_report(data, metrics, timing_stats, f'{args.output_prefix}_report.txt')
    
    print("\n" + "=" * 60)
    print("✅ COMPARISON COMPLETE!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  • {args.output_prefix}_joints.png - Joint tracking comparison")
    print(f"  • {args.output_prefix}_errors.png - Error analysis")
    print(f"  • {args.output_prefix}_timing.png - Timing performance")
    print(f"  • {args.output_prefix}_report.txt - Detailed text report")
    print("\n")


if __name__ == "__main__":
    main()