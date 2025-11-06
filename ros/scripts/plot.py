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

    data = {}
    
    methods = {
        'sync': 'inference',
        'thread': 'thread', 
        'client_server': 'client_server'
    }
    
    
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
    
    for method_name, dir_name in methods.items():
        infer_path = os.path.join(base_dir, dir_name, 'infer.csv')
        if os.path.exists(infer_path):
            data[method_name] = pd.read_csv(infer_path)
            print(f"✓ Loaded {method_name}: {len(data[method_name])} samples")
        else:
            print(f"✗ Not found: {infer_path}")
    
    
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


def plot_joint_comparisons(data, output_file='all_methods_comparison.svg'):
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
        'thread': 'Synchronous_5',
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
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Joint comparison plot saved: {output_file}")


def plot_error_comparison(data, metrics, output_file='error_comparison.svg'):
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
    
    # Plot 3: RMSE comparison
    ax = axes[1, 0]
    
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


def main():
    parser = argparse.ArgumentParser(description="Compare all three inference methods")
    parser.add_argument('--base_dir', type=str, default='.',
                       help='Base directory containing inference/, thread/, client_server/ subdirectories')
    parser.add_argument('--output_prefix', type=str, default='comparison',
                       help='Prefix for output files')
    
    args = parser.parse_args()
    
    # Load data
    data = load_data(args.base_dir)
    
    # Calculate metrics
    metrics = calculate_metrics(data)
    
    # Generate plots
    print("\nGenerating comparison plots...")
    plot_joint_comparisons(data, f'comparison_joints.svg')
    plot_error_comparison(data, metrics, f'comparison_errors.svg')
    

if __name__ == "__main__":
    main()