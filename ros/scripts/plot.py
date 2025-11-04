#!/usr/bin/env python3
"""
Joint Trajectory Analysis and Visualization
Compares inference predictions with original teleoperation data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("Joint Trajectory Analysis")
print("=" * 60)

# Read CSV files
print("\nLoading data...")
infer_df = pd.read_csv('infer.csv')
org_df = pd.read_csv('org.csv')

print(f"✓ Inference data: {len(infer_df)} samples")
print(f"✓ Original data: {len(org_df)} samples")

# Extract joint positions (columns 14-23 are position0 through position9)
joint_names = [
    'L1XP_JOINT',  # Joint 0
    'L2ZR_JOINT',  # Joint 1
    'L3ZP_JOINT',  # Joint 2
    'A1ZR_JOINT',  # Joint 3
    'A2ZR_JOINT',  # Joint 4
    'A3XR_JOINT',  # Joint 5
    'A4YR_JOINT',  # Joint 6
    'A5ZR_JOINT',  # Joint 7
    'H1ZR_JOINT',  # Joint 8
    'EE_GRIPPER_JOINT'  # Joint 9
]

# Extract position columns
position_cols_infer = [f'field.position{i}' for i in range(10)]
position_cols_org = [f'field.position{i}' for i in range(10)]

# Create time axes (using index as time steps)
time_infer = np.arange(len(infer_df))
time_org = np.arange(len(org_df))

# Calculate statistics
print("\n" + "=" * 60)
print("Performance Statistics")
print("=" * 60)

# Align data for comparison (use shorter length)
min_len = min(len(infer_df), len(org_df))
for i, joint_name in enumerate(joint_names):
    pos_col = f'field.position{i}'
    
    infer_vals = infer_df[pos_col][:min_len].values
    org_vals = org_df[pos_col][:min_len].values
    
    mae = np.mean(np.abs(infer_vals - org_vals))
    rmse = np.sqrt(np.mean((infer_vals - org_vals)**2))
    
    print(f"{joint_name:20s} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 14))
fig.suptitle('Joint Position Comparison: Inference vs Original Teleoperation', 
             fontsize=18, fontweight='bold', y=0.995)

# Plot each joint
for i, joint_name in enumerate(joint_names):
    ax = plt.subplot(4, 3, i+1)
    
    pos_col = f'field.position{i}'
    
    # Plot original (ground truth)
    ax.plot(time_org, org_df[pos_col], 
            label='Original (Teleoperation)', 
            color='blue', 
            linewidth=2, 
            alpha=0.7)
    
    # Plot inference (prediction)
    ax.plot(time_infer, infer_df[pos_col], 
            label='Inference (Model)', 
            color='orange', 
            linewidth=2, 
            alpha=0.7,
            linestyle='--')
    
    ax.set_title(f'{joint_name}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Step', fontsize=9)
    ax.set_ylabel('Position (rad)', fontsize=9)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    if i < min_len:
        infer_vals = infer_df[pos_col][:min_len].values
        org_vals = org_df[pos_col][:min_len].values
        mae = np.mean(np.abs(infer_vals - org_vals))
        ax.text(0.02, 0.98, f'MAE: {mae:.4f}', 
                transform=ax.transAxes, 
                fontsize=8, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Hide empty subplot if there is one
if len(joint_names) < 12:
    for i in range(len(joint_names), 12):
        ax = plt.subplot(4, 3, i+1)
        ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.97)

# Save the plot
output_file = 'all_joints_comparison.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved: {output_file}")

# Create additional visualization: Error over time
fig2, axes = plt.subplots(2, 1, figsize=(16, 10))
fig2.suptitle('Tracking Error Analysis', fontsize=16, fontweight='bold')

# Plot 1: Absolute error for all joints
ax1 = axes[0]
for i, joint_name in enumerate(joint_names):
    pos_col = f'field.position{i}'
    infer_vals = infer_df[pos_col][:min_len].values
    org_vals = org_df[pos_col][:min_len].values
    error = np.abs(infer_vals - org_vals)
    
    ax1.plot(error, label=joint_name, alpha=0.7, linewidth=1.5)

ax1.set_xlabel('Time Step', fontsize=12)
ax1.set_ylabel('Absolute Error (rad)', fontsize=12)
ax1.set_title('Absolute Tracking Error Over Time', fontsize=14)
ax1.legend(loc='upper right', ncol=2, fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Mean absolute error comparison (bar chart)
ax2 = axes[1]
mae_values = []
for i in range(len(joint_names)):
    pos_col = f'field.position{i}'
    infer_vals = infer_df[pos_col][:min_len].values
    org_vals = org_df[pos_col][:min_len].values
    mae = np.mean(np.abs(infer_vals - org_vals))
    mae_values.append(mae)

colors = plt.cm.viridis(np.linspace(0, 1, len(joint_names)))
bars = ax2.bar(range(len(joint_names)), mae_values, color=colors, alpha=0.8)
ax2.set_xlabel('Joint', fontsize=12)
ax2.set_ylabel('Mean Absolute Error (rad)', fontsize=12)
ax2.set_title('Mean Absolute Error per Joint', fontsize=14)
ax2.set_xticks(range(len(joint_names)))
ax2.set_xticklabels([name.replace('_JOINT', '') for name in joint_names], 
                     rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, mae) in enumerate(zip(bars, mae_values)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{mae:.4f}',
             ha='center', va='bottom', fontsize=8)

plt.tight_layout()
error_plot_file = 'tracking_error_analysis.png'
plt.savefig(error_plot_file, dpi=150, bbox_inches='tight')
print(f"✓ Error analysis plot saved: {error_plot_file}")

# Summary statistics
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Total samples compared: {min_len}")
print(f"Average MAE across all joints: {np.mean(mae_values):.4f} rad")
print(f"Maximum MAE: {np.max(mae_values):.4f} rad ({joint_names[np.argmax(mae_values)]})")
print(f"Minimum MAE: {np.min(mae_values):.4f} rad ({joint_names[np.argmin(mae_values)]})")
print("\n✓ Analysis complete!")
print("=" * 60)

# Show the plots (optional - comment out if running headless)
# plt.show()