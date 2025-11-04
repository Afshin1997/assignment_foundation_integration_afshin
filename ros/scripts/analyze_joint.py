import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
from typing import Tuple, List


# Configuration constants
DEFAULT_TARGET_FILE = '/org.csv'
DEFAULT_OUTPUT_FILE = '/infer.csv'
DEFAULT_DATASET_PATH = "/workspace/scripts"
JOINT_START_IDX = 14
JOINT_END_IDX = 24
NANO_SEC = 1.0e9
PLOT_COLS = 2
TIME_LIMIT = 52
OUTPUT_IMAGE = "all_joints_comparison.png"


def load_data(dataset_path: str, target_file: str, output_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load target and output data from CSV files."""
    target_path = os.path.join(dataset_path, target_file.lstrip('/'))
    output_path = os.path.join(dataset_path, output_file.lstrip('/'))
    
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Target file not found: {target_path}")
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Output file not found: {output_path}")
    
    target_data = np.genfromtxt(target_path, delimiter=',', skip_header=1)
    output_data = np.genfromtxt(output_path, delimiter=',', skip_header=1)
    
    return target_data, output_data


def get_normalized_time(data: np.ndarray) -> np.ndarray:
    """Convert timestamp data to normalized time in seconds."""
    return (data[:, 0] - data[0, 0]) / NANO_SEC


def calculate_mse_analysis(target_data: np.ndarray, output_data: np.ndarray, 
                          joint_start: int, joint_end: int) -> Tuple[List[float], float]:
    """Calculate MSE for each joint and overall MSE."""
    target_times = get_normalized_time(target_data)
    output_times = get_normalized_time(output_data)
    
    mse_per_joint = []
    
    for joint_idx in range(joint_start, joint_end):
        interp_target = np.interp(output_times, target_times, target_data[:, joint_idx])
        output_joint = output_data[:, joint_idx]
        mse = np.mean((interp_target - output_joint) ** 2)
        mse_per_joint.append(mse)
        print(f"Joint{joint_idx - joint_start}: MSE = {mse:.6f}")
    
    overall_mse = np.mean(mse_per_joint)
    print(f"Overall MSE (joints {joint_start}-{joint_end-1}): {overall_mse:.6f}")
    
    return mse_per_joint, overall_mse


def create_joint_comparison_plot(target_data: np.ndarray, output_data: np.ndarray,
                               joint_start: int, joint_end: int, output_file: str):
    """Create comparison plots for all joints."""
    num_joints = joint_end - joint_start
    nrows = (num_joints + PLOT_COLS - 1) // PLOT_COLS
    
    fig, axes = plt.subplots(nrows, PLOT_COLS, figsize=(12, 3 * nrows), sharex=True)
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    target_times = get_normalized_time(target_data)
    output_times = get_normalized_time(output_data)
    
    for idx, joint_idx in enumerate(range(joint_start, joint_end)):
        joint_name = f"Joint{joint_idx - joint_start}"
        axes[idx].plot(target_times, target_data[:, joint_idx], 
                      label="Ground Truth", color="blue", linewidth=1)
        axes[idx].plot(output_times, output_data[:, joint_idx], 
                      label="Predicted", color="red", linestyle="--", linewidth=1)
        axes[idx].legend()
        axes[idx].set_title(joint_name)
        axes[idx].set_ylabel("Angle (rad)")
        axes[idx].set_xlim(0, TIME_LIMIT)
    
    # Hide unused subplots
    for j in range(num_joints, len(axes)):
        fig.delaxes(axes[j])
    
    # Set x-axis label on the last row
    for ax in axes[num_joints - PLOT_COLS:num_joints]:
        if ax.get_visible():
            ax.set_xlabel("Time (s)")
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze joint trajectory comparison")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH,
                       help="Path to dataset directory")
    parser.add_argument("--target-file", default=DEFAULT_TARGET_FILE,
                       help="Target CSV file name")
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE,
                       help="Output CSV file name")
    parser.add_argument("--joint-start", type=int, default=JOINT_START_IDX,
                       help="Starting joint index")
    parser.add_argument("--joint-end", type=int, default=JOINT_END_IDX,
                       help="Ending joint index")
    parser.add_argument("--plot-output", default=OUTPUT_IMAGE,
                       help="Output plot file name")
    
    return parser.parse_args()


def main():
    """Main function to run the joint analysis."""
    args = parse_arguments()
    
    try:
        # Load data
        target_data, output_data = load_data(args.dataset_path, args.target_file, args.output_file)
        
        # Calculate MSE analysis
        mse_per_joint, overall_mse = calculate_mse_analysis(
            target_data, output_data, args.joint_start, args.joint_end
        )
        
        # Create comparison plots
        create_joint_comparison_plot(
            target_data, output_data, args.joint_start, args.joint_end, args.plot_output
        )
        
        print(f"Analysis complete. Plot saved as {args.plot_output}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())