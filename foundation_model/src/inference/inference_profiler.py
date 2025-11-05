try:
    profile  # Check if profile is already defined
except NameError:
    def profile(func):
        return func  # No-op decorator when not profiling
    
import argparse
import time
import datetime
from pathlib import Path
from rosif import (
    RosIf,
    convert_joint_state_to_numpy,
    convert_ros_image_to_numpy,
    convert_wrench_to_numpy,
)
from measure_timer import MeasureTimer
import numpy as np
from scipy.interpolate import interp1d


def loginfo(msg: str):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}][INFO] {msg}")


def logwarn(msg: str):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}][WARN] {msg}")

@profile
def interpolate_actions_np(actions: np.ndarray, target_length: int = 100, kind: str = 'cubic') -> np.ndarray:
    """
    Interpolates a sequence of actions to a higher resolution using spline interpolation.
    
    Args:
        actions (np.ndarray): Array of shape (original_len, action_dim)
        target_length (int): Number of steps to interpolate to (e.g., 100)
        kind (str): Interpolation type: 'linear', 'quadratic', or 'cubic'
    
    Returns:
        np.ndarray: Interpolated actions of shape (target_length, action_dim)
    """
    original_length, action_dim = actions.shape

    x = np.linspace(0, 1, original_length)
    x_new = np.linspace(0, 1, target_length)

    interpolated = np.stack([
        interp1d(x, actions[:, d], kind=kind)(x_new)
        for d in range(action_dim)
    ], axis=1)

    return interpolated


class InferenceNode:
    def __init__(self, backend, model_config, interpolation_steps, inference_frequency, low_pass_filter, operation_mode, is_dry_run: bool = False, state_smoothing: bool = False, do_blending: bool = True):
        print("Inference configuration:")
        print(f"\tbackend: {backend}")
        print(f"\tinterpolation_steps: {interpolation_steps}")
        print(f"\tinference_frequency: {inference_frequency}")
        print(f"\tlow_pass_filter: {low_pass_filter}")
        print(f"\toperation_mode: {operation_mode}")
        print(f"\tdry_run: {is_dry_run}")
        print(f"\tstate_smoothing: {state_smoothing}")
        print(f"\tdo_blending: {do_blending}")
            
        self.interpolation_steps = interpolation_steps
        self.loop_rate_hz = inference_frequency
        self.low_pass_filter_coeff = low_pass_filter
        self.state_smoothing = state_smoothing
        self.do_blending = do_blending

        self.rosif = RosIf()
        if operation_mode == "teleop_mode":
            self.teleop_mode = True
            self.rosif.text_instruction = "upright the bottle"
        else:
            self.teleop_mode = False
            self.rosif.text_instruction = "Move bottle to a different shelf"

        self.inference_backend = None
        from backend.lerobot import LeRobotInferenceBackend
        self.backend = LeRobotInferenceBackend(model_config)

        self.is_dry_run = is_dry_run
        if self.is_dry_run:
            print("\t/!\\ Dry run mode -> no actions will be sent published by the ROS node! /!\\")

        # TODO: import from config.json
        self.dataset_joint_order = [
            "L1XP_JOINT",
            "L2ZR_JOINT",
            "L3ZP_JOINT",
            "A1ZR_JOINT",
            "A2ZR_JOINT",
            "A3XR_JOINT",
            "A4YR_JOINT",
            "A5ZR_JOINT",
            "H1ZR_JOINT",
            "EE_GRIPPER_JOINT",
        ]

        # Joint limits (rad or meter)
        # tx-parameters/blob/develop/ddc/robot_config/tx_mocap_calibration/default/urdf_offset_lm.xacro
        self.joint_min = np.array([0, -3.665, 0.2, # L1, L2, L3
                                    -2.268, -2.356, -1.623, -1.570, -1.919, # A1-A5
                                    -2.268, 0], # H1, Gripper
                                  dtype=np.float32)
        self.joint_max = np.array([ 1.55,  0.523,  1.53, # L1, L2, L3
                                    2.268, 2.356, 3.141, 1.570, 1.745, # A1-A5
                                    2.268, 1.0], # H1, Gripper
                                  dtype=np.float32)

        # Maximum joint speed in rad/s  (or m/s for linear axes)
        # ref: tx_robot_ros/blob/develop/src/tx_ddc_moveit_config/config/joint_limits.yaml
        speed_scale = 1.0
        self.max_speed = speed_scale * np.array([
            0.5,   # L1XP_JOINT (m/s)
            1.98,  # L2ZR_JOINT
            0.35,  # L3ZP_JOINT (m/s)
            1.6,   # A1ZR_JOINT
            3.14,  # A2ZR_JOINT
            3.14,  # A3XR_JOINT
            3.14,  # A4YR_JOINT
            3.14,  # A5ZR_JOINT
            1.6,   # H1ZR_JOINT
            0.5    # EE_GRIPPER_JOINT (m/s)
        ], dtype=np.float32)

        self.action_queue = np.array([])
        self.timer = MeasureTimer(counts_log=30, label="inference")
        self.previous_action = None

    @profile
    def start_update(self):
        self.rosif.set_periodic_callback(self.loop_rate_hz, self.update)
    
    @profile
    def get_last_observation(self):
        while not self.rosif.is_observation_available(is_teleop=self.teleop_mode):
            self.rosif.sleep_spin(0.001)

        # PnP mode will not wait for head image to be available so there's a risk it's None.
        converted_head_image = None
        if self.rosif.latest_image_head is not None:
            converted_head_image = convert_ros_image_to_numpy(self.rosif.latest_image_head, self.rosif.cv_bridge)
        elif self.teleop_mode:
            print("WARNING: HEAD image is None but TELEOP mode is True.")

        observation = {
            "observation.images.cam_0": convert_ros_image_to_numpy(self.rosif.latest_image_cam0, self.rosif.cv_bridge),
            "observation.images.cam_1": convert_ros_image_to_numpy(self.rosif.latest_image_cam1, self.rosif.cv_bridge),
            "observation.images.usb_cam": convert_ros_image_to_numpy(self.rosif.latest_image_palm, self.rosif.cv_bridge),
            "observation.images.head_cam": converted_head_image,
            "observation.state": convert_joint_state_to_numpy(self.rosif.latest_joints, self.dataset_joint_order),
            "observation.force": convert_wrench_to_numpy(self.rosif.latest_force),
            "task": self.rosif.text_instruction
        }

        return observation
    
    @profile
    def process_new_actions(self, new_actions, current_state, interpolate_from_current_state = False, do_blending = False):
        # do nothing if all of option disabled
        if self.interpolation_steps <= 0 and not interpolate_from_current_state and not do_blending:
            self.action_queue = new_actions
            return

        # new_actions = new_actions[:new_actions.shape[0] // 2]

        # Interpolate from current state to first action position.
        if interpolate_from_current_state:
            current_to_start = np.asarray([current_state, new_actions[0]])
            smoothing_interpolations = interpolate_actions_np(current_to_start, self.interpolation_steps, kind='linear')

        if self.interpolation_steps > 0:
            new_actions = interpolate_actions_np(new_actions, self.interpolation_steps)

        # Interpolate from current state to first action position.
        if interpolate_from_current_state:
            self.action_queue = np.concatenate([smoothing_interpolations, new_actions])
        elif do_blending and len(self.action_queue) > 0:
            blend_len = min(10, len(self.action_queue), len(new_actions))

            if blend_len > 0:
                k = 3.0 / blend_len  # controls exponential rate
                for i in range(blend_len):
                    alpha = 1 - np.exp(-k * (i + 1))  # exponential ramp
                    alpha = np.clip(alpha, 0.0, 1.0)
                    old_action = self.action_queue[-blend_len + i]
                    new_action = new_actions[i]
                    blended = (1 - alpha) * old_action + alpha * new_action
                    self.action_queue[-blend_len + i] = blended
                    new_actions[i] = blended  # optionally update for consistency

            print(f"self.action_queue: {self.action_queue[:blend_len].shape}")
            print(f"new_actions: {new_actions[blend_len:].shape}")
            self.action_queue = np.concatenate([self.action_queue[:blend_len], new_actions[blend_len:]])
        else:
            self.action_queue = new_actions

    # Safety: clip per-frame step size (velocity) to avoid jump faults
    @profile
    def _clip_action_step(self, action: np.ndarray) -> np.ndarray:
        """
        Limit joint speed so that |Δq|/dt ≤ max_speed (rad/s or m/s).
        A single global scale keeps joint coupling intact.
        """
        dq = action - self.previous_action
        # Convert per-second speed limit to per-frame position change limit.
        limit = self.max_speed / self.loop_rate_hz
        scale_vec = np.minimum(1.0, np.abs(limit) / (np.abs(dq) + 1e-6))
        global_scale = np.min(scale_vec)           # same factor for all joints
        if global_scale < 1.0:
            #logwarn(f"[safety] Step size clipped (scale={global_scale:.3f})")
            action = self.previous_action + dq * global_scale
        return action

    @profile
    def update(self, _event):
        # If no leftover actions, build a new batch from current sensor data.
        do_inference = len(self.action_queue) < 10 #(self.loop_rate_hz / 16)

        if do_inference:
            loginfo(f"[update()] Starting inference...")
            self.timer.tick()
            loginfo(f"[update()] \tgetting observation...")
            observation = self.get_last_observation()
            self.last_observation = observation
            loginfo(f"[update()] \tgenerating actions...")
            time.sleep(0.8)
            new_actions = self.backend.generate_actions(observation)
            loginfo(f"[update()] \tprocessing new actions...")
            self.process_new_actions(new_actions, observation["observation.state"], interpolate_from_current_state=self.state_smoothing, do_blending=self.do_blending)
            loginfo(f"[update()] \tDone!")

        action = self.action_queue[0]
        self.action_queue = self.action_queue[1:]

        if self.previous_action is None:
            self.previous_action = self.last_observation["observation.state"]
        
        # Do smoothing.
        action = self.low_pass_filter_coeff * action + (1 - self.low_pass_filter_coeff) * self.previous_action

        if do_inference:
            self.timer.tock()

        # Motion range limitation
        action = np.clip(action, self.joint_min, self.joint_max)

        # Apply per-frame step-size limiter (prevents jump faults)
        action = self._clip_action_step(action)

        # loginfo(f"[update()] Policy action => {action}, queue left-over count: {len(self.action_queue)}")
        self.previous_action = action

        if not self.is_dry_run:
            self.rosif.publish_motor_state(self.dataset_joint_order, action.tolist())
            self.rosif.publish_joint_state(self.dataset_joint_order, action.tolist())


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--backend", type=str, choices=["lerobot"], help="Inference backend.")
    parser.add_argument("-c", "--model_config", type=str, help="Path to the model config file.")
    parser.add_argument("-i", "--interpolation_steps", default=80, type=int, help="Number of actions to interpolate to (-1 to disable). Default: 80.")
    parser.add_argument("-f", "--inference_frequency", default=5, type=int, help="Frequency of the action publishing. Default: 5Hz.")
    parser.add_argument("-l", "--low_pass_filter", default=0.25, type=float, help="Low pass filter (blending) coefficient. Default: 0.25.")
    parser.add_argument("-o", "--operation_mode", type=str, choices=["teleop_mode", "pnp_mode"], help="Operation mode.")
    parser.add_argument("--state-smoothing", action="store_true", help="Add an interpolation to from the current state to the first action (doubles the amount of actions)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode = no actions sent to the robot")
    parser.add_argument("--do-blending", action="store_true", help="Adds blending between two action chunks. Cannot be used with state-smoothing.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        loginfo(f"Initializing InferenceNode (backend={args.backend})...")
        node = InferenceNode(args.backend, args.model_config, args.interpolation_steps, args.inference_frequency, args.low_pass_filter, args.operation_mode, args.dry_run, args.state_smoothing, args.do_blending)
        loginfo("InferenceNode initialized")

        input_str = ""
        while input_str != "y":
            input_str = input("Press 'y' key to start...")

        node.start_update()
        node.rosif.spin()
    except Exception as e:
        print(e)
