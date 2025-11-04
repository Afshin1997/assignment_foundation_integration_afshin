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
import threading
import queue


def loginfo(msg: str):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}][INFO] {msg}")


def logwarn(msg: str):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}][WARN] {msg}")


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


class ThreadedInferenceNode:
    """
    Asynchronous inference node using threading.
    
    Key changes from original:
    1. Inference runs in a separate thread (non-blocking)
    2. Observations are passed via queue
    3. Actions are received via queue
    4. ROS update loop continues without waiting for inference
    """
    
    def __init__(self, backend, model_config, interpolation_steps, inference_frequency, 
                 low_pass_filter, operation_mode, is_dry_run: bool = False, 
                 state_smoothing: bool = False, do_blending: bool = True):
        print("=" * 60)
        print("THREADED Inference configuration:")
        print(f"\tbackend: {backend}")
        print(f"\tinterpolation_steps: {interpolation_steps}")
        print(f"\tinference_frequency: {inference_frequency}")
        print(f"\tlow_pass_filter: {low_pass_filter}")
        print(f"\toperation_mode: {operation_mode}")
        print(f"\tdry_run: {is_dry_run}")
        print(f"\tstate_smoothing: {state_smoothing}")
        print(f"\tdo_blending: {do_blending}")
        print("=" * 60)
            
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

        # Load inference backend
        self.inference_backend = None
        from backend.lerobot import LeRobotInferenceBackend
        self.backend = LeRobotInferenceBackend(model_config)

        self.is_dry_run = is_dry_run
        if self.is_dry_run:
            print("\t/!\\ Dry run mode -> no actions will be sent published by the ROS node! /!\\")

        # Dataset joint order
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
        self.joint_min = np.array([0, -3.665, 0.2,
                                    -2.268, -2.356, -1.623, -1.570, -1.919,
                                    -2.268, 0], dtype=np.float32)
        self.joint_max = np.array([1.55, 0.523, 1.53,
                                    2.268, 2.356, 3.141, 1.570, 1.745,
                                    2.268, 1.0], dtype=np.float32)

        # Maximum joint speed
        speed_scale = 1.0
        self.max_speed = speed_scale * np.array([
            0.5, 1.98, 0.35, 1.6, 3.14, 3.14, 3.14, 3.14, 1.6, 0.5
        ], dtype=np.float32)

        # ============ THREADING COMPONENTS ============
        # Action queue (thread-safe)
        self.action_queue_lock = threading.Lock()
        self.action_queue = np.array([])
        
        # Communication queues
        self.observation_queue = queue.Queue(maxsize=2)  # Observations to process
        self.result_queue = queue.Queue(maxsize=2)       # Processed actions
        
        # Thread control
        self.inference_thread = None
        self.running = False
        
        # Statistics
        self.inference_count = 0
        self.inference_times = []
        self.queue_empty_count = 0
        
        self.timer = MeasureTimer(counts_log=30, label="inference")
        self.previous_action = None
        self.last_observation = None

    def _inference_worker(self):
        """
        Worker thread that continuously processes observations.
        This runs in parallel with the main ROS loop.
        """
        loginfo("[InferenceThread] Starting inference worker thread...")
        
        while self.running:
            try:
                # Wait for observation (blocking, with timeout)
                observation = self.observation_queue.get(timeout=0.1)
                
                # Run inference
                start_time = time.time()
                loginfo(f"[InferenceThread] Processing observation {self.inference_count + 1}...")
                time.sleep(0.8)
                new_actions = self.backend.generate_actions(observation)
                
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)
                self.inference_count += 1
                
                loginfo(f"[InferenceThread] Inference #{self.inference_count} took {inference_time*1000:.2f}ms")
                
                # Put result in queue (non-blocking)
                try:
                    self.result_queue.put((new_actions, observation["observation.state"]), block=False)
                except queue.Full:
                    logwarn("[InferenceThread] Result queue full, dropping old result")
                    # Remove old result and add new one
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.result_queue.put((new_actions, observation["observation.state"]), block=False)
                
            except queue.Empty:
                # No observation to process, continue waiting
                continue
            except Exception as e:
                logwarn(f"[InferenceThread] Error in inference worker: {e}")
                import traceback
                traceback.print_exc()
        
        loginfo("[InferenceThread] Inference worker thread stopped")

    def start_inference_thread(self):
        """Start the inference worker thread"""
        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.inference_thread.start()
        loginfo("[ThreadedNode] Inference thread started")

    def stop_inference_thread(self):
        """Stop the inference worker thread"""
        self.running = False
        if self.inference_thread:
            self.inference_thread.join(timeout=2.0)
            loginfo("[ThreadedNode] Inference thread stopped")

    def start_update(self):
        """Start the ROS update loop and inference thread"""
        # Start inference thread
        self.start_inference_thread()
        
        # Start ROS periodic callback
        self.rosif.set_periodic_callback(self.loop_rate_hz, self.update)
    
    def get_last_observation(self):
        """Get current observation from ROS"""
        while not self.rosif.is_observation_available(is_teleop=self.teleop_mode):
            self.rosif.sleep_spin(0.001)

        converted_head_image = None
        if self.rosif.latest_image_head is not None:
            converted_head_image = convert_ros_image_to_numpy(self.rosif.latest_image_head, self.rosif.cv_bridge)
        elif self.teleop_mode:
            logwarn("HEAD image is None but TELEOP mode is True.")

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
    
    def process_new_actions(self, new_actions, current_state, interpolate_from_current_state=False, do_blending=False):
        """Process and queue new actions (thread-safe)"""
        # Do nothing if all options disabled
        if self.interpolation_steps <= 0 and not interpolate_from_current_state and not do_blending:
            with self.action_queue_lock:
                self.action_queue = new_actions
            return

        # Interpolate from current state to first action position
        if interpolate_from_current_state:
            current_to_start = np.asarray([current_state, new_actions[0]])
            smoothing_interpolations = interpolate_actions_np(current_to_start, self.interpolation_steps, kind='linear')

        if self.interpolation_steps > 0:
            new_actions = interpolate_actions_np(new_actions, self.interpolation_steps)

        # Thread-safe action queue update
        with self.action_queue_lock:
            if interpolate_from_current_state:
                self.action_queue = np.concatenate([smoothing_interpolations, new_actions])
            elif do_blending and len(self.action_queue) > 0:
                blend_len = min(10, len(self.action_queue), len(new_actions))

                if blend_len > 0:
                    k = 3.0 / blend_len
                    for i in range(blend_len):
                        alpha = 1 - np.exp(-k * (i + 1))
                        alpha = np.clip(alpha, 0.0, 1.0)
                        old_action = self.action_queue[-blend_len + i]
                        new_action = new_actions[i]
                        blended = (1 - alpha) * old_action + alpha * new_action
                        self.action_queue[-blend_len + i] = blended
                        new_actions[i] = blended

                self.action_queue = np.concatenate([self.action_queue[:blend_len], new_actions[blend_len:]])
            else:
                self.action_queue = new_actions

    def _clip_action_step(self, action: np.ndarray) -> np.ndarray:
        """Limit joint speed to avoid jump faults"""
        dq = action - self.previous_action
        limit = self.max_speed / self.loop_rate_hz
        scale_vec = np.minimum(1.0, np.abs(limit) / (np.abs(dq) + 1e-6))
        global_scale = np.min(scale_vec)
        if global_scale < 1.0:
            action = self.previous_action + dq * global_scale
        return action

    def update(self, _event):
        """
        Main ROS update loop - NON-BLOCKING!
        This runs at high frequency without waiting for inference.
        """
        # Check if we have new actions from inference thread
        try:
            # Non-blocking check for results
            new_actions, current_state = self.result_queue.get_nowait()
            loginfo(f"[update()] Received new actions from inference thread, processing...")
            self.process_new_actions(
                new_actions, 
                current_state, 
                interpolate_from_current_state=self.state_smoothing, 
                do_blending=self.do_blending
            )
        except queue.Empty:
            # No new results yet, continue with existing queue
            pass

        # Thread-safe check of action queue
        with self.action_queue_lock:
            queue_size = len(self.action_queue)
            do_request_inference = queue_size < 10

        # If queue is low, request new inference (non-blocking)
        if do_request_inference:
            try:
                observation = self.get_last_observation()
                self.last_observation = observation
                
                # Send observation to inference thread (non-blocking)
                try:
                    self.observation_queue.put_nowait(observation)
                    loginfo(f"[update()] Sent observation to inference thread (queue: {queue_size})")
                except queue.Full:
                    logwarn("[update()] Observation queue full, skipping this request")
            except Exception as e:
                logwarn(f"[update()] Error getting observation: {e}")

        # Get next action from queue (thread-safe)
        with self.action_queue_lock:
            if len(self.action_queue) > 0:
                action = self.action_queue[0]
                self.action_queue = self.action_queue[1:]
            else:
                # Queue empty! Use previous action or current state
                self.queue_empty_count += 1
                if self.queue_empty_count % 10 == 1:
                    logwarn(f"[update()] Action queue EMPTY! (count: {self.queue_empty_count})")
                
                if self.previous_action is not None:
                    action = self.previous_action
                elif self.last_observation is not None:
                    action = self.last_observation["observation.state"]
                else:
                    logwarn("[update()] No action available, skipping")
                    return

        # Initialize previous action
        if self.previous_action is None:
            if self.last_observation is not None:
                self.previous_action = self.last_observation["observation.state"]
            else:
                self.previous_action = action

        # Apply smoothing
        action = self.low_pass_filter_coeff * action + (1 - self.low_pass_filter_coeff) * self.previous_action

        # Motion range limitation
        action = np.clip(action, self.joint_min, self.joint_max)

        # Apply per-frame step-size limiter
        action = self._clip_action_step(action)

        self.previous_action = action

        # Publish action to robot
        if not self.is_dry_run:
            self.rosif.publish_motor_state(self.dataset_joint_order, action.tolist())
            self.rosif.publish_joint_state(self.dataset_joint_order, action.tolist())

    def print_statistics(self):
        """Print performance statistics"""
        print("\n" + "=" * 60)
        print("THREADING PERFORMANCE STATISTICS")
        print("=" * 60)
        print(f"Total inferences: {self.inference_count}")
        if self.inference_times:
            print(f"Average inference time: {np.mean(self.inference_times)*1000:.2f}ms")
            print(f"Min inference time: {np.min(self.inference_times)*1000:.2f}ms")
            print(f"Max inference time: {np.max(self.inference_times)*1000:.2f}ms")
        print(f"Queue empty events: {self.queue_empty_count}")
        print(f"Frequency: {self.loop_rate_hz} Hz")
        print("=" * 60 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Threaded Asynchronous Inference Node")

    parser.add_argument("-b", "--backend", type=str, choices=["lerobot"], help="Inference backend.")
    parser.add_argument("-c", "--model_config", type=str, help="Path to the model config file.")
    parser.add_argument("-i", "--interpolation_steps", default=80, type=int, help="Number of actions to interpolate to (-1 to disable). Default: 80.")
    parser.add_argument("-f", "--inference_frequency", default=5, type=int, help="Frequency of the action publishing. Default: 5Hz.")
    parser.add_argument("-l", "--low_pass_filter", default=0.25, type=float, help="Low pass filter (blending) coefficient. Default: 0.25.")
    parser.add_argument("-o", "--operation_mode", type=str, choices=["teleop_mode", "pnp_mode"], help="Operation mode.")
    parser.add_argument("--state-smoothing", action="store_true", help="Add interpolation from current state to first action")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode = no actions sent to the robot")
    parser.add_argument("--do-blending", action="store_true", help="Adds blending between two action chunks")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        loginfo("=" * 60)
        loginfo(f"Initializing THREADED InferenceNode (backend={args.backend})...")
        loginfo("=" * 60)
        
        node = ThreadedInferenceNode(
            args.backend, 
            args.model_config, 
            args.interpolation_steps, 
            args.inference_frequency, 
            args.low_pass_filter, 
            args.operation_mode, 
            args.dry_run, 
            args.state_smoothing, 
            args.do_blending
        )
        
        loginfo("ThreadedInferenceNode initialized")

        input_str = ""
        while input_str != "y":
            input_str = input("Press 'y' key to start...")

        node.start_update()
        node.rosif.spin()
        
    except KeyboardInterrupt:
        loginfo("Shutting down...")
        node.stop_inference_thread()
        node.print_statistics()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        if 'node' in locals():
            node.stop_inference_thread()
            node.print_statistics()