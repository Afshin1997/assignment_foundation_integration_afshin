#!/usr/bin/env python3
"""
Inference Client - Server-Client Architecture
ROS node that connects to inference server and manages action queue.

Based on LeRobot's async inference approach.
Adapted for tx-pizero with ROS integration.
"""

import argparse
import time
import datetime
import socket
import pickle
import threading
import queue
import numpy as np
from scipy.interpolate import interp1d

from rosif import (
    RosIf,
    convert_joint_state_to_numpy,
    convert_ros_image_to_numpy,
    convert_wrench_to_numpy,
)
from measure_timer import MeasureTimer


def loginfo(msg: str):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}][INFO][CLIENT] {msg}")


def logwarn(msg: str):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}][WARN][CLIENT] {msg}")


def interpolate_actions_np(actions: np.ndarray, target_length: int = 100, kind: str = 'cubic') -> np.ndarray:
    """Interpolate actions to higher resolution"""
    original_length, action_dim = actions.shape
    x = np.linspace(0, 1, original_length)
    x_new = np.linspace(0, 1, target_length)
    interpolated = np.stack([
        interp1d(x, actions[:, d], kind=kind)(x_new)
        for d in range(action_dim)
    ], axis=1)
    return interpolated


class InferenceClient:
    """
    ROS Client that communicates with Inference Server.
    Manages action queue and publishes to robot.
    
    Key features from LeRobot:
    - Action queue with threshold-based refill
    - Asynchronous observation sending
    - Non-blocking ROS loop
    """
    
    def __init__(
        self,
        server_address='127.0.0.1',
        server_port=50051,
        interpolation_steps=80,
        inference_frequency=50,
        low_pass_filter=0.25,
        operation_mode='teleop_mode',
        is_dry_run=False,
        state_smoothing=False,
        do_blending=True,
        actions_per_chunk=50,
        chunk_size_threshold=0.5
    ):
        """
        Initialize inference client
        
        Args:
            server_address: Inference server IP
            server_port: Inference server port
            interpolation_steps: Number of interpolation steps
            inference_frequency: Control loop frequency (Hz)
            low_pass_filter: Low-pass filter coefficient
            operation_mode: 'teleop_mode' or 'pnp_mode'
            is_dry_run: If True, don't publish actions
            state_smoothing: Add interpolation from current state
            do_blending: Blend action chunks
            actions_per_chunk: Expected actions per inference
            chunk_size_threshold: Queue refill threshold (0-1)
        """
        loginfo("=" * 60)
        loginfo("Initializing Inference Client (Server-Client Architecture)")
        loginfo("=" * 60)
        loginfo(f"Server: {server_address}:{server_port}")
        loginfo(f"Interpolation steps: {interpolation_steps}")
        loginfo(f"Frequency: {inference_frequency} Hz")
        loginfo(f"Low-pass filter: {low_pass_filter}")
        loginfo(f"Operation mode: {operation_mode}")
        loginfo(f"Actions per chunk: {actions_per_chunk}")
        loginfo(f"Chunk threshold: {chunk_size_threshold}")
        loginfo(f"Dry run: {is_dry_run}")
        loginfo("=" * 60)
        
        # ROS interface
        self.rosif = RosIf()
        if operation_mode == "teleop_mode":
            self.teleop_mode = True
            self.rosif.text_instruction = "upright the bottle"
        else:
            self.teleop_mode = False
            self.rosif.text_instruction = "Move bottle to a different shelf"
        
        # Parameters
        self.interpolation_steps = interpolation_steps
        self.loop_rate_hz = inference_frequency
        self.low_pass_filter_coeff = low_pass_filter
        self.state_smoothing = state_smoothing
        self.do_blending = do_blending
        self.is_dry_run = is_dry_run
        
        # LeRobot-style action queue parameters
        self.actions_per_chunk = actions_per_chunk
        self.chunk_size_threshold = chunk_size_threshold
        
        # Server connection
        self.server_address = server_address
        self.server_port = server_port
        self.socket = None
        self._connect_to_server()
        
        # Joint configuration
        self.dataset_joint_order = [
            "L1XP_JOINT", "L2ZR_JOINT", "L3ZP_JOINT",
            "A1ZR_JOINT", "A2ZR_JOINT", "A3XR_JOINT",
            "A4YR_JOINT", "A5ZR_JOINT", "H1ZR_JOINT",
            "EE_GRIPPER_JOINT",
        ]
        
        # Joint limits
        self.joint_min = np.array([0, -3.665, 0.2, -2.268, -2.356, -1.623, -1.570, -1.919, -2.268, 0], dtype=np.float32)
        self.joint_max = np.array([1.55, 0.523, 1.53, 2.268, 2.356, 3.141, 1.570, 1.745, 2.268, 1.0], dtype=np.float32)
        
        speed_scale = 1.0
        self.max_speed = speed_scale * np.array([0.5, 1.98, 0.35, 1.6, 3.14, 3.14, 3.14, 3.14, 1.6, 0.5], dtype=np.float32)
        
        # Action queue (thread-safe)
        self.action_queue_lock = threading.Lock()
        self.action_queue = np.array([])
        
        # Communication queues
        self.observation_to_send = queue.Queue(maxsize=2)
        self.result_from_server = queue.Queue(maxsize=2)
        
        # Network thread
        self.network_thread = None
        self.running = False
        
        # State
        self.previous_action = None
        self.last_observation = None
        
        # Statistics
        self.inference_count = 0
        self.queue_empty_count = 0
        self.request_count = 0
        self.network_errors = 0
        
        self.timer = MeasureTimer(counts_log=30, label="client")
        
        if self.is_dry_run:
            loginfo("/!\\ Dry run mode -> no actions published /!\\")
        
        loginfo("Client initialized successfully")
        loginfo("=" * 60)
    
    def _connect_to_server(self):
        """Establish connection to inference server"""
        loginfo(f"Connecting to server at {self.server_address}:{self.server_port}...")
        
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.server_address, self.server_port))
                loginfo(f"✓ Connected to server!")
                return
            except ConnectionRefusedError:
                logwarn(f"Connection attempt {attempt + 1}/{max_retries} failed")
                if attempt < max_retries - 1:
                    loginfo(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise Exception(f"Could not connect to server after {max_retries} attempts")
    
    def _send_message(self, data):
        """Send message to server with size prefix"""
        serialized = pickle.dumps(data)
        size = len(serialized)
        self.socket.sendall(size.to_bytes(4, 'big'))
        self.socket.sendall(serialized)
    
    def _receive_message(self):
        """Receive message from server with size prefix"""
        size_data = self.socket.recv(4)
        if not size_data or len(size_data) < 4:
            return None
        
        msg_size = int.from_bytes(size_data, 'big')
        
        data = b''
        while len(data) < msg_size:
            packet = self.socket.recv(min(msg_size - len(data), 4096))
            if not packet:
                return None
            data += packet
        
        return pickle.loads(data)
    
    def _network_worker(self):
        """
        Worker thread that handles server communication.
        Sends observations and receives actions asynchronously.
        """
        loginfo("[NetworkThread] Starting network worker...")
        
        while self.running:
            try:
                # Wait for observation to send
                observation = self.observation_to_send.get(timeout=0.1)
                
                self.request_count += 1
                loginfo(f"[NetworkThread] Sending observation #{self.request_count}...")
                
                # Send observation to server
                start_time = time.time()
                self._send_message(observation)
                
                # Receive response
                response = self._receive_message()
                
                if response is None:
                    logwarn("[NetworkThread] Received None response from server")
                    self.network_errors += 1
                    continue
                
                roundtrip_time = time.time() - start_time
                
                if 'error' in response:
                    logwarn(f"[NetworkThread] Server error: {response['error']}")
                    self.network_errors += 1
                    continue
                
                actions = response['actions']
                inference_time = response.get('inference_time', 0)
                
                self.inference_count += 1
                
                loginfo(f"[NetworkThread] Received actions #{self.inference_count} "
                       f"(inference: {inference_time*1000:.2f}ms, "
                       f"roundtrip: {roundtrip_time*1000:.2f}ms)")
                
                # Put result in queue
                try:
                    self.result_from_server.put_nowait((actions, observation["observation.state"]))
                except queue.Full:
                    logwarn("[NetworkThread] Result queue full, dropping old result")
                    try:
                        self.result_from_server.get_nowait()
                    except queue.Empty:
                        pass
                    self.result_from_server.put_nowait((actions, observation["observation.state"]))
                
            except queue.Empty:
                continue
            except Exception as e:
                logwarn(f"[NetworkThread] Error: {e}")
                self.network_errors += 1
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        
        loginfo("[NetworkThread] Network worker stopped")
    
    def start_network_thread(self):
        """Start the network worker thread"""
        self.running = True
        self.network_thread = threading.Thread(target=self._network_worker, daemon=True)
        self.network_thread.start()
        loginfo("Network thread started")
    
    def stop_network_thread(self):
        """Stop the network worker thread"""
        self.running = False
        if self.network_thread:
            self.network_thread.join(timeout=2.0)
        if self.socket:
            self.socket.close()
        loginfo("Network thread stopped")
    
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
    
    def process_new_actions(self, new_actions, current_state):
        """Process and queue new actions (thread-safe)"""
        if self.interpolation_steps <= 0 and not self.state_smoothing and not self.do_blending:
            with self.action_queue_lock:
                self.action_queue = new_actions
            return
        
        if self.state_smoothing:
            current_to_start = np.asarray([current_state, new_actions[0]])
            smoothing_interpolations = interpolate_actions_np(current_to_start, self.interpolation_steps, kind='linear')
        
        if self.interpolation_steps > 0:
            new_actions = interpolate_actions_np(new_actions, self.interpolation_steps)
        
        with self.action_queue_lock:
            if self.state_smoothing:
                self.action_queue = np.concatenate([smoothing_interpolations, new_actions])
            elif self.do_blending and len(self.action_queue) > 0:
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
    
    def should_request_new_chunk(self):
        """
        Check if we should request new actions from server.
        Based on LeRobot's threshold logic.
        """
        with self.action_queue_lock:
            queue_size = len(self.action_queue)
        
        threshold_size = self.actions_per_chunk * self.chunk_size_threshold
        return queue_size <= threshold_size
    
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
        Manages action queue and publishes to robot.
        """
        # Check for new results from server
        try:
            new_actions, current_state = self.result_from_server.get_nowait()
            loginfo(f"[update()] Processing new actions from server (queue: {len(self.action_queue)})")
            self.process_new_actions(new_actions, current_state)
        except queue.Empty:
            pass
        
        # Check if we need to request new actions (LeRobot's threshold logic)
        if self.should_request_new_chunk():
            try:
                observation = self.get_last_observation()
                self.last_observation = observation
                
                # Send to network thread (non-blocking)
                try:
                    self.observation_to_send.put_nowait(observation)
                    with self.action_queue_lock:
                        queue_size = len(self.action_queue)
                    loginfo(f"[update()] Requesting new chunk (queue: {queue_size}/{self.actions_per_chunk})")
                except queue.Full:
                    logwarn("[update()] Observation queue full")
            except Exception as e:
                logwarn(f"[update()] Error getting observation: {e}")
        
        # Get next action from queue (thread-safe)
        with self.action_queue_lock:
            if len(self.action_queue) > 0:
                action = self.action_queue[0]
                self.action_queue = self.action_queue[1:]
            else:
                self.queue_empty_count += 1
                if self.queue_empty_count % 10 == 1:
                    logwarn(f"[update()] Action queue EMPTY! (count: {self.queue_empty_count})")
                
                if self.previous_action is not None:
                    action = self.previous_action
                elif self.last_observation is not None:
                    action = self.last_observation["observation.state"]
                else:
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
        
        # Publish action
        if not self.is_dry_run:
            self.rosif.publish_motor_state(self.dataset_joint_order, action.tolist())
            self.rosif.publish_joint_state(self.dataset_joint_order, action.tolist())
    
    def start_update(self):
        """Start the client (network thread + ROS loop)"""
        self.start_network_thread()
        self.rosif.set_periodic_callback(self.loop_rate_hz, self.update)
    
    def print_statistics(self):
        """Print client statistics"""
        loginfo("\n" + "=" * 60)
        loginfo("CLIENT STATISTICS")
        loginfo("=" * 60)
        loginfo(f"Requests sent: {self.request_count}")
        loginfo(f"Responses received: {self.inference_count}")
        loginfo(f"Queue empty events: {self.queue_empty_count}")
        loginfo(f"Network errors: {self.network_errors}")
        loginfo(f"Control frequency: {self.loop_rate_hz} Hz")
        loginfo("=" * 60 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference Client for Server-Client Architecture")
    
    # Server connection
    parser.add_argument('--server_address', type=str, default='127.0.0.1',
                        help='Inference server address')
    parser.add_argument('--server_port', type=int, default=50051,
                        help='Inference server port')
    
    # Original parameters
    parser.add_argument('-i', '--interpolation_steps', default=80, type=int,
                        help='Number of interpolation steps')
    parser.add_argument('-f', '--inference_frequency', default=50, type=int,
                        help='Control loop frequency (Hz)')
    parser.add_argument('-l', '--low_pass_filter', default=0.25, type=float,
                        help='Low-pass filter coefficient')
    parser.add_argument('-o', '--operation_mode', type=str,
                        choices=['teleop_mode', 'pnp_mode'],
                        help='Operation mode')
    parser.add_argument('--state-smoothing', action='store_true',
                        help='Add interpolation from current state')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run mode (no actions published)')
    parser.add_argument('--do-blending', action='store_true',
                        help='Blend action chunks')
    
    # LeRobot-style parameters
    parser.add_argument('--actions_per_chunk', type=int, default=50,
                        help='Expected actions per inference call')
    parser.add_argument('--chunk_size_threshold', type=float, default=0.5,
                        help='Queue refill threshold (0-1)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    try:
        client = InferenceClient(
            server_address=args.server_address,
            server_port=args.server_port,
            interpolation_steps=args.interpolation_steps,
            inference_frequency=args.inference_frequency,
            low_pass_filter=args.low_pass_filter,
            operation_mode=args.operation_mode,
            is_dry_run=args.dry_run,
            state_smoothing=args.state_smoothing,
            do_blending=args.do_blending,
            actions_per_chunk=args.actions_per_chunk,
            chunk_size_threshold=args.chunk_size_threshold
        )
        
        input_str = ""
        while input_str != "y":
            input_str = input("Press 'y' key to start...")
        
        client.start_update()
        client.rosif.spin()
        
    except KeyboardInterrupt:
        loginfo("Shutting down...")
        client.stop_network_thread()
        client.print_statistics()
    except Exception as e:
        logwarn(f"Client error: {e}")
        import traceback
        traceback.print_exc()
        if 'client' in locals():
            client.stop_network_thread()
            client.print_statistics()