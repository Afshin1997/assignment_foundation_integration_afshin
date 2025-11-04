#!/usr/bin/env python3
"""
Inference Server - Server-Client Architecture
Runs the policy model and handles inference requests from clients.

Based on LeRobot's async inference approach.
Adapted for tx-pizero ACT policy.
"""

import argparse
import socket
import pickle
import time
import datetime
import numpy as np
import threading
import queue


def loginfo(msg: str):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}][INFO][SERVER] {msg}")


def logwarn(msg: str):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}][WARN][SERVER] {msg}")


class InferenceServer:
    """
    Policy Server that runs inference on GPU.
    Listens for observation requests and returns action predictions.
    """
    
    def __init__(self, model_config, host='0.0.0.0', port=50051, device='cuda'):
        """
        Initialize inference server
        
        Args:
            model_config: Path to model config/checkpoint
            host: Server host address
            port: Server port
            device: Device for inference ('cuda' or 'cpu')
        """
        loginfo("=" * 60)
        loginfo("Initializing Inference Server")
        loginfo("=" * 60)
        
        self.host = host
        self.port = port
        self.device = device
        
        # Load inference backend
        loginfo(f"Loading model from: {model_config}")
        from backend.lerobot import LeRobotInferenceBackend
        self.backend = LeRobotInferenceBackend(model_config)
        loginfo(f"Model loaded successfully on {device}")
        
        # Server socket
        self.server_socket = None
        
        # Statistics
        self.inference_count = 0
        self.inference_times = []
        self.total_bytes_received = 0
        self.total_bytes_sent = 0
        self.client_count = 0
        
        # Thread control
        self.running = False
        
        loginfo(f"Server initialized at {host}:{port}")
        loginfo("=" * 60)
    
    def _send_message(self, client_socket, data):
        """Send message with size prefix"""
        serialized = pickle.dumps(data)
        size = len(serialized)
        
        # Send size (4 bytes, big-endian)
        client_socket.sendall(size.to_bytes(4, 'big'))
        
        # Send data
        client_socket.sendall(serialized)
        
        self.total_bytes_sent += size + 4
        return size
    
    def _receive_message(self, client_socket):
        """Receive message with size prefix"""
        # Receive size (4 bytes)
        size_data = client_socket.recv(4)
        if not size_data or len(size_data) < 4:
            return None
        
        msg_size = int.from_bytes(size_data, 'big')
        
        # Receive data
        data = b''
        while len(data) < msg_size:
            packet = client_socket.recv(min(msg_size - len(data), 4096))
            if not packet:
                return None
            data += packet
        
        self.total_bytes_received += msg_size + 4
        return pickle.loads(data)
    
    def _handle_client(self, client_socket, client_address):
        """Handle a client connection"""
        self.client_count += 1
        client_id = self.client_count
        
        loginfo(f"Client #{client_id} connected from {client_address}")
        
        try:
            while self.running:
                # Receive observation
                observation = self._receive_message(client_socket)
                
                if observation is None:
                    logwarn(f"Client #{client_id} disconnected")
                    break
                
                # Run inference
                start_time = time.time()
                
                try:
                    time.sleep(0.8)
                    actions = self.backend.generate_actions(observation)
                    inference_time = time.time() - start_time
                    
                    self.inference_count += 1
                    self.inference_times.append(inference_time)
                    
                    loginfo(f"Inference #{self.inference_count} completed in {inference_time*1000:.2f}ms")
                    
                    # Send actions back
                    response = {
                        'actions': actions,
                        'inference_time': inference_time,
                        'inference_count': self.inference_count
                    }
                    
                    self._send_message(client_socket, response)
                    
                except Exception as e:
                    logwarn(f"Inference error: {e}")
                    error_response = {
                        'error': str(e),
                        'actions': None
                    }
                    self._send_message(client_socket, error_response)
                    
        except Exception as e:
            logwarn(f"Client #{client_id} error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            client_socket.close()
            loginfo(f"Client #{client_id} connection closed")
    
    def serve(self):
        """Start the server and listen for connections"""
        # Create socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Bind and listen
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        
        self.running = True
        
        loginfo("=" * 60)
        loginfo(f"🚀 Server listening on {self.host}:{self.port}")
        loginfo("Waiting for client connections...")
        loginfo("=" * 60)
        
        try:
            while self.running:
                # Set timeout to allow checking self.running
                self.server_socket.settimeout(1.0)
                
                try:
                    client_socket, client_address = self.server_socket.accept()
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.timeout:
                    continue
                    
        except KeyboardInterrupt:
            loginfo("Shutting down server...")
        finally:
            self.running = False
            if self.server_socket:
                self.server_socket.close()
            self.print_statistics()
    
    def print_statistics(self):
        """Print server statistics"""
        loginfo("\n" + "=" * 60)
        loginfo("SERVER STATISTICS")
        loginfo("=" * 60)
        loginfo(f"Total clients: {self.client_count}")
        loginfo(f"Total inferences: {self.inference_count}")
        
        if self.inference_times:
            loginfo(f"Average inference time: {np.mean(self.inference_times)*1000:.2f}ms")
            loginfo(f"Min inference time: {np.min(self.inference_times)*1000:.2f}ms")
            loginfo(f"Max inference time: {np.max(self.inference_times)*1000:.2f}ms")
            loginfo(f"Std inference time: {np.std(self.inference_times)*1000:.2f}ms")
        
        loginfo(f"Total bytes received: {self.total_bytes_received / 1024 / 1024:.2f} MB")
        loginfo(f"Total bytes sent: {self.total_bytes_sent / 1024 / 1024:.2f} MB")
        loginfo("=" * 60 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference Server for Server-Client Architecture")
    
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Server host address (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=50051,
                        help='Server port (default: 50051)')
    parser.add_argument('-c', '--model_config', type=str, required=True,
                        help='Path to model config/checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for inference (default: cuda)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    try:
        # Create and start server
        server = InferenceServer(
            model_config=args.model_config,
            host=args.host,
            port=args.port,
            device=args.device
        )
        
        # Start serving
        server.serve()
        
    except Exception as e:
        logwarn(f"Server error: {e}")
        import traceback
        traceback.print_exc()