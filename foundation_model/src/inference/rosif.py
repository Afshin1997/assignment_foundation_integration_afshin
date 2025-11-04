#!/usr/bin/env python3

# ------------------------------------------------------------
# Workaround for rospy-all deserialization issue in Python 3.10
# ------------------------------------------------------------
# Problem:
# rospy-all expects a "rosmsg" error handler for deserializing complex messages
# (e.g., sensor_msgs/JointState, sensor_msgs/Image). Without this handler,
# deserialization fails, and subscriber callbacks are not triggered.
# This is due to the `codecs.lookup_error("rosmsg")` line in rospy-all's message
# handling code, which expects a custom error handler named "rosmsg" to be registered.
# Fix:
# Register a custom "rosmsg" error handler that re-raises exceptions, allowing
# rospy-all to proceed without errors.
import codecs


# Custom error handler for "rosmsg" deserialization
def rosmsg_error_handler(exception):
    raise exception


# Register the custom error handler
codecs.register_error("rosmsg", rosmsg_error_handler)
# ------------------------------------------------------------

import torch
import numpy as np
import rospy
from sensor_msgs.msg import JointState, Image, CompressedImage
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from MotorState import MotorState


def convert_ros_image_to_tensor(ros_img_msg, cv_bridge: CvBridge) -> torch.FloatTensor:
    """
    Convert either a raw sensor_msgs/Image or a sensor_msgs/CompressedImage
    into a Torch FloatTensor of shape (3, H, W), with range [0,1], in RGB order.

    Args:
        ros_img_msg: A ROS Image or CompressedImage message.
        bridge (CvBridge): An instance of cv_bridge's CvBridge for image conversions.

    Returns:
        A torch.FloatTensor of shape (3,H,W), scaled to [0,1], in RGB.

    Raises:
        RuntimeError if the cv_bridge conversion fails or if the image encoding isn't supported.
    """

    try:
        # Distinguish between raw or compressed
        if isinstance(ros_img_msg, CompressedImage):
            # Handle compressed data (e.g. JPEG/PNG)
            cv_img = cv_bridge.compressed_imgmsg_to_cv2(
                ros_img_msg, desired_encoding="rgb8"
            )
        elif isinstance(ros_img_msg, Image):
            # Handle raw image data
            cv_img = cv_bridge.imgmsg_to_cv2(ros_img_msg, desired_encoding="rgb8")
        else:
            raise TypeError(
                f"Expected sensor_msgs/Image or sensor_msgs/CompressedImage, got {type(ros_img_msg)}"
            )

        # cv_img is now (H, W, 3), uint8, RGB order

    except CvBridgeError as e:
        raise RuntimeError(f"cv_bridge conversion failed: {e}")

    # Convert [0..255] => [0..1] and transpose to (3, H, W)
    # .contiguous() ensures a well-defined memory layout, which can help speed in further ops
    tensor = torch.from_numpy(cv_img).permute(2, 0, 1).float().div(255.0).contiguous()
    return tensor


def convert_joint_state_to_tensor(
    joint_msg: JointState, joint_order
) -> torch.FloatTensor:
    """
    Reorders the JointState positions into the same order as joint_order.
    If a joint is missing, fill with 0.0.
    Returns np.float32 array of shape (10,).
    """
    # Convert the name list to a dict for quick lookup:
    #   name_to_index["A1ZR_JOINT"] = idx in joint_msg.name
    name_to_index = {name: i for i, name in enumerate(joint_msg.name)}

    # Create output array
    ordered_joints = np.zeros(len(joint_order), dtype=np.float32)

    # For each expected name, find index in joint_msg, or fill with 0 if missing
    for i, desired_name in enumerate(joint_order):
        if desired_name in name_to_index:
            idx = name_to_index[desired_name]
            ordered_joints[i] = joint_msg.position[idx]
        else:
            # e.g., log a warning or do nothing (it's already zero)
            pass

    return torch.from_numpy(ordered_joints)  # shape (1,10)


def convert_ros_image_to_numpy(ros_img_msg, cv_bridge: CvBridge) -> np.ndarray:
    """
    Convert either a raw sensor_msgs/Image or a sensor_msgs/CompressedImage
    into a Torch FloatTensor of shape (3, H, W), with range [0,1], in RGB order.

    Args:
        ros_img_msg: A ROS Image or CompressedImage message.
        bridge (CvBridge): An instance of cv_bridge's CvBridge for image conversions.

    Returns:
        A torch.FloatTensor of shape (3,H,W), scaled to [0,1], in RGB.

    Raises:
        RuntimeError if the cv_bridge conversion fails or if the image encoding isn't supported.
    """

    try:
        # Distinguish between raw or compressed
        if isinstance(ros_img_msg, CompressedImage):
            # Handle compressed data (e.g. JPEG/PNG)
            cv_img = cv_bridge.compressed_imgmsg_to_cv2(
                ros_img_msg, desired_encoding="rgb8"
            )
        elif isinstance(ros_img_msg, Image):
            # Handle raw image data
            cv_img = cv_bridge.imgmsg_to_cv2(
                ros_img_msg, desired_encoding="rgb8"
            )
        else:
            raise TypeError(
                f"Expected sensor_msgs/Image or sensor_msgs/CompressedImage, got {type(ros_img_msg)}"
            )

        # cv_img is now (H, W, 3), uint8, RGB order

    except CvBridgeError as e:
        raise RuntimeError(f"cv_bridge conversion failed: {e}")

    return cv_img


def convert_joint_state_to_numpy(joint_msg: JointState, joint_order) -> np.ndarray:
    """
    Reorders the JointState positions into the same order as joint_order.
    If a joint is missing, fill with 0.0.
    Returns np.float32 array of shape (10,).
    """
    # Convert the name list to a dict for quick lookup:
    #   name_to_index["A1ZR_JOINT"] = idx in joint_msg.name
    name_to_index = {name: i for i, name in enumerate(joint_msg.name)}

    # Create output array
    ordered_joints = np.zeros(len(joint_order), dtype=np.float32)

    # For each expected name, find index in joint_msg, or fill with 0 if missing
    for i, desired_name in enumerate(joint_order):
        if desired_name in name_to_index:
            idx = name_to_index[desired_name]
            ordered_joints[i] = joint_msg.position[idx]
        else:
            # e.g., log a warning or do nothing (it's already zero)
            pass

    return ordered_joints # shape (1,10)

def convert_wrench_to_numpy(msg: WrenchStamped) -> np.ndarray:
    """
    geometry_msgs/WrenchStamped → np.float32[6]
    Order  = [fx, fy, fz, tx, ty, tz]
    """
    f = msg.wrench.force
    t = msg.wrench.torque
    return np.array([f.x, f.y, f.z, t.x, t.y, t.z], dtype=np.float32)

class RosIf:
    def __init__(self, node_name="pi_zero_rosif_node"):
        self.use_dummy = {
            "cam0": False,
            "cam1": False,
            "palm": False,
            "head": False,
            "joints": False,
            "force": False,
        }
        self.init(node_name)

    def init(self, node_name):
        rospy.init_node(node_name, anonymous=True)
        self.cv_bridge = CvBridge()
        use_compressed_image = True
        if use_compressed_image:
            img_key = "/compressed"
            img_type = CompressedImage
        else:
            img_key = ""
            img_type = Image
        self.img_cam0_sub = rospy.Subscriber(
            f"/cam_0/color/image_raw{img_key}",
            img_type,
            self._comp_image_callback_cam0,
            queue_size=1,
        )
        self.img_cam1_sub = rospy.Subscriber(
            f"/cam_1/color/image_raw{img_key}",
            img_type,
            self._comp_image_callback_cam1,
            queue_size=1,
        )
        self.img_palm_sub = rospy.Subscriber(
            f"/usb_cam/image_raw{img_key}",
            img_type,
            self._comp_image_callback_palm,
            queue_size=1,
        )
        # for now, head only publishes compressed image
        self.img_head_sub = rospy.Subscriber(
            f"/tx/head/image_raw/compressed",
            CompressedImage,
            self._comp_image_callback_head,
            queue_size=1,
        )
        self.joint_state_sub = rospy.Subscriber(
            "/tx/robot/hardware/joints",
            JointState,
            self._joint_state_callback,
            queue_size=1,
        )
        self.force_sub = rospy.Subscriber(
            "/tx/tx_driver/force_sensor",
            WrenchStamped,
            self._force_callback,
            queue_size=1,
        )
        self.prompt_sub = rospy.Subscriber(
            "/tx/prompt", String, self._prompt_sub_callback, queue_size=1
        )

        self.cmd_pub = rospy.Publisher(
            "/tx/tx_driver/mds_cmd", MotorState, queue_size=1
        )
        self.joint_pub = rospy.Publisher(
            "/tx/tx_driver/joint_cmd", JointState, queue_size=1
        )
        self.text_instruction = ""

        self.joint_names = [
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

        self._dummy_img = self._make_dummy_img()
        self._dummy_joints = self._make_dummy_joint_state()
        self._dummy_force = self._make_dummy_force()
        self.clear_observation()
        for cam, flag in self.use_dummy.items():
            print(f"[RosIf] camera '{cam}' dummy mode: {flag}")
        print("RosIf initialization done")

    def _comp_image_callback_cam0(self, msg: CompressedImage | Image):
        self.latest_image_cam0 = msg

    def _comp_image_callback_cam1(self, msg: CompressedImage | Image):
        self.latest_image_cam1 = msg

    def _comp_image_callback_head(self, msg: CompressedImage | Image):
        self.latest_image_head = msg

    def _comp_image_callback_palm(self, msg: CompressedImage | Image):
        self.latest_image_palm = msg

    def _joint_state_callback(self, msg: JointState):
        self.latest_joints = msg

    def _force_callback(self, msg: WrenchStamped):
        self.latest_force = msg

    def _prompt_sub_callback(self, msg: String):
        print(f"Received prompt: {msg.data}")
        self.text_instruction = msg.data

    def _make_dummy_joint_state(self):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = self.joint_names
        n = len(self.joint_names)
        zeros = [0.0] * n
        msg.position = zeros
        msg.velocity = zeros
        msg.effort = zeros
        return msg

    def _make_dummy_force(self):
        msg = WrenchStamped()
        msg.header.stamp = rospy.Time.now()
        msg.wrench.force.x = 0.0
        msg.wrench.force.y = 0.0
        msg.wrench.force.z = 0.0
        msg.wrench.torque.x = 0.0
        msg.wrench.torque.y = 0.0
        msg.wrench.torque.z = 0.0
        return msg

    def _make_dummy_img(self):
        h, w = 64, 64
        blank = np.zeros((h, w, 3), np.uint8)
        msg = self.cv_bridge.cv2_to_imgmsg(blank, "bgr8")
        msg.header.stamp = rospy.Time.now()
        return msg

    def publish_joint_state(self, joint_order, positions):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = joint_order  # ["L1XP_JOINT", "L2ZR_JOINT", ...]
        msg.position = positions
        joint_count = len(joint_order)
        msg.velocity = [0.0] * joint_count
        msg.effort = [0.0] * joint_count
        self.joint_pub.publish(msg)
        # print(f"Published PiZero action as JointState => {msg}")

    def publish_motor_state(self, joint_order, positions):
        msg = MotorState()
        msg.header.stamp = rospy.Time.now()
        msg.name = joint_order  # ["L1XP_JOINT", "L2ZR_JOINT", ...]
        msg.position = positions
        joint_count = len(joint_order)
        msg.velocity = [0.0] * joint_count
        msg.acceleration = [0.0] * joint_count
        msg.effort = [0.0] * joint_count
        msg.Id_debug = [0.0] * joint_count
        self.cmd_pub.publish(msg)
        # print(f"Published PiZero action as JointState => {msg}")

    def is_observation_available(self, is_teleop=True, debug_print=False):
        if debug_print:
            print(f"latest_joints: {self.latest_joints is not None}")
            print(f"latest_image_cam0: {self.latest_image_cam0 is not None}")
            print(f"latest_image_cam1: {self.latest_image_cam1 is not None}")
            print(f"latest_image_palm: {self.latest_image_palm is not None}")

        # Base requirements (always needed)
        ready = (
            self.latest_joints is not None
            and self.latest_image_cam0 is not None
            and self.latest_image_cam1 is not None
            and self.latest_image_palm is not None
            and self.latest_force is not None
        )
        # In tele-op mode we also need the head camera
        if is_teleop:
            ready = ready and (self.latest_image_head is not None)
        return ready

    def clear_observation(self):
        self.latest_joints = self._dummy_joints if self.use_dummy["joints"] else None
        self.latest_image_cam0 = self._dummy_img if self.use_dummy["cam0"] else None
        self.latest_image_cam1 = self._dummy_img if self.use_dummy["cam1"] else None
        self.latest_image_palm = self._dummy_img if self.use_dummy["palm"] else None
        self.latest_image_head = self._dummy_img if self.use_dummy["head"] else None
        self.latest_force = self._dummy_force if self.use_dummy["force"] else None

    def sleep_spin(self, duration_sec):
        duration_msec = int(duration_sec * 1000.0)
        rate = rospy.Rate(1000)
        for _ in range(duration_msec):
            rate.sleep()

    def spin(self):
        rospy.spin()

    def set_periodic_callback(self, loop_rate_hz, control_loop):
        rospy.Timer(rospy.Duration(1.0 / loop_rate_hz), control_loop)
