from rosif import (
    RosIf,
    convert_joint_state_to_numpy,
    convert_ros_image_to_numpy,
)
import matplotlib.pyplot as plt
import datetime


def loginfo(msg: str):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}][INFO] {msg}")


class ObservationListenerNode:
    def __init__(self):
        self.rosif = RosIf()
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
        self.teleop_mode = False

    def start_update(self):
        self.rosif.set_periodic_callback(1, self.update)
    
    def get_last_observation(self):
        while not self.rosif.is_observation_available():
            self.rosif.sleep_spin(0.001)

        observation = {
            "observation.images.cam_0": convert_ros_image_to_numpy(self.rosif.latest_image_cam0, self.rosif.cv_bridge),
            "observation.images.cam_1": convert_ros_image_to_numpy(self.rosif.latest_image_cam1, self.rosif.cv_bridge),
            "observation.state": convert_joint_state_to_numpy(self.rosif.latest_joints, self.dataset_joint_order),
            "task": self.rosif.text_instruction,
            "observation.images.head_cam": None,
            "observation.images.usb_cam": None,
        }

        if self.teleop_mode:
            observation["observation.images.head_cam"] = convert_ros_image_to_numpy(
                self.rosif.latest_image_head, self.rosif.cv_bridge
            )
        else:
            observation["observation.images.usb_cam"] = convert_ros_image_to_numpy(
                self.rosif.latest_image_palm, self.rosif.cv_bridge
            )

        self.rosif.clear_observation()
        return observation

    def update(self, _event):
        observation = self.get_last_observation()
        print(observation)
        plt.imshow(observation["observation.images.cam_0"])
        plt.savefig("tx-pizero/outputs/observation.images.cam_0.png")
        plt.imshow(observation["observation.images.cam_1"])
        plt.savefig("tx-pizero/outputs/observation.images.cam_1.png")
        plt.imshow(observation["observation.images.usb_cam"])
        plt.savefig("tx-pizero/outputs/observation.images.usb_cam.png")
        plt.imshow(observation["observation.images.head_cam"])
        plt.savefig("tx-pizero/outputs/observation.images.head_cam.png")


if __name__ == "__main__":
    try:
        loginfo(f"Initializing ObservationListenerNode...")
        node = ObservationListenerNode()

        node.start_update()
        node.rosif.spin()
    except Exception as e:
        print(e)