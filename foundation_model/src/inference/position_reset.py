import time
import argparse
from rosif import RosIf, convert_joint_state_to_numpy, convert_ros_image_to_numpy, convert_ros_image_to_tensor, convert_joint_state_to_tensor
import numpy as np


DEFAULT_FILEPATH = "saved_pos.npy"
DATASET_JOINT_ORDER = [
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
EPSILON = 0.001
MAX_ACTION_NORM = 0.1


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--save", action="store_true", help="Flag to save the current robot position")
    parser.add_argument("-m", "--move", action="store_true", help="Flag to move the robot to the saved position")
    parser.add_argument("-f", "--file", type=str, help="The filepath to save or load the position from (file extension: npy).")

    return parser.parse_args()


def wait_and_save(rosif, save_filepath):
    # Wait for observation.
    while not rosif.is_observation_available(is_teleop=False):
        rosif.sleep_spin(0.001)
    
    # Save joint state.
    joints = convert_joint_state_to_numpy(rosif.latest_joints, DATASET_JOINT_ORDER)
    np.save(save_filepath, joints)

    print(f"Saved position:")
    for i, k in enumerate(DATASET_JOINT_ORDER):
        print(f"\t{k}: {joints[i]}")

    # Just quit brutally.
    quit()


def compute_interpolated_sequence(start, end, num_steps):
    interpolated_sequences = []
    for i in range(len(start)):
        interpolated_sequences.append(np.linspace(start, end, num=num_steps, endpoint=True))
    
    return np.asarray(interpolated_sequences)


def publish(rosif, goal_position):
    if "time_step" not in publish.__dict__:
        publish.time_step = 0

    # Wait for observation.
    while not rosif.is_observation_available(is_teleop=False):
        rosif.sleep_spin(0.001)
    
    # Check distance to target.
    joints = convert_joint_state_to_numpy(rosif.latest_joints, DATASET_JOINT_ORDER)

    if "interpolated_sequence" not in publish.__dict__:
        publish.time_horizon = 100
        publish.interpolated_sequence = compute_interpolated_sequence(joints, goal_position, publish.time_horizon)

    delta = goal_position - joints
    if not np.all(delta < EPSILON) and publish.time_step < publish.time_horizon:
        # Clip delta and use as action.
        # action = delta.clip(-MAX_ACTION_NORM, MAX_ACTION_NORM)
        # Take the next step in the interpolation sequence.
        action = publish.interpolated_sequence[:, publish.time_step][0]
        publish.time_step += 1
        print(f"Action: {action}")
        rosif.publish_motor_state(DATASET_JOINT_ORDER, action.tolist())
    else:
        # Just quit brutally.
        quit()


def main(args):
    if not args.save and not args.move:
        print("You need to specify either --save or --move.")
        return
    
    if args.save:
        # Args check.
        save_filepath = DEFAULT_FILEPATH
        if args.file:
            save_filepath = args.file
        
        print(f"Saving position in {save_filepath}.")

        # Start ROS Interface.
        rosif = RosIf()
        rosif.set_periodic_callback(1, lambda _: wait_and_save(rosif, save_filepath))
        rosif.spin()

    elif args.move:
        # Args check.
        load_filepath = DEFAULT_FILEPATH
        if args.file:
            load_filepath = args.file
        
        print("Loading position from {save_filepath}.")

        goal_position = np.load(load_filepath)
        print("Goal position:")
        for i, k in enumerate(DATASET_JOINT_ORDER):
            print(f"\t{k}: {goal_position[i]}")

        # Start ROS Interface.
        rosif = RosIf()
        rosif.set_periodic_callback(10, lambda _: publish(rosif, goal_position))
        rosif.spin()

    print("Done. Bye!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
