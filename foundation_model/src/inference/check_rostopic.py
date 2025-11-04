#!/usr/bin/env python3
"""
check_rostopic.py
Verifies that required ROS topics are alive.  Prints an explicit list of
✓ OK and ✗ missing topics, then exits 0 on success or 1 on failure.
"""
import sys
import rospy
from threading import Event

from sensor_msgs.msg import JointState, CompressedImage
from std_msgs.msg import String


# ------------------------------------------------------------
# Workaround for rospy-all deserialization issue in Python 3.10
# ------------------------------------------------------------
import codecs
# Custom error handler for "rosmsg" deserialization
def rosmsg_error_handler(exception):
    raise exception
# Register the custom error handler
codecs.register_error("rosmsg", rosmsg_error_handler)

# ----------------------------------------------------------------------
# Topics we MUST see at least one message from
# (name,  ROS msg type,  user-friendly label)
# ----------------------------------------------------------------------
REQUIRED = [
    ("/tx/robot/hardware/joints",        JointState,       "joints"),
    ("/cam_0/color/image_raw/compressed", CompressedImage, "cam0"),
    ("/cam_1/color/image_raw/compressed", CompressedImage, "cam1"),
    ("/usb_cam/image_raw/compressed",     CompressedImage, "palm"),
    ("/tx/head/image_raw/compressed", CompressedImage, "head"),
]

TIMEOUT = rospy.Duration(3.0)   # seconds
# ----------------------------------------------------------------------

def main() -> None:
    rospy.init_node("topic_checker", anonymous=True)

    # One Event per required topic
    seen = {label: Event() for *_, label in REQUIRED}

    def make_cb(label):
        return lambda *_: seen[label].set()

    for topic, msg_t, label in REQUIRED:
        rospy.Subscriber(topic, msg_t, make_cb(label), queue_size=1)

    start = rospy.Time.now()
    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        if all(ev.is_set() for ev in seen.values()):
            break
        if rospy.Time.now() - start > TIMEOUT:
            break
        rate.sleep()

    # ---------------- summary ----------------
    ok      = [lbl for lbl, ev in seen.items() if ev.is_set()]
    missing = [lbl for lbl in seen if lbl not in ok]

    print("\n=== Topic check summary ===")
    for lbl in ok:
        print(f"  ✓  {lbl:7s} topic received")
    for lbl in missing:
        print(f"  ✗  {lbl:7s} NO messages")

    if missing:
        print(f"\n• {len(missing)} required topic(s) missing. Exiting with error.")
        sys.exit(1)
    else:
        print("\n• All required topics OK. Exiting with success.")
        sys.exit(0)

if __name__ == "__main__":
    main()