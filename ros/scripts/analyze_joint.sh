ROSBAG_ANALYSIS=rosbag_analysis.bag
source /opt/ros/noetic/setup.bash

# Start rosbag play in the background
rosbag play $1 /tx/tx_driver/mds_cmd:=/tx/tx_driver/mds_cmd_org &
PLAY_PID=$!

sleep 7 # TODO: block until /tx/tx_driver/mds_cmd_org is published

# sleep 1
# # Wait until /tx/tx_driver/mds_cmd appears in the topic list, with timeout
# echo "Waiting for /tx/tx_driver/mds_cmd to appear in topic list..."
# timeout=20
# elapsed=0
# while ! rostopic list | grep -q "^/tx/tx_driver/mds_cmd$"; do
#     sleep 1
#     elapsed=$((elapsed+1))
#     if [ $elapsed -ge $timeout ]; then
#         echo "Timeout waiting for /tx/tx_driver/mds_cmd. Exiting."
#         kill $PLAY_PID
#         exit 1
#     fi
# done
# echo "/tx/tx_driver/mds_cmd is now in the topic list."

# Start rosbag record in the background and save its PID
rosbag record -O $ROSBAG_ANALYSIS /tx/tx_driver/mds_cmd /tx/tx_driver/mds_cmd_org &
RECORD_PID=$!

# Wait for rosbag play to finish
wait $PLAY_PID

# Stop rosbag record after play finishes
kill $RECORD_PID

sleep 3 

rostopic echo -b $ROSBAG_ANALYSIS -p /tx/tx_driver/mds_cmd > infer.csv
rostopic echo -b $ROSBAG_ANALYSIS -p /tx/tx_driver/mds_cmd_org > org.csv
python analyze_joint.py