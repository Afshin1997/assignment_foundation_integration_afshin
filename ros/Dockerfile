FROM osrf/ros:noetic-desktop-full

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    git \
    wget \
    vim \
    tmux \
    iputils-ping \
    python3-catkin-tools \
    python3-rosdep \
 && ln -s /usr/bin/python3.8 /usr/bin/python

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

CMD ["/bin/bash"]
