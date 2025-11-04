FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV WORKSPACE="/root/workspace"

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        curl \
        tzdata \
        vim \
        ffmpeg
ENV TZ Asia/Tokyo

ADD https://astral.sh/uv/install.sh /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"
RUN sh /uv-installer.sh && rm /uv-installer.sh

WORKDIR ${WORKSPACE}
COPY pyproject.toml ${WORKSPACE}/
COPY lerobot ${WORKSPACE}/lerobot
RUN uv sync
RUN uv pip install -e ./lerobot \
 && uv pip install -e "./lerobot/[pi0]" \
 && uv pip install scipy \
 && uv pip install num2words \
 && uv pip install -U git+https://github.com/huggingface/transformers.git@4f93cc9 \
 && uv pip install -U frozendict etils importlib_resources simplejpeg matplotlib sentencepiece jax[cuda12] orbax absl-py jaxtyping equinox flax flatbuffers \
 && uv pip uninstall torch torchvision torchcodec \
 && uv pip install --pre torch torchvision torchcodec --index-url https://download.pytorch.org/whl/nightly/cu128 \
 && rm -rf ${WORKSPACE}/lerobot pyproject.toml

# Append to ~/.bashrc to install 'lerobot' in editable mode.
RUN echo 'if ! (cd /root/workspace && uv run python -c "import lerobot") 2>/dev/null; then (cd /root/workspace && uv pip install -e tx-pizero/lerobot); fi' \
 >> /root/.bashrc

# install ROS integration related packages
RUN uv pip install opencv-python-headless \
 && uv pip install --extra-index-url https://rospypi.github.io/simple rospy-all cv_bridge

CMD ["/bin/bash"]
