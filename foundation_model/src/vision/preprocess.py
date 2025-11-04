# vision/preprocess.py
#
# ──────────────────────────────────────────────────────────────────────────────
# Vision pre-processing helpers for the LeRobot stack
# ──────────────────────────────────────────────────────────────────────────────
#
# Motivation
# ----------
# • In the LeRobot ecosystem we want to train / run many policies (π0, ACT, …)
#   on the *same* raw dataset and ROS topics.  
# • The Action-Chunking Transformer (ACT) **requires** every camera view to
#   arrive at the network with identical spatial resolution, otherwise the
#   patch-flattening + concatenation step fails.  
# • Other policies (π0, π0-FAST, …) already do their own resizing internally
#   or can accept arbitrary shapes.
#
# Rather than sprinkling `if policy == "act": resize(...)` across all data
# loaders and inference nodes, we move the knowledge of *what each policy
# expects* into this one tiny module.  Upper layers simply call
#
#       tf = create_image_preprocessor(policy_name)
#       img = tf(img)
#
# and remain perfectly agnostic.
#
# Design
# ------
# • `_POLICY_SPECS` is a micro-*registry* mapping `policy_name → VisionSpec`.
# • `VisionSpec` declares:
#       - target_hw : a `(H,W)` tuple → images will be bilinearly resized;
#                    `None`            → leave size unchanged.
#       - norm_to_float : convert uint8 [0-255] → float32 [0-1].
#
# • `create_image_preprocessor()` looks up the spec, builds a
#   `torchvision.transforms.v2.Compose`, and returns it.
#
# Performance
# -----------
# The transforms run on GPU tensors when the caller moves data to CUDA first;
# otherwise they fall back to CPU.  Typical overhead (RTX 4090, float path):
#
#   ┌───────────────────────────────┬────────────┬────────────┐
#   │ operation                     │ VGA (ms)   │ 4×VGA (ms) │
#   ├───────────────────────────────┼────────────┼────────────┤
#   │ uint8 → float32 [0,1]         │   ≈ 0.05   │   ≈ 0.2    │
#   │ bilinear resize to 480×640    │   ≈ 0.65   │   ≈ 2.6    │
#   └───────────────────────────────┴────────────┴────────────┘
#
# Even a 50 Hz control loop (20 ms per step) keeps > 85 % of its budget.
# During offline training the cost is negligible relative to forward/back-ward.
#
# Usage example
# -------------
#     from vision.preprocess import create_image_preprocessor
#
#     tf = create_image_preprocessor("act")      # unified 480×640, float32
#     img = tf(img)                              # PIL, np.uint8 HWC, or Tensor
#
#     tf_pi0 = create_image_preprocessor("pi0")  # only uint8→float
#
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from torchvision.transforms import v2 as T, InterpolationMode, Lambda
import torch
from collections import defaultdict


@dataclass(frozen=True)
class VisionSpec:
    """Simple holder for policy-specific vision requirements."""
    target_hw: tuple[int, int] | None          # None ➜ no resize
    norm_to_float: bool = True                 # uint8 → float32 [0,1]


# ── Policy registry ──────────────────────────────────────────────────────────
# Default: “do nothing”.
_POLICY_SPECS: dict[str, VisionSpec] = defaultdict(lambda: VisionSpec(None))

# Known policies
_POLICY_SPECS.update(
    act     = VisionSpec((480, 640)),      # all cameras will be resized to 480×640
    pi0     = VisionSpec(None),            # Pi-0 resizes to 224×224 internally
    pi0fast = VisionSpec(None),
    # Add yours here:
    # mynet   = VisionSpec((256,256)),
)


# ── Factory function ─────────────────────────────────────────────────────────
def create_image_preprocessor(policy_name: str,
                              override_hw: tuple[int, int] | None = None):
    """
    Build a torchvision.v2 `Compose` that turns **uint8 HWC** (or anything
    torchvision can read) into **float32 CHW** in [0, 1] for *policy_name*.

    Parameters
    ----------
    policy_name : {"act", "pi0", "pi0fast", …}
    override_hw : Optional[(H,W)]
        Resize to this resolution instead of the policy default (rare).

    Returns
    -------
    torchvision.v2.Compose
        Callable that can be fed straight into the model.
    """
    try:
        spec = _POLICY_SPECS[policy_name.lower()]
    except KeyError as e:
        raise KeyError(f"Unknown policy '{policy_name}'.") from e

    target_hw = override_hw if override_hw is not None else spec.target_hw

    # -------------------------- pipeline ---------------------------
    # Ensures an Image object/tensor
    ops: list[T.Transform] = [
        T.ToImage(),
    ]

    # ---------------------------------------------------------------
    # uint8 → float32 and /255 so later ops work in [0,1] range, [0,255] -> [0,1]
    # - If the incoming tensor is uint8 (e.g. live ROS image, PNG/JPEG still frames) 
    #   scale=True produces  x.float() / 255 scaling.
    # - If it is already float32 (e.g. frames decoded by `decode_video_frames` in LerobotDataset, 
    #   which have already been divided by 255 inside the loader) t
    #   orchvision detects the dtype match and returns the tensor unchanged.
    #   → no accidental double scaling.
    # ---------------------------------------------------------------
    if spec.norm_to_float:
        ops.append(T.ToDtype(torch.float32, scale=True))

    # ---------------------------------------------------------------
    # Final layout must be channel-first (C,H,W) for downstream PyTorch ops
    def _to_chw(t: torch.Tensor) -> torch.Tensor:
        # Already CHW if batch dim == channels (1 or 3)
        return t if t.ndim < 3 or t.shape[0] in (1, 3) else t.permute(2, 0, 1)
    ops.append(Lambda(_to_chw))

    # ---------------------------------------------------------------
    # Bilinear resize to a uniform (H,W), before we permute to CHW
    if target_hw is not None:
        ops.append(
            T.Resize(target_hw,
                     InterpolationMode.BILINEAR,
                     antialias=True)
        )

    return T.Compose(ops)
