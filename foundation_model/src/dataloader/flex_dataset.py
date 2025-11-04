import torch
import numpy as np
import logging
from copy import deepcopy
from typing import List, Dict, Optional
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
)
from lerobot.datasets.utils import (
    get_delta_indices,
    check_delta_timestamps,
)
from lerobot.datasets.transforms import ImageTransforms
from lerobot.datasets.factory import resolve_delta_timestamps, IMAGENET_STATS

from configs.lerobot_extra_configs import TxTrainPipelineConfig

# ----------------------------------------------------------------------
# Merge per-feature statistics (mean / std / min / max)
# ----------------------------------------------------------------------
def merge_stats(stats: dict, specs: List[Dict], *, drop_original=True):
    """
    Returns a NEW stats dict in which every rule in `specs` has been applied.
    No mutation of the input dict.
    """
    stats = deepcopy(stats)
    for rule in specs:
        out_k, in_ks = rule["out_key"], rule["in_keys"]
        merged = {
            stat: np.concatenate([np.asarray(stats[k][stat]) for k in in_ks], axis=0)
            for stat in ("mean", "std", "min", "max")
        }
        stats[out_k] = merged
        if drop_original:
            for k in in_ks[1:]:
                stats.pop(k, None)
    return stats

# ----------------------------------------------------------------------
# Merge feature tensors/arrays in a sample dict (e.g. to merge force to state)
# ----------------------------------------------------------------------
def apply_concat(specs: List[Dict], sample: Dict, *, drop_original: bool=True):
    """
    In-place (returns same dict) concatenation according to `specs`.

      - Concats along last dim (torch.cat / np.concatenate).
      - Raises KeyError if a source key is missing.
      - When drop_original=True, auxiliary keys (all but first) are removed.
    """

    for rule in specs:
        out_k, in_ks = rule["out_key"], rule["in_keys"]
        parts = [sample[k] for k in in_ks] # may raise KeyError

        if isinstance(parts[0], torch.Tensor):
            dev = parts[0].device
            parts = [p if isinstance(p, torch.Tensor)
                    else torch.as_tensor(p, device=dev) for p in parts]
            sample[out_k] = torch.cat(parts, dim=-1)
        else:
            parts = [np.asarray(p) for p in parts]
            sample[out_k] = np.concatenate(parts, axis=-1)

        if drop_original:
            for k in in_ks[1:]:
                sample.pop(k, None)
    return sample

class FlexDataset(LeRobotDataset):
    """
    ------------------------------------------------------------------------------
    FlexDataset  - A drop-in super-set of LeRobotDataset
    ------------------------------------------------------------------------------
    GOAL
    ----
    Some projects want to "remap" or concatenate low-level features at run-time
    (e.g.   state + force  →  state ) without rewriting their huge Parquet files.
    FlexDataset lets you declare that mapping once in your YAML:

        concat_specs:
          - out_key: observation.state
            in_keys: [observation.state, observation.force]

    • If concat_specs is omitted or empty, the class behaves exactly
      like the original LeRobotDataset - perfect backward compatibility.

    • When a spec is provided, FlexDataset:
        1.  Updates the in-memory metadata (`meta.features`, `meta.stats`)
            so shapes and normalisation statistics are consistent with the
            merged vector.
        2.  Concatenates the source tensors on-the-fly in `__getitem__`.
        3.  Optionally drops the auxiliary source keys from the returned item
            (`drop_original=True`, default).

    Nothing is written back to disk; the Parquet / video files stay untouched.
    ------------------------------------------------------------------------------
    """
    _RESERVED_KEYS: set[str] = {
        "timestamp",
        "episode_index",
        "frame_index",
        "next.reward",
        "next.done",
        "index",
        "task_index",
    }
    def __init__(self, *args, concat_specs=None, drop_original=True, allowed_keys: Optional[set] = None, **kw):
        """
        Parameters
        ----------
        concat_specs : list[dict] | None
            Each dict must contain:
              - 'out_key':  the feature name that downstream code will read.
              - 'in_keys': list[str]  ordered source keys to concatenate.
            Example:
              {'out_key': 'observation.state',
               'in_keys': ['observation.state', 'observation.force']}

            If None / empty → identity mapping (dataset behaves as vanilla
            LeRobotDataset).

        drop_original : bool
            When True, all source keys *except* the first one are removed from
            both metadata and returned samples.  Set False if you still want
            the originals for debugging.
        """
        super().__init__(*args, **kw)

        # user-provided remap rules or empty list -> identity behaviour
        self.concat_specs: list[dict] = concat_specs or []
        self.drop_original: bool = drop_original

        # Patch dataset-wide metadata ONCE so shapes / stats stay coherent
        self._apply_specs_to_meta()
        if allowed_keys is not None:
            self._filter_to_allowed_keys(allowed_keys | self._RESERVED_KEYS)

    # ---------- metadata patching so stats / shapes stay consistent ---- #
    def _apply_specs_to_meta(self):
        """
        Merges shapes & statistics for every spec in `self.concat_specs`.
        - Shapes: reshape out_key to (Σ in_key lengths,)
        - Stats:  concatenate mean/std/min/max
        - Drop:   Optionally remove aux keys from metadata
        """
        if not self.concat_specs: # identity behaviour
            return

        # merge statistics with the shared util
        self.meta.stats = merge_stats(self.meta.stats,
                                      self.concat_specs,
                                      drop_original=self.drop_original)

        # update shapes & (optionally) drop feature entries
        for rule in self.concat_specs:
            out_k, in_ks = rule["out_key"], rule["in_keys"]
            total = sum(self.meta.features[k]["shape"][0] for k in in_ks)
            self.meta.features[out_k]["shape"] = [total]
            if self.drop_original:
                for k in in_ks[1:]:
                    self.meta.features.pop(k, None)

    def _filter_to_allowed_keys(self, keep_keys: set[str]):
        feats = self.meta.features
        current = list(feats)
        dropped = [k for k in current if k not in keep_keys]
        for k in dropped:
            feats.pop(k, None)
            self.meta.stats.pop(k, None)

        if not dropped:
            return

        logging.info(
            f"FlexDataset: dropped unused feature keys {dropped} - kept {list(feats)}"
        )

        existing = set(self.hf_dataset.column_names)
        to_remove = [k for k in dropped if k in existing]
        if to_remove:
            self.hf_dataset = self.hf_dataset.remove_columns(to_remove)

        # Update cached key list if LeRobotDataset defined one
        if hasattr(self, "keys"):
            self.keys = [k for k in self.keys if k in feats]

    # ----------- RUNTIME CONCATENATION - executed per sample ----------- #
    def __getitem__(self, idx):
        """
        1. Get the raw item from LeRobotDataset.
        2. Apply every concat spec in order.
        3. Return the modified dict.
        """
        item = super().__getitem__(idx)
        if self.concat_specs:
            item = apply_concat(self.concat_specs, item,
                        drop_original=self.drop_original)
        return item

# inspired by ./lerobot/common/datasets/factory.py
def make_dataset(cfg: TxTrainPipelineConfig) -> FlexDataset:
    # build (optional) image transform pipeline
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    # keys required by the policy
    policy_keys = set(cfg.policy.input_features) | set(cfg.policy.output_features)
    concat_src_keys = {
        k for spec in cfg.dataset.concat_specs or []
        for k in spec["in_keys"]
    }
    allowed_keys = policy_keys | concat_src_keys

    # build the dataset itself
    dataset = FlexDataset(
        cfg.dataset.repo_id,
        concat_specs = cfg.dataset.concat_specs, # extra arg for FlexDataset
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        delta_timestamps=None, # to be updated later
        image_transforms=image_transforms,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        allowed_keys=allowed_keys,
    )

    # Update timestamp here because we want to apply after metadata concatenation
    delta_timestamps = resolve_delta_timestamps(cfg.policy, dataset.meta)
    # Propagate the same Δ-timestamps from each out_key to its sources
    for spec in dataset.concat_specs:
        out_k = spec["out_key"]
        if out_k in delta_timestamps:
            for k in spec["in_keys"]:
                delta_timestamps.setdefault(k, delta_timestamps[out_k])
    check_delta_timestamps(delta_timestamps, dataset.fps, dataset.tolerance_s)
    dataset.delta_timestamps = delta_timestamps
    dataset.delta_indices = get_delta_indices(delta_timestamps, dataset.fps)

    # Optionally apply ImageNet statistics
    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset

