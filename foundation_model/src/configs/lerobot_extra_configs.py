from dataclasses import dataclass, field
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig

@dataclass
class FlexDatasetConfig(DatasetConfig):
    """Adds optional run-time concatenation spec."""
    concat_specs: list[dict] = field(
        default_factory=list,
        metadata={
            "help": (
                "List of remap rules. Example:\n"
                "  - out_key: observation.state\n"
                "    in_keys: [observation.state, observation.force]\n"
                "If left empty, the dataset behaves exactly like the "
                "original LeRobotDataset."
            )
        },
    )

# This is the top config we managed as a yaml file under ./config/*.yaml
@dataclass
class TxTrainPipelineConfig(TrainPipelineConfig):
    dataset: FlexDatasetConfig

    # Optional path to pretrained weights. None means start from scratch.
    # train.py moves this value to cfg.policy.pretrained_path so make_policy can load the weights.
    # Using this field instead of --policy.path keeps the YAML config intact.
    pretrained_path: str | None = None

    sampler_drop_n_first_frames: int = 0
    sampler_drop_n_last_frames: int = 0
    grad_acc_steps: int = 1 # e.g. 256 // 8

