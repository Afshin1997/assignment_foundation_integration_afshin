from .backend import InferenceBackend
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
from lerobot.datasets.utils import (
    load_json,
    flatten_dict,
    unflatten_dict,
)
from lerobot.policies.factory import get_policy_class
import datetime
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from vision.preprocess import create_image_preprocessor
from dataloader.flex_dataset import apply_concat, merge_stats
from configs.lerobot_extra_configs import TxTrainPipelineConfig

def loginfo(msg: str):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}][INFO] {msg}")


def logwarn(msg: str):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}][WARN] {msg}")


def make_policy(cfg, meta_stats: dict, device: str | torch.device) -> nn.Module:
    policy_cls = get_policy_class(cfg.type)

    # Prepare the kwargs to pass to the policy constructor
    kwargs = {}
    kwargs["config"] = cfg
    kwargs["dataset_stats"] = meta_stats

    # If there's a pretrained path, load weights from there
    if getattr(cfg, "pretrained_path", None):
        loginfo(f"Loading policy from pretrained path: {cfg.pretrained_path}")
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        raise Exception("Instantiating policy from scratch with no pretrained_path.")

    policy.to(device)
    assert isinstance(policy, nn.Module), "Policy must be a torch.nn.Module."
    return policy


def load_stats(path_stats: Path) -> dict:
    if not (path_stats).exists():
        raise FileNotFoundError(f"Stats file not found: {path_stats}")
    stats = load_json(path_stats)
    stats = {key: torch.tensor(value) for key, value in flatten_dict(stats).items()}
    return unflatten_dict(stats)


class LeRobotInferenceBackend(InferenceBackend):
    """Implementation of the inference backend for LeRobot models (Pi0, Pi0-FAST, ACT)."""
    _RESERVED_KEYS = { "task" }
    def __init__(self, model_config):
        PI0_PATH = "/root/models/pi0/teleop_smpl_170k/pretrained_model"
        PI0_FAST_PATH = "/root/models/pi0fast/teleop_smpl_170k/pretrained_model"
        ACT_PATH = "/root/models/2025-06-25-17-17-09_act_clean_data/checkpoints/450000/pretrained_model"

        pretrained_path = ACT_PATH
        dataset_path =  "/root/dataset/tx_ghost/teleop_smpl"

        train_cfg = TxTrainPipelineConfig.from_pretrained(pretrained_path)
        policy_cfg = train_cfg.policy
        policy_cfg.pretrained_path = pretrained_path
        dataset_metadata = LeRobotDatasetMetadata(dataset_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Specifications for concatenating observations into a single state (may be empty if concat not required)
        self.concat_specs = train_cfg.dataset.concat_specs
        if self.concat_specs:
            updated_stats = merge_stats(dataset_metadata.stats, self.concat_specs)
        else:
            updated_stats = dataset_metadata.stats

        self.policy = make_policy(
            cfg=policy_cfg,
            meta_stats=updated_stats,
            device=self.device,
        )

        # For input feature, we filter the features we used in training (and originaly used before concat)
        self.required_input_feature_keys = self._RESERVED_KEYS | set(train_cfg.policy.input_features) | {
            k
            for spec in (self.concat_specs or [])
            for k in spec["in_keys"]
        }

        # Build the same image pre-processing pipeline used during training
        self.preprocess = create_image_preprocessor(policy_cfg.type)

        self.policy.eval()

    def prepare_batch(self, observation):
        """
        Prepares a batch for the inference with a LeRobot model.

        Args:
            observation (dict): contains the observations in either numpy or raw format (e.g. task text instruction).
        Returns:
            dict: a dictionary with the proper keys in the proper data type to be used directly by a LeRobot model.
        """

        batch: Dict[str, Any] = {}
        missing: list[str] = []

        # ------------------------------------------------------------------
        #  keep only the keys that were used in training
        # ------------------------------------------------------------------
        for k in self.required_input_feature_keys:
            if k in observation:
                batch[k] = observation[k]
            else:
                missing.append(k)
        if missing:
            logwarn(f"prepare_batch: required visual keys missing from observation: {missing}")

        # ------------------------------------------------------------------
        # apply concat-rules if any
        # ------------------------------------------------------------------
        if self.concat_specs:
            batch = apply_concat(self.concat_specs, batch)

        # ------------------------------------------------------------------
        # Move batch tensors to the same device as the policy
        # ------------------------------------------------------------------
        policy_device = next(self.policy.parameters()).device
        for key, val in batch.items():
            # Only move torch.Tensors, skip lists/strings
            if isinstance(val, torch.Tensor):
                batch[key] = val.unsqueeze(0).to(policy_device)
            elif isinstance(val, np.ndarray):
                tensor_val = torch.from_numpy(val)

                # Images.
                if len(tensor_val.shape) == 3:
                    batch[key] = self.preprocess(tensor_val).unsqueeze(0).to(policy_device)
                # State.
                else:
                    batch[key] = tensor_val.unsqueeze(0).to(policy_device)

            else:
                batch[key] = [val]

        return batch

    def generate_actions(self, observation):
        """
        Generates a sequence of actions.

        Args:
            observation (dict): contains the observations in either numpy or raw format (e.g. task text instruction).
        Returns:
            np.ndarray: the generated actions.
        """
        batch = self.prepare_batch(observation)

        # debug line. we can remove once the logic working well.
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                print(f"key: {key}, shape: {val.shape}") 

        with torch.no_grad():
            # --- Pi-0 / Pi-0-FAST -----------------------------------------
            if isinstance(self.policy, (PI0Policy, PI0FASTPolicy, SmolVLAPolicy)):
                self.policy.reset() # clear internal deque
                step_actions = []
                for _ in range(self.policy.config.n_action_steps):
                    a = self.policy.select_action(batch)  # (1, action_dim) tensor
                    step_actions.append(a.squeeze(0))     # remove batch dim

                action_tensor = torch.stack(step_actions, dim=0)  # (T, dim)
            # --- ACT -------------------------------------------------------
            elif isinstance(self.policy, ACTPolicy):
                # ACT can produce the whole chunk in one forward pass
                batch_n = self.policy.normalize_inputs(batch)
                batch_n["observation.environment_state"] = batch_n["observation.state"]
                batch_n["observation.images"] = [
                    batch_n[k] for k in self.policy.config.image_features
                ]
                chunk = self.policy.model(batch_n)[0][:, : self.policy.config.n_action_steps]  # (1,T,D)
                action_tensor = self.policy.unnormalize_outputs({"action": chunk})["action"].squeeze(0)  # (T,D)

                # Trim to dataset action dim (some ACT checkpoints are 11-D)
                action_dim = self.policy.config.action_feature.shape[0]
                action_tensor = action_tensor[:, :action_dim] # (T, dim)
            else:
                raise NotImplementedError(
                    f"Unsupported policy type: {type(self.policy.model)}"
                )

        action_np = action_tensor.cpu().numpy()
        return action_np
