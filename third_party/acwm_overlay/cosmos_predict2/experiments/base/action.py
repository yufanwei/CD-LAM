# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict, LazyConfig
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

# Use the post-trained checkpoint which has the correct experiment reference
DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[
    ModelKey(post_trained=False)
]  # This uses post_trained=True by default
DEFAULT_CHECKPOINT_14B = MODEL_CHECKPOINTS[ModelKey(post_trained=False, size="14B")]
_SKIP_14B_EXPERIMENT_CONFIGS = (
    os.environ.get("COSMOS_SKIP_14B_EXPERIMENT_CONFIGS") == "1"
)
_SKIP_DEFAULT_CHECKPOINT_DOWNLOADS = (
    os.environ.get("COSMOS_SKIP_DEFAULT_CHECKPOINT_DOWNLOADS") == "1"
)


def load_experiment_config(experiment_name: str, default_config: LazyDict) -> LazyDict:
    """
    Load experiment configuration from YAML file and merge with default Python configuration.

    Args:
        experiment_name: Name of the experiment to load.
        default_config: Default Python configuration to use as base.

    Returns:
        LazyDict containing the merged experiment configuration.
    """

    # Get the directory of this file
    project_dir = Path(__file__).parent.parent.parent.parent
    yaml_file = project_dir / "configs" / f"{experiment_name}.yaml"

    if yaml_file.exists():
        # Load YAML overrides
        yaml_config = LazyConfig.load(str(yaml_file))

        # Recursively merge YAML overrides into the default config
        def merge_configs(default_dict, override_dict):
            for key, value in override_dict.items():
                if (
                    key in default_dict
                    and isinstance(default_dict[key], dict)
                    and isinstance(value, dict)
                ):
                    merge_configs(default_dict[key], value)
                else:
                    default_dict[key] = value

        # Convert to regular dict for merging, then back to LazyDict
        default_dict = OmegaConf.to_container(default_config, resolve=True)
        override_dict = OmegaConf.to_container(yaml_config, resolve=True)
        merge_configs(default_dict, override_dict)

        return LazyDict(default_dict, flags={"allow_objects": True})
    else:
        # No YAML file, return default config
        return default_config


"""
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py  -- experiment=ac_reason_embeddings_rectified_flow_2b_256_320
"""
_default_groot_config = LazyDict(
    dict(
        defaults=[
            DEFAULT_CHECKPOINT.experiment,
            {"override /model": "action_conditioned_video2world_fsdp_rectified_flow"},
            {"override /net": "cosmos_v1_2B_action_chunk_conditioned"},
            {"override /conditioner": "action_conditioned_video_conditioner"},
            {"override /data_train": "dreamdojo_13frame_480_640_train"},
            {"override /data_val": "dreamdojo_13frame_480_640_val"},
            "_self_",
        ],
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="cosmos_predict_v2p5",
            name="2b_groot_action_conditioned",
        ),
        optimizer=dict(
            # lr=2 ** (-14.5),  # 2**(-14.5) = 3.0517578125e-05
            lr=16e-5,
            weight_decay=0.1,
        ),
        checkpoint=dict(
            save_iter=10_000,
            # pyrefly: ignore  # missing-attribute
            load_path=(
                os.environ.get("COSMOS_LOCAL_2B_CHECKPOINT", "")
                if _SKIP_DEFAULT_CHECKPOINT_DOWNLOADS
                else get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri)
            ),
            load_training_state=False,
            strict_resume=False,
            load_from_object_store=dict(
                enabled=False,
            ),
            save_to_object_store=dict(
                enabled=False,
            ),
        ),
        trainer=dict(
            max_iter=100_000,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=5000,
                    do_x0_prediction=False,
                    guidance=[0],
                    fps=16,
                    save_s3=False,
                ),
                every_n_sample_ema=dict(
                    every_n=5000,
                    do_x0_prediction=False,
                    guidance=[0],
                    fps=16,
                    save_s3=False,
                ),
                heart_beat=dict(
                    save_s3=False,
                ),
                iter_speed=dict(
                    hit_thres=100,
                    save_s3=False,
                ),
                device_monitor=dict(
                    save_s3=False,
                ),
                wandb=dict(
                    save_s3=False,
                ),
                wandb_10x=dict(
                    save_s3=False,
                ),
                dataloader_speed=dict(
                    save_s3=False,
                ),
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        model=dict(
            config=dict(
                # Enable LoRA training
                # use_lora=True,
                # lora_rank=32,              # Rank of LoRA adaptation matrices
                # lora_alpha=32,             # LoRA scaling parameter
                # lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
                # init_lora_weights=True,    # Properly initialize LoRA weights
                # NOTE: this should be 1 for the action conditioned model
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                # overwrite the probs to disable random num of conditional frames
                conditional_frames_probs=None,
                state_t=1 + 12 // 4,
                net=dict(
                    action_dim=29,
                    temporal_compression_ratio=4,
                    num_action_per_chunk=12,
                    zero_init_action_embedder=False,
                ),
            ),
        ),
        dataloader_train=dict(
            batch_size=4,
            dataset=dict(
                num_frames=13,
                dataset_path="datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot",
                data_split="train",
            ),
        ),
    ),
    flags={"allow_objects": True},
)

_default_groot_config_14b = LazyDict(
    dict(
        defaults=[
            DEFAULT_CHECKPOINT_14B.experiment,
            {"override /model": "action_conditioned_video2world_fsdp_rectified_flow"},
            {"override /net": "cosmos_v1_14B_action_chunk_conditioned"},
            {"override /conditioner": "action_conditioned_video_conditioner"},
            {"override /data_train": "dreamdojo_13frame_480_640_train"},
            {"override /data_val": "dreamdojo_13frame_480_640_val"},
            "_self_",
        ],
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="cosmos_predict_v2p5",
            name="2b_groot_action_conditioned",
        ),
        optimizer=dict(
            # lr=2 ** (-14.5),  # 2**(-14.5) = 3.0517578125e-05
            lr=16e-5,
            weight_decay=0.1,
        ),
        checkpoint=dict(
            save_iter=5_000,
            # pyrefly: ignore  # missing-attribute
            load_path=(
                os.environ.get("COSMOS_LOCAL_14B_CHECKPOINT", "")
                if _SKIP_DEFAULT_CHECKPOINT_DOWNLOADS or _SKIP_14B_EXPERIMENT_CONFIGS
                else get_checkpoint_path(DEFAULT_CHECKPOINT_14B.s3.uri)
            ),
            load_training_state=False,
            strict_resume=False,
            load_from_object_store=dict(
                enabled=False,
            ),
            save_to_object_store=dict(
                enabled=False,
            ),
        ),
        trainer=dict(
            max_iter=1000000,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=5000,
                    do_x0_prediction=False,
                    guidance=[0],
                    fps=16,
                    save_s3=False,
                ),
                every_n_sample_ema=dict(
                    every_n=5000,
                    do_x0_prediction=False,
                    guidance=[0],
                    fps=16,
                    save_s3=False,
                ),
                heart_beat=dict(
                    save_s3=False,
                ),
                iter_speed=dict(
                    hit_thres=100,
                    save_s3=False,
                ),
                device_monitor=dict(
                    save_s3=False,
                ),
                wandb=dict(
                    save_s3=False,
                ),
                wandb_10x=dict(
                    save_s3=False,
                ),
                dataloader_speed=dict(
                    save_s3=False,
                ),
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        model=dict(
            config=dict(
                # Enable LoRA training
                # use_lora=True,
                # lora_rank=32,              # Rank of LoRA adaptation matrices
                # lora_alpha=32,             # LoRA scaling parameter
                # lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
                # init_lora_weights=True,    # Properly initialize LoRA weights
                # NOTE: this should be 1 for the action conditioned model
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                # overwrite the probs to disable random num of conditional frames
                conditional_frames_probs=None,
                state_t=1 + 12 // 4,
                net=dict(
                    action_dim=29,
                    temporal_compression_ratio=4,
                    num_action_per_chunk=12,
                    zero_init_action_embedder=False,
                ),
            ),
        ),
        dataloader_train=dict(
            batch_size=4,
            dataset=dict(
                num_frames=13,
                dataset_path="datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot",
                data_split="train",
            ),
        ),
    ),
    flags={"allow_objects": True},
)

# Automatically load all config files from the configs directory
_configs_dir = Path(__file__).parent.parent.parent.parent / "configs"
_experiment_configs = {}
_public_aliases = {
    "2b_480_640_pretrain": "cdlam_pretrain",
    "2b_480_640_agibot": "cdlam_posttrain",
}

# Scan for all YAML files in the configs directory
for yaml_file in sorted(_configs_dir.glob("*.yaml")):
    # Extract experiment name (filename without .yaml extension)
    experiment_name = yaml_file.stem
    if _SKIP_14B_EXPERIMENT_CONFIGS and "14b" in experiment_name:
        continue

    # Preserve the upstream experiment-registry prefix required by Hydra configs.
    var_name = f"dreamdojo_{experiment_name}"

    # Load the config and store in both dict and globals for backward compatibility
    if "14b" in experiment_name:
        config = load_experiment_config(experiment_name, _default_groot_config_14b)
    else:
        config = load_experiment_config(experiment_name, _default_groot_config)
    _experiment_configs[var_name] = config
    globals()[var_name] = config
    public_alias = _public_aliases.get(experiment_name)
    if public_alias:
        _experiment_configs[public_alias] = config
        globals()[public_alias] = config

cs = ConfigStore.instance()

# Register all dynamically loaded configs
for var_name, config_item in _experiment_configs.items():
    cs.store(
        group="experiment", package="_global_", name=var_name.lower(), node=config_item
    )
