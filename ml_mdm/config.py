# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.

from dataclasses import dataclass, field
from typing import Optional, List
import simple_parsing import ArgumentParser
from simple_parsing.wrappers.field_wrapper import ArgumentGenerationMode

from ml_mdm import reader

MODEL_CONFIG_REGISTRY = {}
MODEL_REGISTRY = {}
PIPELINE_CONFIG_REGISTRY = {}
PIPELINE_REGISTRY = {}


def register_model_config(*names):
    arch, main = names

    def register_config_cls(cls):
        MODEL_CONFIG_REGISTRY[arch] = {}
        MODEL_CONFIG_REGISTRY[arch]["model"] = main
        MODEL_CONFIG_REGISTRY[arch]["config"] = cls
        return cls

    return register_config_cls


def register_model(*names):
    def register_model_cls(cls):
        for name in names:
            MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls


def register_pipeline_config(*names):
    def register_pipeline_cls(cls):
        for name in names:
            PIPELINE_CONFIG_REGISTRY[name] = cls
        return cls

    return register_pipeline_cls


def register_pipeline(*names):
    def register_pipeline_cls(cls):
        for name in names:
            PIPELINE_REGISTRY[name] = cls
        return cls

    return register_pipeline_cls


def get_model(name):
    if name not in MODEL_CONFIG_REGISTRY:
        raise NotImplementedError
    return MODEL_REGISTRY[MODEL_CONFIG_REGISTRY[name]["model"]]


def get_pipeline(name):
    if name not in MODEL_CONFIG_REGISTRY:
        raise NotImplementedError
    return PIPELINE_REGISTRY[MODEL_CONFIG_REGISTRY[name]["model"]]


@dataclass
class CommonConfig:
    loglevel: str = field(default="INFO", metadata={"help": "Logging level"})
    device: str = field(default="cuda", metadata={"help": "Device to run on"})
    fp16: int = field(default=0, metadata={"help": "Using fp16 to speed-up training"})
    seed: int = field(default=-1, metadata={"help": "Random number seed"})
    output_dir: str = field(default="", metadata={"help": "Output directory"})

    # Common language configs
    vocab_file: str = field(default="data/c4_wpm.vocab", metadata={"help": "WPM model file"})
    pretrained_vision_file: Optional[str] = field(default=None, metadata={"help": "Choose either ema or non-ema file to start from"})
    categorical_conditioning: int = field(default=0)
    text_model: str = field(default="google/flan-t5-xl", metadata={"help": "text model for encoding the caption"})
    
    model: str = field(default="unet", metadata={
        "help": "Vision model", 
        "choices": list(MODEL_CONFIG_REGISTRY.keys())
    })

    # Currently, only one option
    # Pre-computing text embeddings (we use it in trainer only)
    use_precomputed_text_embeddings: int = field(default=0, metadata={"help": "use precomputed text embeddings for conditioning"})

    # Batch information
    batch_size: int = field(default=2, metadata={"help": "Batch size to use"})
    num_training_steps: int = field(default=850000, metadata={"help": "# of training steps to train for"})
    num_epochs: int = field(default=20000, metadata={"help": "# of epochs to train for"})


@dataclass
class TrainerConfig(CommonConfig):
    multinode: int = field(default=1, metadata={"help": "Whether to use multi node training"})
    local_rank: int = field(default=0, metadata={"help": "for debugging"})
    use_adamw: bool = field(default=False)
    file_list: str = field(default="cifar10-32/train.csv", metadata={
        "help": "List of training files in dataset. "
        "in multinode model, this list is different per device,"
        "otherwise the list is shared with all devices in current node"
    })
    log_freq: int = field(default=100, metadata={"help": "Logging frequency"})
    save_freq: int = field(default=1000, metadata={"help": "Saving frequency"})
    lr: float = field(default=0.001, metadata={"help": "Learning rate"})
    lr_scaling_factor: float = field(default=0.8, metadata={"help": "Factor to reduce maximum learning rate"})
    gradient_clip_norm: float = field(default=2.0, metadata={"help": "Gradient Clip Norm"})
    warmup_steps: int = field(default=5000, metadata={"help": "# of warmup steps"})
    num_gradient_accumulations: int = field(default=1, metadata={"help": "# of steps to accumulate gradients"})
    loss_factor: float = field(default=1, metadata={"help": "multiply the loss by a factor, to simulate old behaviors"})
    resume_from_ema: bool = field(default=False, metadata={"help": "If enabled, by default loading ema checkpoint when resume"})


@dataclass
class SamplerConfig(CommonConfig):
    model_file: str = field(default="", metadata={"help": "Path to saved model"})
    test_file_list: str = field(default="", metadata={"help": "List of test files in dataset"})
    sample_dir: str = field(default="samples", metadata={"help": "directory to keep all samples"})
    eval_freq: int = field(default=1000, metadata={"help": "Minimum Evaluation interval"})
    sample_image_size: int = field(default=-1, metadata={"help": "Size of image"})
    port: int = field(default=19231)
    min_examples: int = field(default=10000, metadata={"help": "minimum number of examples to generate"})


@dataclass
class EvaluatorConfig(CommonConfig):
    test_file_list: str = field(default="", metadata={"help": "List of test files in dataset"})
    sample_dir: str = field(default="samples", metadata={"help": "directory to keep all samples"})
    eval_freq: int = field(default=1000, metadata={"help": "Minimum Evaluation interval"})
    sample_image_size: int = field(default=-1, metadata={"help": "Size of image"})
    num_eval_batches: int = field(default=500, metadata={"help": "# of batches to evaluate on"})


@dataclass
class DemoConfig(CommonConfig):
    sample_dir: str = field(default="samples", metadata={"help": "directory to keep all samples"})
    sample_image_size: int = field(default=-1, metadata={"help": "Size of image"})


@dataclass
class PreloadConfig:
    model: str = field(default="unet", metadata={
        "help": "Vision model", 
        "choices": list(MODEL_CONFIG_REGISTRY.keys())
    })
    # Currently, only one option
    reader_config_file: Optional[str] = field(default=None, metadata={"help": "Config file for reader"})
    model_config_file: Optional[str] = field(default=None, metadata={"help": "Config file for model"})


def get_arguments(args=None, mode="trainer", additional_config_paths: Optional[List[str]] = None):

    from ml_mdm import diffusion, models

    if additional_config_paths is None:
        additional_config_paths = []

    pre_parser = ArgumentParser(description="pre-loading architecture")
    pre_parser.add_arguments(PreloadConfig, dest="preload_config")
    pre_args, _ = pre_parser.parse_known_args(args)
    
    pre_config = pre_args.preload_config
    model_name = pre_config.model
    config_path = additional_config_paths.copy()
    
    if pre_config.reader_config_file is not None:
        config_path.append(pre_config.reader_config_file)
    if pre_config.model_config_file is not None:
        config_path.append(pre_config.model_config_file)

    parser = ArgumentParser(
        argument_generation_mode=ArgumentGenerationMode.BOTH,
        add_config_path_arg=True,
        config_path=config_path,
        description=f"{mode.capitalize()} configuration for ML MDM"
    )

    if mode == "trainer":
        config_cls = TrainerConfig
        dest = "trainer_config"
    elif mode == "sampler":
        config_cls = SamplerConfig
        dest = "sampler_config"
    elif mode == "evaluator":
        config_cls = EvaluatorConfig
        dest = "evaluator_config"
    elif mode == "demo":
        config_cls = DemoConfig
        dest = "demo_config"
    else:
        raise NotImplementedError(f"Unsupported mode: {mode}")

    parser.add_arguments(config_cls, dest=dest)

    # Add submodule args
    parser.add_arguments(reader.ReaderConfig, dest="reader_config")

    # Add vision model configs
    parser.add_arguments(
        MODEL_CONFIG_REGISTRY[model_name]["config"], 
        dest="unet_config"
    )
    parser.add_arguments(
        PIPELINE_CONFIG_REGISTRY[MODEL_CONFIG_REGISTRY[model_name]["model"]],
        dest="diffusion_config"
    )

    parser.confilct_resolver_max_attempts = 5000

    # Parse known args
    args, _ = parser.parse_known_args(args)
    return args