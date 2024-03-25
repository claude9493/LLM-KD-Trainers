from dataclasses import dataclass, field
from typing import Optional
from src.trainer import KD_ARGS_DICT


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    reference: https://github.com/huggingface/transformers/blob/235e5d4991e8a0984aa78db91087b49622c7740e/examples/pytorch/language-modeling/run_clm.py#L71
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )

    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    tokenizer_kwargs: Optional[dict] = field(
        default=None, metadata={"help": "Kwargs for AutoTokenizer.from_pretrained()"}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    checkpoint: Optional[str] = field(
        default=None, metadata={"help": "Resume training from specified checkpoint."}
    )

    lora_config: Optional[dict] = field(
        default=None, metadata={"help": "Config for the LoRA."}
    )
    # def __post_init__(self):
    #     if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
    #         raise ValueError(
    #             "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
    #         )


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the src.dataset).",
            "choices": ["dolly", "samsum", "gmk8k"],
        },
    )


@dataclass
class KDArguments:
    kd_type: str = field(
        default=None,
        metadata={
            "help": "Type of the KD trainer.",
            "choices": list(KD_ARGS_DICT.keys()),
        },
    )
    teacher_model_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the teacher model checkpoint."}
    )
    kd_args: Optional[dict] = field(
        default=None, metadata={"help": "Key-value pairs of the KD arguments."}
    )
    tensor_parallel: bool = field(default=False)

    def __post_init__(self):
        if self.kd_type == None:
            return
        self.kd_type = self.kd_type.lower()
        assert (
            self.kd_type in KD_ARGS_DICT.keys()
        ), f"Unknown KD trainer type {self.kd_type}."
        self.kd_args = KD_ARGS_DICT[self.kd_type](**self.kd_args)
