# NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 deepspeed --include localhost:0,1 train.py configs/your_config_file.yaml

import os
import sys
import time
from loguru import logger
from pathlib import Path
import torch
from transformers import HfArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from src.dataset import get_dataset
from src.utils.archive import ArchiveScriptCallback
from arguments import ModelArguments, DataArguments

parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))

print(sys.argv)
config_file = None
for _arg in sys.argv:
    if _arg.endswith(".yaml") and _arg.startswith("configs"):
        config_file = os.path.abspath(_arg)
        break
if config_file:
    model_args, data_args, training_args = parser.parse_yaml_file(
        yaml_file=config_file, allow_extra_keys=False
    )
    logger.debug(f"Config file: {os.path.abspath(sys.argv[-1])}")
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


os.environ["WANDB_PROJECT"] = "LLM.KD"
training_args.output_dir = f"{training_args.output_dir}-{time.time_ns()}"
training_args.run_name = f"{data_args.dataset_name}-{training_args.output_dir}"

# Handle arguments
MODEL_PATH = model_args.model_name_or_path
WORK_DIR = Path("results") / data_args.dataset_name / training_args.output_dir
training_args.output_dir = str(WORK_DIR)
training_args.logging_dir = str(
    WORK_DIR / __import__("transformers").training_args.default_logdir()
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, **model_args.tokenizer_kwargs)
tokenizer.pad_token = tokenizer.eos_token

logger.debug(f"The padding token id is {tokenizer.pad_token_id}")

# Load data
_data_class = get_dataset(data_args.dataset_name)
train_data = _data_class.get_train(tokenizer)
val_data = _data_class.get_val(tokenizer)

# Load model
torch_dtype = (
    model_args.torch_dtype
    if model_args.torch_dtype in ["auto", None]
    else getattr(torch, model_args.torch_dtype)
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch_dtype, load_in_8bit=False, use_cache=False
)


logger.debug(
    f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)

if model_args.lora_config is not None:
    from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
    from src.utils import find_all_linear_names

    lora_config = LoraConfig(**model_args.lora_config)
    if lora_config.target_modules == "all":
        lora_config.target_modules = find_all_linear_names(model)
    logger.debug(f"Lora target modules: {lora_config.target_modules}")
    model = get_peft_model(prepare_model_for_kbit_training(model), lora_config)
    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.debug(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
    )


# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model,
    label_pad_token_id=tokenizer.pad_token_id,
    return_tensors="pt",
    pad_to_multiple_of=8,
)

# Trainer
training_args.eval_steps /= training_args.num_train_epochs
training_args.save_steps /= training_args.num_train_epochs
training_args.logging_steps /= training_args.num_train_epochs
training_args.label_names = ["labels"]


def compute_metrics(pred):
    # To-Do: generate and compute rougeL during evaluation
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    pass


archive_script_callback = ArchiveScriptCallback(
    training_args.output_dir, config_file=config_file
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=val_data,
    callbacks=[archive_script_callback],
)

logger.debug(
    "Trainer callbacks: " + trainer.callback_handler.callback_list.replace("\n", ", ")
)
logger.debug("Start Training!!!")
trainer.train(resume_from_checkpoint=model_args.checkpoint)
