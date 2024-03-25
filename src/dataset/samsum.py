from copy import deepcopy
from loguru import logger
from pathlib import Path
from functools import partial
from datasets import load_dataset, load_from_disk
from .base import DataInfo, generate_and_tokenize_prompt, columns

NAME = "Samsum"

prompt_template = """[INST] <<SYS>>
Use the Input to provide a summary of a conversation.
<</SYS>>

Input:
{dialogue}

Summary:
{summary}
"""

info = DataInfo(
    name="Samsum",
    path="./data/samsum",
    prompt_template=prompt_template,
    label_split="Summary:\n",
    label_column="summary",
    cutoff_len=512
)

generate_and_tokenize_prompt = partial(generate_and_tokenize_prompt, info=info)

dataset = load_from_disk(info.path)
logger.debug(f"Dataset: {dataset}")

def get_train(tokenizer):
    train_data = dataset['train'].map(partial(generate_and_tokenize_prompt, tokenizer=tokenizer), num_proc=1) \
                             .filter(lambda instance: instance['is_label_complete']) \
                             .select_columns(columns) \
                             .with_format(type='torch')
    logger.debug("Train data example:\n" + prompt_template.format(**dataset['train'][0]))
    logger.debug(f"Training data usage: {train_data.num_rows}/{dataset['train'].num_rows}.")
    return train_data

def get_val(tokenizer):
    val_data = dataset['validation'].map(partial(generate_and_tokenize_prompt, tokenizer=tokenizer), num_proc=1) \
                          .filter(lambda instance: instance['is_label_complete']) \
                          .select_columns(columns) \
                          .with_format(type='torch', columns=columns)
    logger.debug(f"Validation data usage: {val_data.num_rows}/{dataset['validation'].num_rows}.")      
    return val_data

def get_test(tokenizer):
    columns_test = deepcopy(columns)
    columns_test.append('id')
    test_data = dataset['test'].map(partial(generate_and_tokenize_prompt, tokenizer=tokenizer, is_test=True), num_proc=1) \
                          .select_columns(columns_test) \
                          .with_format(type='torch', columns=columns, output_all_columns=True)
    logger.debug(f"Test data usage: {test_data.num_rows}/{dataset['test'].num_rows}.")      
    return test_data