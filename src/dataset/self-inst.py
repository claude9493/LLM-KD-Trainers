from loguru import logger
from copy import deepcopy
from pathlib import Path
from functools import partial
from datasets import load_dataset, load_from_disk, concatenate_datasets
from .base import generate_and_tokenize_prompt, columns, DataInfo

NAME = "Self-Inst"

# prompt_template = (  # with context
#     "Below is an instruction that describes a task, paired with an input that provides further context. "
#     "Write a response that appropriately completes the request.\n\n"
#     "### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n{response}"
# )

# prompt_template2 = (  # No context
#     "Below is an instruction that describes a task. "
#     "Write a response that appropriately completes the request.\n\n"
#     "### Instruction:\n{instruction}\n\n### Response:\n{response}"
# )

prompt_template = "{text}"

prompt_template_mini = "{prompt}{output}"

info = DataInfo(  # mini version
    name="dolly",
    path=Path("./data/mini/self-inst"),
    prompt_template=prompt_template_mini,
    label_split="### Response:\n",
    label_column="output",
    cutoff_len=512
)

# info = DataInfo(  # Raw
#     name="dolly",
#     path=Path("./data/self-inst"),
#     prompt_template=prompt_template,
#     label_split="### Response:\n",
#     label_column="response",
#     cutoff_len=512
# )

dataset = load_from_disk(info.path)
logger.debug(f"Dataset: {dataset}")

def get_train(tokenizer):
    gtp = partial(generate_and_tokenize_prompt, info=info, tokenizer=tokenizer)
    train_data = dataset['train'].map(gtp, num_proc=1) \
            .filter(lambda instance: instance['is_label_complete']) \
            .select_columns(columns) \
            .with_format(type='torch')
    logger.debug("Train data example:\n" + info.prompt_template.format(**dataset['train'][0]))
    logger.debug(f"Training data usage: {train_data.num_rows}/{dataset['train'].num_rows}.")
    return train_data

def get_val(tokenizer):
    gtp = partial(generate_and_tokenize_prompt, info=info, tokenizer=tokenizer)
    val_data = dataset['validation'].map(gtp, num_proc=1) \
            .filter(lambda instance: instance['is_label_complete']) \
            .select_columns(columns) \
            .with_format(type='torch')
    logger.debug(f"Validation data usage: {val_data.num_rows}/{dataset['validation'].num_rows}.")      
    return val_data

def get_test(tokenizer):
    columns_test = deepcopy(columns)
    columns_test.append('id')
    gtp = partial(generate_and_tokenize_prompt, info=info, tokenizer=tokenizer, is_test=True)
    if 'id' not in dataset['test'].column_names:
        dataset['test'] = dataset['test'].add_column('id', list(range(dataset['test'].num_rows)))
    test_data = dataset['test'].map(gtp, num_proc=1) \
            .select_columns(columns_test) \
            .with_format(type='torch', columns=columns, output_all_columns=True)

    logger.debug(f"Testing data usage: {test_data.num_rows}/{dataset['validation'].num_rows}.")      
    return test_data