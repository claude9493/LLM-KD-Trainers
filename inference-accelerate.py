from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from dataclasses import dataclass
import torch
import evaluate
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    DataCollatorForSeq2Seq,
)
from src.dataset import get_dataset

from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import unwrap_model
from statistics import mean
import torch, time, json

accelerator = Accelerator()


def write_pretty_json(file_path, data):
    import json

    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)


parser = ArgumentParser("LLM inference")
# testing arguments
parser.add_argument("-m", "--model", type=str, required=True)
parser.add_argument("-t", "--tokenizer", type=str, required=True)
parser.add_argument("-d", "--data", type=str, default="samsum")
parser.add_argument(
    "--metric", type=str, default="rouge", choices=["rouge", "accuracy"]
)
parser.add_argument("--seed", type=int, default=2023)
parser.add_argument("--bs", type=int, default=96)  # batch_size
# generation arguments
parser.add_argument("--max-tokens", type=int, default=256)
parser.add_argument("--top-k", type=int, default=-1)
parser.add_argument("--top-p", type=float, default=1.0)
parser.add_argument("--temperature", type=float, default=1.0)

args = parser.parse_args()

SEED = args.seed
evaluation_metric = args.metric

__import__("random").seed(SEED)
__import__("numpy").random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MODEL_PATH = Path(args.model)
TOKENIZER_PATH = Path(args.tokenizer)
WORK_DIR = Path(args.model)

DATA_NAME = args.data

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
tokenizer.padding_side = "left"
tokenizer.pad_token_id = tokenizer.eos_token_id
logger.debug(f"The padding token id is {tokenizer.pad_token_id}")

_data_class = get_dataset(DATA_NAME)
LABEL_SPLIT = _data_class.info.label_split
label_column = _data_class.info.label_column
test_data = _data_class.get_test(tokenizer)  # .select(range(10))

logger.debug(f"Test data: {test_data}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    load_in_8bit=False,
    device_map={"": accelerator.process_index},
    torch_dtype=torch.float16,
)

label_pad_token_id = -100


@dataclass
class idCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        _id = (
            [feature.pop("id") for feature in features]
            if "id" in features[0].keys()
            else None
        )
        return _id, super().__call__(features, return_tensors)


data_collator = idCollator(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    return_tensors="pt",
    pad_to_multiple_of=8,
)

dataloader = DataLoader(test_data, collate_fn=data_collator, batch_size=args.bs)

generation_config = GenerationConfig(
    max_length=1024,
    max_new_token=512,
    min_length=None,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    return_dict_in_generate=True,
    output_scores=False,
)

results = []

model, dataloader = accelerator.prepare(model, dataloader)

# sync GPUs and start the timer
accelerator.wait_for_everyone()
start = time.time()

for _, (_ids, data) in tqdm(enumerate(dataloader), total=len(dataloader)):
    with torch.no_grad():
        out = (
            unwrap_model(model)
            .generate(**data, generation_config=generation_config)
            .sequences
        )
    # print(out)
    results.extend(
        [
            {"id": _id, label_column: summary}
            for _id, summary in zip(_ids, tokenizer.batch_decode(out))
        ]
    )

accelerator.wait_for_everyone()

# collect results from all the GPUs
results_gathered = gather_object(results)

if accelerator.is_main_process:
    timediff = time.time() - start

    print(f"time: {timediff}")
    dataset = _data_class.dataset["test"]
    id_type = str if dataset.features["id"].dtype == "string" else int
    predictions = []

    for result in tqdm(results_gathered):
        if LABEL_SPLIT in result[label_column]:
            predictions.append(
                {
                    "id": id_type(result["id"]),
                    label_column: result[label_column]
                    .replace(tokenizer.pad_token, "")
                    .replace(tokenizer.unk_token, "")
                    .split(LABEL_SPLIT)[-1],
                }
            )

    suffix = time.time_ns()
    with open(WORK_DIR / f"predictions-{DATA_NAME}-{suffix}.json", "w") as f:
        json.dump(predictions, f)

    pred_ref = pd.merge(
        pd.DataFrame.from_records(predictions).drop_duplicates(subset=["id"]),
        pd.DataFrame.from_records(dataset, columns=["id", label_column]),
        on="id",
        how="inner",
        suffixes=["_pred", "_ref"],
    )

    if evaluation_metric == "rouge":
        rouge = evaluate.load("src/metrics/rouge")
        metrics = rouge.compute(
            predictions=pred_ref[label_column + "_pred"],
            references=pred_ref[label_column + "_ref"],
        )
    elif evaluation_metric == "accuracy":
        acc = evaluate.load("src/metrics/accuracy")

        def ans_parse(ans_str: str) -> int:
            try:
                ans = int(ans_str.split("####")[-1].strip().replace(",", ""))
            except Exception:
                ans = -9999
            return ans

        print("predictions:")
        print(pred_ref[label_column + "_pred"].map(ans_parse))
        print("Reference:")
        print(pred_ref[label_column + "_ref"].map(ans_parse))
        metrics = acc.compute(
            predictions=pred_ref[label_column + "_pred"].map(ans_parse),
            references=pred_ref[label_column + "_ref"].map(ans_parse),
        )

    logger.info(metrics)

    with open(WORK_DIR / f"metrics-{DATA_NAME}-{suffix}.json", "w") as f:
        json.dump(metrics, f)
