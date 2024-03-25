# CUDA_VISIBLE_DEVICES=0 python inference-vllm.py -m results/dolly/gpt2-base-sft/checkpoint-4290 -t ../models/gpt2/base/ -d dolly --metric rouge


import json
import os
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer

import evaluate
import pandas as pd
import torch
import vllm
from datasets import load_from_disk
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq

from src.dataset import get_dataset

parser = ArgumentParser("LLM inference")
# testing arguments
parser.add_argument("-m", "--model", type=str, required=True)
parser.add_argument("-t", "--tokenizer", type=str, default=None)
parser.add_argument("-d", "--data", type=str, default="samsum")
parser.add_argument("--lora", type=str, default=None)
parser.add_argument(
    "--metric", type=str, default="rouge", choices=["rouge", "accuracy"]
)
parser.add_argument("--seed", type=int, default=2023)
# generation arguments
parser.add_argument("--max-tokens", type=int, default=512)
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
TOKENIZER_PATH = Path(args.tokenizer) if args.tokenizer else MODEL_PATH
WORK_DIR = Path(args.lora) if args.lora else Path(args.model)

DATA_NAME = args.data
_data_class = get_dataset(DATA_NAME)
dataset = _data_class.dataset["test"]

logger.debug(dataset)

label_column = _data_class.info.label_column
prompt_template = _data_class.info.prompt_template
LABEL_SPLIT = _data_class.info.label_split
logger.debug("Test data example:\n" + prompt_template.format(**dataset[0]))

model = vllm.LLM(model=str(MODEL_PATH), tokenizer=str(TOKENIZER_PATH), seed=SEED)

if args.lora:
    from src.utils import lora_merge_unmerge_state_dict
    from safetensors.torch import load_file
    from peft.config import PeftConfig

    lora_state_dict = load_file(os.path.join(args.lora, "adapter_model.safetensors"))
    peft_config = PeftConfig.from_pretrained(args.lora)
    lora_merge_unmerge_state_dict(model, lora_state_dict, peft_config, merge=True)
    logger.debug("LoRA adapter merged.")

tokenizer = model.get_tokenizer()
tokenizer.padding_side = "left"
tokenizer.pad_token_id = tokenizer.eos_token_id

if "id" not in dataset.column_names:
    dataset = dataset.add_column("id", list(range(dataset.num_rows)))


@dataclass
class idCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        _id = (
            [feature.pop("id") for feature in features]
            if "id" in features[0].keys()
            else None
        )
        prompts = [
            prompt_template.format(**instance).split(LABEL_SPLIT)[0] + LABEL_SPLIT
            for instance in features
        ]
        return _id, prompts


data_collator = idCollator(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    return_tensors="pt",
    pad_to_multiple_of=8,
)

dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=8)

beam_search_params = vllm.SamplingParams(
    max_tokens=256,
    temperature=0,
    use_beam_search=True,
    length_penalty=0.75,
    best_of=3,
    top_p=1.0,
    top_k=-1,
)

sampling_params = vllm.SamplingParams(
    max_tokens=args.max_tokens,  # 512,
    top_k=args.top_k,  # -1,
    top_p=args.top_p,  # 1.0,
    temperature=args.temperature,  # 1.0,
)

torch.cuda.empty_cache()
predictions = []

start = timer()
for it, (_ids, data) in tqdm(enumerate(dataloader), total=len(dataloader)):
    results = model.generate(data, sampling_params, use_tqdm=False)
    predictions.extend(
        [
            {"id": _id, label_column: pred.strip()}
            for _id, pred in zip(_ids, [result.outputs[0].text for result in results])
        ]
    )

end = timer()
logger.info(f"Testing time: {end-start:.6f}s.")

logger.info(f"Example prediction: {predictions[0]}")

suffix = time.time_ns()
with open(WORK_DIR / f"predictions-{DATA_NAME}-{suffix}.json", "w") as f:
    json.dump(predictions, f)

pred_ref = pd.merge(
    pd.DataFrame.from_records(predictions),
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

    metrics = acc.compute(
        predictions=pred_ref[label_column + "_pred"].map(ans_parse),
        references=pred_ref[label_column + "_ref"].map(ans_parse),
    )

logger.info(metrics)

with open(WORK_DIR / f"metrics-{DATA_NAME}-{suffix}.json", "w") as f:
    json.dump(metrics, f)
