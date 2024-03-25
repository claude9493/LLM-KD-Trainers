from dataclasses import dataclass, field
from pathlib import Path

columns = ['input_ids', 'attention_mask', 'labels']

@dataclass(frozen=True)
class DataInfo:
    name: str
    path: Path
    prompt_template: str
    label_split: str
    label_column: str
    cutoff_len: int

def generate_and_tokenize_prompt(instance, tokenizer, info, is_test=False):
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=info.cutoff_len,
            padding=True,  # "longest",
            return_tensors=None
        )
        if(
            result['input_ids'][-1] != tokenizer.eos_token_id
            and len(result['input_ids']) < info.cutoff_len
            and add_eos_token
        ):
            result['input_ids'].append(tokenizer.eos_token_id)
            result['attention_mask'].append(1)
        result['labels'] = result['input_ids'].copy()
        return result
    tokenized_full_prompt = tokenize(info.prompt_template.format(**instance))
    tokenized_user_prompt = tokenize(info.label_split.join(info.prompt_template.format(**instance).split(info.label_split)[:-1])  # In case there are few-shot prompts
                                      + info.label_split, add_eos_token=False)
    user_prompt_len = len(tokenized_user_prompt['input_ids'])
    tokenized_full_prompt['labels'] = [-100]*user_prompt_len + tokenized_full_prompt['labels'][user_prompt_len:]
    if is_test:
        tokenized_user_prompt['_id'] = instance['id']
        return tokenized_user_prompt
    
    len_labels = len(tokenizer(instance[info.label_column])['input_ids'])
    tokenized_full_prompt['is_label_complete'] = len(tokenized_full_prompt['labels'][user_prompt_len:]) >= len_labels
    return tokenized_full_prompt

