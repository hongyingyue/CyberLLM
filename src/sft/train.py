
import os
import sys
import copy
import json
import logging
import torch
from torch import nn
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import transformers
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    PreTrainedModel,
    default_data_collator,
)


IGNORE_INDEX = -100

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="trendmicro-ailab/Llama-Primus-Base"
    )
    use_flash_attn: bool = field(
        default=False,
        metadata={"help": "Enable FlashAttention-2 for faster training."}
    )
    

@dataclass
class DataArguments:
    data_path: str = field(
        default="trendmicro-ailab/Primus-Instruct",
        metadata={"help": "Path to the training data."},
    )
    data_cache_dir: Optional[str] = field(default=None, metadata={"help": "The datasets processed stored"})
    max_seq_length: Optional[int] = field(default=512)
    source_length: int = field(default=512)
    target_length: int = field(default=512)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default="./sft/")
    cache_dir: Optional[str] = field(default=None)
    label_names: List[str] = field(default_factory=lambda: ["labels"])  # PeftModel hides the base model so you need to re-specify the labels.
    num_train_epochs: int = (field(default=1))
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=4)
    optim: str = field(default="adamw_torch")  # paged_adamw_32bit
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = field(default=True)
    lora_rank: int = field(default=64, metadata={"help": "The rank of lora."})
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    train_attention_params_only: bool = field(default=False, metadata={
        "help": "Whether to train attention parameters only."}
    )
    save_steps: int = field(default=100)
    logging_steps: int = field(default=50)
    learning_rate: float = field(default=2e-4)
    max_grad_norm: float = field(default=0.3)
    # max_steps: int = field(default=1000)
    warmup_ratio: float = field(default=0.)
    lr_scheduler_type: str = field(default="cosine")
    remove_unused_columns: bool = field(default=False)
    group_by_length: bool = field(
        default=False,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )


def print_args(args, name='arguments'):
    """Print arguments."""
    if torch.distributed.get_rank() == 0:
        print(f'------------------------ {name} ------------------------', flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print(f'-------------------- end of {name} ---------------------', flush=True)


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)

    valid_examples = []
    valid_labels = []
    valid_input_ids = []

    for i, (label, source_len) in enumerate(zip(labels, sources_tokenized["input_ids_lens"])):
        # Create a copy to avoid modifying the original
        label_copy = copy.deepcopy(label)
        label_copy[:source_len] = IGNORE_INDEX

        # Check if there are any valid labels (non-IGNORE_INDEX tokens)
        valid_token_count = (label_copy != IGNORE_INDEX).sum()

        if valid_token_count == 0:
            print(f"Invalid example {i}: All tokens are IGNORE_INDEX")
            print(f"Source: {sources[i][:30]}")
            print(f"Target: {targets[i][:30]}")
            print(f"Source length: {source_len}, Total length: {len(label_copy)}")
            continue

        valid_examples.append(i)
        valid_labels.append(label_copy)
        valid_input_ids.append(input_ids[i])

    print(f"Kept {len(valid_examples)} out of {len(sources)} examples")
    return dict(input_ids=valid_input_ids, labels=valid_labels)


class SupervisedDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len):
        tokenizer.model_max_length = max_len
        logging.warning("Formatting inputs...")
        sources = []
        targets = []

        for row in dataset:
            messages = row["messages"]

            user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
            assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), None)

            if user_msg and assistant_msg:
                sources.append(user_msg)
                targets.append(assistant_msg + tokenizer.eos_token)

        print(f"Found {len(sources)} examples before preprocessing")
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict.get("attention_mask")
        print(f"Final dataset size: {len(self.input_ids)} examples")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return dict(input_ids=self.input_ids[idx], labels=self.labels[idx])


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def find_all_linear_names(model: PreTrainedModel, linear_type: Optional[object] = None) -> List[str]:
    """
    Find all linear layer names

    :param model: PreTrainedModel
    :param linear_type: Optional[object] = None, linear type, such as nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt

    :return: List[str], linear layer names
    """
    if linear_type is None:
        linear_type = nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_type):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


# @dataclass
# class DataCollatorForSupervisedDataset(object):
#     """Collate examples for supervised fine-tuning."""

#     tokenizer: transformers.PreTrainedTokenizer

#     def __call__(self, instances):
#         input_ids = [instance['input_ids'] for instance in instances]
#         labels = [instance['labels'] for instance in instances]
#         input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
#         labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
#         return dict(
#             input_ids=input_ids,
#             labels=labels,
#             attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
#         )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = outputs.loss  # Assumes model returns loss
        return (loss, outputs) if return_outputs else loss


def get_optimizer_grouped_parameters(
        model,
        weight_decay,
        lora_lr=5e-4,
        no_decay_name_list=["bias", "LayerNorm.weight"],
        lora_name_list=["lora_right_weight", "lora_left_weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n
                                                    for nd in lora_name_list))
            ],
            "weight_decay":
                weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n
                                                for nd in lora_name_list))
            ],
            "weight_decay":
                weight_decay,
            "lr":
                lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
                0.0,
        },
    ]
    if not optimizer_grouped_parameters[1]["params"]:
        optimizer_grouped_parameters.pop(1)
    return optimizer_grouped_parameters


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print_args(model_args, 'model arguments')
    print_args(data_args, 'data arguments')
    print_args(training_args, 'training arguments')

    additional_special_tokens = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        encode_special_tokens=True,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )
    if additional_special_tokens is not None:
        additional_special_tokens = [additional_special_tokens] if isinstance(additional_special_tokens,
                                                                              str) else additional_special_tokens
        tokenizer.add_special_tokens(
            {'additional_special_tokens': additional_special_tokens})

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        # device_map='auto',  # can't train a model that has been loaded with `device_map='auto'` in any distributed mode.
    )
    model.enable_input_require_grads()  # RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
    if training_args.use_lora:
        from peft import LoraConfig, PeftModel, get_peft_model, TaskType
        
        TARGET_MODULES = find_all_linear_names(model)
        config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=TARGET_MODULES,
            lora_dropout=training_args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()    
        model.train()

    dataset = load_dataset("trendmicro-ailab/Primus-Instruct")
    dataset = dataset["train"]
    split_data = dataset.train_test_split(test_size=0.2, seed=42)
    train_data = split_data['train']
    eval_data = split_data['test']
    print(f"train data size: {len(train_data)}")
    print(f"eval data size: {len(eval_data)}")

    train_dataset = SupervisedDataset(train_data, tokenizer, max_len=data_args.max_seq_length)
    eval_dataset = SupervisedDataset(eval_data, tokenizer, max_len=data_args.max_seq_length)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    # test
    batch = next(iter(DataLoader(train_dataset, batch_size=1, collate_fn=data_collator)))
    for k, v in batch.items():
        print(f"{k}: {v.shape} | device: {v.device}")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    model.config.use_cache = False
    trainer.train()

    # save_model(args, model, tokenizer, f"epoch_model")

if __name__ == "__main__":
    train()
