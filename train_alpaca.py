#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import json
from datasets import load_dataset
from multiprocessing import cpu_count
import random
random.seed(42)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:"
    ),
    "prompt_input_with_demo": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}\n\n {}\n\n### Input:\n{}\n\n### Response:"
    ),
}

TASK_PROMPTS = {p['task_name']:[p['prompt_template'],p['input_template'],p['label_map']] for p in json.load(open("prompt_and_demo/task_prompts.json"))}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    train_subset: str = field(default=None, metadata={"help": "train subset name if loading from huggingface datasets"})
    test_subset: str = field(default=None, metadata={"help": "test subset name if loading from huggingface datasets"})

    prompt_type: str = field(default="attribution-no-definition", metadata={"help": "prompt engineering: which prompt to use"})
    input_has_query: bool = field(default=True, metadata={"help": "whether to include query in the input for evaluating attribution"})
    num_train_samples: int = field(
        default=-1,
        metadata={"help": "number of train samples."},
    )
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


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

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class SupervisedDataset(Dataset):
    def __init__(self, data_args: str, tokenizer: transformers.PreTrainedTokenizer, split='train'):
        self.tokenizer = tokenizer
        self.dataset_path = data_args.data_path
        self.subset_name = data_args.train_subset if split=='train' else data_args.test_subset
        self.prompt_template, self.input_template, _ = TASK_PROMPTS[data_args.prompt_type]
        self.prompt_input = PROMPT_DICT["prompt_input"]
        self.num_train_samples = data_args.num_train_samples
        self.input_has_query = data_args.input_has_query
        self.split = split
        self.input_ids, self.labels = self.load_and_tokenize_dataset()

    def _tokenize_fn(self, text: str) -> Dict:
        """Tokenize a list of strings."""
        tokenized = self.tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )

        input_ids = labels = tokenized.input_ids[0]
        input_ids_lens = labels_lens = tokenized.input_ids.ne(self.tokenizer.pad_token_id).sum().item()

        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )


    def process_function(self, example):

        label_name = 'label'

        if self.input_has_query:
            query = example['query'] if example['query'] and example['query'] not in ["nan",""] else ""
            answer = example['answer'] if example['answer'] and example['answer'] not in ["nan",""] else ""
            input = self.input_template.format(query + " " + answer, example['reference'])

        else:
            answer = example['answer'] if example['answer'] and example['answer'] not in ["nan",""] else ""
            input = self.input_template.format(answer, example['reference'])

        source = self.prompt_input.format(self.prompt_template, input)
        target = f"{example[label_name]} {self.tokenizer.eos_token}"
        source_target = source + target

        example_tokenized = self._tokenize_fn(source_target)
        source_tokenized = self._tokenize_fn(source)

        input_ids = example_tokenized["input_ids"]
        label = copy.deepcopy(input_ids)

        label[:source_tokenized["input_ids_lens"]] = IGNORE_INDEX

        return {"input_ids": input_ids, "labels": label}

    def load_and_tokenize_dataset(self):
        # Load the dataset
        try:
            dataset = load_dataset(self.dataset_path, self.subset_name)[self.split]
        except:
            dataset = load_dataset(self.dataset_path.split(".")[-1], data_files=self.dataset_path)['train']
        # If num_train_samples is specified and less than the total dataset length
        if 0 < self.num_train_samples < len(dataset):
            dataset = dataset.select(range(self.num_train_samples))

        # Tokenize the dataset in a batched way
        tokenized_dataset = dataset.map(self.process_function, batched=False, num_proc=cpu_count()-1)

        input_ids = [torch.tensor(d,dtype=torch.int64) for d in tokenized_dataset['input_ids']]
        labels = [torch.tensor(l,dtype=torch.int64) for l in tokenized_dataset['labels']]

        #debug
        logging.info(f"{self.tokenizer.decode(input_ids[0],skip_special_tokens=True)}")

        return input_ids, labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args, split='train')

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, data_collator=data_collator)



def train():
    transformers.logging.set_verbosity_info()
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    #Suppress wandb
    training_args.report_to = []

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
