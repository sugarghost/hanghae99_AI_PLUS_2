import os
import sys
import json
import torch
import logging
import wandb
import random
from itertools import chain
from dataclasses import dataclass, field
from typing import Optional
from sklearn.model_selection import train_test_split
from huggingface_hub import login

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from datasets import Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from transformers.trainer_utils import get_last_checkpoint

wandb.init(project='Hanghae99')
wandb.run.name = 'advance_homework'


@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field(default="openai-community/gpt2")
    torch_dtype: Optional[str] = field(default="auto", metadata={'choices': ['auto', 'bfloat16', 'float16', 'float32']})
    block_size: int = field(default=1024)
    num_workers: Optional[int] = field(default=4)
    corpus_path: str = field(default="./corpus.json")

args = Arguments()

with open(args.corpus_path, "r", encoding="utf-8") as f:
    corpus = json.load(f)

random.shuffle(corpus)

dataset = Dataset.from_dict({
    "question": [item["question"] for item in corpus],
    "answer": [item["answer"] for item in corpus]
})

train_split = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
train_dataset, eval_dataset = train_split['train'], train_split['test']

model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 1024
def formatting_prompts_func(data):
    output_texts = []
    for i in range(len(data['question'])):
        text = f"### Question: {data['question'][i]}\n ### Answer: {data['answer'][i]}"
        output_texts.append(text)
    return output_texts
response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

training_config = SFTConfig(
    output_dir="./homework/advance_output",
    save_total_limit=1,
    logging_steps=300,
    eval_steps=300,
    fp16=True,
    bf16=False,
    do_train=True,
    do_eval=True,
    gradient_accumulation_steps=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_strategy="epoch",
    logging_strategy="steps",
    evaluation_strategy="steps",
    overwrite_output_dir=True,
    num_train_epochs=3
)


trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

checkpoint = get_last_checkpoint(training_config.output_dir)

train_result = trainer.train(resume_from_checkpoint=checkpoint)
trainer.save_model()

metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
