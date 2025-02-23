import torch
import os
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt
from datasets import load_dataset
from utils.constants import max_seq_length

def train_model(model, dataset, tokenizer):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps=60,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",  # Use this for WandB etc
        ),
    )
    
    # We also use Unsloth's `train_on_completions` method to only train on the assistant outputs and ignore the loss on the user's inputs.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    # We verify masking is actually done:

    print(tokenizer.decode(trainer.train_dataset[5]["input_ids"]))

    space = tokenizer(" ", add_special_tokens=False).input_ids[0]
    print(
        tokenizer.decode(
            [space if x == -100 else x for x in trainer.train_dataset[5]["labels"]]
        )
    )
    
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    # @title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    
    return model, tokenizer

def data_prep(tokenizer):
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )


    def formatting_prompts_func(examples):
        # check if the examples contain the key "conversations" and replace examples with the value of the key
        if "conversations" in examples:
            examples = examples["conversations"]
        texts = []
        for example in examples:
            text = tokenizer.apply_chat_template(
            example, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {
            "text": texts,
        }


    local_dataset_path = "datasets/numbers_dataset.json"

    if not os.path.exists(local_dataset_path):
        # Provide the correct path to the dataset or download the dataset if necessary
        raise FileNotFoundError(f"Dataset not found at {local_dataset_path}. Please provide the correct path to the dataset.")     

    dataset = load_dataset("json", data_files=local_dataset_path, split="train")
    dataset = standardize_sharegpt(dataset)
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )

    return dataset, tokenizer
