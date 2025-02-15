import os
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt
#------------------------------------------------
from utils.inference import inference
from utils.training import train_model
from utils.load_models import load_base_model, save_lora, add_adapters, load_lora


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
    #? example data 
    # dataset = load_dataset("mlabonne/FineTome-100k", split="train")
    # ----
    dataset = standardize_sharegpt(dataset)
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )

    return dataset, tokenizer


def main():
    token = "<|CMD-N-U-M-B-E-R|>"
    # model, tokenizer = load_base_model()
    #model = add_adapters(model)
    #dataset, tokenizer = data_prep(tokenizer)
    #model, tokenizer = train_model(model, dataset, tokenizer) # TODO probar si esto va a devolver el modelo entrenado o el modelo que ya tenemos basta
    
    # save_lora(model, tokenizer, "x_lora_modelxx")
    model, tokenizer = load_lora("x_lora_modelxx")
    
    while True:
        user_input = input("\nEnter your prompt (or type 'exit' to quit): \n")
        if user_input.lower() == 'exit':
            break
        inference(user_input, model, tokenizer, stream=True)
    
if __name__ == "__main__":
    main()