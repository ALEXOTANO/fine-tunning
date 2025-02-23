from utils.inference import inference
from utils.training import train_model, data_prep
from utils.load_models import load_base_model, save_lora, add_adapters, load_lora


def main():
    
    model, tokenizer = load_base_model()
    
    # model = add_adapters(model)
    # dataset, tokenizer = data_prep(tokenizer)
    # model, tokenizer = train_model(model, dataset, tokenizer)
    
    # save_lora(model, tokenizer, "final_lora_numbers")
    # model, tokenizer = load_lora("x_lora_modelxx")
    
    while True:
        user_input = input("\nEnter your prompt (or type 'exit' to quit): \n")
        if user_input.lower() == 'exit':
            break
        inference(user_input, model, tokenizer, stream=True)
    
if __name__ == "__main__":
    main()