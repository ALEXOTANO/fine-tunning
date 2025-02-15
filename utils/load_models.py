from unsloth import FastLanguageModel
from utils.constants import max_seq_length, dtype, load_in_4bit

def load_base_model(model_name="unsloth/Llama-3.2-3B-Instruct"):
    # Removed local max_seq_length, dtype, load_in_4bit

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/Meta-Llama-3.1-8B-bnb-4bit",  # Llama-3.1 2x faster
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-405B-bnb-4bit",  # 4bit for 405b!
        "unsloth/Mistral-Small-Instruct-2409",  # Mistral 22b 2x faster!
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "unsloth/Phi-3.5-mini-instruct",  # Phi-3.5 2x faster!
        "unsloth/Phi-3-medium-4k-instruct",
        "unsloth/gemma-2-9b-bnb-4bit",
        "unsloth/gemma-2-27b-bnb-4bit",  # Gemma 2x faster!
        "unsloth/Llama-3.2-1B-bnb-4bit",  # NEW! Llama 3.2 models
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-3B-bnb-4bit",
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    ]  # More models at https://huggingface.co/unsloth

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,  # or choose "unsloth/Llama-3.2-1B-Instruct"
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    return  model, tokenizer

def add_adapters(model):
    """
    Add adapters to a model so it can be train 
    """
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    return model


def save_lora(model, tokenizer, adapter_name="lora_model"):
    model.save_pretrained(adapter_name)  
    tokenizer.save_pretrained(adapter_name)
    
def load_lora(adapter_name="lora_model"):
    model, tokenizer = FastLanguageModel.from_pretrained(adapter_name)
    return model, tokenizer