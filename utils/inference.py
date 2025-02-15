from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from transformers import TextStreamer

def inference(prompt, model, tokenizer, stream=False, skip_prompt=True):
   
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    messages = [
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    if not stream:
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=64,
            use_cache=True,
            temperature=1.5,
            min_p=0.1
        )
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if skip_prompt:
            response = response[0].split('assistant\n\n')[1]
        else :
            response = response[0]
        return response
    else:
        text_streamer = TextStreamer(tokenizer, skip_prompt=skip_prompt)
        _ = model.generate(
            input_ids=inputs,
            streamer=text_streamer,
            max_new_tokens=128,
            use_cache=True,
            temperature=1.5,
            min_p=0.1,
        )
