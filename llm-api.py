import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoTokenizer,
    AutoModelForCausalLM
)
# GPT-2 MEDIUM
def generate_gpt2(prompt, max_length=200):
    model_name = "gpt2-medium"

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# QWEN 2.5 CODER

def generate_qwen(prompt, max_new_tokens=256):
    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )

    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


# PARALLELNI TEST

if __name__ == "__main__":
    prompt = "Write a quick sort algorithm in Python."

    print("\n" + "="*50)
    print(" GPT-2 MEDIUM OUTPUT ")
    print("="*50 + "\n")
    print(generate_gpt2(prompt))

    print("\n" + "="*50)
    print(" QWEN 2.5 CODER OUTPUT ")
    print("="*50 + "\n")
    print(generate_qwen(prompt))
