# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B")
result = pipe(messages)
print(result)



# from transformers import AutoTokenizer, Qwen2ForCausalLM

# model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# prompt = "Who are you?"
# inputs = tokenizer(prompt, return_tensors="pt")

# # Generate
# generate_ids = model.generate(inputs.input_ids, max_length=30)
# outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
# print(outputs)