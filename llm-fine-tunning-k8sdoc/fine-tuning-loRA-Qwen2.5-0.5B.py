from transformers import Qwen2ForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig

def create_prompt_format(en_text: str, zh_text: str) -> str:
    """创建符合 ChatML 格式的 prompt"""
    return f"""<|im_start|>system
You are a professional translator who can translate English to Chinese accurately while preserving the original formatting and technical terms.
<|im_end|>
<|im_start|>user
Translate the following English text to Chinese:
{en_text}
<|im_end|>
<|im_start|>assistant
{zh_text}
<|im_end|>"""

def preprocess_function(examples):
    """预处理函数，将数据转换为模型所需的格式"""
    # 创建完整的对话格式
    prompts = [create_prompt_format(en, zh) 
              for en, zh in zip(examples['en'], examples['zh'])]
    
    # 编码输入
    model_inputs = tokenizer(
        prompts,
        max_length=1024,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # 创建标签，-100表示不计算损失的位置
    labels = model_inputs['input_ids'].clone()
    
    # 找到每个序列中assistant回复的起始位置
    for idx, prompt in enumerate(prompts):
        # 找到assistant回复的起始位置
        assistant_start = prompt.find('<|im_start|>assistant\n') + len('<|im_start|>assistant\n')
        # 将非助手回复的部分标记为-100
        prompt_tokens = tokenizer(prompt[:assistant_start])['input_ids']
        labels[idx, :len(prompt_tokens)] = -100
    
    model_inputs['labels'] = labels
    return model_inputs

# 加载模型和分词器
model_name = 'Qwen/Qwen2.5-0.5B'
model = Qwen2ForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side="right"
)
# 配置 LoRA
lora_config = LoraConfig(
    r=4,               # 低秩矩阵的秩
    lora_alpha=16,      # 学习率
    lora_dropout=0.1,   # dropout 概率
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]  # 适配的目标模块
)
# 应用 LoRA 到模型
model = get_peft_model(model, lora_config)

# Special tokens
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# 加载和处理数据
data = load_dataset('json', data_files='docs_concepts_dataset.jsonl', split='train[:1%]')

# 数据集划分
train_val_test = data.train_test_split(test_size=0.2, seed=42)
train_data = train_val_test['train']
val_test = train_val_test['test'].train_test_split(test_size=0.5, seed=42)
val_data = val_test['train']
test_data = val_test['test']

# 数据预处理
tokenized_train = train_data.map(
    preprocess_function,
    batched=True,
    remove_columns=train_data.column_names
)
tokenized_val = val_data.map(
    preprocess_function,
    batched=True,
    remove_columns=val_data.column_names
)

# 训练参数
training_args = TrainingArguments(
    output_dir="./qwen2.5-finetuned",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    fp16=True,
    report_to="none",  # 禁用报告以减少依赖
)

# 创建训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
)

# 训练模型
trainer.train()

# 保存模型
trainer.save_model("./qwen2.5-finetuned-final")

# 测试生成
def generate_translation(text: str):
    prompt = f"""<|im_start|>system
You are a professional translator who can translate English to Chinese accurately while preserving the original formatting and technical terms.
<|im_end|>
<|im_start|>user
Translate the following English text to Chinese:
{text}
<|im_end|>
<|im_start|>assistant
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=1024,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 测试示例
test_text = """This document highlights and consolidates configuration best practices that are introduced
throughout the user guide, Getting Started documentation, and examples."""
print(generate_translation(test_text))