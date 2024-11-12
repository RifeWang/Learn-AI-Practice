from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig

# 基于 MarianMT 框架的翻译模型，属于机器翻译（Machine Translation, MT）模型，并不属于典型意义上的 LLM
model_name = 'Helsinki-NLP/opus-mt-en-zh'
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# # 打印模型的所有模块
# for name, module in model.named_modules():
#     print(name)

# 配置 LoRA
lora_config = LoraConfig(
    r=16,               # 低秩矩阵的秩
    lora_alpha=32,      # 学习率
    lora_dropout=0.1,   # dropout 概率
    target_modules=[
        "self_attn.q_proj", 
        "self_attn.k_proj", 
        "self_attn.v_proj", 
        "self_attn.out_proj"
    ]  # 适配的目标模块
)
# 应用 LoRA 到模型
model = get_peft_model(model, lora_config)

# 加载数据集
data = load_dataset('json', data_files='docs_concepts_dataset.jsonl', split='train[:5%]') # 只加载数据集的前百分比数据，因为内存不够用

# 先划分出 80% 的数据作为训练集，剩下的 20% 用于验证和测试
train_val_test_split = data.train_test_split(test_size=0.2, seed=42)
train_data = train_val_test_split['train']
val_test_data = train_val_test_split['test']

# 再将验证+测试数据按 50%/50% 划分为验证集和测试集
val_test_split = val_test_data.train_test_split(test_size=0.5, seed=42)
val_data = val_test_split['train']
test_data = val_test_split['test']

print(f"训练集大小: {len(train_data)}")
print(f"验证集大小: {len(val_data)}")
print(f"测试集大小: {len(test_data)}")


def preprocess_function(examples):
    inputs = examples['en']
    targets = examples['zh']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 批量预处理数据
tokenized_train_data = train_data.map(preprocess_function, batched=True)
tokenized_val_data = val_data.map(preprocess_function, batched=True)
tokenized_test_data = test_data.map(preprocess_function, batched=True)

# 设置训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir="./opus-mt-en-zh-finetuned",   # 模型输出目录
    eval_strategy="epoch",             # 评估策略
    learning_rate=2e-5,                # 学习率
    per_device_train_batch_size=2,     # 训练批次大小，批次越多越占用内存
    per_device_eval_batch_size=2,      # 评估批次大小，批次越多越占用内存
    weight_decay=0.01,                 # 权重衰减
    save_total_limit=3,                # 保存的模型数量限制
    num_train_epochs=2,                # 训练的轮次
    predict_with_generate=True,        # 生成时使用的模式
    fp16=True,                         # 使用混合精度加速训练
)

# 训练
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,   
    eval_dataset=tokenized_val_data,      
    tokenizer=tokenizer,
)
trainer.train()

# 在验证集上评估并打印结果
val_metrics = trainer.evaluate(eval_dataset=tokenized_val_data)
print("验证集上的评估结果：", val_metrics)

# 在测试集上进行最终评估
test_metrics = trainer.evaluate(eval_dataset=tokenized_test_data)
print("测试集上的评估结果：", test_metrics)

# 保持模型
model.save_pretrained("./opus-mt-en-zh-finetuned")
tokenizer.save_pretrained("./opus-mt-en-zh-finetuned")
