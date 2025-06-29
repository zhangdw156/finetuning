from peft import LoraConfig, TaskType,PeftModel,get_peft_model
from transformers import AutoTokenizer,AutoModelForCausalLM,TrainingArguments,Trainer,DataCollatorForSeq2Seq
import os
from datasets import load_dataset
import torch
import swanlab
from accelerate import Accelerator


# 启用accelerator
accelerator=Accelerator()

model_path = "/data/download-model/Qwen3-0.6B"
# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=False, torch_dtype=torch.float16)
tokenizer.padding_side = "left"

# 设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

datset_path='../dataset/delicate_medical_r1/'
train_dataset_path = os.path.join(datset_path,"train.jsonl")
test_dataset_path = os.path.join(datset_path,"val.jsonl")
dataset=load_dataset('json',data_files={
    'train':train_dataset_path,
    'validation':test_dataset_path
})

PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_LENGTH = 2048

def process_func(example):
    """
    将数据集进行预处理
    """ 
    instruction = f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{example['question']}<|im_end|>\n<|im_start|>assistant\n"
    response = f"<think>{example['think']}</think>\n{example['answer']}<|im_end|>"

    # 对完整对话进行编码
    full_input = instruction + response
    
    # tokenize
    tokenized = tokenizer(
        full_input,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_tensors=None
    )
    
    # 对仅指令部分进行编码以计算标签位置
    instruction_ids = tokenizer(
        instruction,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_tensors=None
    )['input_ids']
    
    # 创建labels，instruction部分设为-100，response部分保持原值
    labels = tokenized['input_ids'].copy()
    instruction_len = len(instruction_ids)
    for i in range(instruction_len):
        if i < len(labels):
            labels[i] = -100
    
    tokenized['labels'] = labels
    
    return tokenized

dataset=dataset.map(process_func, remove_columns=dataset['train'].column_names)
train_dataset=dataset['train']
val_dataset=dataset['validation']

# 配置lora
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

model = get_peft_model(model, config)

# 在prepare之前打印可训练参数
if accelerator.is_main_process:
    model.print_trainable_parameters()  # 打印可训练参数，确认是否正确

# 准备
model,train_dataset,val_dataset=accelerator.prepare(model,train_dataset,val_dataset)

# 确保模型处于训练模式
model.train()

if accelerator.is_main_process:
    swanlab.config.update({
        "model": "Qwen3-0.6B-acc",
        "prompt": PROMPT,
        "data_max_length": MAX_LENGTH,
        })

args = TrainingArguments(
    output_dir="../output/Qwen3-0.6B-acc",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=400,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=False,  # 暂时关闭梯度检查点
    report_to="swanlab" if accelerator.is_main_process else None,
    run_name="qwen3-0.6B-acc",
    dataloader_drop_last=True,
    fp16=True,  # 使用混合精度训练
    remove_unused_columns=False,  # 重要：防止列被错误移除
)

# 创建数据整理器
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    return_tensors="pt"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

trainer.train()
