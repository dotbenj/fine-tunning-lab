import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import classification_report
import numpy as np

# ----------- CONFIG ----------
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
max_length = 1024
num_epochs = 3
batch_size = 1  # QLoRA = micro batch
gradient_accumulation_steps = 16
eval_patience = 1

# ----------- LOAD MODEL (QLoRA) ----------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Required for Mistral padding

# ----------- PEFT (QLoRA) ----------
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ----------- LOAD DATASET -----------
dataset = load_dataset("json", data_files={
    "train": "../data/train-instruct-sftt.jsonl",
    "eval": "../data/eval-instruct-sftt.jsonl"
})

def tokenize(batch):
    prompts = batch["prompt"]
    completions = batch["completion"]
    texts = [p + "\n<|assistant|>\n" + c for p, c in zip(prompts, completions)]
    outputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["prompt", "completion"])

# ----------- TRAINING ARGUMENTS -----------
training_args = TrainingArguments(
    output_dir="./qlora-mistral",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=10,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=2e-4,
    bf16=False,
    fp16=True,
    lr_scheduler_type="cosine",
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=eval_patience)],
)

trainer.train()

# ----------- CLASSIFICATION REPORT ----------
print("🔎 Évaluation qualitative (sample):")
eval_preds = trainer.predict(tokenized_dataset["eval"])
decoded_preds = tokenizer.batch_decode(np.argmax(eval_preds.predictions, axis=-1), skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(eval_preds.label_ids, skip_special_tokens=True)
for pred, label in zip(decoded_preds[:5], decoded_labels[:5]):
    print(f"\n💬 Pred: {pred}\n✅ Label: {label}")
