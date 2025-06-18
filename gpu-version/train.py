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

# 🧬 Save PEFT model (adapters) correctly
print("💾 Saving QLoRA adapter to ./qlora-mistral/")
model.save_pretrained("./qlora-mistral")
tokenizer.save_pretrained("./qlora-mistral")

# -------------------
# 🧼 Free up memory before evaluation
# -------------------
import gc
torch.cuda.empty_cache()
gc.collect()

# -------------------
# ✅ Safe evaluation on small eval set
# -------------------
print("🔎 Running safe evaluation...")

# Optional: reduce the number of eval examples to avoid OOM
eval_subset = tokenized_dataset["eval"].select(range(10))

# Just get metrics (no full decode to avoid logits storage)
metrics = trainer.evaluate(eval_dataset=eval_subset)
print("📊 Eval loss:", metrics["eval_loss"])

# -------------------
# 🧠 Sample decoding (manual generation)
# -------------------
print("\n🔎 Manual generation preview:")

# Get a few raw prompts to generate on
raw_eval_set = dataset["eval"].select(range(3))

model.eval()
for sample in raw_eval_set:
    input_text = sample["prompt"] + "\n<|assistant|>\n"
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n💬 Prompt:\n{input_text}")
    print(f"🤖 Output:\n{generated}")
