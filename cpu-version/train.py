import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import f1_score, classification_report


# ----------------------
# 🔧 Hyperparamètres
# ----------------------
# model_name = "prajjwal1/bert-tiny" # Use this model for low CPU
model_name = "distilbert-base-uncased"
num_labels = 16  # nombre total de départements
batch_size = 8
epochs = 35
max_length = 512
eval_patience = 2

# ----------------------
# 📦 Chargement du modèle et tokenizer
# ----------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# ----------------------
# 🧹 Fonction de pré-traitement
# ----------------------
def preprocess(example):
    return tokenizer(
        example["prompt"],
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

# ----------------------
# 📂 Chargement des données instruct (JSONL)
# ----------------------
dataset = load_dataset("json", data_files={
    "train": "../data/train-instruct-sftt.jsonl",
    "eval": "../data/eval-instruct-sftt.jsonl"
})

# Map labels en entier
unique_labels = sorted(list(set(example["completion"] for example in dataset["train"])))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

def encode_labels(example):
    example["labels"] = label2id[example["completion"]]
    return example

dataset = dataset.map(encode_labels)
tokenized_dataset = dataset.map(preprocess, batched=True)

# Supprimer les champs inutiles pour éviter les erreurs
tokenized_dataset = tokenized_dataset.remove_columns(["prompt", "completion"])

# ----------------------
# 🧪 Métriques personnalisées
# ----------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"macro_f1": macro_f1}

# ----------------------
# 🏋️ TrainingArguments (CPU Friendly)
# ----------------------
training_args = TrainingArguments(
    output_dir="./model-cpu",
    eval_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"
)

# ----------------------
# 🚀 Entraînement
# ----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=eval_patience)]
)

trainer.train()

# ----------------------
# 📊 Rapport final
# ----------------------
predictions = trainer.predict(tokenized_dataset["eval"])
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids
print("\n🎯 Classification Report:\n")
print(classification_report(labels, preds, target_names=unique_labels))

# ----------------------
# 💾 Sauvegarde pour Hugging Face
# ----------------------
model_dir = "./model-cpu-final"

# Ajoute explicitement ces champs à la config
model.config.id2label = id2label
model.config.label2id = label2id
model.config.num_labels = len(label2id)

# 🔧 Sauvegarde manuelle de la config propre
model.config.save_pretrained(model_dir)

# 💾 Sauvegarde du modèle et du tokenizer
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
