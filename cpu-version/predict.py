from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import sys
import numpy as np

model_path = "./model-cpu-final"

# Chargement
print("🔧 Chargement du modèle...")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

# Texte à prédire
text = sys.argv[1]

# Préparation
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    predicted_id = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_id].item()

# Mapping label ID → label texte
label = model.config.id2label[predicted_id]

print(f"✅ Département prédit : {label} (confiance : {confidence:.2%})")
