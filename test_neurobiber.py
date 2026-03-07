import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "Blablablab/neurobiber"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

text = "I think that this approach could work quite well."

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=512
)

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
predictions = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()

print(predictions)
print(predictions.shape)