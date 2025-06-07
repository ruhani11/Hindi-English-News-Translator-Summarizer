# train_model.py
import pandas as pd
from datasets import Dataset
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from evaluate import load
from tqdm import tqdm
import torch

# 1. Load and prepare dataset
df = pd.read_csv('scrapped_clean.csv')
df.columns = ["hindi", "english"]
df.dropna(inplace=True)
dataset = Dataset.from_pandas(df)

def format_translation(example):
    return {
        "translation": {
            "hi": example["hindi"],
            "en": example["english"]
        }
    }

formatted_dataset = dataset.map(format_translation)

model_name = "Helsinki-NLP/opus-mt-hi-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)

def tokenize(batch):
    hi_texts = [item["hi"] for item in batch["translation"]]
    en_texts = [item["en"] for item in batch["translation"]]
    inputs = tokenizer(hi_texts, padding="max_length", truncation=True, max_length=64)
    targets = tokenizer(en_texts, padding="max_length", truncation=True, max_length=64)
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": targets["input_ids"]
    }

tokenized_dataset = formatted_dataset.map(tokenize, batched=True)

# 2. Load model
model = MarianMTModel.from_pretrained(model_name)

# 3. Training setup
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# 4. Evaluate before training
metric = load("sacrebleu")

def get_predictions(df, model, batch_size=8):
    preds = []
    hindi_texts = df["hindi"].tolist()
    for i in tqdm(range(0, len(hindi_texts), batch_size)):
        batch = hindi_texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64)
        with torch.no_grad():
            outputs = model.generate(**inputs)
        preds.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return preds

print("Evaluating before fine-tuning...")
base_model = MarianMTModel.from_pretrained(model_name)
baseline_preds = get_predictions(df, base_model)
bleu_before = metric.compute(predictions=baseline_preds, references=[[ref] for ref in df["english"]])
print("BLEU before fine-tuning:", bleu_before)

# 5. Train
trainer.train()

# 6. Evaluate after training
print("Evaluating after fine-tuning...")
fine_tuned_preds = get_predictions(df, model)
bleu_after = metric.compute(predictions=fine_tuned_preds, references=[[ref] for ref in df["english"]])
print("BLEU after fine-tuning:", bleu_after)

# 7. Save model
model.save_pretrained("fine_tuned_hi_en")
tokenizer.save_pretrained("fine_tuned_hi_en")
