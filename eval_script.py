import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    AutoModelForQuestionAnswering,
    default_data_collator
)
from evaluate import load


# Use GPU if able to
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Local Path for Model and Tokenizer
local_model_path = "./hybert_pretraining_model"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# Preprocess GLUE 
def preprocess_glue(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

glue_data = load_dataset("glue", "sst2")
glue_data = glue_data.map(preprocess_glue, batched=True)

# Preprocess SQuAD
def preprocess_squad(examples):
    inputs = tokenizer(
        examples["question"], 
        examples["context"], 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )
    return inputs

squad_data = load_dataset("squad")
squad_data = squad_data.map(preprocess_squad, batched=True)

# Preprocess SuperGLUE
def preprocess_superglue(examples):
    return tokenizer(
        examples["premise"], 
        examples["hypothesis"], 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )

superglue_data = load_dataset("super_glue", "rte", trust_remote_code=True)
superglue_data = superglue_data.map(preprocess_superglue, batched=True)

# Train GLUE 
def train_glue():
    model = AutoModelForSequenceClassification.from_pretrained(local_model_path, num_labels=2).to(device)
    training_args = TrainingArguments(
        output_dir="./glue_results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        logging_steps=10,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=glue_data["train"],
        eval_dataset=glue_data["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    trainer.train()
    results = trainer.evaluate()
    print("GLUE Results:", results)

# SQuAD
def evaluate_squad():
    model = AutoModelForQuestionAnswering.from_pretrained(local_model_path).to(device)
    metric = load("squad")
    squad_eval_data = squad_data["validation"]
    
    current_batches = 0

    for example in squad_eval_data:

        inputs = tokenizer(
            example["question"], 
            example["context"], 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        start_logits, end_logits = outputs.start_logits, outputs.end_logits
        start_idx = torch.argmax(start_logits)
        end_idx = torch.argmax(end_logits)
        prediction = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx + 1])
        metric.add(prediction=prediction, reference=example["answers"]["text"][0])
        
        current_batches += 1

    results = metric.compute()
    print("SQuAD Results:", results)

# 6. Train SuperGLUE
def train_superglue():
    model = AutoModelForSequenceClassification.from_pretrained(local_model_path, num_labels=2).to(device)
    training_args = TrainingArguments(
        output_dir="./superglue_results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        logging_steps=10,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=superglue_data["train"],
        eval_dataset=superglue_data["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    trainer.train()
    results = trainer.evaluate()
    print("SuperGLUE Results:", results)

if __name__ == "__main__":
    print("Starting GLUE Evaluation...")
    train_glue()
    print("Starting SQuAD Evaluation...")
    evaluate_squad()  # This now stops after 50 batches
    print("Starting SuperGLUE Evaluation...")
    train_superglue()
