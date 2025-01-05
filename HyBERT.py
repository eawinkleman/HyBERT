import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForPreTraining, AdamW
from datasets import load_dataset
import matplotlib.pyplot as plt
import os

save_dir = "./hybert_pretraining_checkpoint"
os.makedirs(save_dir, exist_ok=True)

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForPreTraining.from_pretrained("bert-base-uncased")
model.train()

# Load the WikiText dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Prepare the dataset for pretraining
class PretrainingDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.texts = [text for text in texts if len(text.split(". ")) >= 2]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        sentences = text.split(". ")

        # Randomly select one sentence to mask for ICT
        masked_idx = torch.randint(0, len(sentences), (1,)).item()
        masked_sentence = sentences[masked_idx]
        context_sentences = sentences[:masked_idx] + sentences[masked_idx + 1:]
        context = ". ".join(context_sentences)
        
        # Tokenize context and masked sentence
        context_tokens = self.tokenizer(
            context,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        masked_tokens = self.tokenizer(
            masked_sentence,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Prepare ICT labels
        ict_labels = masked_tokens["input_ids"].clone()
        ict_labels[ict_labels == self.tokenizer.pad_token_id] = -100  # Ignore padding

        # Prepare MLM and SOP as before
        tokens = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = tokens["input_ids"].clone()
        rand = torch.rand(labels.shape)
        mask_arr = (rand < 0.15) & (labels != 101) & (labels != 102) & (labels != 0)
        labels[~mask_arr] = -100  # Ignore non-masked tokens
        tokens["mlm_labels"] = labels
        tokens["sop_labels"] = torch.randint(0, 2, (1,))  # Random SOP labels (0 or 1)

        return {
            "context_input_ids": context_tokens["input_ids"],
            "context_attention_mask": context_tokens["attention_mask"],
            "ict_labels": ict_labels,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "mlm_labels": tokens["mlm_labels"],
            "sop_labels": tokens["sop_labels"],
        }

train_dataset = PretrainingDataset(dataset["train"]["text"], tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True)

# Model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Loss tracking
mlm_loss_trend = []
sop_loss_trend = []
ict_loss_trend = []

# Training loop
with open("HyBERT_loss_log.txt", "a") as log_file:
    epochs = 2
    for epoch in range(epochs):
        total_mlm_loss = 0
        total_sop_loss = 0
        total_ict_loss = 0

        for batch_num, batch in enumerate(train_dataloader, start=1):

            print(len(train_dataloader))
            
            # Move data to the device
            input_ids = batch["input_ids"].squeeze(1).to(device)
            attention_mask = batch["attention_mask"].squeeze(1).to(device)
            mlm_labels = batch["mlm_labels"].squeeze(1).to(device)
            sop_labels = batch["sop_labels"].squeeze(1).to(device)

            context_input_ids = batch["context_input_ids"].squeeze(1).to(device)
            context_attention_mask = batch["context_attention_mask"].squeeze(1).to(device)
            ict_labels = batch["ict_labels"].squeeze(1).to(device)

            # Forward pass for MLM and SOP
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=mlm_labels,
                return_dict=True,
            )

            # MLM Loss
            mlm_logits = outputs.prediction_logits
            mlm_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            mlm_loss = mlm_loss_fct(
                mlm_logits.view(-1, mlm_logits.size(-1)),
                mlm_labels.view(-1),
            )

            # SOP Loss
            sop_logits = outputs.seq_relationship_logits
            sop_loss_fct = torch.nn.CrossEntropyLoss()
            sop_loss = sop_loss_fct(sop_logits, sop_labels)

            # Forward pass for ICT (context modeling)
            context_outputs = model(
                input_ids=context_input_ids,
                attention_mask=context_attention_mask,
            )

            # ICT Loss
            ict_logits = context_outputs.prediction_logits  # Use prediction_logits for context modeling
            ict_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            ict_loss = ict_loss_fct(
                ict_logits.view(-1, ict_logits.size(-1)),
                ict_labels.view(-1),
            )

            if mlm_loss is None or sop_loss is None or ict_loss is None:
                raise ValueError(f"MLM loss: {mlm_loss}, SOP loss: {sop_loss}, ICT loss: {ict_loss}. Check inputs and forward pass.")

            # Combined Loss
            loss = mlm_loss + sop_loss + ict_loss

            # Print Losses
            # print("MLM Loss:", mlm_loss.item())
            # print("SOP Loss:", sop_loss.item())
            # print("ICT Loss:", ict_loss.item())
            # print("Total Loss (MLM + SOP + ICT):", loss.item())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track losses
            total_mlm_loss += mlm_loss.item()
            total_sop_loss += sop_loss.item()
            total_ict_loss += ict_loss.item()

            log_file.write(f"Batch: {batch_num}, MLM Loss: {mlm_loss:.4f}, SOP Loss: {sop_loss:.4f}, ICT: {ict_loss:.4f}, Total Loss: {loss:.4f}\n")
            print(f"Batch: {batch_num}, MLM Loss: {mlm_loss:.4f}, SOP Loss: {sop_loss:.4f}, ICT: {ict_loss:.4f}, Total Loss: {loss:.4f}")

            if batch_num >= 100: 
                break
                


        # Average losses
        avg_mlm_loss = total_mlm_loss / len(train_dataloader)
        avg_sop_loss = total_sop_loss / len(train_dataloader)
        avg_ict_loss = total_ict_loss / len(train_dataloader)
        total_loss = avg_mlm_loss + avg_sop_loss + avg_ict_loss
        mlm_loss_trend.append(avg_mlm_loss)
        sop_loss_trend.append(avg_sop_loss)
        ict_loss_trend.append(avg_ict_loss)

        log_file.write(f"\nEpoch {epoch+1}/{epochs}, MLM Loss: {avg_mlm_loss:.4f}, SOP Loss: {avg_sop_loss:.4f}, ICT Loss: {avg_ict_loss:.4f}, Total Loss: {total_loss:.4f}\n")
        print(f"Epoch {epoch+1}/{epochs}, MLM Loss: {avg_mlm_loss:.4f}, SOP Loss: {avg_sop_loss:.4f}, ICT Loss: {avg_ict_loss:.4f}, Total Loss: {total_loss:.4f}")
        
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))

# Save the model and loss trends
model.save_pretrained("./hybert_pretraining_model")
tokenizer.save_pretrained("./hybert_pretraining_model")
torch.save({"mlm_loss": mlm_loss_trend, "sop_loss": sop_loss_trend, "ict_loss": ict_loss_trend}, "loss_trends.pt")

# Load loss trends
loss_trends = torch.load("loss_trends.pt")
mlm_loss_trend = loss_trends["mlm_loss"]
sop_loss_trend = loss_trends["sop_loss"]
ict_loss_trend = loss_trends["ict_loss"]

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(mlm_loss_trend, label="MLM Loss", marker="o")
plt.plot(sop_loss_trend, label="SOP Loss", marker="o")
plt.plot(ict_loss_trend, label="ICT Loss", marker="o")
plt.title("Loss Trends During Pretraining")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.savefig("hybert_loss_trends_plot.png")
plt.show()
