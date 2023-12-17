import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizerFast 
import os
from datasets import load_dataset

# checks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load dataset
dataset = load_dataset("imdb")

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
tokenized_dataset = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_dataset["train"]
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(device)

training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=1,              
    per_device_train_batch_size=32,  
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./logs',            
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()
