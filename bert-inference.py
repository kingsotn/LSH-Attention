from transformers import DistilBertForSequenceClassification
import torch

example_texts = [
    "I loved this movie, it was fantastic!",
    "The movie was okay, but I probably wouldn't watch it again.",
    "It was an okay movie, nothing special but not terrible either",
    "Honestly, it was quite disappointing. I had high expectations and it just didn't deliver.",
    "An absolute masterpiece! The performances were outstanding and the story was captivating",
    "What a waste of my time and mone"
]

from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

encoded_input = tokenizer(example_texts, padding=True, truncation=True, return_tensors='pt')

from transformers import DistilBertForSequenceClassification
import torch

model = DistilBertForSequenceClassification.from_pretrained('/Users/kingston/algoml/project/results/checkpoint-500')
# Load your trained model
model.eval()

# If you have a GPU, move your model and inputs there
if torch.cuda.is_available():
    model.cuda()
    encoded_input = encoded_input.to('cuda')

# Perform inference
with torch.no_grad():
    outputs = model(**encoded_input)

# Get the predictions
predictions = torch.argmax(outputs.logits, dim=-1)

predicted_labels = ["positive" if pred == 1 else "negative" for pred in predictions]
for text, label in zip(example_texts, predicted_labels):
    print(f"'{text}' - Sentiment: {label}")
