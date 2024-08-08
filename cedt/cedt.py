from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.optim as optim
import os
import attention

# Define custom dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label

# Define a simple transformer model for text classification
class TransformerClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(TransformerClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, inputs):
        outputs = self.bert(**inputs)
        logits = self.classifier(outputs.pooler_output)
        return logits

# Training loop
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    for batch in dataloader:
        inputs, labels = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
def collate_fn(batch,tokenizer):
    texts, labels = zip(*batch)
    inputs = tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True, max_length=128)
    labels = torch.tensor(labels)
    return inputs, labels

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')

def main():
    num_epochs = 5
    num_labels = 2
    model_name = "bert-base-uncased"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # Load the IMDB dataset
    dataset = load_dataset("imdb")
 
    # Get positive and negative samples
    positive_samples = [sample for sample in dataset['train'] if sample['label'] == 1][:50]
    negative_samples = [sample for sample in dataset['train'] if sample['label'] == 0][:50]
    
    # Prepare balanced texts and labels
    texts = [sample['text'] for sample in positive_samples + negative_samples]
    labels = [sample['label'] for sample in positive_samples + negative_samples]
    
    print(set(labels))
    # Create dataset and dataloader
    text_dataset = TextDataset(texts, labels, tokenizer)
    dataloader = DataLoader(text_dataset, batch_size=32, shuffle=True, collate_fn=lambda batch: collate_fn(batch, tokenizer))
    #print(dataloader)

    # Initialize model, optimizer, and criterion
    model = TransformerClassifier(model_name, num_labels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        train(model, dataloader, optimizer, criterion, device)
    '''
    # Save the model
    model_save_path = "transformer_classifier.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Load the model to check if saved correctly
    loaded_model = TransformerClassifier(model_name, num_labels).to(device)
    loaded_model.load_state_dict(torch.load(model_save_path))
    print("Model loaded successfully")
    '''

    evaluate(model, dataloader, device)
    sample_texts = ["I don't like this movie", "This movie was fantastic!", "The plot was boring."]
    sample_input = tokenizer(sample_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
    sample_input = {k: v.to(device) for k, v in sample_input.items()}
    with torch.no_grad():
        outputs = model(sample_input)
    print("Sample output:", outputs)

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(outputs, dim=-1)
    print("Probabilities:", probabilities)

    # Get the predicted class
    predicted_classes = torch.argmax(outputs, dim=-1)
    print("Predicted classes:", predicted_classes)

if __name__ == "__main__":
    main()
    # test attention
    
    # Example tensors for query, key, and value
    query = torch.randn(2, 8, 50, 64, device='cuda')
    key = torch.randn(2, 8, 50, 64, device='cuda')
    value = torch.randn(2, 8, 50, 64, device='cuda')

    # Call the forward function from the CUDA extension
    output = attention.forward(query, key, value)
    print(output)
