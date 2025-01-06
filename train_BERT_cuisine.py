import json
import torch
from transformers import logging
logging.set_verbosity_error() # To supress the warning message for fine tuning

####################
# Load sentences_dict from the JSON file
with open('sentences_dict.json', 'r') as f:
    sentences_dict = json.load(f)

from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Map cuisines to numerical labels
cuisine_labels = {
    "Italian": 0,
    "French": 1,
    "American": 2,
    "Mexican": 3,
    "Thai": 4,
    "Greek": 5,
    "Indian": 6,
    "Japanese": 7,
    "Spanish": 8,
    "Chinese": 9,
    "Unknown": 10,
}

# Prepare lists to store inputs and labels
all_input_ids = []
all_attention_masks = []
all_labels = []

# Tokenize and process sentences
max_length = 128
for cuisine, sentences in sentences_dict.items():

    '''
    Go cuisine by cuisine and:
    - Get the label index for the cuisine. The model expects the label as being the index and not the one hot vector
    - Tokenize all sentences for the cuisine. It generages 'input_ids' and 'attention_mask'
    - Give the same label to all the senteces
    '''
    label = cuisine_labels[cuisine]

    # Tokenize the sentences
    encoded = tokenizer(
        sentences,
        max_length=max_length,
        padding='max_length',  # Explicitly pad to `max_length`
        truncation=True,
        return_tensors="pt"
    )

    # Append to the lists
    all_input_ids.append(encoded['input_ids'])
    all_attention_masks.append(encoded['attention_mask'])
    all_labels.extend([label] * len(sentences))

''''
all_input_ids dimension is a list with a total number of elements equal to the total number of sentences
each element of the list is a tensor with size equal to max_length
By concatenating all_input_ids we get a single tensor [total_number_of_sentences, max_length]

Same goes for all_attention_masks

Labels was a list of intergers which becomes a tensor [total_number_of_sentences]
'''

input_ids = torch.cat(all_input_ids, dim=0)
attention_masks = torch.cat(all_attention_masks, dim=0)
labels = torch.tensor(all_labels)


# Put all the tensors in a TensorDataset
dataset = TensorDataset(input_ids, attention_masks, labels)

from torch.utils.data import random_split

# Assume `dataset` is the original TensorDataset used in the DataLoader
dataset_size = len(dataset)
validation_split = 0.2
validation_size = int(validation_split * dataset_size)
train_size = dataset_size - validation_size

# Split the dataset
train_dataset, val_dataset = random_split(dataset, [train_size, validation_size])

# Create DataLoaders for training and validation
batch_size = 20

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

from transformers import BertForSequenceClassification, AdamW
import torch.nn as nn


# Define tokenizer and model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(sentences_dict))

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def validate_model(model, val_dataloader, device):
    model.eval()
    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_masks, labels = [item.to(device) for item in batch]
            outputs = model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits

            loss = criterion(logits, labels)
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()

    avg_loss = total_loss / len(val_dataloader)
    accuracy = total_correct / len(val_dataloader.dataset)

    return avg_loss, accuracy

def train_model_with_validation(
    model, train_dataloader, val_dataloader, optimizer, criterion, device, num_epochs=10, patience=3
):
    model.to(device)
    best_val_loss = float("inf")
    patience_counter = 0

    early_stopping = False
    for epoch in range(num_epochs):

        if early_stopping:
            break

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            input_ids, attention_masks, labels = [item.to(device) for item in batch]

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits

            loss = criterion(logits, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            print(f"\rProcessing batch {batch_idx + 1}/{len(train_dataloader)} - Batch Loss: {loss.item():.4f}", end='')

            # Validate every 1/y of the epoch
            if (batch_idx) % (len(train_dataloader) // 5) == 0:
                val_loss, val_accuracy = validate_model(model, val_dataloader, device)
                print(f"\nValidation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}\n")

                # Check early stopping criteria
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), "best_model.pt")
                else:
                    patience_counter += 2
                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        early_stopping = True
                        break

    print("\nTraining complete.")

# Example usage
train_model_with_validation(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    num_epochs=3,
    patience=1
)