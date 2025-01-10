import json
import torch
from transformers import logging
logging.set_verbosity_error() # To supress the warning message for fine tuning


###############################
# Load cuisine sentences dictionary
###############################

with open('data_preprocess/cuisine_sentences_dict.json', 'r', encoding='utf-8') as f:
    cuisine_sentences_dict = json.load(f)

from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch

# Get the cuisine labels from the keys of the sentences_dict
cuisine_labels = {cuisine: i for i, cuisine in enumerate(cuisine_sentences_dict.keys())}

###############################
# Tokenize using BERT tokenizer
###############################

# Prepare lists to store inputs and labels
all_input_ids = []
all_attention_masks = []
all_labels = []

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 128

for cuisine, sentences in cuisine_sentences_dict.items():

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

###############################
# Create the dataloader
###############################

# Put all the tensors in a TensorDataset
dataset = TensorDataset(input_ids, attention_masks, labels)

# Assume `dataset` is the original TensorDataset used in the DataLoader
dataset_size = len(dataset)
validation_split = 0.2
validation_size = int(validation_split * dataset_size)
train_size = dataset_size - validation_size

# Split the dataset
from torch.utils.data import random_split
torch.manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [train_size, validation_size])

# Create DataLoaders for training and validation
batch_size = 20
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

###############################
# Load BERT model
###############################
from transformers import BertForSequenceClassification, AdamW
import torch.nn as nn

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(cuisine_sentences_dict))
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

###############################
# Training functions
###############################

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

def train_model_with_validation(model, train_dataloader, val_dataloader, optimizer, criterion, device, num_epochs=10, patience=3):
    
    model.to(device)
    best_val_loss = float("inf")
    patience_counter = 0
    
    epoch_batches = len(train_dataloader)

    training_batch_losses = []
    validation_losses = []
    val_accuracies = []
    batch_val = []
    batch_train = []

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

            training_batch_losses.append(loss.item())
            batch_train.append(epoch* epoch_batches + batch_idx)

            print(f"\rProcessing batch {batch_idx + 1}/{epoch_batches} - Batch Loss: {loss.item():.4f}", end='')


            # Validate every 50 batches
            if (batch_idx) % (50) == 0:
                val_loss, val_accuracy = validate_model(model, val_dataloader, device)
                validation_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                batch_val.append(epoch* epoch_batches + batch_idx)

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
    return training_batch_losses, validation_losses, val_accuracies, batch_train, batch_val

###############################
# Training
###############################

training_batch_losses, validation_losses, val_accuracies, batch_train, batch_val = train_model_with_validation(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    num_epochs=6,
    patience=3
)

###############################
# Store training and validation losses for plotting
###############################

losses = {
    "training_batch_losses": training_batch_losses,
    "validation_losses": validation_losses,
    "batch_train": batch_train,
    "batch_val": batch_val,
    "validation_accuracies": val_accuracies,
}

with open('results/losses.json', 'w') as f:
    json.dump(losses, f)