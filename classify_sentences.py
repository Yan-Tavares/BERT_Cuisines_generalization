import torch
import json
import time
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import logging
logging.set_verbosity_error() # To supress the warning message for fine tuning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################
# Collect the submissions dictionary
############################################

# Load the filtered submissions from the JSON file
with open('data_preprocess/submissions_dict.json', 'r', encoding='utf-8') as f:
    submissions_dict = json.load(f)

with open('data_preprocess/cuisine_sentences_dict.json', 'r', encoding='utf-8') as f:
    cuisine_sentences_dict = json.load(f)

cuisine_labels = {cuisine: i for i, cuisine in enumerate(cuisine_sentences_dict.keys())}
del cuisine_sentences_dict  # Remove the sentences_dict to free up memory

#############################
# Cuisine Prediction Function
#############################

def cuisine_probs(sentences):
    inputs = tokenizer_cuisine(sentences, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = BERT_cuisine(**inputs)
        logits = outputs.logits

    probs = torch.nn.functional.softmax(logits, dim=1)
    return probs

#############################
# Emotion Prediction Function
#############################

def get_emotions(sentences):
    # Tokenize the input text
    inputs = tokenizer_emotions(sentences, max_length= 128 , return_tensors="pt", truncation=True, padding=True).to(device)
    
    # Perform forward pass to get logits
    outputs = roBERTa_emotions(**inputs)
    
    # Apply softmax to get probabilities
    outputs = torch.softmax(outputs.logits, dim=1).detach()
 
    return outputs

if __name__ == "__main__":
    ############################################
    # Load BERT sentiment
    ############################################

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer_emotions = AutoTokenizer.from_pretrained(model_name)
    roBERTa_emotions = AutoModelForSequenceClassification.from_pretrained(model_name)
    roBERTa_emotions.to(device)

    ############################################
    # Load BERT cuisines
    ############################################

    # Load BERT cuisines
    tokenizer_cuisine = BertTokenizer.from_pretrained("bert-base-uncased")
    BERT_cuisine = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(cuisine_labels))

    # Load the trained model weights
    BERT_cuisine.load_state_dict(torch.load("best_model.pt"))
    BERT_cuisine.eval()
    BERT_cuisine.to(device)

    #############################
    # Create a dictionary to store classified sentences per cuisine
    #############################

    classified_sentences_dict = {}
    for cuisine in cuisine_labels:
        classified_sentences_dict[cuisine] = {'sentences emotions': []}

    def get_cuisine_emotions_and_store(sentences):
        # Classify the titles
        cuisine_probs_batch = cuisine_probs(sentences)
        emotions_batch = get_emotions(sentences)

        for i, sent in enumerate(sentences):
            text_cuisine_index = torch.argmax(cuisine_probs_batch[i]).item()
            text_cuisine = list(cuisine_labels.keys())[text_cuisine_index]
            text_emotions = emotions_batch[i].squeeze().tolist()

            classified_sentences_dict[text_cuisine]['sentences emotions'].append(text_emotions)

    #############################
    # Compute cuisines probabilities and sentiment scores for submissions
    #############################

    #Shuffle the submissions_dict such that the testing is representative if part of the submissions are used
    import random
    random.seed(42)
    random.shuffle(submissions_dict)

    print(f"Total submissions: {len(submissions_dict)}")

    counter = 0
    batch_size = 50
    batch_texts = []

    start_time = time.time()

    for sub in submissions_dict:
        estimated_time = ((len(submissions_dict) - counter) * (time.time() - start_time) / counter)/3600  if counter > 0 else 0

        print(f"Processing submission {counter} | Estimated time in hours {estimated_time:.2f}", end="\r")

        title = sub['processed_title']
        selftext = sub['processed_selftext']
        comments = sub['processed_comments']

        # Collect texts in batches
        batch_texts.append(title)
        if selftext != '':
            batch_texts.append(selftext)
        batch_texts.extend(comments)

        # Process in batches of 10
        while len(batch_texts) >= batch_size:
            get_cuisine_emotions_and_store(batch_texts[:batch_size])
            batch_texts = batch_texts[batch_size:]

        counter += 1

        # if counter == 10000:
        #     break

    # Process any remaining texts
    if batch_texts:
        get_cuisine_emotions_and_store(batch_texts)

    print("\nDone.")
    with open('results/classified_sentences.json', 'w') as f:
        json.dump(classified_sentences_dict, f)