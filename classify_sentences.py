import torch
import json
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import logging
logging.set_verbosity_error() # To supress the warning message for fine tuning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################
# Collect the submissions dictionary
############################################

# Load the filtered submissions from the JSON file
with open('data_preprocess/submissions_dict.json', 'r') as f:
    submissions_dict = json.load(f)

with open('data_preprocess/sentences_dict.json', 'r') as f:
    sentences_dict = json.load(f)

cuisine_labels = {cuisine: i for i, cuisine in enumerate(sentences_dict.keys())}
del sentences_dict  # Remove the sentences_dict to free up memory

#############################
# Cuisine Prediction Function
#############################

def cuisine_probs(text):
    inputs = tokenizer_cuisine(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = BERT_cuisine(**inputs)
        logits = outputs.logits

    probs = torch.nn.functional.softmax(logits, dim=1).to('cpu')
    return probs

#############################
# Emotion Prediction Function
#############################

def get_emotions(text):
    # Tokenize the input text
    inputs = tokenizer_emotions(text, return_tensors="pt", truncation=True, padding=True).to(device)
    
    # Perform forward pass to get logits
    outputs = roBERTa_emotions(**inputs)
    
    # Apply softmax to get probabilities
    outputs = torch.softmax(outputs.logits, dim=1).detach()[0].unsqueeze(0).to('cpu')
 
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

    def get_cuisine_emotions_and_store(text):
        # Classify the title
        text_cuisine_index = torch.argmax(cuisine_probs(text)).item()
        text_cuisine = list(cuisine_labels.keys())[text_cuisine_index]
        text_emotions = get_emotions(text).squeeze().tolist()

        classified_sentences_dict[text_cuisine]['sentences emotions'].append(text_emotions)

    #############################
    # Compute cuisines probabilities and sentiment scores for submissions
    #############################

    print(f"Total submissions: {len(submissions_dict)}")

    counter = 0
    for sub in submissions_dict:

        print(f"Processing submission {counter}", end="\r")

        title = sub['processed_title']
        selftext = sub['processed_selftext']
        comments = sub['processed_comments']

        # Classify and store the title
        get_cuisine_emotions_and_store(title)

        # Classify and store the selftext if not empty
        if selftext != '':
            get_cuisine_emotions_and_store(selftext)

        # Classify the comments
        for comment in comments:
            get_cuisine_emotions_and_store(comment)

        counter += 1

        if counter == 100:
            break

    print("\nDone.")
    with open('results/classified_sentences.json', 'w') as f:
        json.dump(classified_sentences_dict, f)
