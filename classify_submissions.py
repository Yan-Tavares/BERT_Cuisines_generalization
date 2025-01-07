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
    # Compute cuisines probabilities and sentiment scores for submissions
    #############################
    print(f"Total submissions: {len(submissions_dict)}")

    counter = 0
    list_of_classified_submissions = []
    for sub in submissions_dict:
        classified_submission = {}

        print(f"Processing submission {counter}", end="\r")

        combined_comments = " ".join(sub['processed_comments'])
        combined_text = sub['processed_title'] + " " + sub['processed_selftext'] + " " + combined_comments

        classified_submission['id'] = sub['id']
        classified_submission['class'] = torch.argmax(cuisine_probs(combined_text)).item()
        classified_submission['emotion_probs'] = get_emotions(combined_text).squeeze().tolist()

        list_of_classified_submissions.append(classified_submission)
        counter += 1

        # if counter == 1000:
        #     break

    print("\nDone.")

    with open('results/classified_submissions.json', 'w') as f:
        json.dump(list_of_classified_submissions, f)
