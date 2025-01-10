import json
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

########################
# Load dictionaries
########################

# Load the filtered submissions from the JSON file
with open('data_preprocess/submissions_dict.json', 'r') as f:
    prcessed_submissions = json.load(f)

# Load sentences_dict from the JSON file
with open('data_preprocess/cuisine_sentences_dict.json', 'r') as f:
    cuisine_sentences_dict = json.load(f)

print("Sentences dictionary loaded from sentences_dict.json.")

# Get the cuisine labels from the sentences_dict
cuisine_labels = {cuisine: i for i, cuisine in enumerate(cuisine_sentences_dict.keys())}
del cuisine_sentences_dict  # Remove the sentences_dict to free up memory

#############################
# Set up BERT for testing
#############################
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(cuisine_labels))

# Load the trained model weights
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


#############################
# Prediction Function
#############################

def predict_logits(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    return logits

def classify_text(text,cuisine_labels):
    logits = predict_logits(text)
    predicted_number = torch.argmax(logits, dim=1).item()

    predicted_class = list(cuisine_labels.keys())[predicted_number]
    return predicted_class

#############################
# Manual Inspection
#############################
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string


submission_inpection = 18010
 #  ####16060
submission = prcessed_submissions[submission_inpection]

# Classify the title
title = submission['processed_title']
title_class = classify_text(title, cuisine_labels)
print(f"\nTitle: {title}")
print(f"Title classification: {title_class}\n")

# Classify the selftext
selftext = submission['processed_selftext']
selftext_class = classify_text(selftext, cuisine_labels)

print(f"Selftext: {selftext}")
print(f"Selftext classification: {selftext_class}\n")

# Classify each comment
for i, comment in enumerate(submission['processed_comments']):
    print(f"Comment {i+1}: {comment}")
    comment_class = classify_text(comment, cuisine_labels)
    print(f"Classification: {comment_class}\n")


# Test a specific sentence
sentence = "if I add grated parm to my deep fried mac n cheese balls will they burn?"
sentence_class = classify_text(sentence, cuisine_labels)
print(f"Specific sentence: {sentence}")
print(f"Specific sentence classification: {sentence_class}\n")