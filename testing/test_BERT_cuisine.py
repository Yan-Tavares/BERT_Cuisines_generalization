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
with open('data_preprocess/sentences_dict.json', 'r') as f:
    sentences_dict = json.load(f)

print("Sentences dictionary loaded from sentences_dict.json.")

# Get the cuisine labels from the sentences_dict
cuisine_labels = {cuisine: i for i, cuisine in enumerate(sentences_dict.keys())}
del sentences_dict  # Remove the sentences_dict to free up memory

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
    inputs = tokenizer(sentence, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
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


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

sentence = "How about a vinaigrette and add feta, chickpeas, dill, cucumber, grape tomatoes?"
processed_sentence = preprocess_text(sentence)

logits = predict_logits(processed_sentence)
print(logits)
predicted_class = classify_text(processed_sentence, cuisine_labels)
print(f"Predicted class for the given sentence: {predicted_class}")

submission_inpection = 16035 # Interesting ones : 16002 , 16003, 16008, 16009
combined_text = prcessed_submissions[submission_inpection]['processed_title'] + " " + prcessed_submissions[submission_inpection]['processed_selftext'] + " " + " ".join(prcessed_submissions[submission_inpection]['processed_comments'])

print(f"Submission {submission_inpection} title:\n {prcessed_submissions[submission_inpection]['title']}")
print(f"Submission {submission_inpection} comments:\n {prcessed_submissions[submission_inpection]['comments']}")

logits = predict_logits(combined_text)
probabilities = F.softmax(logits, dim=-1)

for cuisine, prob in zip(cuisine_labels.keys(), probabilities[0]):
    print(f"{cuisine}: {prob:.4f}")
