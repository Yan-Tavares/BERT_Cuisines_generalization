import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the filtered submissions from the JSON file
with open('data_preprocess/submissions_dict.json', 'r') as f:
    prcessed_submissions = json.load(f)


# Load pre-trained BERT tokenizer and model for sentiment analysis
tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

def analyze_sentiment(text, tokenizer, model):
    """
    Analyze sentiment of the given text using BERT.

    Args:
        text (str): Input text to analyze.
        tokenizer: Pre-loaded tokenizer for the BERT model.
        model: Pre-loaded BERT model for sentiment analysis.

    Returns:
        dict: Dictionary containing probabilities for positive, negative, and neutral sentiment.
    """
    # Tokenize input text
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=100,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Perform sentiment analysis
    with torch.no_grad():
        outputs = model(**inputs)

    # Get probabilities
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze()

    # Map probabilities to sentiment labels
    sentiment_labels = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    sentiment_scores = {sentiment_labels[i]: float(probabilities[i]) for i in range(len(sentiment_labels))}

    return sentiment_scores


submission_inpection = 16002
title = prcessed_submissions[submission_inpection]['processed_title']
combined_comments = " ".join(prcessed_submissions[submission_inpection]['processed_comments'])

print(f"Submission {submission_inpection} title: {title}")
print(prcessed_submissions[submission_inpection]['comments'])

sentiment = analyze_sentiment(title, tokenizer, model)

def compute_score(probabilities):
    # Total score should range from 0 to 1
    # Getting a prob of 0.2 for all stars should give a score of 0.5
    # Gettin a prob of 1 for 5 stars should give a score of 1
    # Getting a prob of 1 for 1 star should give a score of 0
    # Create a tensor with rows equals to ne number of rows in probabilities, columns equals to 1

    scores = torch.zeros(probabilities.shape[0],1)

    for i, entry in enumerate(probabilities):
        weight = 0
        score = 0
        for value in entry:
            score += value * weight
            weight += 1

        scores[i] = score/probabilities.shape[1]
        

    return scores
    

print(sentiment)
score = compute_score(sentiment)
print(score)