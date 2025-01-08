import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the filtered submissions from the JSON file
with open('data_preprocess/submissions_dict.json', 'r') as f:
    prcessed_submissions = json.load(f)


def get_emotions(text, model, tokenizer):
    # Tokenize the input text
    inputs = tokenizer(text, max_length = 128,return_tensors="pt", truncation=True, padding=True)
    
    # Perform forward pass to get logits
    outputs = model(**inputs)
    
    # Apply softmax to get probabilities
    probabilities = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
    
    # Map emotion IDs to labels and scores
    emotion_labels = model.config.id2label
    emotion_scores = {emotion_labels[i]: float(probabilities[i]) for i in range(len(probabilities))}
    return emotion_scores


if __name__ == "__main__":
    # Load the model and tokenizer
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Inspect a specific submission
    submission_inpection = 16001
    title = prcessed_submissions[submission_inpection]['processed_title']
    comments = prcessed_submissions[submission_inpection]['processed_comments']

    print(f"Submission {submission_inpection} title: {title}")

    # Get emotions for combined comments
    for idx, comment in enumerate(comments):
        comment_emotions = get_emotions(comment, model, tokenizer)
        print(f"\nComment {idx}:")
        print(comment)
        print(comment_emotions)