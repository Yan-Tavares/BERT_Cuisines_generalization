import json
import numpy as np


if __name__ == "__main__":
    # Get the emotion labels
    from transformers import AutoModelForSequenceClassification
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    roBERTa_emotions = AutoModelForSequenceClassification.from_pretrained(model_name)
    emotion_labels = roBERTa_emotions.config.id2label
    
    print(emotion_labels)

    # Collect the classified submissions dictionary 
    with open('results/classified_submissions.json', 'r') as f:
        classified_submissions = json.load(f)

    # Colect the cuisine labels from the sentences_dict
    with open('data_preprocess/sentences_dict.json', 'r') as f:
        sentences_dict = json.load(f)

    cuisine_labels = {cuisine: i for i, cuisine in enumerate(sentences_dict.keys())}
    del sentences_dict  # Remove the sentences


    # Create a dictionary to store the cuisine and emotions for each cuisine
    cuisine_emotions_dict = {}
    for cuisine in cuisine_labels:
        cuisine_emotions_dict[cuisine] = {'submissions_emotions': []}

    # Go thtough the classified submissions and assign the cuisine and emotions to the cyisine_emotions_dict

    for sub in classified_submissions:
        # Collect the cuisine name using the cuisine_labels and sub['class']
        sub_cuisine = list(cuisine_labels.keys())[sub['class']]
        cuisine_emotions_dict[sub_cuisine]['submissions_emotions'].append(sub['emotion_probs'])

    # Sum all the lists of emotion_probs for each cuisine

    for cuisine in cuisine_emotions_dict:
        submissions_emotions = np.array(cuisine_emotions_dict[cuisine]['submissions_emotions'])
        summed_emotions = np.sum(submissions_emotions, axis=0)
        average_emotions = summed_emotions / len(cuisine_emotions_dict[cuisine]['submissions_emotions'])
        
        cuisine_emotions_dict[cuisine]['summed_emotions'] = summed_emotions.tolist()
        cuisine_emotions_dict[cuisine]['average_emotions'] = average_emotions.tolist()
        
    # Use the emotion_labels to print the average emotions
    for index, emotion in emotion_labels.items():
        print(f"\n{emotion}")
        for cuisine in cuisine_emotions_dict:
            print(f"{cuisine}: {cuisine_emotions_dict[cuisine]['average_emotions'][index]}")

    # Save the cuisine_emotions_dict to a JSON file
    with open('results/cuisine_emotions_dict.json', 'w') as f:
        json.dump(cuisine_emotions_dict, f, indent=4)