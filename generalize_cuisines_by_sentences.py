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
    with open('results/classified_sentences.json', 'r') as f:
        classified_sentences = json.load(f)

    for cuisine in classified_sentences:
        cuisine_emotions = np.array(classified_sentences[cuisine]['sentences emotions'])
        summed_emotions = np.sum(cuisine_emotions, axis=0)
        average_emotions = summed_emotions / len(classified_sentences[cuisine]['sentences emotions'])
        classified_sentences[cuisine]['summed_emotions'] = summed_emotions.tolist()
        classified_sentences[cuisine]['average_emotions'] = average_emotions.tolist()
        
    # Use the emotion_labels to print the average emotions
    for index, emotion in emotion_labels.items():
        print(f"\n{emotion}")
        for cuisine in classified_sentences:
            print(f"{cuisine}: {classified_sentences[cuisine]['average_emotions'][index]}")

    #Print the number of sentences for each cuisine
    print("\nNumber of sentences for each cuisine:")
    for cuisine in classified_sentences:
        print(f"{cuisine}: {len(classified_sentences[cuisine]['sentences emotions'])}")

    # Save the cuisine_emotions_dict to a JSON file
    with open('results/classified_sentences.json', 'w') as f:
        json.dump(classified_sentences, f, indent=4)