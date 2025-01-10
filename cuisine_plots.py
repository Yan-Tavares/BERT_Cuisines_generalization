import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the classified sentences
with open('results/classified_sentences.json', 'r', encoding='utf-8') as f:
    classified_sentences = json.load(f)

# Get the cuisine labels from the keys of the calssified sentences
cuisine_labels = {cuisine: i for i, cuisine in enumerate(classified_sentences.keys())}


#Load emotions from roberta go emotions

if __name__ == "__main__":
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    emotion_labels = model.config.id2label

    print("Emotion labels:")
    print(emotion_labels)


    #emotion_indexes = [0, 1]
    #colors = ['red','purple']
    
    # emotion_indexes = [3, 4]  # Replace with your desired emotion indexes
    # colors = ['yellow','gray']

    emotion_indexes = [5, 6]  # Replace with your desired emotion indexes
    colors = ['blue','green']

    fig, axes = plt.subplots(1, 2)  # Create a figure with 2 subplots side by side

    for i, emotion_index in enumerate(emotion_indexes):
        emotion_name = emotion_labels[emotion_index]
        emotion_scores = {}
        for cuisine in classified_sentences:
            average_emotions = classified_sentences[cuisine]["average_emotions"]
            emotion_scores[cuisine] = average_emotions[emotion_index]

        # Sort the emotion scores by value in descending order
        sorted_emotion_scores = dict(sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True))

        # Plot the emotion scores as a bar chart with transparency and border
        axes[i].bar(sorted_emotion_scores.keys(), sorted_emotion_scores.values(), alpha=0.7, edgecolor='black', color=colors[i])
        axes[i].set_xlabel('Cuisine')
        axes[i].set_ylabel(f'Emotion Score')
        axes[i].set_title(f'{emotion_name}')
        axes[i].tick_params(axis='x', rotation=45)  # Tilt the x-axis labels

    plt.subplots_adjust(wspace=2)  # Add more space between the graphs
    plt.tight_layout()
    plt.show()




    # Get the emotion scores for each sentence