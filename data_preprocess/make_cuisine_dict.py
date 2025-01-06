import ndjson
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import torch



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


"""#**RQ 3**

- Pull wikpidea articles describing the cuisine, take also another article descrining how to make the 10 most popular foods from that cuisine.
- Preprocess the articles, chop them in sentences, label as the cuisine
- Dowload a pre-trained uncased BERT with a softmax at the end to perform the classification
- Use the text pieces and their label to perform the training of the FC layer
- Dowload a version of BERT that can perform sentiment alysis.
- Apply the submissions to BERT Cuisines and classify which cuisine the submission refers to
- Appply the same submission to BERT Sentiment and classify the sentiment score
- Group the sentment scores per cuisine.
- Average all the scores --> Sentiment per cuisine based on reddit
"""

import wikipediaapi
import re
import nltk
# nltk.download('punkt')

# Set the custom user-agent header
headers = {
    "User-Agent": "MyApp/1.0 (https://www.myapp.com; support@myapp.com)"
}

# Create the Wikipedia object with the headers
wiki_wiki = wikipediaapi.Wikipedia('en', headers=headers)

def fetch_article(title):
    page = wiki_wiki.page(title)
    if page.exists():
        # Fetch the article text and clean it
        text = page.text
        clean_text = remove_references(text)
        return clean_text
    else:
        print(f"Page '{title}' does not exist.")
        return None

def remove_references(text):
    # Remove reference tags like [1], [2], ... and HTML reference links
    clean_text = re.sub(r'\[\d+\]', '', text)  # Remove simple references like [1]
    clean_text = re.sub(r'<ref.*?>.*?</ref>', '', clean_text)  # Remove <ref> tags if they exist
    clean_text = re.sub(r'<[^>]+>', '', clean_text)  # Remove any other HTML tags if necessary
    return clean_text

def split_into_sentence_pairs(text):
    # Use NLTK's sent_tokenize to split the text into sentences
    sentences = nltk.sent_tokenize(text)

    # Group sentences into pairs
    sentence_pairs = [' '.join(sentences[i:i+2]) for i in range(0, len(sentences), 2)]

    return sentence_pairs


###############################
# Fetch articles for each cuisine
###############################
articles_dict = {}

# 1 Italian articles:
italian_titles = ["Italian cuisine", "Cuisine of Abruzzo", "Apulian cuisine","Cuisine of Basilicata"]#, "List of Italian foods and drinks","Cuisine of Abruzzo","Apulian cuisine","Cuisine of Basilicata", "Pizza", "Pasta", "Lasagne", "Risotto", "Sfogliatella", "Prosciutto"]
italian_articles = {title: fetch_article(title) for title in italian_titles}
articles_dict['Italian'] = italian_articles

# 2 French articles:
french_titles = ["French cuisine", "List of French dishes", "List of French desserts"]#,,"List of French desserts", "Croissant", "Baguette", "Ratatouille", "Coq au vin", "Quiche Lorraine", "Bouillabaisse", "Macaron","Soufflé"]
french_articles = {title: fetch_article(title) for title in french_titles}
articles_dict['French'] = french_articles

# 3 American articles:
american_titles = ["American cuisine", "Cuisine of California","Cuisine of New England","Cuisine of the Southern United States" ]#,"Cuisine of California","Cuisine of New England","Cuisine of the Southern United States" ,"Hamburger", "Hot dog", "BBQ", "Mac and cheese", "Fried chicken"]
american_articles = {title: fetch_article(title) for title in american_titles}
articles_dict['American'] = american_articles

# 4 Mexican articles:
mexican_titles = ["Mexican cuisine", "Antojito", "Cuisine of Mexico City", "Oaxacan cuisine"]#, "Taco", "Burrito", "Enchilada", "Guacamole", "Tamale", "Quesadilla"]
mexican_articles = {title: fetch_article(title) for title in mexican_titles}
articles_dict['Mexican'] = mexican_articles

# 5 Thai articles:
thai_titles = ["Thai cuisine","List of Thai ingredients","List of Thai dishes", "List of Thai desserts and snacks"]#, "Pad Thai", "Green curry", "Tom Yum", "Green papaya salad", "Massaman curry", "Mango sticky rice"]
thai_articles = {title: fetch_article(title) for title in thai_titles}
articles_dict['Thai'] = thai_articles

# 6 Greek articles:
greek_titles = ["Greek cuisine","List of Greek dishes","Greek Macedonian cuisine"]#, "Souvlaki", "Moussaka", "Tzatziki", "Feta", "Greek salad", "Baklava", "Pita", "Gyros", "Baklava"]
greek_articles = {title: fetch_article(title) for title in greek_titles}
articles_dict['Greek'] = greek_articles

# 7 Indian articles:
indian_titles = ["Indian cuisine","List of Indian dishes","Arunachali cuisine","Assamese cuisine"]#, "Biryani", "Butter chicken", "Samosa", "Tandoori", "Rogan josh", "Chole bhature"]
indian_articles = {title: fetch_article(title) for title in indian_titles}
articles_dict['Indian'] = indian_articles

# 8 Japanese articles:
japanese_titles = ["Japanese cuisine","List of Japanese dishes", "Japanese regional cuisine", "Yōshoku"]#, "Sushi", "Ramen", "Tempura", "Sashimi", "Okonomiyaki", "Takoyaki"]
japanese_articles = {title: fetch_article(title) for title in japanese_titles}
articles_dict['Japanese'] = japanese_articles

# 9 Spanish articles:
spanish_titles = ["Spanish cuisine","List of Spanish dishes","Aragonese cuisine","Basque cuisine","Manchego cuisine"]#, "Paella", "Gazpacho", "Churros", "Jamón ibérico", "Pisto"]
spanish_articles = {title: fetch_article(title) for title in spanish_titles}
articles_dict['Spanish'] = spanish_articles

# 10 Chinese articles:
chinese_titles = ["Chinese cuisine","List of Chinese dishes","List of Chinese desserts","Chinese regional cuisine"]#, "Dim sum", "Peking duck", "Kung Pao chicken", "Spring roll", "Hot pot", "Chow mein"]
chinese_articles = {title: fetch_article(title) for title in chinese_titles}
articles_dict['Chinese'] = chinese_articles

# Unknow cuisines and not cuisine specific articles:
unknown_titles = ["Cooking","Vietnamese cuisine", "Philippine cuisine", "Dutch cuisine", "Australian cuisine", "Korean cuisine", "Brazilian cuisine", "Turkish cuisine", "Russian cuisine", "German cuisine", "British cuisine", "Swedish cuisine"]
unknown_articles = {title: fetch_article(title) for title in unknown_titles}
articles_dict['Unknown'] = unknown_articles

sentences_dict = {}
for cuisine, articles in articles_dict.items():
    sentences_dict[cuisine] = []  # Initialize the list for sentences of this cuisine
    for title, article_text in articles.items():

        # Split the article text into sentence pairs, each element of sentences is two sentences combined
        sentences = split_into_sentence_pairs(article_text)
        
        # Preprocess each sentence
        for i in range(len(sentences)):
            sentences[i] = preprocess_text(sentences[i])

        # Add the processed sentences to the cuisine's list
        sentences_dict[cuisine].extend(sentences)

###############################
# Compute ingredient frequency 
###############################

# Collect the number of occurrences of each ingredient in the sentences dict 

print(f"\n---- Italian sentences ---- \nNumber of sentences: {len(sentences_dict['Italian'])} \nSentence example: {sentences_dict['Italian'][0] }")
print(f"---- French sentences ---- \nNumber of sentences: {len(sentences_dict['French'])} \nSentence example: {sentences_dict['French'][0] }")
print(f"n---- Unknown sentences ---- \nNumber of sentences: {len(sentences_dict['Unknown'])} \nSentence example: {sentences_dict['Unknown'][0] }\n")
# Save sentences_dict to a JSON file
with open('data_preprocess/sentences_dict.json', 'w') as f:
    json.dump(sentences_dict, f, indent=4)
    
print("Sentences dictionary saved to sentences_dict.json.")