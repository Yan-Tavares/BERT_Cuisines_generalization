import json
import nltk
import wikipediaapi
import re
import nltk


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

    # Remove everything after 'See also' or 'Further reading' sections
    # This includes content in the sections and the sections themselves
    
    # Remove 'See also' section and everything that follows
    text = re.sub(r"See also.*", "", text, flags=re.DOTALL)
    
    # Remove 'Further reading' section and everything that follows
    text = re.sub(r"Further reading.*", "", text, flags=re.DOTALL)
    
    # Optionally remove any other sections you don't want, like 'External links'
    text = re.sub(r"External links.*", "", text, flags=re.DOTALL)
    
    return text

def split_into_sentence_groups(text):
    # Use NLTK's sent_tokenize to split the text into sentences
    sentences = nltk.sent_tokenize(text)

    # Group sentences into three
    sentence_groups = [' '.join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]

    # # Split by paragraphs \n and keep them in the sentence_pairs list
    # final_sentences = []
    # for group in sentence_groups:
    #     paragraphs = group.split('\n')
    #     final_sentences.extend(paragraphs)

    # Remove empty sentences
    final_sentences = sentence_groups
    final_sentences = [sentence for sentence in final_sentences if sentence.strip() != '']

    return final_sentences


###############################
# Fetch articles for each cuisine
###############################
articles_dict = {}

# 1 Italian articles:
italian_titles = ["Italian cuisine", "List of Italian foods and drinks","Cuisine of Abruzzo", "Apulian cuisine", "Cuisine of Abruzzo", "Cuisine of Basilicata"]#, "List of Italian foods and drinks","Cuisine of Abruzzo","Apulian cuisine","Cuisine of Basilicata", "Pizza", "Pasta", "Lasagne", "Risotto", "Sfogliatella", "Prosciutto"]
italian_articles = {title: fetch_article(title) for title in italian_titles}
articles_dict['Italian'] = italian_articles

# 2 French articles:
french_titles = ["French cuisine", "List of French dishes", "List of French desserts", "Lyonnaise cuisine","Basque cuisine","Cuisine of Corsica","Cuisine of Reunion"]#,,"List of French desserts", "Croissant", "Baguette", "Ratatouille", "Coq au vin", "Quiche Lorraine", "Bouillabaisse", "Macaron","Soufflé"]
french_articles = {title: fetch_article(title) for title in french_titles}
articles_dict['French'] = french_articles

# 3 American articles:
american_titles = ["American cuisine", "List of American foods", "List of American desserts", "Cuisine of California", "Cuisine of New England"]#,"","","" ,"Hamburger", "Hot dog", "BBQ", "Mac and cheese", "Fried chicken"]
american_articles = {title: fetch_article(title) for title in american_titles}
articles_dict['American'] = american_articles

# 4 Mexican articles:
mexican_titles = ["Mexican cuisine", "List of Mexican dishes","Mexican-American cuisine", "Cuisine of Mexico City", "Oaxacan cuisine","Cuisine of Chiapas","Cuisine of Veracruz","Cuisine of the Southwestern United States"]#, "Taco", "Burrito", "Enchilada", "Guacamole", "Tamale", "Quesadilla"]
mexican_articles = {title: fetch_article(title) for title in mexican_titles}
articles_dict['Mexican'] = mexican_articles

# 5 Thai articles:
thai_titles = ["Thai cuisine","List of Thai ingredients","List of Thai dishes", "List of Thai desserts and snacks","Thai salads"]#, "Pad Thai", "Green curry", "Tom Yum", "Green papaya salad", "Massaman curry", "Mango sticky rice"]
thai_articles = {title: fetch_article(title) for title in thai_titles}
articles_dict['Thai'] = thai_articles

# 6 Greek articles:
greek_titles = ["Greek cuisine","List of Greek dishes","Greek Macedonian cuisine","Cypriot cuisine","Cretan cuisine"]#, "Souvlaki", "Moussaka", "Tzatziki", "Feta", "Greek salad", "Baklava", "Pita", "Gyros", "Baklava"]
greek_articles = {title: fetch_article(title) for title in greek_titles}
articles_dict['Greek'] = greek_articles

# 7 Indian articles:
indian_titles = ["Indian cuisine","List of Indian dishes","Cuisine of Haryana","Arunachali cuisine","Andhra cuisine","Assamese cuisine"]#, "Biryani", "Butter chicken", "Samosa", "Tandoori", "Rogan josh", "Chole bhature"]
indian_articles = {title: fetch_article(title) for title in indian_titles}
articles_dict['Indian'] = indian_articles

# 8 Japanese articles:
japanese_titles = ["Japanese cuisine","List of Japanese dishes", "Japanese regional cuisine","Okinawan cuisine","Sushi"]#, "Sushi", "Ramen", "Tempura", "Sashimi", "Okonomiyaki", "Takoyaki"]
japanese_articles = {title: fetch_article(title) for title in japanese_titles}
articles_dict['Japanese'] = japanese_articles

# 9 Spanish articles:
spanish_titles = ["Spanish cuisine","List of Spanish dishes","Asturian cuisine","Aragonese cuisine","Basque cuisine","Manchego cuisine","Cuisine of Menorca","Canarian cuisine","Manchego cuisine","Catalan cuisine"]#, "Paella", "Gazpacho", "Churros", "Jamón ibérico", "Pisto"]
spanish_articles = {title: fetch_article(title) for title in spanish_titles}
articles_dict['Spanish'] = spanish_articles

# 10 Chinese articles:
chinese_titles = ["Chinese cuisine","List of Chinese dishes","List of Chinese desserts","Chinese regional cuisine","Shandong cuisine","Sichuan cuisine","Cantonese cuisine","Fujian cuisine"]#, "Dim sum", "Peking duck", "Kung Pao chicken", "Spring roll", "Hot pot", "Chow mein"]
chinese_articles = {title: fetch_article(title) for title in chinese_titles}
articles_dict['Chinese'] = chinese_articles

# Unknow cuisines and not cuisine specific articles:
unknown_titles = ["Cooking","Outline of food preparation","List of cooking appliances","Meal","Vietnamese cuisine", "Philippine cuisine", "Dutch cuisine", "Australian cuisine", "Korean cuisine", "Brazilian cuisine", "Turkish cuisine", "Russian cuisine", "German cuisine", "British cuisine", "Swedish cuisine"]
unknown_articles = {title: fetch_article(title) for title in unknown_titles}
articles_dict['Unknown'] = unknown_articles

cuisine_sentences_dict = {}
for cuisine, articles in articles_dict.items():
    cuisine_sentences_dict[cuisine] = []  # Initialize the list for sentences of this cuisine
    for title, article_text in articles.items():

        # Split the article text into sentence pairs, each element of sentences is two sentences combined
        sentences = split_into_sentence_groups(article_text)
        
        # Add the processed sentences to the cuisine's list
        cuisine_sentences_dict[cuisine].extend(sentences)

###############################
# Compute ingredient frequency 
###############################

for cuisine, sentences in cuisine_sentences_dict.items():
    print(f"{cuisine} sentences: {len(sentences)}")

with open('data_preprocess/cuisine_sentences_dict.json', 'w', encoding='utf-8') as f:
    json.dump(cuisine_sentences_dict, f, indent=4)
    
print("Sentences dictionary saved to sentences_dict.json.")