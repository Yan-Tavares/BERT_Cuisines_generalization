# BERT_Cuisines_generalization
This research shows training and testing of BERT_Cuisines and also the generalization of the emotions related to the 10 most famous couisines based on data collected from subreddits r/Recipes and r/Cooking.

* Collect data from wikepia to train a BERT model to identify cuisines. Make a dictionary with sentences related to each cuisine. Select the relevant articles and use wikiapi to fetch the articles. Store cuisine_sentences_dict.
* Train BERT to classify the the sentences of wikipedia articles
* Use pushshift method to collect r/Recipes and r/Cooking post submissions and comments. Zip files for both datasets should be added to the data_preprocess folder as cooking.zip and recipes.zip.
* Preprocess and filter the data. Link the comments to the submissions as the key ['comments']
* Classify the sentences present in the submissions by cuisines
* Use roBERTa go emotions to classify the sentences
* Store the emotions from each sentence in the classified_sentences_dict. Each key refers to a cuisine and it contains thousands of emotion scores for the sentences.
* Generalize cuisines by averaging all the emotion scores for each cuisine