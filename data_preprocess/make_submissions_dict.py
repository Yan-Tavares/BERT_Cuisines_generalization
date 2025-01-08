import json
import zipfile
import os

###################################
# Unzip the data files
###################################

zip_file_path = 'data_preprocess/recipes.zip'
extraction_dir = 'data_preprocess/unzipped_recipes'

# Create the extraction directory if it doesn't exist
os.makedirs(extraction_dir, exist_ok=True)

# Unzip the file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_dir)

print("Recipes unzipped.")

zip_file_path = 'data_preprocess/cooking.zip'
extraction_dir = 'data_preprocess/unzipped_cooking'

# Create the extraction directory if it doesn't exist
os.makedirs(extraction_dir, exist_ok=True)

# Unzip the file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_dir)

print("Cooking unzipped.")

###################################
# Load the submissions and comments
###################################

# Delete directory and its contents
def delete_directory(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            delete_directory(item_path)
        else:
            os.remove(item_path)

    os.rmdir(directory)

def filter_and_collect_submission_file(file_path):
    '''
    Generator function to process an NDJSON file and yield each line as a dictionary.
    This function reads the file line by line to reduce memory usage.
    '''
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            processed_data = {
                'id': data.get('id'),
                'title': data.get('title', ''),
                'selftext': data.get('selftext', ''),
            }

            yield processed_data

def filter_and_collect_comments_file(file_path):

    with open(file_path, encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            processed_data = {
                'id': data.get('id'),
                'link_id': data.get('link_id').split('_')[-1],  # Match to submission `id`
                'body': data.get('body', '')
            }

            yield processed_data

def link_comments_to_submissions(submissions, comments):
    '''
    Combine the comments with their corresponding submissions.
    '''
    # Create a comments_dict to map submission IDs to lists of comments
    # Iterate over each comment
    # Extract the link_id from each comment, which corresponds to a submission ID
    # If the link_id is not already in the dictionary, add it with an empty list
    # Append the comment's body to the list of comments for this link_id
    # In this way each key of comments_dict is a submission ID and the value is a list of comments for that submission

    comments_dict = {}
    for comment in comments:
        link_id = comment['link_id']

        if link_id not in comments_dict:
            comments_dict[link_id] = []
        
        comments_dict[link_id].append(comment['body'])

    # Iterate over each submission
    # Extract the submission ID
    # Link the comments to the submission by setting the 'comments' key
    # If there are no comments for this submission, set an empty list

    for sub in submissions:
        # Extract the submission ID
        sub_id = sub['id']
        
        sub['comments'] = comments_dict.get(sub_id, [])

    # Return the submissions with their linked comments
    return submissions

recipes_submissions = list(filter_and_collect_submission_file('data_preprocess/unzipped_recipes/recipes_submissions.ndjson'))
recipes_comments = list(filter_and_collect_comments_file('data_preprocess/unzipped_recipes/recipes_comments.ndjson'))

print(f"Loaded {len(recipes_submissions)} recipes submissions and {len(recipes_comments)} comments.")
delete_directory('data_preprocess/unzipped_recipes') # Free up disk space

# Combine the comments with their corresponding submissions
recipes_submissions = link_comments_to_submissions(recipes_submissions, recipes_comments)
del recipes_comments  # Free up memory

cooking_submissions = list(filter_and_collect_submission_file('data_preprocess/unzipped_cooking/cooking_submissions.ndjson'))
cooking_comments = list(filter_and_collect_comments_file('data_preprocess/unzipped_cooking/cooking_comments.ndjson'))

print(f"Loaded {len(cooking_submissions)} cooking submissions and {len(cooking_comments)} comments.")
delete_directory('data_preprocess/unzipped_cooking') # Free up disk space

# Combine the comments with their corresponding submissions
cooking_submissions = link_comments_to_submissions(cooking_submissions, cooking_comments)
del cooking_comments  # Free up memory

###################################
# Join the two datasets
###################################
joined_submissions = recipes_submissions + cooking_submissions
del recipes_submissions, cooking_submissions  # Free up memory

print(f"Joined {len(joined_submissions)} submissions.")


###################################
# Preprocess the text data
###################################
import emoji
import re

def preprocess_text(text):
    # stop_words = set(stopwords.words('english'))
    # lemmatizer = WordNetLemmatizer()
    
    # # Lowercase
    # text = text.lower()
    # # Remove punctuation and numbers
    # text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    # # Tokenize
    # words = word_tokenize(text)
    # # Remove stopwords and lemmatize
    # words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # Remove URLs using regex
    text = re.sub(r'http\S+', '', text)

    # Remove emojis
    text = emoji.replace_emoji(text, replace='')

    output = text

    return output


###################################
# Filter submissions without comments or with only 'removed' or 'deleted' comments
###################################

# Remove all comments that are only 'removed','deleted' or if starts with "Your submission has been removed"
for sub in joined_submissions:
    sub['processed_comments'] = [comment for comment in sub['comments'] if comment not in ['removed', 'deleted','[removed]', '[deleted]'] and not comment.startswith("Your submission has been removed")]

# Remove all selftexts that are only 'removed' or 'deleted'
for sub in joined_submissions:
    sub['processed_selftext'] = sub['selftext'] if sub['selftext'] not in ['removed', 'deleted','[removed]', '[deleted]'] else ''

# Remove all comments that have less than 10 words
min_words = 10
for sub in joined_submissions:
    sub['processed_comments'] = [comment for comment in sub['processed_comments'] if len(("word one"+ comment).split()) >= min_words+2]
    # Add a dummy word to the comment to account for the case where the comment is a single word

# Remove submissions with no comments
joined_submissions = [sub for sub in joined_submissions if len(sub['processed_comments']) != 0]
print(f"Submissions with comments: {len(joined_submissions)} ")

# Apply preprocessing
count = 0
for sub in joined_submissions:
    count += 1
    print(f"Processing submission {count}", end="\r")
    sub['processed_title'] = preprocess_text(sub['title'])
    sub['processed_selftext'] = preprocess_text(sub['processed_selftext'])
    sub['processed_comments'] = [preprocess_text(comment) for comment in sub['processed_comments']]


###################################
# Save data
###################################

# Save to a JSON file
with open('data_preprocess/submissions_dict.json', 'w') as f:
    json.dump(joined_submissions, f, indent=4)

print("Processed data saved.")