from transformers import BertTokenizer

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Sample sentence
sentence = "I had this EXACT problem when I started cooking. Italian food was the easiest to learn because my first step was boiling pasta correctly or baking a chicken breasts at the right temperature for the right amount of time. It's warm, comfortable, but you get complacent really quickly.It's really easy to make good tasting (if non authentic) Italian food because tomato sauce and cheese is delicious on everything. Easiest way out of it is to just pick an entirely different culture and surprise yourself.To break my Italian rut, I immediately leapt into CHINESE and GREEK food.Use the familiarity you have gained with rice and noodles to make some tangy spicy Asian dishes or use the familiarity you've gained with meats to make kebabs, pita sandwiches, etc.I would also experiment on veggie dishes/roasted vegetables as a side with some pretty basic meats. Like you I am a carnivore and I think a good meal requires meat, But I got stuck in ruts of not learning how to properly make vegetables because I consider them the s*** I had to eat to get to the meat. Look at some specific vegetarian dishes and try them so you learn how to make vegetables stand on their own, and have a good baked chicken breast or something to go with it.I would also make a point to try making a meal out of a side, appetizer, or even a dip. Italian food has a lot of warm fatty flavors and it's good to experience things that taste fresh/crisp/non-complex. I would highly recommend trying homemade Tzatziki (Greek yogurt/cucumber dip) and eat the whole damn batch with pita bread and sticks of carrot and bell pepper, or on the Asian side make spring rolls with a simple peanut sauce.From there you can spread out where you feel. I'm not a fan of French cooking in general so I skipped right over it and landed in Spain to start practicing some of those dishes. Asia has such a wide variety of food so I started dipping specifically into Filipino and Vietnamese."

# Tokenize the sentence
tokens = tokenizer.tokenize(sentence)

# Output the result
print("Number of tokens:", len(tokens))
print("Tokens:", tokens[:30])  # Show the first 30 tokens for reference