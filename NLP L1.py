import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Ensure necessary datasets are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Input text
text = """Natural Language Processing with Python is interesting. 
It helps in analyzing and understanding human language."""

# Tokenize the text
tokens = word_tokenize(text)

# Load English stop words
stop_words = set(stopwords.words('english'))

# POS tagging with all tokens (including stop words)
pos_tags_with_stopwords = pos_tag(tokens)

# Filter out stop words and perform POS tagging
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
pos_tags_without_stopwords = pos_tag(filtered_tokens)

# Display results
print("POS Tags with Stop Words:")
print(pos_tags_with_stopwords)

print("\nPOS Tags without Stop Words:")
print(pos_tags_without_stopwords)
