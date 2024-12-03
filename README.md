code 1: 

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')

# Input text
text = "This is an example sentence demonstrating part of speech tagging."

# Tokenize and remove stop words
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in word_tokenize(text) if word.lower() not in stop_words]

# Perform POS tagging
pos_tags = nltk.pos_tag(filtered_words)

# Output
print(pos_tags)

------------------------------------------------------------------------------------------------------------



CODE 2


from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "This is a sample document.",
    "This document is another sample document.",
    "And this is yet another example document."
]

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Compute TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(documents)

# Display TF-IDF values
for word, idx in vectorizer.vocabulary_.items():
    print(f"{word}: {vectorizer.idf_[idx]:.3f}")



-----------------------------------------------------------------------------------------



CODE 3


import nltk
from collections import defaultdict, Counter
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk import ConditionalFreqDist

# Download required resources
nltk.download('punkt')

class NgramModel:
    def __init__(self, n):
        self.n = n
        self.model = defaultdict(Counter)

    def train(self, text):
        tokens = word_tokenize(text.lower())
        n_grams = ngrams(tokens, self.n)
        for gram in n_grams:
            prefix, next_word = tuple(gram[:-1]), gram[-1]
            self.model[prefix][next_word] += 1

    def predict(self, context):
        context = tuple(context[-(self.n - 1):])
        if context in self.model:
            return self.model[context].most_common(1)[0][0]
        return None

# Example usage
text = "This is a simple example. This example is for N-gram language modeling."
model = NgramModel(n=2)  # Bigram model
model.train(text)

# Predict next word
context = ["this", "is"]
print("Next word:", model.predict(context))




----------------------------------------------------------------------------


CODE 4A:

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Download required resources
nltk.download('punkt')

# Sample corpus
corpus = [
    "This is a simple example.",
    "Word embeddings are useful for NLP tasks.",
    "Word2Vec captures semantic relationships between words."
]

# Preprocess: Tokenize sentences
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

# Get vector for a word
word_vector = model.wv['word']  # Replace 'word' with any word in the vocabulary
print(f"Vector for 'word': {word_vector}")

# Find most similar words
similar_words = model.wv.most_similar('word', topn=5)  # Replace 'word' as needed
print("Most similar words:", similar_words)





CODE 4B:


from google.colab import files
uploaded = files.upload()




from gensim.models import KeyedVectors

# Load GloVe pre-trained embeddings (download required .txt file first)
glove_path = 'glove.6B.100d.txt'  # Update path to GloVe file
glove_model = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)

# Get vector for a word
glove_vector = glove_model['word']  # Replace 'word' with your word
print(f"GloVe vector for 'word': {glove_vector}")











CODE 4C:


from gensim.models import FastText

# Train FastText model on the tokenized corpus
fasttext_model = FastText(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

# Get vector for a word
fasttext_vector = fasttext_model.wv['word']  # Replace 'word' with your word
print(f"FastText vector for 'word': {fasttext_vector}")

# Similarity
fasttext_similar = fasttext_model.wv.most_similar('word', topn=5)
print("Most similar words (FastText):", fasttext_similar)
