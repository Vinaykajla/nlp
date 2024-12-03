import gensim
from gensim.models import Word2Vec, FastText
import gensim.downloader as api

# Sample text corpus (ensure consistent casing)
sentences = [
    "This is a sentence about Dogs.",  # Consider including both uppercase and lowercase
    "Dogs are furry friends.",
    "Cats are also cute animals."
]

# 1. Word2Vec (address case sensitivity and minimum word count)
# Create a Word2Vec model (adjust min_count if necessary)
model_w2v = Word2Vec(sentences, min_count=1, vector_size=100, window=5)  # Lower min_count for flexibility

# Check if 'dog' is in the vocabulary (lowercase for consistency)
if 'dog' in model_w2v.wv:
    word_vector_w2v = model_w2v.wv['dog']
    print("Word2Vec vector for 'dog':", word_vector_w2v)
else:
    print("'dog' not found in Word2Vec vocabulary.")

# 2. GloVe
# Download pre-trained GloVe embeddings
model_glove = api.load("glove-wiki-gigaword-100")

# Get the vector representation of a word (GloVe might be less case-sensitive)
try:
    word_vector_glove = model_glove['dog']
    print("GloVe vector for 'dog':", word_vector_glove)
except KeyError:
    print("'dog' not found in GloVe vocabulary (try 'dogs' or adjust case sensitivity).")

# 3. FastText (potentially better with rare words)
# Create a FastText model
model_fasttext = FastText(sentences, min_count=1, vector_size=100, window=5)

# Get the vector representation of a word (FastText might handle rare words better)
if 'dog' in model_fasttext.wv:
    word_vector_fasttext = model_fasttext.wv['dog']
    print("FastText vector for 'dog':", word_vector_fasttext)
else:
    print("'dog' not found in FastText vocabulary.")