import math
from sklearn.feature_extraction.text import CountVectorizer

# Input documents
documents = [
    "Natural Language Processing with Python is interesting.",
    "Python helps in analyzing and understanding human language.",
    "TF-IDF is a statistical measure for text analysis.",
]

# Step 1: Tokenization and TF Calculation
def compute_tf(doc):
    tf_dict = {}
    total_words = doc.split()
    total_count = len(total_words)
    for word in total_words:
        tf_dict[word] = tf_dict.get(word, 0) + 1
    # Normalize by total word count
    for word in tf_dict:
        tf_dict[word] = tf_dict[word] / total_count
    return tf_dict

# Step 2: IDF Calculation
def compute_idf(corpus):
    idf_dict = {}
    total_docs = len(corpus)
    for doc in corpus:
        for word in set(doc.split()):
            idf_dict[word] = idf_dict.get(word, 0) + 1
    for word in idf_dict:
        idf_dict[word] = math.log(total_docs / idf_dict[word]) + 1
    return idf_dict

# Step 3: TF-IDF Calculation
def compute_tfidf(tf_dict, idf_dict):
    tfidf_dict = {}
    for word, tf_value in tf_dict.items():
        tfidf_dict[word] = tf_value * idf_dict.get(word, 0)
    return tfidf_dict

# Preprocess: Convert documents to lowercase for consistency
documents = [doc.lower() for doc in documents]

# Calculate TF, IDF, and TF-IDF
tf_results = [compute_tf(doc) for doc in documents]
idf_result = compute_idf(documents)
tfidf_results = [compute_tfidf(tf, idf_result) for tf in tf_results]

# Display results
for i, tfidf in enumerate(tfidf_results):
    print(f"Document {i+1} TF-IDF:")
    for word, score in tfidf.items():
        print(f"  {word}: {score:.4f}")
    print()
