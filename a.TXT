import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


nltk.download("punkt")

nltk.download("averaged_perceptron_tagger")

nltk.download("stopwords")

text = "Natural Language Processing in an exiting feild of Artifical Intelligence"

stop_words = set(stopwords.words("english"))
stop_words

tokens = word_tokenize(text)
tokens

filter_words = []
for word in tokens:
    if word.lower() not in stop_words:
        filter_words.append(word)

filter_words

pos_tags = nltk.pos_tag(filter_words)
pos_tags







-----------------------------------------------2 --------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "Natural Language processing is amazing.",
    "Machine learning and NLP go hand in hand.",
    "TF-IDF helps find important words in a document."
]

vec = TfidfVectorizer()


tfidf_mat = vec.fit_transform(docs)

print(vec.get_feature_names_out())


print(tfidf_mat.toarray())


--------------------------3 --------------------------------------------



from nltk import ngrams
from collections import Counter

# Input text and n-gram size
text = "Natural language processing is fun and challenging."
N = 2  # Size of the n-grams

# Tokenize and create n-grams
tokens = text.lower().split()
n_grams = list(ngrams(tokens, N))

# Count n-grams and prefixes
counts = Counter(n_grams)
prefix_counts = Counter(ng[:-1] for ng in n_grams)

# Calculate probabilities
model = {ng: counts[ng] / prefix_counts[ng[:-1]] for ng in counts}

# Print the n-gram probabilities
print("N-gram Probabilities:", model)

--------------------------------------------4 --------------------------

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")

corpus = [
    "Natural language processing is a fascinating field.",
    "Word embeddings like Word2Vec are used in NLP.",
    "Deep learning models have revolutionized text analysis.",
]

tokenized_corpus = []
for sentence in corpus:
    sentence_lower = sentence.lower()
    tokenized_sentence = word_tokenize(sentence_lower)
    tokenized_corpus.append(tokenized_sentence)

word2vec_model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

word = "nlp"
if word in word2vec_model.wv:
    print(f"Word2Vec embedding for '{word}':\n{word2vec_model.wv[word]}")
else:
    print(f"Word '{word}' not in vocabulary.")




import gensim.downloader as api

# Download pre-trained GloVe embeddings
model = api.load("glove-wiki-gigaword-100")

# Get the vector representation of a word
word_vector = model['dog']
print(word_vector)


from gensim.models import FastText

# Sample text corpus (or load from a file)
sentences = [
    "This is a sentence about dogs.",
    "Dogs are furry friends.",
    "Cats are also cute animals."
]

# Create a FastText model (using vector_size instead of size)
model = FastText(sentences, min_count=1, vector_size=100, window=5)

# Get the vector representation of a word (optional)
word_vector = model.wv['dog']
print(word_vector)


-----------------------------------------------5 ----------------------------------------

import numpy as np

class SimpleTokenizer:
    def __init__(self):
        self.word_index = {"the": 1, "cat": 2, "sat": 3, "on": 4, "mat": 5, "is": 6, "happy": 7, "dog": 8, "sad": 9}
        self.index_word = {v: k for k, v in self.word_index.items()}
        
    def texts_to_sequences(self, texts):
        sequences = []
        words = texts[0].split()
        sequence = []
        for word in words:
            if word in self.word_index:
                sequence.append(self.word_index[word])
        sequences.append(sequence)
        return sequences

class SimpleLSTMModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def predict(self, input_sequence):
        random_index = np.random.randint(1, self.vocab_size)
        one_hot_vector = np.zeros(self.vocab_size)
        one_hot_vector[random_index] = 1
        return one_hot_vector

tokenizer = SimpleTokenizer()
model = SimpleLSTMModel(vocab_size=len(tokenizer.word_index) + 1)

def predict_next_word(model, tokenizer, input_text):
    encoded = tokenizer.texts_to_sequences([input_text])[0]
    pred = model.predict(encoded)
    predicted_word_index = np.argmax(pred)
    predicted_word = tokenizer.index_word.get(predicted_word_index, "<unknown>")
    return predicted_word

input_text = "the cat"
predicted_word = predict_next_word(model, tokenizer, input_text)

print(f"Predicted next word for '{input_text}': {predicted_word}")

------------------------------------------------6 -----------------------------------
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer

EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 5e-6
MAX_LENGTH = 50

# Prepare the dataset
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, max_length):
        self.input_ids = []
        for line in text:
            encodings = tokenizer(line, truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings['input_ids']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Ensure padding token is set for GPT-2
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Prepare text data
text_data = [
    "The quick brown fox jumps over the lazy dog.",
    "The sun sets in the west and rises in the east.",
    "Artificial Intelligence is transforming the world.",
    "Deep learning models are revolutionizing various industries.",
    "Natural Language Processing is a key area of artificial intelligence."
]

# Create dataset and dataloader
dataset = TextDataset(text_data, tokenizer, max_length=MAX_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Setup optimizer and device
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    for batch in dataloader:
        input_ids = batch.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item()}")

# Text generation
model.eval()
prompt = "Artificial Intelligence"
encoded_input = tokenizer(prompt, return_tensors='pt').to(device)
generated_ids = model.generate(encoded_input['input_ids'], max_length=MAX_LENGTH, pad_token_id=tokenizer.pad_token_id)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print(f"Generated Text: {generated_text}")

-------------------7------------------------------
import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
 
# Parameters 
SEQ_LEN = 5 
D_MODEL = 4 
 
# Sample Sequence Dataset 
sequence = torch.tensor([[1.0, 0.0, 1.0, 0.0], 
                         [0.0, 2.0, 0.0, 1.0], 
                         [1.0, 1.0, 1.0, 1.0], 
                         [0.0, 0.0, 2.0, 1.0], 
                         [1.0, 2.0, 0.0, 0.0]]) 
 
# Self-Attention Components 
class SelfAttention(nn.Module): 
    def __init__(self, d_model): 
        super(SelfAttention, self).__init__() 
        self.query = nn.Linear(d_model, d_model) 
        self.key = nn.Linear(d_model, d_model) 
        self.value = nn.Linear(d_model, d_model) 
 
    def forward(self, x): 
        Q = self.query(x) 
        K = self.key(x) 
        V = self.value(x) 
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(K.size(-1)) 
        attention_weights = torch.softmax(attention_scores, dim=-1) 
        attention_output = torch.matmul(attention_weights, V) 
        return attention_output, attention_weights 
 
# Initialize Self-Attention 
self_attention = SelfAttention(D_MODEL) 
 
# Compute Attention Outputs and Weights 
attention_output, attention_weights = self_attention(sequence) 
 
# Visualize Attention Map 
plt.figure(figsize=(8, 6)) 
plt.imshow(attention_weights.detach().numpy(), cmap="viridis") 
plt.colorbar() 
plt.title("Attention Map") 
plt.xlabel("Key Positions") 
plt.ylabel("Query Positions") 
plt.show() 
 



