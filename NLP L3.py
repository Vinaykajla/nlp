from collections import defaultdict, Counter
import math

class NGramLanguageModel:
    def __init__(self, n):
        self.n = n  # The size of the n-grams
        self.ngrams = defaultdict(Counter)  # Stores n-grams and their counts
        self.context_counts = Counter()  # Stores context counts for probability calculations

    def tokenize(self, text):
        # Tokenize text into words and add special start/end tokens
        tokens = text.lower().split()
        tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']
        return tokens

    def train(self, corpus):
        # Train the model with the given corpus
        for sentence in corpus:
            tokens = self.tokenize(sentence)
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i + self.n - 1])
                word = tokens[i + self.n - 1]
                self.ngrams[context][word] += 1
                self.context_counts[context] += 1

    def calculate_probability(self, context, word):
        # Calculate the probability of a word given its context
        if context in self.ngrams:
            word_count = self.ngrams[context][word]
            context_count = self.context_counts[context]
            return word_count / context_count
        return 0.0

    def generate_sentence(self, max_words=20):
        # Generate a sentence using the language model
        context = ('<s>',) * (self.n - 1)
        sentence = list(context)

        for _ in range(max_words):
            if context not in self.ngrams:
                break
            word = max(self.ngrams[context], key=self.ngrams[context].get)  # Greedy selection
            if word == '</s>':
                break
            sentence.append(word)
            context = tuple(sentence[-(self.n - 1):])

        return ' '.join(sentence[(self.n - 1):])  # Exclude the start tokens

# Example usage
if __name__ == "__main__":
    corpus = [
        "I love natural language processing.",
        "Language models are a part of AI.",
        "I love building AI models.",
    ]

    ngram_model = NGramLanguageModel(n=2)  # Bigram model
    ngram_model.train(corpus)

    # Test the model
    context = ('i',)
    word = 'love'
    prob = ngram_model.calculate_probability(context, word)
    print(f"Probability of '{word}' given context {context}: {prob:.4f}")

    # Generate a sentence
    print("Generated Sentence:", ngram_model.generate_sentence())
