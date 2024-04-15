import os
import re


def load_reviews(directory):
    # Load and return a list of reviews from a directory
    reviews = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                reviews.append(file.read())
    return reviews

def preprocess_text(text):
    # Preprocess text by removing non-alphanumeric characters and lowercasing
    return re.sub('[^a-z0-9]+', ' ', text.lower()).split()

def build_vocabulary(reviews, max_words, min_frequency, max_frequency):
    # Build a vocabulary from the reviews based on word frequency
    word_counts = {}
    for review in reviews:
        for word in preprocess_text(review):
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1

    # Sort words by frequency and select the top 'max_words' words
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
    reduced_words = sorted_words[min_frequency:-max_frequency] if max_frequency > 0 else sorted_words[min_frequency:]
    return set(reduced_words[:max_words])

def vectorize_review(review, vocabulary):
    # Convert a review to a binary vector based on the presence of vocabulary words
    words = set(preprocess_text(review))
    return [1 if word in words else 0 for word in vocabulary]