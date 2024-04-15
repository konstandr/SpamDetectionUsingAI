import re
import os
import numpy as np
import basicFunctions

def train_naive_bayes(training_reviews, training_labels):
    # Train a Bernoulli Naive Bayes Classifier
    num_documents = len(training_reviews)
    num_words = len(training_reviews[0])
    classes = set(training_labels)

    class_priors = {cls: 0 for cls in classes}
    word_probabilities = {cls: [1] * num_words for cls in classes}

    # Count the presence of words in each class
    for document, label in zip(training_reviews, training_labels):
        class_priors[label] += 1
        for i, word in enumerate(document):
            word_probabilities[label][i] += word

    # Convert counts to probabilities
    for cls in classes:
        class_priors[cls] /= num_documents
        total_presences = sum(word_probabilities[cls])
        word_probabilities[cls] = [word / total_presences for word in word_probabilities[cls]]

    return class_priors, word_probabilities

def predict_class(review, class_priors, word_probabilities):
    # Predict the class of a review using the Bernoulli Naive Bayes model
    max_class, max_prob = None, float('-inf')

    for cls in class_priors.keys():
        log_prob = np.log(class_priors[cls])
        for i, word in enumerate(review):
            if word:  # If the word is present
                log_prob += np.log(word_probabilities[cls][i])
            else:  # If the word is absent
                log_prob += np.log(1 - word_probabilities[cls][i])

        if log_prob > max_prob:
            max_class, max_prob = cls, log_prob

    return max_class

def evaluate_classifier(classified_reviews, true_labels, class_priors, word_probabilities):
    # Evaluate the classifier's accuracy
    correct_predictions = 0
    for review, label in zip(classified_reviews, true_labels):
        prediction = predict_class(review, class_priors, word_probabilities)
        if prediction == label:
            correct_predictions += 1

    accuracy = correct_predictions / len(classified_reviews)
    return accuracy

