from basicFunctions import *
from bayesImpl import *

if __name__ == "__main__":
    # Define file paths
    positive_training_path = ''
    negative_training_path = ''
    positive_testing_path = ''
    negative_testing_path = ''

    # Load reviews and labels
    positive_training_reviews = load_reviews(positive_training_path)
    negative_training_reviews = load_reviews(negative_training_path)
    positive_testing_reviews = load_reviews(positive_testing_path)
    negative_testing_reviews = load_reviews(negative_testing_path)

    # Combine reviews and create labels
    all_training_reviews = positive_training_reviews + negative_training_reviews
    labels = [1] * len(positive_training_reviews) + [0] * len(negative_training_reviews)

    # Create vocabulary and vectorize reviews
    vocabulary = build_vocabulary(all_training_reviews, max_words=1000, min_frequency=50, max_frequency=30)
    vectorized_training_reviews = [vectorize_review(review, vocabulary) for review in all_training_reviews]

    # Train and evaluate Bernoulli Naive Bayes classifier
    class_priors, word_probabilities = train_naive_bayes(vectorized_training_reviews, labels)
    positive_test_vectors = [vectorize_review(review, vocabulary) for review in positive_testing_reviews]
    negative_test_vectors = [vectorize_review(review, vocabulary) for review in negative_testing_reviews]
    test_reviews = positive_test_vectors + negative_test_vectors
    test_labels = [1] * len(positive_test_vectors) + [0] * len(negative_test_vectors)

    training_accuracy = evaluate_classifier(vectorized_training_reviews, labels, class_priors, word_probabilities)
    test_accuracy = evaluate_classifier(test_reviews, test_labels, class_priors, word_probabilities)

    print(f"Training Accuracy: {training_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")