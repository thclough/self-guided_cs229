#%%

import collections

import numpy as np

import util
import svm
import re

def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    # remove all punctuation, special characters, and numbers
    mod = message
    replace_chars = [" ',' "]
    for symbol in message:
        if symbol not in replace_chars and not symbol.isalpha() and symbol not in ["'", " "]:
            replace_chars.append(symbol)

    for char in replace_chars:
        mod = mod.replace(char, " ")

    mod = mod.lower()

    # split on one or more spaces
    word_list = re.split(r'\s+', mod)

    return word_list

    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    master_word_list = []
    for message in messages:
        word_list = get_words(message)
        master_word_list += word_list
    
    word_counts = collections.Counter(master_word_list)

    # filter to words that only only occur 4 times
    words_filtered = [word for word, count in word_counts.items() if count >= 5]

    # mapping 
    word_mapping = {word: idx for idx, word in enumerate(words_filtered)}

    return word_mapping
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    # create array to fill
    occurrences = np.zeros((len(messages), len(word_dictionary)))

    # go through messages and 
    for i, message in enumerate(messages):
        message_words = get_words(message)
        for word in message_words:
            # find word index and count up frequency if in dictionary
            j = word_dictionary.get(word, -1) #-1 if not in dictionary
            if j >= 0:
                occurrences[i,j] += 1

    return occurrences
    # *** END CODE HERE ***

""" train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')

word_dict = create_dictionary(train_messages[100:200])
freq_array = transform_text(train_messages[100:200], word_dict) """



def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    ham_matrix = matrix[labels==0, :]
    spam_matrix = matrix[labels==1, :]

    # calculate phi y=1 
    phi_1 = len(spam_matrix)/len(matrix)

    # calculate phi k|y=n with Laplace Smoothing, n in {0,1}
    V = matrix.shape[1] # number of vocab words

    phi_k_0 = (1 + np.sum(ham_matrix, axis=0)) / (V + np.sum(ham_matrix))
    phi_k_1 = (1 + np.sum(spam_matrix, axis=0)) / (V + np.sum(spam_matrix))
    
    return phi_1, phi_k_0, phi_k_1
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    phi_1, phi_k_0, phi_k_1 = model

    # use logarithms to compare the probabilities
    logprob_0 = np.sum(matrix * np.log(phi_k_0),axis=1) + np.log(1 - phi_1)
    logprob_1 = np.sum(matrix * np.log(phi_k_1),axis=1) + np.log(phi_1)

    labels = (logprob_1 > logprob_0).astype(int)

    return labels
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    _, phi_k_0, phi_k_1 = model

    token_strength = np.log(phi_k_1/phi_k_0)

    top_indices = np.argsort(token_strength)[-5::]
    top_indices = np.sort(top_indices)[::-1]
    
    top_nb_words = [word for word, idx in dictionary.items() if idx in top_indices]

    return top_nb_words
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    best_accuracy = 0
    best_radius = radius_to_consider[0]
    # *** START CODE HERE ***
    for radius in radius_to_consider:
        predicted_labels = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        accuracy = (predicted_labels == val_labels).mean()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_radius = radius

    return best_radius
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
