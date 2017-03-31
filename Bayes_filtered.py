import glob
from collections import Counter
from nltk.corpus import stopwords
import string
import numpy
import sys

ham_train = sys.argv[1]
spam_train = sys.argv[2]
ham_test = sys.argv[3]
spam_test = sys.argv[4]
Lamda = sys.argv[5]
iteration = sys.argv[6]
learning_rate = sys.argv[7]

# Training Data  Set
ham_train = glob.glob(ham_train+"/*.txt")
spam_train = glob.glob(spam_train+"/*.txt")

# Testing Data set
ham_test = glob.glob(ham_test+"/*.txt")
spam_test = glob.glob(spam_test+"/*.txt")

# the function reads the file and returns a counter of words for each file
def read_file(filename, stop_words, bayes):
    train_file = ""
    for file_names in filename:
        files = open(file_names)
        train_file += files.read()
        # remove for bayes
    #if bayes == 1:
    translator = str.maketrans('', '', string.punctuation)
    train_file = train_file.translate(translator)
    translator = str.maketrans('', '', string.digits)
    train_file = train_file.translate(translator)
    train_file = Counter(train_file.split())

    if stop_words:
        train_file = Counter([word for word in train_file if word not in stopwords.words('english')])
    return train_file

# the function calculate sthe probability of each word occuring in a particular file
def word_probability(unique_words):
    word_prob = {}
    total_count = 0
    for i in unique_words:
        total_count += unique_words[i]
    for i in unique_words:
        word_prob[i] = float(unique_words[i]) / int(total_count)
    return word_prob

# calculates the probability of class given the word in the testing file
def probability_given_word(unique_word):
    prob = {}
    for word_bag in bag_of_words:
        prob[word_bag] = float(unique_word[word_bag]) / float(bag_of_words[word_bag] + len(bag_of_words))
    return prob

# the function calculates the finnal probability of the testing file if it is a spam or not
def calculate_probability(test_file):
    count_spam = 0.0
    count_ham = 0.0

    for file_name in test_file:
        ln_prob_spam = 0.0
        ln_prob_ham = 0.0
        prob_ham = 1.0
        prob_spam = 1.0
        # base = 1.0

        file = open(file_name)
        test_file_ham = file.read()
        test_file_ham = Counter(test_file_ham.split())
        # print(test_file_ham)
        for word in test_file_ham:
            if word in bag_of_words:
                # ln_prob_ham = ln_prob_ham + numpy.log(prob_ham_given_word[word])
                # ln_prob_spam = ln_prob_spam + numpy.log(prob_spam_given_word[word])
                # base = ln_prob_ham + ln_prob_spam
                prob_ham = prob_ham * prob_ham_given_word[word]
                prob_spam = prob_spam * prob_spam_given_word[word]
                # print(str(base) + "    " + str(math.exp(base)) + "-" + word)

        # print(-ln_prob_spam + base)
        if prob_ham + prob_spam:
            if prob_ham / (prob_ham + prob_spam) > 0.5:
                count_ham += 1
            else:
                count_spam += 1
    return count_ham, count_spam

# performs add one smoothing on the set of words from the training files and on bag of words
def laplace_smoothing(train_1, train_2, bag):
    for word in bag:
        train_1[word] += 1
        train_2[word] += 1
    return train_1, train_2

# calculate the final accuracy of the classifiers
def accuracy_bayes(stop_word):
    unique_train_ham = read_file(ham_train, stop_word, 0)
    # SPAM dictionary of training data with their word count
    unique_train_spam = read_file(spam_train, stop_word, 0)

    # HAM word probability
    global ham_word_prob
    ham_word_prob = word_probability(unique_train_ham)

    # SPAM word probability
    global spam_word_prob
    spam_word_prob = word_probability(unique_train_spam)

    # Bag of words
    global bag_of_words
    bag_of_words = unique_train_ham + unique_train_spam

    # Laplace smoothing function
    unique_train_ham, unique_train_spam = (laplace_smoothing(unique_train_ham, unique_train_spam, bag_of_words))

    # probability of ham and spam using laplace smoothing (Add one)
    global prob_ham_given_word
    global prob_spam_given_word
    prob_ham_given_word = probability_given_word(unique_train_ham)
    prob_spam_given_word = probability_given_word(unique_train_spam)

    count_ham, count_spam = calculate_probability(ham_test)

    true = count_ham
    total = count_ham + count_spam

    count_ham, count_spam = calculate_probability(spam_test)

    true += count_spam

    total += count_ham + count_spam

    if stop_word == 1:
        try:
            print("Naive Bayes - filtered - Total Accuracy - " + str(float(true / total) * 100))
        except ZeroDivisionError as err:
            print('Handling run-time error:', err)
    else:
        try:
            print("Naive Bayes - unfiltered - Total Accuracy - " + str(float(true / total) * 100))
        except ZeroDivisionError as err:
            print('Handling run-time error:', err)


accuracy_bayes(stop_word=0) # accuracy without removing stop-words
accuracy_bayes(stop_word=1) # accuracy with removing stop-words

