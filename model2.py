import json
import sys
from nbutil import *

train_file = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

train_data = [json.loads(line) for line in open(train_file, 'r')]
test_data = [json.loads(line) for line in open(test_file, 'r')]
m_train = len(train_data)
m_test = len(test_data)

reviews_train = [train_data[i]['text'] for i in range(m_train)]
stars_train = [train_data[i]['stars'] for i in range(m_train)]
reviews_test = [test_data[i]['text'] for i in range(m_test)]
stars_test = [test_data[i]['stars'] for i in range(m_test)]

print("Loaded Train & Test Data")

stemmed_reviews_train = getStemmedDocuments(reviews_train)
stemmed_reviews_test = getStemmedDocuments(reviews_test)
print("Loaded Stemmed Data")
stemmed_dictionary = Dictionary(stemmed_reviews_train)
print("Number of Words in Dictionary = {}".format(len(stemmed_dictionary)))
labels, label_counts = tokensWithCount(stars_train)
class_label_map = Map(labels)
stemmedFeatures = Features(stemmed_reviews_train, stars_train, stemmed_dictionary, labels, class_label_map)
print("Stemmed Features Loaded")
stemmed_phi, stemmed_theta = computeParameters(labels, label_counts, m_train, stemmed_dictionary, stemmedFeatures)
stemmed_log_theta = np.log(stemmed_theta)
stemmed_log_phi = np.log(stemmed_phi)
stemmed_train_pred, stemmed_train_accuracy, stemmed_train_pred_prob, stemmed_train_binpred = NaiveBayesAccuracy(stemmed_reviews_train, stars_train, stemmed_log_theta, stemmed_log_phi, stemmed_dictionary, labels)
stemmed_test_pred, stemmed_test_accuracy, stemmed_test_pred_prob, stemmed_test_binpred = NaiveBayesAccuracy(stemmed_reviews_test, stars_test, stemmed_log_theta, stemmed_log_phi, stemmed_dictionary, labels)

print("Stemmed Naive Bayes Train Accuracy = {}".format(stemmed_train_accuracy))
print("Stemmed Naive Bayes Test Accuracy = {}".format(stemmed_test_accuracy))

f = open(output_file, 'w')
for pred in stemmed_test_pred:
	print((int)(pred), file=f)