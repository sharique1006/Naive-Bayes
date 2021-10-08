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
ngram_dictionary = NgramDictionary(stemmed_reviews_train, 2)
print("Number of Words in Dictionary = {}".format(len(ngram_dictionary)))
labels, label_counts = tokensWithCount(stars_train)
class_label_map = Map(labels)
ngramFeatures = NgramFeatures(stemmed_reviews_train, stars_train, 2, ngram_dictionary, labels, class_label_map)
print("Ngram Features Loaded")
ngram_phi, ngram_theta = computeParameters(labels, label_counts, m_train, ngram_dictionary, ngramFeatures)
ngram_log_theta = np.log(ngram_theta)
ngram_log_phi = np.log(ngram_phi)
ngram_train_pred, ngram_train_accuracy, ngram_train_pred_prob, ngram_train_binpred = NaiveBayesAccuracy(stemmed_reviews_train, stars_train, ngram_log_theta, ngram_log_phi, ngram_dictionary, labels)
ngram_test_pred, ngram_test_accuracy, ngram_test_pred_prob, ngram_train_binpred = NaiveBayesAccuracy(stemmed_reviews_test, stars_test, ngram_log_theta, ngram_log_phi, ngram_dictionary, labels)

print("Ngram Naive Bayes Train Accuracy = {}".format(ngram_train_accuracy))
print("Ngram Naive Bayes Test Accuracy = {}".format(ngram_test_accuracy))

f = open(output_file, 'w')
for pred in ngram_test_pred:
	print((int)(pred), file=f)
