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

dictionary = Dictionary(reviews_train)
print("Number of Words in Dictionary = {}".format(len(dictionary)))
labels, label_counts = tokensWithCount(stars_train)
class_label_map = Map(labels)
features = Features(reviews_train, stars_train, dictionary, labels, class_label_map)
print("Features Loaded")
phi, theta = computeParameters(labels, label_counts, m_train, dictionary, features)
log_theta = np.log(theta)
log_phi = np.log(phi)
train_pred, train_accuracy, train_pred_prob, train_binpred = NaiveBayesAccuracy(reviews_train, stars_train, log_theta, log_phi, dictionary, labels)
test_pred, test_accuracy, test_pred_prob, test_binpred = NaiveBayesAccuracy(reviews_test, stars_test, log_theta, log_phi, dictionary, labels)
random_accuracy = RandomAccuracy(stars_test, labels)
maxcount_accuracy = MaxCountAccuracy(stars_test, labels, label_counts)

print("Naive Bayes Train Accuracy = {}".format(train_accuracy))
print("Naive Bayes Test Accuracy = {}".format(test_accuracy))
print("Random Test Accuracy = {}".format(random_accuracy))
print("Maximum Count Test Accuracy = {}".format(maxcount_accuracy))

f = open(output_file, 'w')
for pred in test_pred:
	print((int)(pred), file=f)

confusion_matrix = ConfusionMatrix(stars_test, test_pred, labels, class_label_map)
PlotConfusionMatrix(confusion_matrix, labels, 'ConfusionMatrix.png')
ROC(test_binpred, test_pred_prob, 'ROC.png')