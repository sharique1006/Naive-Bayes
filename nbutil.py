import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ngrams
from functools import lru_cache
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
import math

tokenizer = RegexpTokenizer(r'\w+')

def tokenize(doc):
	return tokenizer.tokenize(doc.lower())

def Map(x):
	return dict(zip(list(x), np.arange(len(x))))

def tokensWithCount(x):
	words, count = np.unique(x, return_counts=True)
	return words, count

def _stem(doc, p_stemmer, en_stop):
    tokens = word_tokenize(doc.lower())
    stopped_tokens = filter(lambda token: token not in en_stop, tokens)
    stemmed_tokens = map(lambda token: p_stemmer(token), stopped_tokens)
    return ' '.join(stemmed_tokens)

def getStemmedDocuments(docs):
	en_stop = set(stopwords.words('english'))
	ps = PorterStemmer()
	p_stemmer = lru_cache(maxsize=None)(ps.stem)

	stemmed_docs = []
	for item in docs:
		stemmed_doc = _stem(item, p_stemmer, en_stop)
		stemmed_docs.append(stemmed_doc)
		
	return stemmed_docs

def Dictionary(x):
	words = []
	for doc in x:
		words += set(tokenize(doc))
	words = set(words)
	dictionary = Map(words)
	return dictionary

def Features(x, y, d, l, clm):
	featureCount = np.zeros((l.shape[0], len(d)), dtype=int)
	for i in range(len(x)):
		doc_words, word_count = tokensWithCount(tokenize(x[i]))
		for j in range(len(doc_words)):
			featureCount[clm[y[i]], d[doc_words[j]]] += word_count[j]
	return featureCount

def computeParameters(l, lc, m_train, d, wcpc):
	phi = lc/m_train
	theta = np.zeros((l.shape[0], len(d)))
	for i in range(len(l)):
		x = wcpc[i].sum() + len(d)
		for j in range(len(d)):
			theta[i][j] = (wcpc[i][j] + 1)/x

	return phi, theta

def predict(x, log_phi, log_theta, d, labels):
	prediction = []
	pred_prob = []
	for doc in x:
		words = tokenize(doc)
		probs = np.zeros(len(labels))
		for w in words:
			try:
				probs += log_theta[:, d[w]]
			except KeyError:
				continue
		probs += log_phi
		predicted_label = labels[np.argmax(probs)]
		prediction.append(predicted_label)
		pred_prob.append(np.exp(np.max(probs)))

	return np.array(prediction), np.array(pred_prob)

def Accuracy(y, prediction):
	binpred = (y == prediction).astype(int)
	accuracy = (y == prediction).sum()/float(len(y))
	return accuracy, binpred

def NaiveBayesAccuracy(x, y, log_theta, log_phi, d, labels):
	prediction, pred_prob = predict(x, log_phi, log_theta, d, labels)
	accuracy, binpred = Accuracy(y, prediction)
	return prediction, accuracy, pred_prob, binpred

def RandomAccuracy(y, labels):
	prediction = np.random.randint(0, labels.shape[0], len(y))
	accuracy, binpred = Accuracy(y, prediction)
	return accuracy

def MaxCountAccuracy(y, labels, lc):
	prediction = labels[lc.argmax()]
	accuracy, binpred = Accuracy(y, prediction)
	return accuracy

def ConfusionMatrix(y, pred, labels, clm):
	confusion_matrix = np.zeros((labels.shape[0], labels.shape[0]))
	for l, p in zip(y, pred):
		confusion_matrix[clm[p], clm[l]] += 1

	TP = confusion_matrix.trace()
	FP = confusion_matrix.sum(axis=0)
	FN = confusion_matrix.sum(axis=1)
	Precision = TP/(TP + FP)
	Recall = TP/(TP + FN)
	#print("True Positives = ", TP)
	#print("False Positives = ", FP)
	#print("False Negatives = ", FN)
	#print("Precision = ", Precision)
	#print("Recall = ", Recall)

	return confusion_matrix

def PlotConfusionMatrix(confusion_matrix, labels, file):
	fig = plt.figure(figsize=(8,8))
	ax = fig.gca()
	_ = sns.heatmap(confusion_matrix,annot=True,cmap="Blues",xticklabels=labels,yticklabels=labels,fmt='g')
	ax.set_xlabel("Actual Class")
	ax.set_ylabel("Predicted Class")
	plt.title("Confusion Matrix",y=1.08)
	ax.xaxis.tick_top()
	ax.xaxis.set_label_position('top')
	plt.savefig(file)
	#plt.show()
	plt.close()

def ROC(binpred, pred_prob, file):
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(2):
	    fpr[i], tpr[i], _ = roc_curve(binpred, pred_prob)

	plt.figure()
	plt.plot(fpr[1], tpr[1])
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.savefig(file)
	#plt.show()
	plt.close()

def ngram(w, n):
	p = ngrams(w, n)
	l = []
	for y in p:
		valid = True
		for z in y:
			if len(z) <= 4:
				valid = False
				break
		if valid:
			l.append(' '.join(y))
	return l 

def NgramDictionary(x, n):
	words = []
	bigram = []
	for doc in x:
		w = tokenize(doc)
		words += set(w)
		for i in range(1, n):
			bigram += set(ngram(w, i+1))
	words = set(words)
	bigram = set(bigram)
	final = words.union(bigram)
	dictionary = Map(final)
	return dictionary

def NgramFeatures(x, y, n, d, l, clm):
	featureCount = np.zeros((l.shape[0], len(d)), dtype=int)
	for i in range(len(x)):
		tokens = tokenize(x[i])
		doc_words, word_count = tokensWithCount(tokens)
		for j in range(len(doc_words)):
			featureCount[clm[y[i]], d[doc_words[j]]] += word_count[j]
		for j in range(1, n):
			doc_features, feature_count = tokensWithCount(ngram(tokens, j+1))
			for k in range(len(doc_features)):
				featureCount[clm[y[i]], d[doc_features[k]]] += feature_count[k]
	return featureCount