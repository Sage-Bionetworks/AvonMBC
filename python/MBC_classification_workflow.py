#Creates features
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import svm

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
#NLTK -> tokenizer, stemmer
import nltk
from nltk.collocations import *
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
#nltk.download("stopwords")
import pandas
import string
import re
from time import time
import logging
#Beautiful soup -> Gets rid of HTML and replaces non ascii characters
#pip install Beautifulsoup4
from bs4 import BeautifulSoup
import unicodedata 
import numpy
import pprint
from random import shuffle
#Get common functions
from common_functions import *
# -----------------------------------------------------------------------------------
# This is the workflow document, all functions are in common_functions.py
# -----------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------
#Normalization of text (Remove HTML, replace non-ascii characters, remove punctuation,
#remove lone numbers, stem words, remove stop words recombine into text)
#--------------------------------------------------------------------------------------
grants = pandas.read_csv('metastatic_grant_binary.csv')
total = grants['AwardTitle'] + ", " + grants['TechAbstract']

total_complete = normalize_text(total)

total_complete.reindex(numpy.random.permutation(total_complete.index))


train_data = total_complete[0:1500]
train_result = grants['Metastasis_YN'][0:1500]
test_data = total_complete[1501:2236]
test_result = grants['Metastasis_YN'][1501:2236]

findBestParameters(train_data, train_result)

train_data = total_complete[0:1500]
train_result = grants['mt_binary'][0:1500]
test_data = total_complete[1501:2236]
test_result = grants['mt_binary'][1501:2236]

findBestParameters(train_data, train_result)


#----------------------------------------------------------------
# SCIKIT TOOLS -> Get features by using TfidF Vectorizer
#----------------------------------------------------------------
grants = pandas.read_csv('metastatic_grant_binary.csv')
total = grants['AwardTitle'] + ", " + grants['TechAbstract']

train_data = total_complete[0:1500]
train_result = grants['pw_binary'][0:1500]
test_data = total_complete[1501:2236]
test_result = grants['pw_binary'][1501:2236]

vectorizer = CountVectorizer(max_df=0.0005,stop_words='english')
X = vectorizer.fit_transform(total_complete)
analyze = vectorizer.build_analyzer()
vectorizer.get_feature_names()
#counts = X.toarray()
#tfidf = transformer.fit_transform(counts)
#transformer = TfidfTransformer()

vectorizer = TfidfVectorizer(max_df=0.00045,stop_words='english')
X = vectorizer.fit_transform(total_complete)
vectorizer.get_feature_names()
features_array = X.toarray()[1]


svc = svm.SVC(kernel='linear')
svc.fit(features_array[0:1500], grantInfo['pw_binary'][0:1500])  
predictions = svc.predict(features_array[1501:2236])

numpy.mean(predictions == grantInfo['pw_binary'][1501:2236])


# ----------------------------------------------------------------------
# Classification of text documents using sparse features using variety
# of different machine learning algorithms
# ----------------------------------------------------------------------
grants = pandas.read_csv('metastatic_grant_binary.csv')
total = grants['AwardTitle'] + ", " + grants['TechAbstract']

train_data = total_complete[0:1500]
train_result = grants['pw_binary'][0:1500]
test_data = total_complete[1501:2236]
test_result = grants['pw_binary'][1501:2236]

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_data)
#chi2
ch2 = SelectKBest(chi2, k=2)
X_train = ch2.fit_transform(X_train, train_result)
X_test = ch2.transform(X_test)


results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            dual=False, tol=1e-3)))
    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))


class L1LinearSVC(LinearSVC):

    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        self.transformer_ = LinearSVC(penalty="l1",
                                      dual=False, tol=1e-3)
        X = self.transformer_.fit_transform(X, y)
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)

print('=' * 80)
print("LinearSVC with L1-based feature selection")
results.append(benchmark(L1LinearSVC()))



#------------------------------------------------------------------------------------
# NLTK collocations, and getting all proper nouns after the text has been normalized
#------------------------------------------------------------------------------------
bigrams = []
trigrams = []
pronouns = []
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
for i in total_complete:
	tokens = nltk.word_tokenize(i)
	text = nltk.Text(tokens)
	tags = nltk.pos_tag(text)
	#Get all pronouns
	pronouns.append([word for word in tags if word[1]=="NNP"])
	#bigrams, and trigrams (Collocations)
	finder = BigramCollocationFinder.from_words(tokens)
	scored = finder.score_ngrams(bigram_measures.raw_freq)
	bigrams.append(sorted(finder.nbest(bigram_measures.raw_freq, 10)))
	trifinder = TrigramCollocationFinder.from_words(tokens)
	scored = trifinder.score_ngrams(trigram_measures.raw_freq)
	trigrams.append(sorted(trifinder.nbest(trigram_measures.raw_freq, 10)))



#searching for the word
#text.concordance('HPR1')

#----------------------------------------------------------------------
# semi feature selection
	# Lasso (give this a try)
# How to generate features
	#Feature making process 
# PCA -> treating features as gene names (Upstream data quality and structure)
	#No clusters on PCA then shit out of luck
#Different categories mean 
#Linear discriminant analysis
#----------------------------------------------------------------------



# -----------------------------------------------------
# Classification of grants metastatic stage
# -----------------------------------------------------
grants = pandas.read_csv('metastatic_grant_binary.csv')
MBCgrants = grants[grants['Metastasis.Y.N']=="y"]
#Get unique metastasis stages
stages = set(MBCgrants['X.Metastasis.stage'])
MBCgrants['stageNumerical']= MBCgrants.index
for num,meta in enumerate(stages):
    temp = MBCgrants['X.Metastasis.stage'] == meta
    MBCgrants['stageNumerical'][temp] = num

total = MBCgrants['AwardTitle'] + ", " + MBCgrants['TechAbstract']
total_complete = normalize_text(total,removeNumbers = False, stemDocument = False)

train_data = total_complete[0:1500]
true_train = MBCgrants['stageNumerical'][0:1500]
test_data = total_complete[1501:2236]
true_test = MBCgrants['stageNumerical'][1501:2236]

svmWorkFlow(train_data, true_train, test_data,true_test)
# Accuracy
# 54%


# ------------------------------------------
# classify Metastasis Y/N
# ------------------------------------------
grants = pandas.read_csv('documents/ICRP_allcombined_grants.csv')
shuffledGrants = grants.reindex(numpy.random.permutation(grants.index))
total = shuffledGrants['AwardTitle'] + ", " + shuffledGrants['TechAbstract']

total_complete = normalize_text(total,removeNumbers = False, stemDocument = False)

interval = int(len(total_complete)/5)
a = total_complete[0:interval]
b = total_complete[interval:2*interval]
c = total_complete[2*interval:3*interval]
d = total_complete[3*interval:4*interval]
e = total_complete[4*interval:len(total_complete)]

a_result = shuffledGrants['Metastasis_YN'][0:interval]
b_result = shuffledGrants['Metastasis_YN'][interval+1:2*interval]
c_result = shuffledGrants['Metastasis_YN'][2*interval+1:3*interval]
d_result = shuffledGrants['Metastasis_YN'][3*interval+1:4*interval]
e_result = shuffledGrants['Metastasis_YN'][4*interval+1:len(total_complete)]

train_data = a+b+c+d
true_train = a_result +  b_result+ c_result+ d_result
test_data = e
true_test = e_result

#train_data = total_complete[0:int(len(total_complete)*0.75)]
#true_train = shuffledGrants['Metastasis_YN'][0:int(len(total_complete)*0.75)]
#test_data = total_complete[int(len(total_complete)*0.75)+1:len(total_complete)]
#true_test = shuffledGrants['Metastasis_YN'][int(len(total_complete)*0.75)+1:len(total_complete)]
C = range(1,100,2)
allmetrics = dict()
index = ['TP','FP','FN','TN']
df = pandas.DataFrame(index = index, columns = C)

for i in C:
    metrics = svmWorkFlow(train_data,true_train,test_data,true_test,C = i)
    f[i] = metrics
    df[i] = [metrics['TP'],metrics['FP'],metrics['FN'],metrics['TN']]

df.to_csv("./test.csv")
#Accuracy
#0.953719008264!
#538 yes
#3697 no




