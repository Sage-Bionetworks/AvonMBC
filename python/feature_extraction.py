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
#NLTK -> tokenizer, stemmer
import nltk
from nltk.collocations import *
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
nltk.download("stopwords")
import pandas
import string
import re
#Beautiful soup -> Gets rid of HTML and replaces non ascii characters
pip install Beautifulsoup4
from bs4 import BeautifulSoup
import unicodedata 
import numpy


#-------------------------------------------------------------------------------
#Check if it is a number, if not return the word, if it is return a stop word
#-------------------------------------------------------------------------------
def is_number(s):
    try:
        float(s)
        return 'the'
    except ValueError:
        return s

#--------------------------------------------------------------------------------------
#Normalization of text (Remove HTML, replace non-ascii characters, remove punctuation,
#remove lone numbers, stem words, remove stop words recombine into text)
#--------------------------------------------------------------------------------------
grantInfo = pandas.read_csv('metastatic_grant_binary.csv')
total = grants['AwardTitle'] + ", " + grants['TechAbstract']
stopwords.words('english')
#Remove HTML, replace non-ascii characters, tokenize (remove punctuation), stem words, then reconcatenate
text_noHTML = [BeautifulSoup(i).get_text() for i in total]
total_ascii = [unicodedata.normalize('NFKD', text).encode('ascii','ignore') for text in text_noHTML]
total_noperiod = [re.sub(r'\D\.\D',' ', text) for text in total_ascii]
total_noslash = [re.sub(r'\b\/\b',' ', text) for text in total_noperiod]
total_combinedash = [re.sub(r'\b-\b',"",text) for text in total_noslash]
total_nopunct = [nltk.word_tokenize(text.translate(None, string.punctuation)) for text in total_combinedash]



#Use english stemmer
stemmer = SnowballStemmer("english")
total_complete[:]
for i in range(0,len(total_nopunct)):
	temp  = [is_number(text)  for text in total_nopunct[i] ]
	temp = [stemmer.stem(word) for word in temp]
	temp = [word for word in temp if word not in stopwords.words('english')]
	total_complete[i] = ' '.join(temp)




#----------------------------------------------------------------
# SCIKIT TOOLS -> Get features by using TfidF Vectorizer
#----------------------------------------------------------------
train_data = total_complete[0:1500]
train_result = grantInfo['pw_binary'][0:1500]
test_data = total_complete[1501:2236]
test_result = grantInfo['pw_binary'][1501:2236]


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


#-----------------
# Classification of text documents using sparse features
#-----------------
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_data)
#chi2
ch2 = SelectKBest(chi2, k=15863)
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

#--------
# BENCHMARK
#-------
feature_names=None
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, train_result)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.f1_score(test_result, pred)
    print("f1-score:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time



#-----
# semi feature selection
	# Lasso (give this a try)
# How to generate features
	#Feature making process 
# PCA -> treating features as gene names (Upstream data quality and structure)
	#No clusters on PCA then shit out of luck
#Different categories mean 
#Linear discriminant analysis

#----