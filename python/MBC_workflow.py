#!/usr/bin/env python
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
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

#NLTK -> tokenizer, stemmer
import nltk
from nltk.collocations import *
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
#nltk.download("stopwords")
import pandas as pd
import string
import re
from time import time
import logging
#Beautiful soup -> Gets rid of HTML and replaces non ascii characters
#pip install Beautifulsoup4
from bs4 import BeautifulSoup
import unicodedata 
import numpy as np
import pprint
from random import shuffle
import argparse
import synapseclient


# R files
import rpy2.robjects as robjects


# -------------------------------------------------------------------------------
# Normalization of text
# -------------------------------------------------------------------------------

# Check if it is a number, if not return the word, if it is return a stop word
def is_number(s):
    try:
        float(s)
        return 'the'
    except ValueError:
        return s

#Normalize text
#Remove HTML, replace non-ascii characters, tokenize (remove punctuation), stem words, then reconcatenate
def normalize_text(grants, removeNumbers = True, stemDocument = True):
    #no, hers, and her can be pathway names
    unwantedWords = [i for i in stopwords.words('english') if i not in ['no','her','hers']]
    #Remove html
    grants[grants.isnull()] = ''
    grants = [BeautifulSoup(i,"html.parser").get_text() for i in grants]
    #Replace Ascii
    grants = [unicodedata.normalize('NFKD', text).encode('ascii','ignore') for text in grants]
    #Get rid of numbers
    if removeNumbers:
        grants = [re.sub(r'\d+', '', text) for text in grants]
    #Get rid of periods, slashes, and punctuation all together
    grants = [re.sub(r'\D\.\D',' ', text) for text in grants]
    grants = [re.sub(r'\b\/\b',' ', text) for text in grants]
    #grants = [re.sub(r'\b-\b','',text) for text in grants]
    grants = [nltk.word_tokenize(text.translate(None, string.punctuation)) for text in grants]
    #Use english stemmer
    stemmer = SnowballStemmer("english")
    normalized_grants=[]
    for text in grants:
        #temp  = [is_number(text)  for text in total_nopunct[i] ]
        temp = [word for word in text if word not in unwantedWords]
        if stemDocument:
            temp = [stemmer.stem(word) for word in temp]
        #Remove stop words
        normalized_grants.append(' '.join(temp))
    return normalized_grants


def randomForestWorkflow(train_data, true_train, test_data,true_test=None,
                C=1, 
                vect__max_df=0.75,
                vect__max_features = 50000,
                vect__ngram_range = (1,2),
                tfidf__use_idf = False,
                tfidf__norm='l2'):
    """
    Input training data, training data true values, test data and test data true values.
    Work Flow:
        CountVectorizer
            - max_df
            - max_features
            - ngram_range
        Tfidf transformer
            - norm
            - use_idf
        svm
            - kernel
    """
    vectorizer = CountVectorizer(max_df=vect__max_df,
        max_features = vect__max_features,
        #If not None, build a vocabulary that only consider the top max_features ordered by
        #term frequency across the corpus.
        ngram_range = vect__ngram_range)
    #Figure out what countvectorizer is doing in feature selection 
    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)
    #tfidf transformer
    transformer = TfidfTransformer(norm = tfidf__norm,use_idf = tfidf__use_idf)
    tfidf_train = transformer.fit_transform(X_train)
    tfidf_test = transformer.transform(X_test)
    #features_array = X.toarray()
    training_features = tfidf_train.toarray()
    test_features = tfidf_test.toarray()
    #RF balanced weight adjusts weight proportional to class frequencies
    clf = RandomForestClassifier(n_estimators=C,class_weight = 'balanced')
    clf.fit(training_features, true_train)
    predictions = clf.predict(test_features)
    #get probability of each class
    prob = clf.predict_proba(test_features)
    prob = pd.DataFrame(prob)
    #prob.to_csv("%s_rf_prob.csv" %C)
    if true_test is not None:
        score = clf.score(test_features,true_test)
    else:
        score = 0
    return dict(predictions = predictions,prob = prob,score=score)

# ------------------------------------------------
# Classify Metastasis stage
# ------------------------------------------------

def classify_MBC_Stage(training, test,nTrees=100):
    stages = ['Arrest & extravasation','Immune surveillance/escape',
            'Intravasation & circulation','Invasion',
            'Metabolic deregulation','Metastatic colonization']
    stage_prob = ['arrest_meta','immune_meta','intravasatsion_meta','invasion_meta',
                'metabolic_meta','metastatic_meta']
    training = training[training['Metastasis_stage'].isin(stages)]

    training_text = training['AwardTitle'] + ", " + training['TechAbstract']
    test_text = test['AwardTitle'] + ", " + test['TechAbstract']

    print("NORMALIZING TEXT")
    training_text = normalize_text(training_text,removeNumbers = False, stemDocument = False )
    test_text = normalize_text(test_text, removeNumbers = False, stemDocument = False )

    #C = 1,5,10,50,100,500,1000,5000,10000
    #allmetrics = dict()
    index = ['PRED_SCORE']
    #probs = pd.DataFrame(columns = [nTrees])
    predictions = pd.DataFrame(columns = [nTrees])
    print("RANDOM FOREST")
    #for i in C:
    metrics = randomForestWorkflow(training_text,training['Metastasis_stage'],test_text,C = nTrees)
    predictions[nTrees] = metrics['predictions']
    probs = metrics['prob']
    print("POSTERIOR PROBABILITY")
    print(probs)
    print("PREDICTIONS")
    print(predictions[nTrees])
    probs.to_csv("./meta_stage_probability.csv")
    predictions.to_csv("./meta_stage_predictions.csv")
    return(probs, predictions)


def call_stage(args):
    classify_MBC_Stage(args.training,args.test)


def geneList_matching(testPath, geneListPath):
    robjects.r("source('R/geneList_matching.R')")
    updateGeneList = robjects.r('updateGeneList')
    updateGeneList(testPath, geneListPath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='MBC workflow')

    subparsers = parser.add_subparsers(title='commands',
            description='The following commands are available:')

    #Update beatAML project
    parser_stage = subparsers.add_parser('predictStage',
            help='Predict metastatic stage')
    parser_stage.add_argument('--training',  metavar='training.txt', type=str, required=True,
            help='File that contains training data'),
    parser_stage.add_argument('--test', metavar='test.txt', type=str, required=True,
            help='File that contains testing data')
    parser_stage.set_defaults(func=call_stage)

    #Parse args
    args = parser.parse_args()
    perform_main(args)

    syn = synapseclient.login()
    training_ent = syn.get("syn6136723")
    training = pd.read_csv(training_ent.path)
    geneList_ent = syn.get("syn5594707")
    geneList = pd.read_csv(geneList_ent.path,sep="\t")
    null_training = training[training['Metastasis_stage'].isnull()]
    temp = syn.query('select id,name,run from file where parentId == "syn6047020"')
    for i in temp['results']:
        if i['file.run'] is None:
            test_ent = syn.get(i['file.id'])
            test = pd.read_excel(test_ent.path)
            names = test.columns.values
            for index,name in enumerate(names):
                name = re.sub(" ","_",name)
                name = re.sub("[*]|[(]|[)]|[/]","",name)
                names[index] = name
            test.columns = names
            extras = training.columns[~training.columns.isin(test.columns)]
            for col in extras:
                test[col] = ''
            metaYN = []
            for score in test['Breast_Cancer']:
                if score>=50:
                    metaYN.append("yes")
                else:
                    metaYN.append("no")
            test['Metastasis_YN'] = metaYN
            test['Metastasis_stage'] = "Auto Generated"
            probs, prediction = classify_MBC_Stage(training, test)
            probs.columns = stage_prob
            for i in stage_prob:
                test[i] = probs[i]
            test['Predicted_metastage'] = prediction
            test['Predicted_metastage'][test['Metastasis_YN'] == "no"] = "Not relevant"

            #matched_genes = geneList_matching(test_ent.path, geneList_ent.path)
            os.system("Rscipt geneList_matching.R %s %s" % (test_ent.path, geneList_ent.path)) #pass in arguments to the R script
            test['gene_list'] = matched_genes

            test_ent.run = "completed"
            syn.store(test_end)




#### Clean abstracts of html tags
#### Call metastage classifier
#### Call MBC classifier
#### Distance metric with SA abstracts
#### Match gene list

