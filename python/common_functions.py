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

#--------------------------------------------------------------------
# Best Parameters CountVectorizer, Tfidf transformer, SGD classifier 
#--------------------------------------------------------------------
def findBestParameters(train_data, train_result):
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),
    ])
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
        'clf__n_iter': (10, 50, 80),
    }
    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print(parameters)
    t0 = time()
    grid_search.fit(train_data, train_result)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


# ------------------------------------------
# Machine learning algorithms
# ------------------------------------------
def svmWorkFlow(train_data, true_train, test_data,true_test,
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
    #SVM
    svc = svm.SVC(C,kernel='linear')
    svc.fit(training_features, true_train)  
    predictions = svc.predict(test_features)
    prediction_score = svc.decision_function(test_features)
    #Metrics Calculation confusion matrix AUC
    #5-fold 
    #Divide into 5 parts, and use different 4 training set, will get 5 different metrics
    #Create a plot of the confusion matrix
    #Loss function and penalty function
    #Hinge loss, 
    #Choose different c's (loss function)
    print C
    print "Accuracy"
    print numpy.mean(predictions == true_test)
    confusion = confusion_matrix(true_test,predictions)
    return dict(predictions = predictions,prediction_score = prediction_score, TP = confusion[0,0], FP = confusion[0,1], FN = confusion[1,0], TN = confusion[1,1])

#linear regression
def linearRegressionWorkFlow(train_data, true_train, test_data,true_test, 
                vect__max_df=0.75,vect__max_features = 50000,vect__ngram_range = (1,2),
                tfidf__use_idf = False,tfidf__norm='l2'):
    vectorizer = CountVectorizer(max_df=vect__max_df,
        max_features = vect__max_features, 
        ngram_range = vect__ngram_range)
    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)
    #tfidf transformer
    transformer = TfidfTransformer(norm = tfidf__norm,use_idf = tfidf__use_idf)
    tfidf_train = transformer.fit_transform(X_train)
    tfidf_test = transformer.transform(X_test)
    #features_array = X.toarray()
    training_features = tfidf_train.toarray() 
    test_features = tfidf_test.toarray()
    #SVM
    clf = LogisticRegression(penalty='l1')
    clf.fit(training_features, true_train)  
    predictions = clf.predict(test_features)
    print "Accuracy"
    print numpy.mean(predictions == true_test)
    return(predictions)

def randomForestWorkflow(train_data, true_train, test_data,true_test,
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
    #SVM
    clf = RandomForestClassifier(n_estimators=C)
    clf.fit(training_features, true_train)
    predictions = clf.predict(test_features)
    score = clf.score(test_features,true_test)
    return dict(predictions = predictions,score=score)

#----------------------------------------------------------------------
# BENCHMARK
#----------------------------------------------------------------------
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

def show_top10(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print("%s: %s" % (category, " ".join(feature_names[top10])))
