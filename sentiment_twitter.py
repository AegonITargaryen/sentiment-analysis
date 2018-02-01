"""
supervized learning machine on unstructured data
sentiment extraction
my dataframe is constructed from data from my local database
I encourage you to use pandas to create one with 'text' and 'sentiment' columns
Here you have data cleaning functions, logistic regression
I will post the link to training and testing sets reference
It is made of 25,000 reviews for training and test set each.
12,500 of negative, 12,500 of positive selected 
from 1 to 3 and 7 to 10 (min=1, max=10) rates. We avoided neutral reviews
"""


import re
import sys
import glob
import pickle
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from nltk.corpus import stopwords # Import the stop word list


#read files for defining training and testing sets
ALL_FILES_NEG_TRAIN = glob.glob("../aclImdb/train/neg/*.txt")
ALL_FILES_POS_TRAIN = glob.glob("../aclImdb/train/pos/*.txt")
ALL_FILES_TRAIN = ALL_FILES_NEG_TRAIN + ALL_FILES_POS_TRAIN

#parameters sampled from exponential distribution, randomized parametrization
params_logit = {"penalty": ['l1', 'l2'], "C":st.expon()}
#random number generator for having the same conditions over code re-executions
seed = np.random.seed = 10


def clean_data(tweet):
    """
    purpose: clean Twitter text field
    Inputs: string field from a tweet
    Returns: string in lowercase with one space exactly between words,
    without urls, without # symbol nor emojis. Also, non roman characters 
    are removed (to be completed)
    """
    try:
        #tweet = tweet.lower()
        tweet = re.sub('[\s]+', ' ', tweet)
        tweet = re.sub('((www\.[^\s]+)|(https://[^\s]+))', '', tweet)
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        tweet = myre_UCS2.sub('', tweet)
        tweet = re.sub("[!,?,\",\']", " ", tweet)
        tweet = re.sub(r"<br /><br />", " ", tweet)

    except:
        print "Unexpected error:", sys.exc_info()[0]
    return tweet

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    """draw a matrix to visualise precision and recall
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, dataframe.label.unique())
    plt.yticks(tick_marks, dataframe.label.unique())
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#exclude stop words for english as I only picked tweets in english
def associate_test(row):
    """clean data by removing surplus of space, lowercase as words are case sensitive"""
    with open("{}".format(all_files_test[row]), 'r') as f:
        test = f.readline()
        test = re.sub('[\s]+', ' ', test)
        test = test.lower()
        test = ' '.join([word for word in test.split() if \
        word not in set(stopwords.words('english'))])

    return test

def associate_train(row):
    """clean data by removing surplus of space, lowercase as words are case sensitive"""
    with open("{}".format(ALL_FILES_TRAIN[row]), 'r') as f:
        train = f.readline()
        train = re.sub('[\s]+', ' ', train)
        train = train.lower()
        train = ' '.join([word for word in train.split() \
        if word not in set(stopwords.words('english'))])

    return train

#classification: 1 for positive reviews and 0 for negative ones

if __name__ == "__main__":

    dataframe = pd.DataFrame(columns=['text', 'label'], index=range(50000), dtype=object)
    dataframe.loc[:24999, 'text'] = [associate_train(i) for i in range(0, 25000)]
    #dataframe['text']=dataframe['text'].apply(clean_data)
    dataframe.label = 1
    dataframe.loc[:12499, 'label'] = [0 for i in range(0, 12500)]



    #t2=pd.DataFrame(columns=['text', 'label'], index=range(25000),dtype=object)
    dataframe.loc[25000:, 'text'] = [associate_test(i) for i in range(0, 25000)]
    dataframe.loc[:, 'text'] = dataframe['text'].apply(clean_data)
    #dataframe.label=0
    dataframe.loc[25000:37499]['label'] = [0 for i in range(0, 12500)]


    texdataframe = dataframe.text
    Y1 = dataframe.label



    params_forest = {"n_estimators": st.randint(10, 20), "criterion":['gini', 'entropy']}
    seed = np.random.seed = 10
    cvt = CountVectorizer(min_df=1e-2)
    X1 = cvt.fit_transform(texdataframe)
    pickle.dump(cvt.vocabulary_, open("dictio_NEW", 'w'))



    clf = RandomizedSearchCV(LogisticRegression(), params_logit, n_iter=10, cv=5, n_jobs=-1)
    #clf = RandomizedSearchCV(RandomForestClassifier(), params_forest)
    pred_res = clf.fit(X1[:25000].toarray(), Y1[:25000]).predict(X1[25000:])
    #pred=clf.fit(X1.toarray(), Y1).predict(X2)
    pred_proba = clf.fit(X1[:25000].toarray(), Y1[:25000]).predict_proba(X1[25000:])

    print "Resultats:\n best_estimator = {}\n best_score = {}".format(clf.best_params_, clf.best_score_)
    print "Accuracy of training is: {%0.2f}\n Accuracy of test is: {%0.2f}"\
    %(clf.score(X1[:25000].toarray(), Y1[:25000]), clf.score(X1[25000:].toarray(), Y1[25000:]))
    print "Grid_score is: {}\n  ".format(clf.grid_scores_)
    print "*******************************************************************"



    cm = confusion_matrix(Y1[25000:], pred_res)
