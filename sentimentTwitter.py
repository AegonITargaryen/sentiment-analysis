"""
supervized learning machine on unstructured data
sentiment extraction
my dataframe is constructed from data from my local database
I encourage you to use pandas to create one with 'text' and 'sentiment' columns
Here you have data cleaning functions, logistic regression
I will post the link to training and testing sets reference
It is made of 25,000 reviews for training and test set each.
12,500 of negative, 12,500 of positive selected from 1 to 3 and 7 to 10 (min=1, max=10) rates. We avoided neutral reviews
"""

import scipy.stats as st
import re
import sys
import numpy as np
import pandas as pd
import datetime
from sklearn.grid_search import  RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords # Import the stop word list
from sklearn.linear_model import LogisticRegression
import glob
import joblib
import pickle

#parameters sampled from exponential distribution, randomized parametrization
params_logit={"penalty": ['l1','l2'], "C":st.expon()}
#random number generator for having the same conditions over code re-executions
seed=np.random.seed= 10

def cleanData(tweet):
        """
        purpose: clean Twitter text field
        Inputs: string field from a tweet
        Returns: string in lowercase with one space exactely between words, without urls, without # symbol nor emojis. Also, non roman characters are removed (to be completed)
        """
        try:
            tweet = tweet.lower()
            tweet = re.sub('[\s]+', ' ', tweet)
            tweet = re.sub('((www\.[^\s]+)|(https://[^\s]+))', '', tweet)
            tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
            tweet = myre_UCS4.sub('', tweet)
            tweet = myre_UCS2.sub('', tweet)
            tweet = re.sub("[^a-zA-Z]", " ", tweet)
            #tweet = [w for w in text if not w in stopwords.words("english")]
            # print tweet
        except:
            print "Unexpected error:", sys.exc_info()[0]
        return tweet
        
        
#read files for defining training and testing sets       
all_files_neg_train=glob.glob("../aclImdb/train/neg/*.txt")
all_files_pos_train=glob.glob("../aclImdb/train/pos/*.txt")
all_files_train=all_files_neg_train+all_files_pos_train

#initialize train and test sets
#exclude stop words for english as I only picked tweets in english
train=[]
def associate_train(row):
    with open("{}".format(all_files_train[row]),'r') as f:
        train = f.readline()
        train = train=' '.join([word for word in train.split() if word not in set(stopwords.words('english'))])
        train = re.sub('[\s]+',' ', train)
    return train
 
test=[]
def associate_test(row):
    with open("{}".format(all_files_test[row]),'r') as f:
        test = f.readline()
        test = test=' '.join([word for word in test.split() if word not in set(stopwords.words('english'))])
        test = re.sub('[\s]+',' ', test)
    return test
    
#classification: 1 for positive reviews and 0 for negative ones
    
t1=pd.DataFrame(columns=['text','label'], index=range(50000),dtype=object)
t1.loc[:24999,'text']=[associate_train(i) for i in range(0,25000)]
#t1['text']=t1['text'].apply(cleanData)
t1.label=1
t1.loc[:12499,'label']=[0 for i in range(0,12500)]


t1.loc[25000:,'text']=[associate_test(i) for i in range(0,25000)]
t1['text']=t1['text'].apply(cleanData)
t1.loc[25000:37499]['label']=[1 for i in range(25000,37500)]

#fill your dataframe with the cleaned data
text1=t1.text
Y1=t1.label


time1 = datetime.datetime.now()
cvt = CountVectorizer(min_df=1e-2)
X1 = cvt.fit_transform(text1)
#pickle.dump(cvt.vocabulary_, open("../dictio", 'w'))

clf = RandomizedSearchCV(LogisticRegression(), params_logit, n_iter=2, cv=10)
pred_res=clf.fit(X1[:25000].toarray(), Y1[:25000]).predict(X1[25000:])
pred_proba=clf.fit(X1[:25000].toarray(), Y1[:25000]).predict_proba(X1[25000:])

print "Resultats:\n best_estimator = {}\n best_score = {}".format(clf.best_params_, clf.best_score_)
print "Accuracy of training is: {%0.2f}\n Accuracy of test is: {%0.2f}"%(clf.score(X1[:25000].toarray(),Y1[:25000]), clf.score(X1[25000:].toarray(),Y1[25000:]))
print "Grid_score is: {}\n  ".format(clf.grid_scores_)
print "*******************************************************************"

time2 = datetime.datetime.now()    
print "duree est de {} secondes".format(time2-time1)
