# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 19:24:48 2019

@author: shivansh
"""
import pandas as pd 
import pickle
#from sklearn.externals import joblib
#import os
#import sqlite3
#from nltk import corpus
#conn = sqlite3.connect('model_result.db', check_same_thread=False)
from sklearn.feature_extraction.text import CountVectorizer
df= pd.read_csv("D:/ytb_model/data_csv_files/YoutubeSpamMergedData_spam_ham.csv")
df_data = df[["CONTENT","CLASS"]]

# Features and Labels
df_x = df_data['CONTENT']
df_y = df_data.CLASS

## Extract Feature With CountVectorizer
corpus = df_x
#print(corpus)
cv = CountVectorizer(ngram_range=(1, 2))
X = cv.fit_transform(corpus) # Fit the Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)

# Naive Bayes Classifier
#from sklearn.naive_bayes import MultinomialNB
#clf = MultinomialNB()
##--------- STEP (2). EXECUTE STEP (1) BEFORE LOADING PICKLE FILE--------------
with open('MnNB_new.pkl', 'rb') as fid_1:
    clf = pickle.load(fid_1)
    fid_1.close()
##--------------------------------------------------------------------------    
#train_data = clf.fit(X_train,y_train)
acc = clf.score(X_test,y_test)   # this is to in web application 
print (acc)
#
##------------IMPORTANT STEP (1)---------------------------------
## save the classifier
#with open('MnNB_new.pkl', 'wb') as fid:
#    pickle.dump(train_data, fid)
#    fid.close()
##---------------------------------pICKLE FILE FOR CORPUS- STEP(3)-------------------
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
df= pd.read_csv("D:/ytb_model/data_csv_files/YoutubeSpamMergedData_spam_ham.csv")
df_data = df[["CONTENT","CLASS"]]

# Features and Labels
df_x = df_data['CONTENT']

## Extract Feature With CountVectorizer
corpus = df_x
cv = CountVectorizer(ngram_range=(1, 2))
X = cv.fit_transform(corpus) # Fit the Data
with open('corpus_fit_transform.pkl', 'wb') as fid:
    pickle.dump(X, fid)
    fid.close()

