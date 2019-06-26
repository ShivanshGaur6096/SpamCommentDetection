# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:46:18 2019

@author: shivansh
"""
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer


df= pd.read_csv("D:/ytb_model/data_csv_files/YoutubeSpamMergedData_spam_ham.csv")
df_data = df[["CONTENT","CLASS"]]

# Features and Labels
df_x = df_data['CONTENT']
df_y = df_data.CLASS

corpus = df_x
# Extract Feature With CountVectorizer
cv = CountVectorizer(ngram_range=(1, 2))
X = cv.fit_transform(corpus) # Fit the Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)

# Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()

clf.fit(X_train,y_train)
acc = clf.score(X_test,y_test)   # this is to show in web application 
print(acc)

''' ==== create pickle file for clf.fit(X_train,y_train) or just put "clf" at place of X in pickle.dump(X, fid) 
and,  
also create pickle file for corpus "line 22= X = cv.fit_transform(corpus) " as same given below==========
'''
#with open('enter_name_to_creat_pkl_file.pkl', 'wb') as fid:
#    pickle.dump(X, fid)
#    fid.close()

#with open('enter_name_of_created_pkl_file.pkl', 'rb') as fid_1:
#    clf = pickle.load(fid_1)
#    fid_1.close()