# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 19:24:48 2019

@author: shivansh
"""
import pandas as pd 
import pickle
#from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
df= pd.read_csv("D:/ytb_model/data_csv_files/YoutubeSpamMergedData_spam_ham.csv")
df_data = df[["CONTENT","CLASS"]]
#
## Features and Labels
df_x = df_data['CONTENT'] 
##df_y = df_data.CLASS
corpus = df_x
### Extract Feature With CountVectorizer
##print(corpus)
cv = CountVectorizer(ngram_range=(1, 2))
X = cv.fit_transform(corpus) # Fit the Data
#
#with open('corpus_fit_transform.pkl', 'wb') as fid:
#    pickle.dump(X, fid)
#    fid.close()
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
#
## Naive Bayes Classifier
##from sklearn.naive_bayes import MultinomialNB
##clf = MultinomialNB()
#from sklearn.feature_extraction.text import transform
#with open('corpus_fit_transform.pkl', 'rb') as fid_1:
#    corpus_pkl = pickle.load(fid_1)
#    fid_1.close()

#with open('my_dumped_classifier.pkl', 'rb') as fid:
#    clf = pickle.load(fid)
#    fid.close()
with open('MnNB_new.pkl', 'rb') as fid:
    clf = pickle.load(fid)
    fid.close()

data = ['hey how are you']
print(data)
vect = cv.transform(data).toarray()
##        print(vect)
my_prediction = clf.predict(vect)
predict_proba = clf.predict_proba(vect).tolist()
#predict_proba = clf.predict_proba(vect)[:,1]
print(my_prediction)
print(predict_proba)
        
#train_data = clf.fit(X_train,y_train)
#acc = clf.score(X_test,y_test)   # this is to in web application 
#print (acc)
#
## save the classifier
#with open('MnNB_new.pkl', 'wb') as fid:
#    pickle.dump(train_data, fid)
#    fid.close()
    

# load it again
#with open('my_dumped_classifier.pkl', 'rb') as fid:
#    gnb_loaded1 = cPickle.load(fid)
#data = ['here we go again']
#print (data)
#cv = CountVectorizer(ngram_range=(1, 2))
#X = cv.fit_transform(corpus)
##print(X)
#vect = cv.transform(data).toarray()
##
##print(vect)
#my_prediction = gnb_loaded1.predict(vect)
#print(my_prediction)
#predict_proba = clf.predict_proba(vect)[:,1]
#print(predict_proba)

#my_prediction = clf.predict(X)
#print(my_prediction)
## CREATING PICKLE FILE FOR THE BACKEND
#joblib.dump(clf, 'naivebayes_spam_model_2.pkl')

# LOADING THE PICKLE FILE IN THE PREDICT MODEL +
#ytb_model = open("naivebayes_spam_model.pkl","rb")  
#clf = joblib.load(ytb_model)
