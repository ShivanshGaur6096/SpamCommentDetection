from flask import Flask,render_template,url_for,request, redirect, flash ,abort,send_file
import pandas as pd 
import pickle
from sklearn.externals import joblib
from werkzeug.utils import secure_filename
import os
import sqlite3

conn = sqlite3.connect('model_result.db', check_same_thread=False)

UPLOAD_FOLDER = 'D:/ytb_model/Uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#--------------------------------- BACKEND / MODEL -----------------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
df= pd.read_csv("D:/ytb_model/YoutubeSpamMergedData.csv")
df_data = df[["CONTENT","CLASS"]]

# Features and Labels
df_x = df_data['CONTENT']
df_y = df_data.CLASS

# Extract Feature With CountVectorizer
corpus = df_x
cv = CountVectorizer(ngram_range=(1, 2))
X = cv.fit_transform(corpus) # Fit the Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)

# Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
acc = clf.score(X_test,y_test)   # this is to in web application 

# CREATING PICKLE FILE FOR THE BACKEND
joblib.dump(clf, 'naivebayes_spam_model.pkl')

# LOADING THE PICKLE FILE IN THE PREDICT MODEL   
clf = joblib.load("naivebayes_spam_model.pkl")

#-------------------------------- BACKEND finish HERE --------------------------------------------------------------


@app.route('/', methods=['GET','POST'])
def home():
	return render_template('home.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#------------------------------------ UPLOAD PART --------------------------------------------------------------------
@app.route('/upload')
def upload_file():
    
      # Return 404 if path doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        return abort(404)
    
    # Check if path is a file and serve
    if os.path.isfile(UPLOAD_FOLDER):
        return send_file(UPLOAD_FOLDER)

    # Show directory contents
    files = os.listdir(UPLOAD_FOLDER)
    return render_template('upload.html', files=files)
    
    
   #return render_template('upload.html')

@app.route('/uploader', methods = ['GET','POST'])
def upload_file1():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            flash('Not selected file')
            return redirect(request.url)
                        
        if 'file' not in request.files:
            flash('Not correct extension for file')
            return redirect(request.url)
             
        if file and allowed_file(file.filename):
            if not os.path.exists(UPLOAD_FOLDER):
                return abort(404)
              
            if os.path.isfile(UPLOAD_FOLDER):
                return send_file(UPLOAD_FOLDER)

            
            files = os.listdir(UPLOAD_FOLDER)

            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('upload.html', files=files , msg = file.filename+'uploaded successfully' )
             
#====================================== END OF UPLOAD SECTION ========================================================================                
   
@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
	predict_proba = clf.predict_proba(vect).tolist()
    file = open("sample.txt","a")
    file.write('{} , {}\n'.format(comment, *my_prediction))
    file.close()
    with open("sample.txt", "r") as file:
        file.close()
#------DATABASE WORKING-------------
    conn.execute("INSERT INTO RESULTS (COMMENT,RESULT) VALUES(?, ?)",(comment, my_prediction))
    conn.commit()
    conn.close()	
    return render_template('result.html',prediction = my_prediction, accuracy = acc, pred_prcnt=predict_proba, details = comment)
#-------------------------------------- (1) ADDITIONAL FUNCTIONALITY OF MODEL --------------------------------------------------
@app.route('/content')
def content():
	text = open('testDoc.txt', 'r',encoding="utf8")
	content = text.read()
	text.close()
	return render_template('content.html', text=content)
#---------------------------------------(2) ADDITIONAL FUNCTIONALITY OF MODEL-----------------------------------------------
import time 
import datetime

@app.route('/diff')
def form():
    return render_template('form_action.html')

@app.route('/hello/', methods=['POST','GET'])
def hello():
    global time

    name=request.form['yourname']
    email=request.form['youremail']
    comment=request.form['yourcomment']
    comment_time=time.strftime("%a-%d-%m-%Y %H:%M:%S")
    
    f = open ("user+comments.txt","a")

    f.write(name + '  ' + email + '   ' + comment + "  " + comment_time)
    f.write('\n')
    f.close()
    with open("user+comments.txt", "r") as f:
        details = f.read()
        f.close()
        return render_template('form_action.html', details = details, name=name,   
    email=email, comment=comment, comment_time=comment_time)

#-------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	app.run(debug=True)
