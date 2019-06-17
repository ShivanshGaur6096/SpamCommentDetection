from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from werkzeug import secure_filename
from flask_sqlalchemy import SQLAlchemy 
import os


app = Flask(__name__)
app.config['SQLALCHEMT_DATABASE_URI'] = 'sqlite:////D:/YTSpamDetection/file/filestorage.db'
db = SQLAlchemy(app)

class FileContents(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(300))
    data = db.Column(db.LargeBinary)


@app.route('/', methods=['GET','POST'])
def home():
	return render_template('home.html')

@app.route('/upload')
def upload_file():
   return render_template('upload.html')

@app.route('/uploader', method = ['POST'])
def upload_file1():
    file = request.files['inputFile']
    
    newFile = FileContents(name=file.filename, file.read())
    db.session.add(newFile)
    db.session.commit()
    
    return 'saved ' + file.filename + ' to the database.'
   	

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("YoutubeSpamMergedData.csv")
	df_data = df[["CONTENT","CLASS"]]
	# Features and Labels
	df_x = df_data['CONTENT']
	df_y = df_data.CLASS
    # Extract Feature With CountVectorizer
	corpus = df_x
	cv = CountVectorizer()
	X = cv.fit_transform(corpus) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
	#Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)
	#Alternative Usage of Saved Model
	# ytb_model = open("naivebayes_spam_model.pkl","rb")
	# clf = joblib.load(ytb_model)

	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)