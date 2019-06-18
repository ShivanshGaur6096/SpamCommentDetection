from flask import Flask,render_template,url_for,request, redirect, flash
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy 
import os

UPLOAD_FOLDER = 'D:/YTSpamDetection/Uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['SQLALCHEMT_DATABASE_URI'] = 'sqlite:////D:/YTSpamDetection/file/filestorage.db'
#db = SQLAlchemy(app)
#
#class FileContents(db.Model):
#    id = db.Column(db.Integer, primary_key=True)
#    name = db.Column(db.String(300))
#    data = db.Column(db.LargeBinary)


@app.route('/', methods=['GET','POST'])
def home():
	return render_template('home.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload')
def upload_file():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET','POST'])
def upload_file1():
    errors = []
#    flash('welcome to our new members')
    if request.method == 'POST':
        file = request.files['file']
        try:
            
            if file.filename == '':
                flash('Not selected file')
                return redirect(request.url)
                    
            if 'file' not in request.files:
                flash('Not correct extension for file')
                return redirect(request.url)
        except:
            errors.append(
                "Unable to get URL. Please make sure it's valid and try again."
            )
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return( file.filename + '<h1> uploaded successfully </h1>')
             
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
#----------------------------------------------------------------------------------------------------------

@app.route('/con')
def content():
	text = open('testDoc.txt', 'r+')
	content = text.read()
	text.close()
	return render_template('content.html', text=content)
#------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
	app.run(debug=True)