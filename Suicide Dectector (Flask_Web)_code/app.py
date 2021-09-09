from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
import string
import pickle
import regex as re
import unicodedata
import nltk
from nltk.collocations import *
stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import tree
import joblib


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    # Importing
    suicide_merged = pd.read_csv('suicide_merged.csv', encoding="latin-1")
    
    string.punctuation                             
    # Function to clean text
    def clean_text(text):
        text = ''.join([word.lower() for word in text if word not in string.punctuation])
        tokens = re.split('\W+', text)
    
        text = [ps.stem(word) for word in tokens if word not in stopwords]
        return text

    # TF-IDF
    tfidf_vect = TfidfVectorizer(analyzer = clean_text)
    X_tfidf = tfidf_vect.fit_transform(suicide_merged['text'].values.astype('U'))
    
    # Sparse Matric
    X_tfidf_df = pd.DataFrame(X_tfidf.toarray())
    X_tfidf_df.columns = tfidf_vect.get_feature_names()
    X_tfidf_df
    
    # Modelling (Random Forest)
    # Assigning of X and y
    di = {'suicide': 1, 'non-suicide' : 0}
    suicide_merged['class'] = suicide_merged['class'].map(di)
    
    # assigning X and y 
    X = X_tfidf_df
    y = suicide_merged['class']
    
    # Train, Test, Split (Preparing the data for training the model)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25,
                                                    random_state=42)
    
    # Building a Random Forest model
    classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    y_train_array = np.ravel(y_train)
    classifier.fit(X_train, y_train_array)
    y_pred = classifier.predict(X_test)
    
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        X_tfidf_df = tfidf_vect.transform(data).toarray()
        my_prediction = classifier.predict(X_tfidf_df)
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
    app.run(debug=True)
    
 
    
    
    
    
    
    
    