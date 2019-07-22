import praw
import json
from flask import request, abort
from flask.ext import restful
from flask.ext.restful import reqparse
from flask_rest_service import app, api, mongo
from bson.objectid import ObjectId
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from pymongo import MongoClient
from sklearn.externals import joblib

class MyPredictFlair(restful.Resource):
    def __init__(self, *args, **kwargs):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('url', type=str)
    
    def post(self):
        args = self.parser.parse_args()
        print(args['url'])
        if not args['url']:
            abort(400)

        url = json.loads(args['url'])
        #Put your code to train and predict
        reddit = praw.Reddit(client_id='3g4ggl4suqTAIQ', redirect_uri='http://localhost:8080', client_secret='es9JYYwOFw-Fxjp04PDQ64WSwvg', user_agent='praw-test')
        submission = reddit.submission(url=url)
        text = data = pd.DataFrame([{'title': submission.title}])
        text.dropna(inplace=True)
        data['title'] = [entry.lower() for entry in data['title']]
        data['title']= [word_tokenize(entry) for entry in data['title']]
        tag_map = defaultdict(lambda : wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV
        for index,entry in enumerate(text):
            Final_words = []
            word_Lemmatized = WordNetLemmatizer()
            for word, tag in pos_tag(entry):
                if word not in stopwords.words('english') and word.isalpha():
                    word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                    Final_words.append(word_Final)
            data.loc[index,'text_final'] = str(Final_words)
        Tfidf_vect = joblib.load('Tfidf_vector.sav')
        final_data = Tfidf_vect.transform(text['text_final'])

        SVM = joblib.load('finalized_model.sav')
        predictions_SVM = SVM.predict(final_data)
        print(predictions_SVM[0])
        return {
            'status': 'OK',
            'flair': predictions_SVM[0],
        }
        """predictions_SVM[0]"""

        
class Root(restful.Resource):
    def get(self):
        return {
            'status': 'OK',
            'msg': 'Service Hosting MyPredictFlair',
        }

api.add_resource(Root, '/')
api.add_resource(MyPredictFlair, '/predict/')
