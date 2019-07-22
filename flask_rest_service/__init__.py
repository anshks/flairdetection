import os
from flask import Flask
from flask.ext import restful
from flask.ext.mongoalchemy import MongoAlchemy
#from flask.ext.pymongo import PyMongo
from flask import make_response
from bson.json_util import dumps
import praw
from praw.models import MoreComments
import pandas as pd
import logging
import pymongo	
from pymongo import MongoClient
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
from sklearn.externals import joblib

MONGO_URL = os.environ.get('MONGO_URL')
if not MONGO_URL:
    MONGO_URL = "mongodb://localhost:27017/rest";

app = Flask(__name__)

app.config['MONGOALCHEMY_DATABASE'] = 'redditdata'
app.config['MONGO_URI'] = MONGO_URL
mongo = MongoAlchemy(app)

"""storing"""
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger = logging.getLogger('prawcore')
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

reddit = praw.Reddit(client_id='3g4ggl4suqTAIQ', redirect_uri='http://localhost:8080', client_secret='es9JYYwOFw-Fxjp04PDQ64WSwvg', user_agent='praw-test')
subreddit = reddit.subreddit('india')
top_subreddit = subreddit.rising()
top_subreddit = subreddit.rising(limit=99999)
topics_list = []
for submission in top_subreddit:
    topics_dict = dict()
    topics_dict["title"] = submission.title
    topics_dict["score"] = submission.score
    topics_dict["id"] = submission.id
    topics_dict["url"] = submission.url
    topics_dict["comms_num"] = submission.num_comments
    topics_dict["body"] = submission.selftext
    topics_dict["flair"] = submission.link_flair_text
    tmp = topics_dict["title"] + "\n" + topics_dict["body"] +"\n" 
    for top_level_comment in submission.comments:
    	if isinstance(top_level_comment, MoreComments):
        	continue
    	topics_dict["comments"] = top_level_comment.body
    	tmp += topics_dict["comments"]
    if topics_dict["flair"] == None:
    	continue
    topics_dict["text"] = tmp
    topics_list.append(topics_dict)

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["redditdata"]
mycol = mydb["posts"]
mycol.insert_many(topics_list)

"""training"""
client = MongoClient()
db = client.redditdata
collection = db.posts
data = pd.DataFrame(list(collection.find()))

np.random.seed(100)
data['title'].dropna(inplace=True)
data['title'] = [entry.lower() for entry in data['title']]
data['title']= [word_tokenize(entry) for entry in data['title']]
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(data['title']):
	Final_words = []
	word_Lemmatized = WordNetLemmatizer()
	for word, tag in pos_tag(entry):
		if word not in stopwords.words('english') and word.isalpha():
			word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
			Final_words.append(word_Final)
	data.loc[index,'text_final'] = str(Final_words)

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(data['text_final'],data['flair'],test_size=0.3)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(data['text_final'])
filename = 'Tfidf_vector.sav'
joblib.dump(Tfidf_vect, filename)

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
filename = 'finalized_model.sav'
joblib.dump(SVM, filename)

def output_json(obj, code, headers=None):
    resp = make_response(dumps(obj), code)
    resp.headers.extend(headers or {})
    return resp

DEFAULT_REPRESENTATIONS = {'application/json': output_json}
api = restful.Api(app)
api.representations = DEFAULT_REPRESENTATIONS

import flask_rest_service.resources
