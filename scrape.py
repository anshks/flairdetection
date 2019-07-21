import praw
from praw.models import MoreComments
import pandas as pd
import datetime as dt
import logging
import pymongo	
from pymongo import MongoClient

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger = logging.getLogger('prawcore')
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

reddit = praw.Reddit(client_id='3g4ggl4suqTAIQ', redirect_uri='http://localhost:8080', client_secret='es9JYYwOFw-Fxjp04PDQ64WSwvg', user_agent='praw-test')
subreddit = reddit.subreddit('india')
top_subreddit = subreddit.hot()
top_subreddit = subreddit.hot(limit=99999)
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
    	topics_dict["flair"] = ""
    topics_dict["text"] = tmp
    topics_list.append(topics_dict)
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["redditdata"]
mycol = mydb["posts"]
mycol.insert_many(topics_list)
