import pandas as pd
from joblib import load
from worker import config
from worker import util
import praw
import sys
import re

punctuationRe = util.punctuationRe
symbolsRe = util.symbolsRe
URLRe = util.URLRe
makeString = util.makeString
removePunctuation = util.removePunctuation
removeSymbols = util.removeSymbols
removeURLs = util.removeURLs

# def makeString(text):
#     return str(text)

# def removePunctuation(text):    
#     text = punctuationRe.sub(' ', text)
#     return text

# def removeSymbols(text):
#     text = symbolsRe.sub(' ', text)
#     return text

# def removeURLs(text):
#     text = URLRe.sub('', text)
#     return text

def predictFlair(urlInp):
    model = load('./model/finalModel.joblib')

    credentials = config.global_config

    reddit = praw.Reddit(client_id=credentials['client_id'], client_secret=credentials['client_secret'], password=credentials["password"], user_agent=credentials["user_agent"], username=credentials["username"])

    print(model)

    submission = praw.models.Submission(reddit, url=urlInp)

    title = submission.title
    body = submission.selftext
    comments = ''

    submission.comments.replace_more(limit=None)
    for comment in submission.comments:
        comments += comment.body + ' '

    totalPost = title + ' ' + body + ' ' + comments

    totalPost = removePunctuation(totalPost)
    totalPost = removeSymbols(totalPost)
    totalPost = removeURLs(totalPost)

    return model.predict([totalPost])[0]
