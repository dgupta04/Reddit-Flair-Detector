import pandas as pd
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
import re

client = MongoClient()
database = client.trainTestData

punctuationRe = re.compile(r'[\/.,\[\];{}|:"<>\?!]')
symbolsRe = re.compile(r'[0-9@#$&%\n\*()!\+-]')
URLRe = re.compile(r'https?:\/\/\S+')

def makeString(text):
    return str(text)

def removePunctuation(text):    
    text = punctuationRe.sub(' ', text)
    return text

def removeSymbols(text):
    text = symbolsRe.sub(' ', text)
    return text

def removeURLs(text):
    text = URLRe.sub('', text)
    return text

all_data = pd.read_csv('../csv_data/data.csv')

all_data['comments'] = all_data['comments'].apply(makeString)
all_data['body'] = all_data['body'].apply(makeString)
all_data['flair'] = all_data['flair'].apply(makeString)
all_data['title'] = all_data['title'].apply(makeString)

all_data['comments'] = all_data['comments'].apply(removePunctuation)
all_data['comments'] = all_data['comments'].apply(removeSymbols)
all_data['comments'] = all_data['comments'].apply(removeURLs)

all_data['body'] = all_data['body'].apply(removePunctuation)
all_data['body'] = all_data['body'].apply(removeSymbols)
all_data['body'] = all_data['body'].apply(removeURLs)

all_data['title'] = all_data['title'].apply(removePunctuation)
all_data['title'] = all_data['title'].apply(removeSymbols)
all_data['title'] = all_data['title'].apply(removeURLs)

comments = []
body = []
titles = []
flairs = []
feature_combo = {'title_comment': [], 'comment_body': [], 'title_body': [], 'all_three': []}


for i in range(len(all_data['comments'])):
    comments.append(all_data['comments'][i])
    body.append(all_data['body'][i])
    flairs.append(all_data['flair'][i])
    titles.append(all_data['title'][i])
    feature_combo['title_comment'].append(all_data['comments'][i] + ' ' + all_data['title'][i])
    feature_combo['comment_body'].append(all_data['comments'][i] + ' ' + all_data['body'][i])
    feature_combo['title_body'].append(all_data['body'][i] + ' ' + all_data['title'][i])
    feature_combo['all_three'].append(all_data['body'][i] + ' ' + all_data['title'][i] + ' ' + all_data['comments'][i])

singleFeatureObject = {'comments':comments, 'body': body, 'title':titles}

for feature in singleFeatureObject:
    trainDataX, testDataX, trainDataY, testDataY = train_test_split(singleFeatureObject[feature], flairs, test_size=0.1, random_state=42)
    collection = database[feature]
    insertData = {"trainX": trainDataX, "testX": testDataX, "trainY": trainDataY, "testY": testDataY}
    collection.insert_one(insertData)

for featureCombo in feature_combo:
    trainDataX, testDataX, trainDataY, testDataY = train_test_split(feature_combo[featureCombo], flairs, test_size=0.1, random_state=42)
    collection = database[featureCombo]
    insertData = {"trainX": trainDataX, "testX": testDataX, "trainY": trainDataY, "testY": testDataY}
    collection.insert_one(insertData)

client.close()