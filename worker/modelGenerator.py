import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump
import re
import pathlib

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

def MNB(trainX, trainY, testX, testY):
    NBClassifier = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')), ('nb',MultinomialNB())])
    NBClassifier.fit(trainX, trainY)
    classPred = NBClassifier.predict(testX)
    return accuracy_score(testY, classPred)

def RNF(trainX, trainY, testX, testY):
    RNF = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')), ('rnf',RandomForestClassifier(n_estimators=1000, random_state=42))])
    RNF.fit(trainX, trainY)
    classPred = RNF.predict(testX)
    return accuracy_score(testY, classPred)

def MLP(trainX, trainY, testX, testY):
    MLP = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')), ('mlp',MLPClassifier(hidden_layer_sizes=(30,30,30), activation='relu'))])
    MLP.fit(trainX, trainY)
    classPred = MLP.predict(testX)
    return accuracy_score(testY, classPred)

def LRC(trainX, trainY, testX, testY):
    LRC = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')), ('lr', LogisticRegression(penalty='l2',random_state=10, solver='lbfgs', multi_class='multinomial'))])
    LRC.fit(trainX, trainY)
    classPred = LRC.predict(testX)
    return accuracy_score(testY, classPred)

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
    MNB(trainDataX, trainDataY, testDataX, testDataY)
    RNF(trainDataX, trainDataY, testDataX, testDataY)
    MLP(trainDataX, trainDataY, testDataX, testDataY)
    LRC(trainDataX, trainDataY, testDataX, testDataY)
    

for featureCombo in feature_combo:
    trainDataX, testDataX, trainDataY, testDataY = train_test_split(feature_combo[featureCombo], flairs, test_size=0.1, random_state=42)
    MNB(trainDataX, trainDataY, testDataX, testDataY)
    RNF(trainDataX, trainDataY, testDataX, testDataY)
    MLP(trainDataX, trainDataY, testDataX, testDataY)
    LRC(trainDataX, trainDataY, testDataX, testDataY)

highestAccuracyTrainX, highestAccuracyTestX, higestAccuracyTrainY, highestAccuracyTestY = train_test_split(feature_combo['all_three'], flairs, test_size=0.1, random_state=42 )
RNF_allThree = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')), ('rnf',RandomForestClassifier(n_estimators=1000, random_state=42))])
RNF_allThree.fit(highestAccuracyTrainX, higestAccuracyTrainY)

highestAccuracyModel = RNF_allThree

dump(highestAccuracyModel, '../model/finalModel.joblib')











