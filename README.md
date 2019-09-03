# Reddit Flair Detector

I developed a flair detector, which detects the flair of any post given from the [r/india] subreddit.

# Directory structure

The root directory has the following components:
- [app.py] - The main Flask application while which powers the whole UI
- [worker] - Folder containing the code snippets for data extraction, model generation and prediction
  - [worker/dataCollector.py] - Extracts the data from the [r/india] subreddit and stores it into [csv_data/data.csv]
  - [worker/modelGenerator.py] - Reads the data, performs pre-processing tasks and generates various models using a combination of features. The model with the highest frequency goes to the joblib dump
  - [worker/predictor.py] - the core prediction algorithm running after the URL is fed on the main website
  - [worker/config.py] - the credentials used for accessing the Reddit API. Can be altered according to user preference.
  - [worker/util.py] - utility functions for pre-processing tasks
  - [worker/mongoStorage.py] - Python script to store data in MongoDB collections
- [Procfile] - used for initiating Heroku app
- [requirements.txt] - Python package requirements, installed on deployment
- [.gitignore] - files to be ignored by git
- [db] - contains the MongoDB collections to be accessed by the user
- [templates] - templates for different webpages, with Jinja2 templating format

# Codebase

The entire codebase is written in the Python language due to ease of usage and powerful computation abilities. I developed the web application using the Flask framework, which is deployed using the gunicorn WSGI server on Heroku.

# Running locally

To run the code locally on your machine, please execute the following commands:

1. `git clone https://github.com/dgupta04/Reddit-Flair-Detector` to clone the repository locally
2. `pip install -r requirements.txt` to install all the dependencies
3. `mongod --dbpath="./db"` to run the existing MongoDB instance
4. `mongo` to access all the MongoDB collections
5. `flask run` to run the application locally on the assigned port number.


# The approach

Performing an ML classification on textual data requires the use of many algorithms for tasks like pre-processing the data, extarcting useful features from it and feeding it into a classifier which generates maximum accuracy. 

Although discrete features like upvotes and timestamp; and binary features like Over-18 and isSelf are available to us, these features are not useful in predicting the flair of a post due to huge variations and outlier values. Thus, textual features like comments, the body of the post and its title were used as the main features.

### 1. Collecting data

The data consists of 1018 unique reddit posts from the [r/india] subreddit. The \[R]eddiquette flair consisted of a mere 18 posts while the others had 100 posts per flair. Data collection was the longest part of the whole process, and took about an hour for complete retrieval. The data was stored into a CSV file for further usage.

### 2. Pre-processing data

Pre-processing the data involved removal of symbols, punctuation marks and unwanted URLs to generate a cleaner set of words as a whole feature. This was done using the [re] module in Python.

### 3. Splitting training and testing sets

The cleaned data is split into training and testing sets. In my example, the test data consists of 10% of the overall data, which makes it 102 test samples, leading to a a high number of test samples.

### 4. Processing textual data into meaningful features

Raw textual data cannot be used as a feature since machines understand numbers only. For this purpose, scikit-learn offers powerul processing tools such as the TfidfVectorizer and the CountVectorizer. The text was fed into these vectorizers and analysed further.

### 5. Fitting a suitable model

The TfidfVectorizer was used as the choice of vectorizer to derive better performance of the model. Further, the output yielded by the vectorizer was pipelined with different models. The models used to fit the data include:

  - Multinomial Naive Bayes Classifier
  - Logistic Regression Classifier
  - Multi-Layer Perceptron (MLP) Classifier
  - Random Forrest Classifier
 
 The results yielded by each of these classifiers are as follows:
 
#### 1. Using post comments as the only feature
 
| Classifier                  | Accuracy      |
| :--------------------------: |:-------------:|
| Multinomial Naive Bayes    |51.96078431372549%  |
| Logistic Regression Classifier      |**59.80392156862745%**|
| Multi-Layer Perceptron Classifier |53.92156862745098%       |
| Random Forrest Classifier | **59.80392156862745%**|

#### 2. Using post body as the only feature
 
| Classifier                  | Accuracy      |
| :--------------------------: |:-------------:|
| Multinomial Naive Bayes    |24.50980392156862%  |
| Logistic Regression Classifier      |24.509803921568626%|
| Multi-Layer Perceptron Classifier |21.568627450980393%       |
| Random Forrest Classifier | **31.37254901960784%**|

#### 3. Using post title as the only feature
 
| Classifier                  | Accuracy      |
| :--------------------------: |:-------------:|
| Multinomial Naive Bayes    |65.68627450980392%  |
| Logistic Regression Classifier      |65.68627450980392%|
| Multi-Layer Perceptron Classifier |57.84313725490197%       |
| Random Forrest Classifier | **67.64705882352942%**|

#### 4. Using post title and comments as the features
 
| Classifier                  | Accuracy      |
| :--------------------------: |:-------------:|
| Multinomial Naive Bayes    |64.70588235294118%  |
| Logistic Regression Classifier      |78.43137254901961%|
| Multi-Layer Perceptron Classifier |65.68627450980392%       |
| Random Forrest Classifier | **82.35294117647058%**|

#### 5. Using post comments and body as the features
 
| Classifier                  | Accuracy      |
| :--------------------------: |:-------------:|
| Multinomial Naive Bayes    |59.80392156862745%  |
| Logistic Regression Classifier      |69.6078431372549%|
| Multi-Layer Perceptron Classifier |61.76470588235294%       |
| Random Forrest Classifier |**69.6078431372549%**|



#### 6. Using post comments and body as the features
 
| Classifier                  | Accuracy      |
| :--------------------------: |:-------------:|
| Multinomial Naive Bayes    |71.56862745098039%  |
| Logistic Regression Classifier      |75.49019607843137%|
| Multi-Layer Perceptron Classifier |57.84313725490197%       |
| Random Forrest Classifier | **77.45098039215687%**|

#### 7. Using all three (title + comment + body) as the features
 
| Classifier                  | Accuracy      |
| :--------------------------: |:-------------:|
| Multinomial Naive Bayes    |69.6078431372549%  |
| Logistic Regression Classifier      |79.41176470588235%|
| Multi-Layer Perceptron Classifier |71.56862745098039%       |
| Random Forrest Classifier | **85.29411764705882%**|

It was observed that the Random Forrest Classifier gave the highest accuracy of around 85% using (title + body + comment) as the features. Moreover, using post body as the only feature gave the worst accuracies on all classifiers. Thus, a combination of features was used to predict the flair correctly.

# References

1. https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
2. https://www.kdnuggets.com/2016/10/beginners-guide-neural-networks-python-scikit-learn.html/2

[r/india]: <https://www.reddit.com/r/india>
[app.py]: <https://github.com/dgupta04/Precog2019/blob/master/app.py>
[worker]: <https://github.com/dgupta04/Precog2019/blob/master/worker>
[worker/dataCollector.py]: <https://github.com/dgupta04/Precog2019/blob/master/worker/dataCollector.py>
[csv_data/data.csv]: <https://github.com/dgupta04/Precog2019/blob/master/csv_data/data.csv>
[worker/modelGenerator.py]: <https://github.com/dgupta04/Precog2019/blob/master/worker/modelGenerator.py>
[worker/predictor.py]: <https://github.com/dgupta04/Precog2019/blob/master/worker/predictor.py>
[worker/config.py]: <https://github.com/dgupta04/Precog2019/blob/master/worker/config.py>
[worker/util.py]: <https://github.com/dgupta04/Precog2019/blob/master/worker/util.py>
[Procfile]: <https://github.com/dgupta04/Precog2019/blob/master/Procfile>
[requirements.txt]: <https://github.com/dgupta04/Precog2019/blob/master/requirements.txt>
[.gitignore]: <https://github.com/dgupta04/Precog2019/blob/master/.gitignore>
[re]: <https://docs.python.org/3/library/re.html>
[db]: <https://github.com/dgupta04/Precog2019/blob/master/db>
[worker/mongoStorage.py]: <https://github.com/dgupta04/Precog2019/blob/master/worker/mongoStorage.py>
[templates]: <https://github.com/dgupta04/Precog2019/blob/master/templates>
