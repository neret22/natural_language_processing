# import required libraries including accuracy_score, recall_score and f1_scre in order to estimate the accuracy of predictions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

#importing the dataset 
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting = 3)

#cleaning text for further manipulations
#here we 
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range (0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split() #how do we split, with which symbol here
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word is set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)
  
#creating the Bag of Words model 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1].values

#splitting the dataset to train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Naive Bayes Classification

#Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier_naive = GaussianNB()
classifier_naive.fit(X_train, y_train)

#Predicting the Test set results
y_pred_naive = classifier_naive.predict(X_test)


#Logistic Regression
#Scaling the training and test sets
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.fit_transform(X_test)

#Training the Logistic Regression model on train set
from sklearn.linear_model import LogisticRegression
classifier_logistic = LogisticRegression(random_state = 0)
classifier_logistic.fit(X_train_scaled, y_train)

#Predicting the Test set results
y_pred_logistic = classifier_logistic.predict(X_test_scaled)


#K-Nearest Neighbors

#Training the K-Nearest Neighbors model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier_k = KNeighborsClassifier(n_neighbors = 5,  p=2, metric='minkowski')
classifier_k.fit(X_train_scaled, y_train)

#Predicting the Test set results
y_pred_k = classifier_k.predict(X_test_scaled)


#Support Vector Machine
#Training the Support Vector Machine model on the Training set
from sklearn.svm import SVC
classifier_svc = SVC( kernel = 'linear', random_state = 0)
classifier_svc.fit(X_train_scaled, y_train)

#Predicting the Test set results
y_pred_svc = classifier_svc.predict(X_test_scaled)


#Kernel SVM

#Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
classifier_svm = SVC(kernel = 'rbf', random_state = 0)
classifier_svm.fit(X_train_scaled, y_train)

#Predicting the Test set results
y_pred_svm = classifier_svm.predict(X_test_scaled)


#Decision Tree Classification

#Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier_tree = DecisionTreeClassifier( criterion='entropy', random_state = 0)
classifier_tree.fit(X_train_scaled, y_train)

#Predicting the Test set results
y_pred_tree = classifier_tree.predict(X_test_scaled)


#Random Forest Classification

#Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state = 0)
classifier_forest.fit(X_train_scaled, y_train)

#Predicting the Test set results
y_pred_forest = classifier_forest.predict(X_test_scaled)

