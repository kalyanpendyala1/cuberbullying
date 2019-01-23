import numpy as np 
import pandas as pd 
import os

#print(os.listdir('C:/Users/kalya/Desktop'))
df = pd.read_json('C:/Users/kalya/Desktop/Dataset for Detection of Cyber-Trolls.json', lines=True)
print(df.head())

from nltk.corpus import stopwords
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
corpus = []

for i in range(0, len(df)):
	review = re.sub('[^a-zA-Z]', ' ', df['content'][i])
	review = review.lower()
	review = review.split()
	review = ' '.join(review)
	corpus.append(review)

bow_transformer = CountVectorizer()
bow_transformer = bow_transformer.fit(corpus)

print('Length of the Vocabulary: ',len(bow_transformer.vocabulary_))
messages_bow = bow_transformer.transform(corpus)
tfidf_transformer = TfidfTransformer().fit(messages_bow)
X = tfidf_transformer.transform(messages_bow)

y = []
for i in range(0, len(df)):
	y.append(df.annotation[i]['label'])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean=False)

x_train_std = sc.fit_transform(X_train)
x_test_std = sc.transform(X_test)




#------------Naive Bayes--------------------
from sklearn.naive_bayes  import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print('\n\nResults with naive bayes')
print('Accuracy:%.2f ' %accuracy_score(y_test,y_pred))
confusion_matrix = confusion_matrix(y_test,y_pred)
print("Confusion Matrix", confusion_matrix)
print('F1-score', f1_score(y_test, y_pred, average="macro"))
print('Precision Score: ', precision_score(y_test, y_pred, average="macro"))
print("recall_score: ",recall_score(y_test, y_pred, average="macro"))
#----------------------------------------------


#--------------Decision Tree-------------------
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print('\n\nResults with Decision Tree')
print('Accuracy:%.2f ' %accuracy_score(y_test,y_pred))
confusion_matrix = confusion_matrix(y_test,y_pred)
print("Confusion Matrix", confusion_matrix)
print('F1-score', f1_score(y_test, y_pred, average="macro"))
print('Precision Score: ', precision_score(y_test, y_pred, average="macro"))
print("recall_score: ",recall_score(y_test, y_pred, average="macro"))
#-----------------------------------------------


#------------Random Forest----------------------
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print('\n\nResults with Random Forest')
print('Accuracy:%.2f ' %accuracy_score(y_test,y_pred))
confusion_matrix = confusion_matrix(y_test,y_pred)
print("Confusion Matrix", confusion_matrix)
print('F1-score', f1_score(y_test, y_pred, average="macro"))
print('Precision Score: ', precision_score(y_test, y_pred, average="macro"))
print("recall_score: ",recall_score(y_test, y_pred, average="macro"))
#-------------------------------------------------


#----------------Logistic Regression--------------
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0, solver='lbfgs')
lr.fit(x_train_std, y_train)
y_pred = lr.predict(x_test_std)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print('\n\nResults with LogisticRegression')
print('Accuracy:%.2f ' %accuracy_score(y_test,y_pred))
confusion_matrix = confusion_matrix(y_test,y_pred)
print("Confusion Matrix", confusion_matrix)
print('F1-score', f1_score(y_test, y_pred, average="macro"))
print('Precision Score: ', precision_score(y_test, y_pred, average="macro"))
print("recall_score: ",recall_score(y_test, y_pred, average="macro"))
#-------------------------------------------------


#----------Support Vector Machines---------------
from sklearn.svm import SVC
#svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10, probability=True) # high precision, low recall, why?
svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10)
svm.fit(x_train_std, y_train)
y_pred = svm.predict(x_test_std)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print('\n\nResults with Support Vector Machine')
print('Accuracy:%.2f ' %accuracy_score(y_test,y_pred))
confusion_matrix = confusion_matrix(y_test,y_pred)
print("Confusion Matrix", confusion_matrix)
print('F1-score', f1_score(y_test, y_pred, average="macro"))
print('Precision Score: ', precision_score(y_test, y_pred, average="macro"))
print("recall_score: ",recall_score(y_test, y_pred, average="macro"))
#------------------------------------------------


#-------------K-Nearest Neighbors-----------------
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors =5, metric = 'minkowski')
knn.fit(x_train_std, y_train)
y_pred = knn.predict(x_test_std)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print('\n\nResults with K-Nearest neighbors')
print('Accuracy:%.2f ' %accuracy_score(y_test,y_pred))
confusion_matrix = confusion_matrix(y_test,y_pred)
print("Confusion Matrix", confusion_matrix)
print('F1-score', f1_score(y_test, y_pred, average="macro"))
print('Precision Score: ', precision_score(y_test, y_pred, average="macro"))
print("recall_score: ",recall_score(y_test, y_pred, average="macro"))
#--------------------------------------------------
