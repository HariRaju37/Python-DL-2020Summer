#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier


train_df = pd.read_csv('./train_preprocessed.csv')
test_df = pd.read_csv('./test_preprocessed.csv')
X_train = train_df.drop("Survived",axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId",axis=1).copy()
print(train_df[train_df.isnull().any(axis=1)])
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)

train_df[['Survived', 'Sex']].groupby(['Survived'], as_index=False).mean().sort_values(by='Sex', ascending=False)


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd

glass_df = pd.read_csv('./glass.csv')
glass_df
X_train = glass_df.drop("Type",axis=1)
Y_train = glass_df["Type"]

X_train, X_test, Y_train, Y_test= train_test_split(X_train, Y_train, test_size=0.4, random_state=0)
model = GaussianNB()
model.fit(X_train,Y_train)

y_pred = model.predict(X_test)


score = accuracy_score(Y_test,y_pred)*100
print("accuracy score: " + str(score))

print(classification_report(Y_test, y_pred))


# In[5]:


svc = SVC()
svc.fit(X_train,Y_train)
y_pred1 = svc.predict(X_test)
y_pred1

acc_svc = accuracy_score(Y_test, y_pred1) * 100
print("score: " + str(acc_svc))
print(classification_report(Y_test, y_pred1))

