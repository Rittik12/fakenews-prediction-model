#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from numpy import nan
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# In[2]:


os.chdir(r"D:\news project\fakenews\news-1")


# In[3]:


df1=pd.read_csv("news.csv")
df1.head()


# In[4]:


df1.shape


# In[5]:


df1.isnull().sum()


# In[6]:


df1.columns


# In[7]:


y=df1['label']
df1.drop('label',axis=1, inplace=True)
print(y)


# In[8]:


X_train,X_test,y_train,y_test= train_test_split(df1['text'],y,test_size=0.2,random_state=7)


# In[9]:


tfidf_vectorizer= TfidfVectorizer()
tfidf_train= tfidf_vectorizer.fit_transform(X_train)
tfidf_test= tfidf_vectorizer.transform(X_test)


# In[13]:


clf=MultinomialNB()
clf.fit(tfidf_train,y_train)


# In[15]:


prediction=clf.predict(tfidf_test)
score=accuracy_score(prediction,y_test)
print(f'Accuracy score: {round(score*100,2)}%')


# In[16]:


confusion_matrix(y_test,prediction,labels=['REAL','FAKE'])


# In[ ]:




