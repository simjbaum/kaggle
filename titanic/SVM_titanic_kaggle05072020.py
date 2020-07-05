#!/usr/bin/env python
# coding: utf-8

# # kaggle titanic challenge

# In[184]:


import pandas as pd
import numpy as np
import plotly.express as px
import re


# # load the data

# In[185]:


test = pd.read_csv("Documents/Bioinformatics/kaggle/titanic/data/test.csv")


# In[186]:


test.head(2)


# In[187]:


train = pd.read_csv("Documents/Bioinformatics/kaggle/titanic/data/train.csv")


# In[188]:


train.head(2)


# In[189]:


df =  pd.read_csv("Documents/Bioinformatics/kaggle/titanic/data/gender_submission.csv")


# # data processing - str to int or removing Nans

# In[190]:


test["gender"] = test.apply(lambda x: 0 if x["Sex"] == "male" else 1, axis = 1)
train["gender"] = train.apply(lambda x: 0 if x["Sex"] == "male" else 1, axis = 1)


# In[191]:


test[~test.Name.str.contains("Mr")]


# In[192]:


vec_test = []
for row in test["Embarked"]:
    if row == "Q":
        vec_test.append(0)
    if row == "S":
        vec_test.append(1)
    if row == "C":
        vec_test.append(2)


# In[193]:


test["Embarked_v1"] = vec_test


# In[194]:


vec_train = []
for row in train["Embarked"]:
    if row == "Q":
        vec_train.append(0)
    elif row == "S":
        vec_train.append(1)
    elif row == "C":
        vec_train.append(2)
    else:
        vec_train.append(3)
    


# In[195]:


train["Embarked_v1"] = vec_train


# In[196]:


# explore the age


# In[197]:


px.scatter(test, x = "Age")


# In[198]:


# try first with giving age = 0 if nan


# In[199]:


test["Age"].fillna(0, inplace=True)
test["Age"] = test["Age"].astype('int64')


# In[200]:


train["Age"].fillna(0,inplace=True)
train["Age"] = train["Age"].astype('int64')


# In[213]:


# write Fare as numeric not float # and fill NA


# In[214]:


test["Fare"].fillna(0, inplace=True)
train["Fare"].fillna(0,inplace=True)


# In[215]:


train["Fare"] = train["Fare"].astype('int64')
test["Fare"] = test["Fare"].astype('int64')


# In[ ]:


# fill all Nas with 0 for the beginning


# In[216]:


test.fillna(0, inplace=True)
train.fillna(0,inplace=True)


# In[201]:


# look at the tickets:


# In[202]:


#train["ticket_str"] = train.apply(lambda x: re.sub("\d+", "", x["Ticket"]), axis = 1)


# In[203]:


#pd.unique(train["ticket_str"] )


# In[204]:


# ok - I can come back later to this to make sense out of it


# In[205]:


# Fare


# In[206]:


px.box(train, "Pclass", "Fare")


# In[ ]:


# check cabins


# In[211]:


pd.unique(train["Cabin"])


# # ok train a simple SVM model on the data:

# In[ ]:


#remove PassengerId, Name, Sex (we have in a another variable), Ticket, Cabin, Embarked


# In[218]:


train.drop(columns = ["Name", "PassengerId", "Sex", "Ticket", "Cabin", "Embarked"], inplace=True)
test.drop(columns = ["Name", "PassengerId", "Sex", "Ticket", "Cabin", "Embarked"], inplace=True)


# In[219]:


train


# In[224]:


train.iloc[:,1:]


# In[226]:


train.iloc[:,:1]


# In[228]:


from sklearn import svm


# In[238]:


X = np.array(train.iloc[:,1:])
y = np.array(train.iloc[:,:1]).squeeze()


# In[242]:


x = np.array(test)


# In[239]:


clf = svm.SVC()
clf.fit(X, y)


# In[245]:


res = clf.predict(x)


# In[248]:


test = pd.read_csv("Documents/Bioinformatics/kaggle/titanic/data/test.csv")


# In[252]:





# In[256]:


to_submit = pd.DataFrame({"PassengerId":pd.array(test["PassengerId"]), 
             "Survived":res})


# In[257]:


get_ipython().system('mkdir "Documents/Bioinformatics/kaggle/titanic/submission"')


# In[261]:


to_submit.to_csv("Documents/Bioinformatics/kaggle/titanic/submission/05072020_titanic_SVM.csv", index=False)


# ### example submission data format

# In[262]:


pd.read_csv("Documents/Bioinformatics/kaggle/titanic/submission/05072020_titanic_SVM.csv")


# In[244]:


pd.read_csv("Documents/Bioinformatics/kaggle/titanic/data/gender_submission.csv")


# In[ ]:




