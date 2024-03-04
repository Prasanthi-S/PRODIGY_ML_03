#!/usr/bin/env python
# coding: utf-8

# In[37]:


import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[12]:


df=pd.read_csv('cell_samples.csv')


# In[13]:


df.head()


# In[14]:


df.shape


# In[15]:


df.size


# In[16]:


df.count()


# In[18]:


df['Class'].value_counts()


# In[20]:


benign_df=df[df['Class']==2][0:200]
malignant_df=df[df['Class']==4][0:200]
axes=benign_df.plot(kind='scatter', x='Clump', y='UnifSize', color='red', label='Benign')
malignant_df.plot(kind='scatter', x='Clump', y='UnifSize', color='blue', label='Benign', ax=axes)


# In[22]:


df.dtypes


# In[38]:


df=df[pd.to_numeric(df['BareNuc'],errors='coerce').notnull()]
df['BareNuc']=df['BareNuc'].astype('int')
df.dtypes


# In[27]:


feature_df=df[['Clump','UnifSize','UnifShape','MargAdh','SingEpiSize','BareNuc','BlandChrom','NormNucl','Mit']]
X=np.asarray(feature_df)
y=np.asarray(df['Class'])
y[0:5]


# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


# In[29]:


X_train.shape


# In[30]:


y_train.shape


# In[31]:


X_test.shape


# In[33]:


y_test.shape


# In[34]:


from sklearn import svm


# In[35]:


classifier= svm.SVC(kernel='linear', gamma='auto', C=2)
classifier.fit(X_train, y_train)

y_predict=classifier.predict(X_test)


# In[36]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))


# In[ ]:




