#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sklearn.datasets
from sklearn.model_selection import train_test_split


# In[2]:


from keras.models import Sequential


# In[3]:


from keras.layers.core import Dense, Activation


# In[4]:


from keras.layers.recurrent import LSTM, GRU


# In[7]:


import statsmodels.api as sm


# In[8]:


df=sm.datasets.elnino.load_pandas().data


# In[20]:


x=df.as_matrix()[:,1:-1]


# In[22]:


x=(x-x.min())/(x.max()-x.min())


# In[29]:


y=df.as_matrix()[:,-1].reshape(61)


# In[30]:


y=(y-y.min())/(y.max()-y.min())


# In[31]:


x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=.1)


# In[32]:


model=Sequential()


# In[33]:


model.add(GRU(20, input_shape=(11,1)))


# In[34]:


model.add(Dense(1, activation='sigmoid'))


# In[36]:


model.compile(loss='mean_squared_error', optimizer='adadelta')


# In[37]:


model.fit(x_train.reshape((54,11,1)), y_train, nb_epoch=5)


# In[41]:


proba=model.predict_proba(x_test.reshape((7,11,1)), batch_size=32)
proba


# In[42]:


import pandas as pd
pred=pd.Series(proba.flatten())
pred


# In[43]:


true=pd.Series(y_test.flatten())


# In[44]:


print("Corr. of preds and truth:", pred.corr(true))


# In[ ]:




