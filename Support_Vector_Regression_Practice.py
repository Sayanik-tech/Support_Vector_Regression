#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


dataset = pd.read_csv('Position_Salaries.csv')
dataset


# In[4]:


x = dataset.iloc[: , 1:-1].values
y = dataset.iloc[: ,-1].values


# In[5]:


print(x)


# In[6]:


print(y)


# In[7]:


y = y.reshape(len(y),1)


# In[8]:


print(y)


# In[9]:


## Feature Scaling


# In[10]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)


# In[12]:


print(x)


# In[13]:


print(y)


# In[14]:


## Training the SVR Model on the whole data set


# In[15]:


from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x,y)


# In[16]:


## predict the new result


# In[17]:


sc_y.inverse_transform(regressor.predict(sc_x.transform([[7.5]])))


# In[18]:


## visualization of SVR


# In[20]:


plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x)), color = 'blue')
plt.title('visualization of SVR model')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()


# In[22]:


x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))), color = 'blue')
plt.title('SVR')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




