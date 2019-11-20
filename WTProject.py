#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR


# In[2]:


def normalise_row(row):
    normalise_dict = {1 : 12, 2 : 6, 3 : 3, 4 : 3, 5 : 4 , 6 : 4}
    #print(row)
    #print(row['essay_set'])
    return row['domain1_score'] / normalise_dict[row['essay_set']]


# In[3]:


# read the data file
essays = pd.read_csv("essays.csv", encoding = 'latin1')
print(essays)


# In[4]:


# see domain1_score distribution

get_ipython().run_line_magic('matplotlib', 'inline')
essays.boxplot(column = 'domain1_score', by = 'essay_set', figsize = (10,10))


# In[5]:


#  important for prediction

essay_data = essays[['essay_set', 'essay', 'domain1_score']].copy()
essay = essays['essay']
essay_score = essays['domain1_score']


# In[6]:


# using bag of words
vectorizer = CountVectorizer(stop_words='english')
count_vectors = vectorizer.fit_transform(essay_data[(essay_data['essay_set'] == 1) | (essay_data['essay_set'] == 2)]['essay'])
feature_names = vectorizer.get_feature_names()

X = count_vectors.toarray()
y = essay_data[(essay_data['essay_set'] == 1) | (essay_data['essay_set'] == 2)]['domain1_score'].as_matrix()

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30)


# In[7]:


print(X_train)


# In[8]:


# first lets try linear Regression

linear_regressor = LinearRegression()

linear_regressor.fit(X_train, y_train)

y_pred = linear_regressor.predict(X_test)

#print(y_pred)

print('Coefficients: \n', linear_regressor.coef_)

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

print([i for i in X_test[0] if i!=0])

print("Please enter the test case as per format in essays.csv (sample displayed above): ")
xtest= input()

ypred = linear_regressor.predict([X_test[0]])   #change to xtest[i] where i is the index of test case

print("Predicted - linear regression")
for i in list(ypred):
    print(i)
    print(np.ceil(i))


# In[9]:


# lets try a lasso regression model

alphas = np.array([3, 1, 0.3, 0.1])

lasso_regressor = Lasso()

grid = GridSearchCV(estimator = lasso_regressor, param_grid = dict(alpha=alphas))
grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

#print('Coefficients: \n', lasso_regressor.coef_)

print("Please enter the test case as per format in essays.csv (sample displayed above): ")
xtest= input()

ypred = grid.predict([X_test[0]])   #change to xtest[i] where i is the index of test case

print("Predicted - lasso regression")
for i in list(ypred):
    print(i)
    print(np.ceil(i))


# In[10]:


# lets also try SVMs
svr = SVR(kernel='rbf')
svr.fit(X_train,y_train)

y_pred = svr.predict(X_test)

print("Mean squared error: %.2f" % mean_squared_error(y_test,y_pred))

print("Please enter the test case as per format in essays.csv (sample displayed above): ")
xtest= input()

ypred = svr.predict([X_test[0]])   #change to xtest[i] where i is the index of test case

print("SVM -rbf kernel")
for i in list(ypred):
    print(i)
    print(np.ceil(i))


# In[11]:


# see which svm model is the best

options = ['linear','rbf','poly','sigmoid']
mse = []

for option in options:
    svr = SVR(kernel=option)
    svr.fit(X_train,y_train)
    
    y_pred = svr.predict(X_test)
    
    mse.append(mean_squared_error(y_test,y_pred))
    
print(mse)


# In[12]:



objects = ('linear','rbf','poly','sigmoid')
y_pos = np.arange(len(objects))

plt.bar(y_pos, mse, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Mean Squared Error')
plt.title('SVM for different kernels')

plt.show()


# In[13]:


# lets try with tfidf

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(essay_data[(essay_data['essay_set'] == 1) | (essay_data['essay_set'] == 2)]['essay'])
y = essay_data[(essay_data['essay_set'] == 1) | (essay_data['essay_set'] == 2)]['domain1_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[14]:


print(X_train.shape)


# In[18]:


# first lets try linear Regression

linear_regressor = LinearRegression()

linear_regressor.fit(X_train, y_train)

y_pred = linear_regressor.predict(X_test)

print('Coefficients: \n', linear_regressor.coef_)

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

ypred = linear_regressor.predict(X_test[0].reshape(1,-1))   #change to xtest[i] where i is the index of test case

print("Predicted - linear regression")
for i in list(ypred):
    print(i)
    print(np.ceil(i))


# In[19]:


# lets try a lasso regression model

alphas = np.array([3, 1, 0.3, 0.1])

lasso_regressor = Lasso()

grid = GridSearchCV(estimator = lasso_regressor, param_grid = dict(alpha=alphas))
grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

#print('Coefficients: \n', lasso_regressor.coef_)

ypred = grid.predict(X_test[0].reshape(1,-1))   #change to xtest[i] where i is the index of test case

print("Predicted - lasso regression")
for i in list(ypred):
    print(i)
    print(np.ceil(i))


# In[20]:


# lets also try SVMs
svr = SVR(kernel='linear',C=0.5)
svr.fit(X_train,y_train)

y_pred = svr.predict(X_test)

print("Mean squared error: %.2f" % mean_squared_error(y_test,y_pred))

ypred = svr.predict(X_test[0].reshape(1,-1))   #change to xtest[i] where i is the index of test case

print("Predicted - SVM linear kernel")
for i in list(ypred):
    print(i)
    print(np.ceil(i))


# In[21]:


# see which svm model is the best

options = ['linear','rbf','poly','sigmoid']
mse = []

for option in options:
    svr = SVR(kernel=option)
    svr.fit(X_train,y_train)
    
    y_pred = svr.predict(X_test)
    
    mse.append(mean_squared_error(y_test,y_pred))
    
print(mse)


# In[22]:


import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = ('linear','rbf','poly','sigmoid')
y_pos = np.arange(len(objects))
 
plt.bar(y_pos, mse, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Mean Squared Error')
plt.title('SVM for different kernels')
 
plt.show()


# In[ ]:




