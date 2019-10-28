#!/usr/bin/env python
# coding: utf-8

# Simple implementation of a least-squares solution to linear regression that applies an iterative update to adjust the weights.

# In[1]:


from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load the diabetes dataset
diabetes = datasets.load_diabetes()


# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]


# In[2]:


#We are looking for y = d + w0x where w0 is or minimized wieght.
diabetes_X_mean = np.mean(diabetes_X_test)
diabetes_y_mean = np.mean(diabetes_y_test)

#Now we wanna calculate the numerator and denominator of the Linear Regression weight formula
num = 0
den = 0

#Length of data
l = len(diabetes_y_test)

#Calculating the numerator and denominator 
for i in range(l):
    num += (diabetes_X_test[i] - diabetes_X_mean) * (diabetes_y_test[i] - diabetes_y_mean)
    den += (diabetes_X_test[i] - diabetes_X_mean) ** 2
w = num / den
d = diabetes_y_mean - (w * diabetes_X_mean)

#Lets print our equation 
print("Equation is: y = %f + %fx" %(w,d))

#Getting maximum and minimum of dataset - we add 50% of each to themselves for better view in our plot
max_x = np.max(diabetes_X_test) + np.max(diabetes_X_test)*0.5
min_x = np.min(diabetes_X_test) + np.min(diabetes_X_test)*0.5

#Using linspace to create the line
x = np.linspace(min_x, max_x, 1000)
y = d + w * x

#Plotting the linear regeerssion line.
plt.plot(x, y, color='red', label='line')

#Plotting scatter plot
plt.scatter(diabetes_X_test, diabetes_y_test)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# Using Linear regression and other models to predict revenue of movies.

# In[1]:


### An example to load a csv file
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from ast import literal_eval
import os
os.chdir('D:\\University\\CPS803\\Assignment1\\A1')
meta_data=pd.read_csv('movies_metadata.csv', low_memory=False) # You may wish to specify types, or process columns once read
ratings_small=pd.read_csv('ratings_small.csv')
import warnings; warnings.simplefilter('ignore')
##### YOUR CODE HERE #######
#Set the proper data type for relative data.
meta_data['popularity'] = pd.to_numeric(meta_data['popularity'],errors='coerce')
meta_data['budget'] = pd.to_numeric(meta_data['budget'],errors='coerce')
meta_data['runtime'] = pd.to_numeric(meta_data['runtime'],errors='coerce')


# In[2]:


meta_data.columns


# In[3]:


meta_data.head()


# In[4]:


#Set train data and delete irrelevant columns
train = meta_data
train_id = train['id']
train_imdb_id = train['imdb_id']
train_revenue = train['revenue']
train_status = train['status']
del train['id']
del train['imdb_id']
del train['adult']
del train['status']


# In[5]:


#Only select data that are numbers
data_num = meta_data.select_dtypes(include=[np.number])

#Find correlation between data and plotting heatmap
corr = data_num.corr()
plt.subplots(figsize=(8,5))
sns.heatmap(corr, annot=True)

#Second heatmap showing top 50% of correlative data
top_feature = corr.index[abs(corr['revenue']>0.5)]
plt.subplots(figsize=(5, 3))
top_corr = meta_data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

most_corr = pd.DataFrame(top_feature)
most_corr.columns = ['Most correlated features']
most_corr


# In[6]:


#Scatter plot of ratings(Vote Average) vs revenue
plt.figure(figsize=(15, 8))
plt.scatter(meta_data['vote_average'], meta_data['revenue'], c='red', s=10)
plt.xlabel('Ratings')
plt.ylabel('Revenue')
plt.show()


# Features are information about different movies such as release date, production company and etc. Above code shows the head of data and how they are represented
# 
# we can see the movie ratings are in the range of 0 to 10 however before 0.5 rating the numbers of movies are insignificant therefore we can argue that it is an outlier and remove it in further steps. We can also abserve that most of our ratings are in the range 6-8
# 
# Based on the heat map of the top correlative features we can say that Budget, Votings and popularity are the main indicators
# of the movie revenue, therefore in furhter steps all are caluclations and reconstruction of data will be based on those features as they are the main features that will be used in our models(Other features will not cause huge variance in prediction).

# In[7]:


# The following line is one way of cleaning up the genres field - there are more verbose ways of doing this that are easier for a human to read
#meta_data['genres'] = meta_data['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
#meta_data['year'] = pd.to_datetime(meta_data['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
#meta_data.head()
# Consider how to columns look before and after this 'clean-up' - it is very common to have to massage the data to get the right features

##### YOUR CODE HERE #######
#Lets first see which features have null values 
train.columns[train.isnull().any()]


# In[8]:


#Let's find the columns with any null objects and plot them based on their count
data_na = train.isnull().sum()
data_na = data_na[data_na>0]
data_na = data_na.to_frame()
data_na.columns = ['count']
data_na.index.names = ['name']
data_na['name'] = data_na.index


# In[9]:


#Plotting a bar plot of number null objects in each column
plt.figure(figsize=(25,8))
sns.set(style='ticks')
sns.barplot(x='name', y='count', data = data_na)
plt.show()


# In[10]:


#The numerical features of data. We will used the median of each and group movies based on production companies.
train.select_dtypes(include=[np.number]).columns


# In[11]:


#First drop columns with more than 80% null values
train = train.dropna(thresh=0.80*len(train), axis=1)


# In[12]:


#Check now to see which columns still have null values
train.columns[train.isnull().any()]


# In[13]:


#Now let change the null values
#Lets fill in the missing values that are strings
for col in ('original_language', 'overview','poster_path',
            'production_countries', 'release_date','spoken_languages','title','production_companies',
            'video'):
    train[col] = train[col].fillna('None')
#We set the revenue.votings and popularityy of missing movies to there median value based on their production companies
#The difference of grouping and not grouping is that the features are grouped by thei prodcution companies which will 
#give us a better average for each featurer rather than the general average
train['revenue'] = train.groupby('production_companies')['revenue'].transform(lambda x: x.fillna(x.median()))
train['vote_average'] = train.groupby('production_companies')['vote_average'].transform(lambda x: x.fillna(x.median()))
train['popularity'] = train.groupby('production_companies')['popularity'].transform(lambda x: x.fillna(x.median()))
train['vote_count'] = train.groupby('production_companies')['vote_count'].transform(lambda x: x.fillna(x.median()))
#If any empty rows left in numerical values fill them with their mean
for col in ('revenue','popularity','vote_average', 'vote_count', 'runtime', 'budget', 'popularity'):
     train[col] = train[col].fillna(train[col].mean())


# In[14]:


#We have to now change all string values of data to float, as it is necessary for plotting and training models
col = ('genres','original_language', 'original_title', 'overview',
       'poster_path', 'production_companies', 'production_countries','release_date', 'spoken_languages',
       'title', 'video')
from sklearn.preprocessing import LabelEncoder
for c in col:
    lbl = LabelEncoder()
    lbl.fit(list(train[c].values))
    train[c] = lbl.transform(list(train[c].values))


# In[15]:


#Lets plot scatter plot of the tom correlative features and find their outliers.
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(16, 7))
axes = np.ravel(axes)
col_name = ['budget','vote_count','vote_average','popularity']
for i, c in zip(range(4), col_name):
    train.plot.scatter(ax=axes[i], x=c, y='revenue', sharey=True, colorbar=False, c='blue')


# In[16]:


#Lets remove outliers
train = train[train['budget'] < 260000000]
train = train[train['vote_count'] < 10500]
train = train[train['vote_average'] > 0.5]
train = train[train['popularity'] < 250]


# In[17]:


#Lets draw box plot's for all relative data with high correlation to the revenue after removing the outliers
fig = plt.figure(figsize=(20, 15))
sns.set(style='ticks')
fig2 = fig.add_subplot(221);
fig2.set_title('Budget', fontsize=16)
plt.boxplot(train['budget'])
fig3 = fig.add_subplot(222);
fig3.set_title('Vote Average', fontsize=16)
plt.boxplot(train['vote_average'])
fig4 = fig.add_subplot(223);
fig4.set_title('Popularity', fontsize=16)
plt.boxplot(train['popularity'])
fig5 = fig.add_subplot(224);
fig5.set_title('Vote Count', fontsize=16)
plt.boxplot(train['vote_count'])


# [3 Marks]
# # d

# Train a regression model to predict movie revenue. Plot predicted revenue vs. actual revenue on the test set. Quantify the error in your prediction. (You may use sklearn for this step)

# In[18]:


#Train sets
from sklearn.model_selection import train_test_split
y = train['revenue']
del train['revenue']
X = train.values
y=y.values
#test size is 1/3 of data or ~0.33 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 7)


# In[19]:


#Get the shap of enginered data. We can see that we had a lot of redundant and null data
print(X_train.shape, y_train.shape)


# In[20]:


# Regression model here, plot your fit to the revenue data versus the actual data from the test set as a scatter plot.

##### YOUR CODE HERE #######
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
lr = LinearRegression()
lr.fit(X_train, y_train)
prediction = lr.predict(X_test)
print("Accuracy of LinearRegression: ", r2_score(y_test, prediction)*100)
print("MSE of LinearRegression: ", mean_squared_error(y_test, prediction)*100)


# In[21]:


#Plot our model
plt.figure(figsize=(15, 8))
plt.scatter(x=range(0,y_test.size), y=y_test, color = "red", label='actual', s=20)
plt.scatter(x=range(0,prediction.size), y=prediction, color = "green", label='predicted', s=20)
plt.xlabel("Measures")
plt.ylabel("Prediction")
plt.legend()
plt.show()


# A non-linear fit to the data, with and without regularization.

# In[91]:


##### YOUR CODE HERE WITHOUT REGULARIZATION #######
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, LinearRegression
poly_features = PolynomialFeatures(degree=3)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.fit_transform(X_test)
lr = LinearRegression()
lr.fit(X_train_poly,y_train)
prediction = lr.predict(X_test_poly)
print("MSE of two degree Linear Regression: ", mean_squared_error(y_test, prediction))
print("R2 score of three degree Linear Regression: ", mean_squared_error(y_test, prediction))


# In[93]:


plt.figure(figsize=(15, 8))
plt.scatter(x=range(0,y_test.size), y=y_test, color = "orange", label='actual', s=20, alpha=0.5)
plt.scatter(x=range(0,prediction.size), y=prediction, color = "black", label='predicted', s=20, alpha=0.5)
plt.legend()


# In[88]:


##### YOUR CODE HERE WITH REGULARIZATION #######
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
ridge = linear_model.Ridge(alpha = 1e9)
ridge.fit(X_train,y_train)       


# In[97]:


prediction = ridge.predict(X_test)
print("R2 score of Ridge regression with alpha = 1e9 is: ",r2_score(prediction, y_test)*100)
print("MSE of Ridge regression with alpha = 1e9 is: ",mean_squared_error(prediction, y_test))


# In[90]:


plt.figure(figsize=(15, 8))
plt.scatter(x=range(0,y_test.size), y=y_test, color = "purple", label='actual', s=20, alpha=0.5)
plt.scatter(x=range(0,prediction.size), y=prediction, color = "blue", label='predicted', s=20, alpha=0.5)
plt.legend()

