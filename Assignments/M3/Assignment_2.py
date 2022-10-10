#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set_style('darkgrid')
sns.set(font_scale=1.3)


# In[25]:


df=pd.read_csv("/content/drive/MyDrive/IBM/Assignment - 2 /Churn_Modelling.csv")


# In[26]:


df.head()


# In[29]:


df.drop(["RowNumber","CustomerId","Surname"],axis=1,inplace=True)


# In[30]:


df.info()


# In[28]:


#Perform Univariate Analysis
plt.figure(figsize=(8,8))
sns.countplot(x='Tenure',data=df)
plt.xlabel('0:Customers with Bank, 1: exited from bank')
plt.ylabel('No.of.Customers')
plt.title("Bank Customers viz")
plt.show()


# In[9]:


#Perform Univariate Analysis
plt.figure(figsize=(8,8))
sns.kdeplot(x=df['Balance'])


# In[10]:


#Perform Bivariate Analysis 
plt.scatter(df.Age,df.Balance)


# In[54]:


#Perform Bivariate Analysis
df.corr()


# In[36]:


#Perform Bivariate Analysis
import statsmodels.api as sm

#define response variable
y = df['CreditScore']

#define explanatory variable
x = df[['EstimatedSalary']]

#add constant to predictor variables
x = sm.add_constant(x)

#fit linear regression model
model = sm.OLS(y, x).fit()

#view model summary
print(model.summary())


# In[35]:


#Perform Multivariate Analysis
plt.figure(figsize=(4,4))
sns.pairplot(data=df[["Balance","CreditScore","EstimatedSalary","NumOfProducts","Tenure","Exited"]],hue="Exited")


# In[40]:


#Perform Descriptive Statistics
df=pd.DataFrame(df)
print(df.sum())


# In[39]:


#Perform Descriptive Statistics
print("----Sum Value-----")
print(df.sum(1))
print("----------------------------------")
print("-----Product Value-----")
print(df.prod())
print("----------------------------------")


# In[38]:


#Perform Descriptive Statistics
print("----------Mean Value-----------")
print(df.mean())
print("-------------------------------")
print("----------Median Value---------")
print(df.median())
print("-------------------------------")
print("----------Mode Value------------")
print(df.mode())
print("-------------------------------")


# In[41]:


#Handling with missing Values
df.isnull()#Checking values are null


# In[42]:


#Handling with missing Values
df.notnull()#Checking values are not null


# In[43]:


#Find outliers & replace the outliers
sns.boxplot(df['Balance'])


# In[44]:


#Find outliers & replace the outliers
print(np.where(df['Balance']>100000))


# In[45]:


#Find outliers & replace the outliers
from scipy import stats
import numpy as np
 
z = np.abs(stats.zscore(df["EstimatedSalary"]))
print(z)


# In[48]:


#Check for categorical columns & performs encoding
from sklearn.preprocessing import LabelEncoder
df['Gender'].unique()


# In[49]:


#Check for categorical columns & performs encoding
df['Gender'].value_counts()


# In[57]:


#Check for categorical columns & performs encoding
encoding=LabelEncoder()
df["Gender"]=encoding.fit_transform(df.iloc[:,1].values)
df


# In[ ]:


#Check for categorical columns & performs encoding


# In[ ]:


#Split the data into Dependent & Independent Variables
print("----------Dependent Variables----------")
X=df.iloc[:,1:4]
print(X)
print("---------------------------------------")
print("---------Independent Variables---------")
Y=df.iloc[:,4]
print(Y)
print("---------------------------------------")


# In[ ]:


#Scale the independent Variables
from sklearn.preprocessing import StandardScaler
object= StandardScaler()
# standardization 
scale = object.fit_transform(df) 
print(scale)


# In[ ]:


#Split the data into training & testing
from sklearn.model_selection import train_test_split


# In[ ]:


#Split the data into training & testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=4,random_state=4)
x_train


# In[ ]:


#Split the data into training & testing
x_test


# In[ ]:


#Split the data into training & testing
y_train


# In[ ]:


#Split the data into training & testing
y_test

