#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[13]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[14]:


data=pd.read_excel("ANZ synthesised transaction dataset.xlsx")


# In[15]:


data.columns


# In[16]:


features = ['status', 'card_present_flag', 'bpay_biller_code', 'account',
       'currency', 'long_lat', 'txn_description', 'merchant_id',
       'merchant_code', 'first_name', 'balance', 'date', 'gender', 'age',
       'merchant_suburb', 'merchant_state', 'extraction', 'amount',
       'transaction_id', 'country', 'customer_id', 'merchant_long_lat',
       'movement']

mask = np.zeros_like(data[features].corr(), dtype = np.bool) 
mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation Matrix',fontsize=25)

sns.heatmap(data[features].corr(),linewidths = 0.25,vmax = 0.7,square = True,cmap = "BuGn", 
            linecolor = 'w',annot = True,annot_kws = {"size":8},mask = mask,cbar_kws = {"shrink": 0.9})


# In[59]:


new_data=data[data.txn_description=="PAY/SALARY"]


# In[60]:


new_data.head()


# In[61]:


new_data.drop(['status', 'bpay_biller_code', 'account','currency', 'long_lat', 'txn_description', 'merchant_id','merchant_code', 'first_name','merchant_suburb', 'merchant_state', 'extraction','transaction_id', 'country', 'merchant_long_lat','movement'],inplace=True,axis=1)


# In[62]:


new_data.head()


# In[63]:


new_data.isna().sum()


# In[64]:


new_data[['customer_id','amount']].groupby(["customer_id"]).mean().sort_values(by='amount').plot.bar(color='blue')
plt.show()


# In[76]:


new_data.date


# ## As the data is of three months lets divide the data in months and take one month to predict salary rather than 3

# In[86]:


from datetime import datetime
month = []
for row in new_data['date']:
    try:
        month.append(row.strftime("%m"))
    except:
        month.append(np.NaN)
new_data['month'] = month


# In[87]:


new_data.head(-10)


# In[92]:


octdata=new_data[new_data.month=="08"]
novdata=new_data[new_data.month=="09"]
septdata=new_data[new_data.month=="09"]


# In[96]:


octdata.head()


# In[104]:


octdata=octdata.groupby("customer_id").mean()
octdata.head()


# In[125]:


salaries = []

for i in octdata["amount"]:
    try:
        salaries.append(i*12)
    except:
        salaries.append(NaN)
octdata["annual_salary"] = salaries


# In[145]:


octdata.drop(["card_present_flag"],inplace=True,axis=1)


# In[149]:


octdata.isna().sum()


# In[156]:


annulsal = []

for customer_id in new_data["customer_id"]:
    try:
        annulsal.append(int(octdata.loc[customer_id]["annual_salary"]))
    except:
        annulsal.append(0)
new_data["annual_salary"] = annulsal


# In[165]:


print(new_data.head())
new_data.drop(["date"],axis=1,inplace=True)


# In[171]:


new_data.head()


# Annual_salary is same from october and november data

# In[166]:


features = ['card_present_flag', 'balance', 'age', 'amount', 'annual_salary','gender']

mask = np.zeros_like(new_data[features].corr(), dtype = np.bool) 
mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation Matrix',fontsize=25)

sns.heatmap(new_data[features].corr(),linewidths = 0.25,vmax = 0.7,square = True,cmap = "BuGn", 
            linecolor = 'w',annot = True,annot_kws = {"size":8},mask = mask,cbar_kws = {"shrink": 0.9})


# # Simple regression

# In[190]:


X = new_data.iloc[:, 1:-3].values
y = new_data.iloc[:, 7].values



from sklearn.preprocessing import LabelEncoder
Labelx=LabelEncoder()
X[:,1]=Labelx.fit_transform(X[:,1])
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4 , random_state = 0)


# In[191]:


new_data.dtypes


# In[192]:


from sklearn.linear_model import LinearRegression
rg=LinearRegression()
rg.fit(X_train,y_train)


# In[200]:


#predicting
y_pred = rg.predict(X_test)


# In[ ]:


#predicting
y_pred = rg.predict(X_test)
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,rg.predict(X_train),color='blue')
plt.title('Salary VS Experience(Training set)')
plt.xlabel('Year of ecperience')
plt.ylabel('Salary')
plt.show()

