#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[2]:


import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTEENN


# #### Reading csv

# In[3]:


df=pd.read_csv(r"C:\Users\Admin\Desktop\Python\Churn model\prediction_Expense\cust_churn.csv")
df.head()


# In[4]:


df=df.drop('Unnamed: 0',axis=1)


# In[5]:


X=df.drop('Churn',axis=1)
X


# In[6]:


y=df['Churn']
y


# ##### Train Test Split

# In[7]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# #### Decision Tree Classifier

# In[8]:


model_dt=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)


# In[9]:


model_dt.fit(X_train,y_train)


# In[10]:


y_pred=model_dt.predict(X_test)
y_pred


# In[11]:


model_dt.score(X_test,y_test)


# In[12]:


print(classification_report(y_test, y_pred, labels=[0,1]))


# ###### As you can see that the accuracy is quite low, and as it's an imbalanced dataset, we shouldn't consider Accuracy as our metrics to measure the model, as Accuracy is cursed in imbalanced datasets.
# 
# ###### Hence, we need to check recall, precision & f1 score for the minority class, and it's quite evident that the precision, recall & f1 score is too low for Class 1, i.e. churned customers.
# 
# ###### Hence, moving ahead to call SMOTEENN (UpSampling + ENN)

# In[16]:





# In[18]:


from imblearn.over_sampling import SMOTE

oversample = SMOTE()
X1, y1 = oversample.fit_resample(X, y)


#sm = SMOTEENN()
#X_resampled, y_resampled = sm.fit_sample(X,y)


# In[20]:


xr_train,xr_test,yr_train,yr_test=train_test_split(X1, y1,test_size=0.2)


# In[21]:


model_dt_smote=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)


# In[22]:


model_dt_smote.fit(xr_train,yr_train)
yr_predict = model_dt_smote.predict(xr_test)
model_score_r = model_dt_smote.score(xr_test, yr_test)
print(model_score_r)
print(metrics.classification_report(yr_test, yr_predict))


# In[23]:


print(metrics.confusion_matrix(yr_test, yr_predict))


# ###### Now we can see quite better results, i.e. Accuracy: 92 %, and a very good recall, precision & f1 score for minority class.
# 
# ###### Let's try with some other classifier.

# #### Random Forest Classifier

# In[24]:


from sklearn.ensemble import RandomForestClassifier


# In[25]:


model_rf=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)


# In[27]:


model_rf.fit(X_train,y_train)


# In[28]:


y_pred=model_rf.predict(X_test)


# In[29]:


model_rf.score(X_test,y_test)


# In[30]:


print(classification_report(y_test, y_pred, labels=[0,1]))


# In[ ]:





# In[32]:


from imblearn.over_sampling import SMOTE

oversample = SMOTE()
X2, y2 = oversample.fit_resample(X, y)


# In[23]:




# In[33]:


xr_train1,xr_test1,yr_train1,yr_test1=train_test_split(X2, y2,test_size=0.2)


# In[34]:


model_rf_smote=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)


# In[35]:


model_rf_smote.fit(xr_train1,yr_train1)


# In[36]:


yr_predict1 = model_rf_smote.predict(xr_test1)


# In[37]:


model_score_r1 = model_rf_smote.score(xr_test1, yr_test1)


# In[38]:


print(model_score_r1)
print(metrics.classification_report(yr_test1, yr_predict1))


# In[40]:


print(metrics.confusion_matrix(yr_test1, yr_predict1))


# ###### With RF Classifier, also we are able to get quite good results, infact better than Decision Tree.
# 
# ###### We can now further go ahead and create multiple classifiers to see how the model performance is, but that's not covered here, so you can do it by yourself :)

# #### Performing PCA

# In[41]:


# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(0.9)
xr_train_pca = pca.fit_transform(xr_train1)
xr_test_pca = pca.transform(xr_test1)
explained_variance = pca.explained_variance_ratio_


# In[42]:


model=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)


# In[43]:


model.fit(xr_train_pca,yr_train1)


# In[44]:


yr_predict_pca = model.predict(xr_test_pca)


# In[45]:


model_score_r_pca = model.score(xr_test_pca, yr_test1)


# In[46]:


print(model_score_r_pca)
print(metrics.classification_report(yr_test1, yr_predict_pca))


# ##### With PCA, we couldn't see any better results, hence let's finalise the model which was created by RF Classifier, and save the model so that we can use it in a later stage :)

# In[ ]:





# #### Pickling the model

# In[47]:


import pickle


# In[48]:


filename = 'model.sav'


# In[49]:


pickle.dump(model_rf_smote, open(filename, 'wb'))


# In[50]:


load_model = pickle.load(open(filename, 'rb'))


# In[51]:


model_score_r1 = load_model.score(xr_test1, yr_test1)


# In[52]:


model_score_r1


# ##### Our final model i.e. RF Classifier with SMOTEENN, is now ready and dumped in model.sav, which we will use and prepare API's so that we can access our model from UI.

# In[ ]:




