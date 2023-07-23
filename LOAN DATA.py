#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the libaries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn import tree
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder


# In[2]:


#read the columns of the data
df=pd.read_excel(r'C:\Users\Humphery\Desktop\digital_loan_data.xlsx')
df.isnull().sum()
df.columns


# In[3]:


# print the size of the data
print(df.shape)


# In[4]:


# Rename the name of the feature
df.columns
df.rename(columns={'Timestamp':'Time','1. Please indicate your age range:':'age','2. What is your gender?':'gend',
            '3. How would you rate your level of education?':'loe' ,   '4. What is your current employment status?':'ces',
            '5. Do you use digital payment platforms such as OPay, Palmpay, Paga etc. for payment transactions?':'dpp' ,
    '6. How likely do you expect that utilising digital payment methods (e.g., mobile payments, online transfers) would be more convenient than using traditional payment methods (e.g., cash, cheques)?':'conv',
    '7. How confident are you that digital payment methods, when compared to traditional payment methods, will give you more control over your financial transactions?':'contft',
       '8. How likely do you believe that adopting digital payment methods will give you with more security and fraud protection than traditional payment methods?':' secu','9. How simple do you think it is to understand and use digital payment methods compared to traditional payment methods?':'unds',
'10. How likely do you believe adopting digital payment methods will necessitate significant adjustments in your current financial routines and habits compared to traditional payment methods? ':'habi','11. How much influence do recommendations and pleasant experiences from friends and family have on your decision to choose digital payment methods compared to traditional payment methods?  ':'frnfa',
        
      '12. How satisfied are you with the availability and reliability of digital payment infrastructure and platforms compared to traditional payment methods?':'infr' ,
      
    '13. How probable do you believe it is that having the appropriate financial and technical support services in place will facilitate your use of digital payment methods rather than traditional payment methods?':'tecsu', '14. Are you willing to switch from traditional payment methods to digital payment platforms in the near future?':'switdp',
    
    
     '15. Do you use digital lending platforms such as OKash, Palmcredit, Branch etc. to secure loan for your financial needs?':' dlp',
                   
                   
    '16. How likely do you believe that using digital lending platforms (e.g., peer-to-peer lending, online loan applications) would offer you faster access to credit than traditional lending institutions (e.g., banks, microfinance institutions)?':'fasac',      

    '17. How likely do you believe that using digital lending platforms (e.g., peer-to-peer lending, online loan applications) would offer you faster access to credit than traditional lending institutions (e.g., banks, microfinance institutions)?' :'fascr', ' 18. How much do you believe digital lending platforms demand less paperwork and documentation than traditional lending institutions?':'ppwrk',
    
    '19. How comfortable and secure are you with sharing personal and financial information through digital platforms for loan applications compared to dealing with traditional lending institutions?':'shapi','20. How much influence do recommendations and favourable experiences from friends and family have on your decision to use digital lending platforms rather than traditional lending institutions? ':'favex',
     '21. How likely are you to use online lending platforms if you believe a sizable portion of the people you know is currently using them instead of traditional lending institutions? ':'sizpp',
      '22. To what extent do you believe the regulatory framework in Nigeria promotes and encourages the usage of online lending platforms in comparison to traditional lending institutions? ':'regfw' ,
                    ' 23.Are you willing to switch from traditional lending methods to digital lending platforms in the near future?':'switdl',
                   
     
                   '24. How accessible are fintech services to you in terms of technology requirements (e.g., smartphone, internet connectivity, cost of data)?':'smicd',
                   
                   '25. Would you consider switching to using more digital financial platform (e.g., cryptocurrencies, robo adviser) for financial transactions instead of traditional financial platform?\n':'mordf'
                   
                  },inplace=True)


# CHECK FOR MISSING VALUE THE LOAN DATA

# In[5]:


df.isnull().sum()


# SIZE OF THE DATA
# 

# In[6]:


print(df.shape)


# Drop misssing value

# In[7]:


# Drop misssing value
df.dropna(inplace=True)


# NEW SHAPE OF THE DATA
# 

# In[8]:


df.shape


# # Inspecting the data for LOAN DATA

# In[9]:


loan_data=df.iloc[:,[1,2,3,4,15,16,17,18,19,20,21,22,23,24,25]]
loan_data


# In[10]:


# check if Loan data as a missing value
loan_data.isnull().sum()
loan_data.columns


# LABEL ENCODER FOR GENDER,level of education,current employment status,(CATIGORICAL VARIABLES)
# USING THE AVERAGE TO REPLACE THE AGE

# In[11]:


loan_data['age']=loan_data['age'].replace({'36-45':41, '26-35':31,'18-25':22,'46-55':51,'36?45':41,'56-65':61, 'Prefer not to say':18,'Option 2':31, '36â€\x9045':41})
encoder=LabelEncoder()
loan_data.gend=encoder.fit_transform(loan_data.gend)
loan_data.loe=encoder.fit_transform(loan_data.loe)
loan_data.ces=encoder.fit_transform(loan_data.ces)
loan_data[' dlp']=encoder.fit_transform(loan_data[' dlp'])
loan_data.switdl=encoder.fit_transform(loan_data.switdl)
loan_data.head()


# In[12]:


loan_data.head()


# In[13]:


loan_data.switdl.value_counts()


# In[14]:


loan_data.describe()


# In[15]:


sns.countplot(x='age', data=loan_data)
plt.title(' Age  Distribution on loan data')
plt.show()
#the count plot chart show that the number of age that was consider the most is 26 to 45 years for thw survey


# In[16]:


#  percentage of the gender that was surveyed  for digital payment
loan_data.gend.value_counts()
x=[145,77,1]
label=['Male','Female','Prefer not to say']
plt.pie(x,labels=label,autopct='%1.1f%%')
plt.title('percentage of gender in digital payment ')
plt.show()   #the chart below shows that the number of males are more compare to the number of female that was survyed


# In[17]:


loan_data['loe'].value_counts()
x=[135,75,9,4]
label=["Bachelor's degree","Master's degree",'Secondary school','PhD or equivalent']
plt.pie(x,labels=label,autopct='%1.0f%%')
plt.title('the percentage of Level of Education in digital payment')
plt.show()   #the chart below shows that the number of  "Bachelor's degree holders are more compare to the number of level of education that was survyed


# In[18]:


loan_data['fasac'].value_counts()
x=[61,65,29,46,22]
label=["Strongly Agree",'Agree','Strongly Disagree','Neutral','Disagree']
plt.pie(x,labels=label,autopct='%1.0f%%')
plt.title('percentage of people who believe  that  using digital lending platforms (e.g., peer-to-peer lending, online loan applications) would offer you faster access to credit than traditional lending institutions')
plt.show()


# In[19]:


loan_data[' dlp'].value_counts()
x=[166,57]
label=['people that do not use digital lending','people that use digital lending']
plt.pie(x,labels=label,autopct='%1.0f%%')
plt.title('percentage of people that use either digital lending or do not use digital lending')
plt.show()   #the chart below shows that the number of males are more compare to the number of female that was survyed


# CORRELATION OF PEOPLE WHO ARE WILLING TO SWITCH FROM TRADITIONAL LENDING TO DIGITAL LENDING

# In[20]:


#loan_data.corrwith(loan_data['switdl']).sort_values(ascending=False)


# In[21]:


loan_data['switdl'].value_counts()
x=[70,153]
Label=['people that are not willing to switch from trad to digital lending','people that willing to switch from trad to digital lending']
plt.pie(x,labels=label,autopct='%1.0f%%')
plt.title('percentage of people that use either digital lending or do not use digital lending')
plt.show()   #the chart below shows that the number of males are more compare to the number of female that was survyed


# In[22]:


# Features Selections on digital payment platforms such as OPay, Palmpay, Paga etc. for payment transactions.
x_loan_payment = loan_data.drop(['switdl'], axis=1)
y_loan_payment = loan_data['switdl']
loan_pay_selection=SelectKBest(score_func=chi2,k=3).fit(x_loan_payment,y_loan_payment)
p_value=loan_pay_selection.pvalues_
p_value
#Important featurs are:contft,secu, frnfa, infr,tecsu,smicd


# In[23]:


names=loan_pay_selection.feature_names_in_
names
for p , name in zip(p_value,names):
    print(name,p)


# In[24]:


loan_data.columns


# MODEL BUILDING UPON THE TRAINING DATA

# In[25]:


x_lend=loan_data[[ ' dlp', 'fasac', 'fascr', 'ppwrk', 'shapi',
       'favex', 'sizpp', 'regfw', 'smicd', 'mordf']]
y_lend=loan_data['switdl']
y_lend.unique()


# In[26]:


y_lend.value_counts()


# In[27]:


for i in x_lend:
    plt.figure(figsize=(3,4))
    sns.kdeplot(x_lend[i],shade=True)
#THE density graph below shows that lot of people are willing to switch to digital payment platfrom


# In[28]:


loan_data['cat_switdl']=pd.cut(loan_data['switdl'],bins=2,labels=[1,2])
loan_data['cat_switdl'].hist() # we have more people who prefer using digital payment than tranditional


# In[29]:


#This create digital payment category in ther to cover all the data according to the original
sss=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in sss.split(loan_data,loan_data['cat_switdl']):
    strat_train=loan_data.iloc[train_index]
    strat_test=loan_data.iloc[test_index]
    


# In[30]:


strat_train['switdl'].value_counts()/len(strat_train)*100 # this clearly stratified equallly with the original data
#i.e it captures all the targets variables.


# In[31]:


x_trains=strat_train.drop(columns='switdl')
y_trains=strat_train['switdl']


# In[32]:


y_trains.value_counts()/len(y_trains)


# SELECT TRAINING MODELS

# In[33]:


model1=LogisticRegression()
model1.fit(x_trains,y_trains)
predictions1=model1.predict(x_trains)


# In[34]:


model2=RandomForestClassifier()
model2.fit(x_trains,y_trains)
predictions2=model2.predict(x_trains)


# In[35]:


model3=DecisionTreeClassifier(max_depth=2)
model3.fit(x_trains,y_trains)
predictions3=model3.predict(x_trains)


# In[36]:


#CHECKING THE PERFORMANCE OF THE MODELS
cv1=cross_val_score(model1,x_trains,y_trains,cv=3,scoring='accuracy')
print('the accuracy of the logistic model is:',cv1.mean())


# In[37]:


cv2=cross_val_score(model2,x_trains,y_trains,cv=3,scoring='accuracy')
print('the accuracy of the randomforest model is:',cv2.mean())


# In[38]:


#CHECKING THE PERFORMANCE OF THE MODELS
cv3=cross_val_score(model3,x_trains,y_trains,cv=5,scoring='accuracy')
print('the accuracy of the Decision model is:',cv3.mean())


# In[39]:


# check the confusion matrix
conf1=confusion_matrix(y_trains,predictions1)
print(conf1)


# In[40]:


# check the confusion matrix
conf2=confusion_matrix(y_trains,predictions2)
print(conf2)


# In[41]:


#check the confusion matrix
conf3=confusion_matrix(y_trains,predictions3)
print(conf3)


# In[42]:


print(classification_report(y_trains,predictions1))


# In[43]:


print(classification_report(y_trains,predictions2))


# In[44]:


print(classification_report(y_trains,predictions3))


# Classification report shows that since the data is uneven , the the F1-score is a good way to measure how
# our model was able to predict well. 
# this clearly show that Randomforest classifer, Decision tree classifyer and the logistic regression model are a good model for prediction.
# with 1.00 confident that the prediction was correctly done with help of this 2 models.

# Using Random sample  crossvalidation (train_test _split)
# 

# In[45]:


x_train,x_test,y_train,y_test=train_test_split(x_lend,y_lend,test_size=0.2,random_state=42)


# In[46]:


model1=LogisticRegression()
model1.fit(x_train,y_train)
prediction1=model1.predict(x_train)


# In[47]:


model2=RandomForestClassifier()
model2.fit(x_train,y_train)
prediction2=model2.predict(x_train)


# In[48]:


model3=DecisionTreeClassifier(max_depth=4)
model3.fit(x_train,y_train)
prediction3=model3.predict(x_train)


# In[49]:


#CHECKING THE PERFORMANCE OF THE MODELS
cv1=cross_val_score(model1,x_train,y_train,cv=5,scoring='accuracy')
print('the accuracy of the logistic model is:',cv1.mean())


# In[50]:


#CHECKING THE PERFORMANCE OF THE MODELS
cv2=cross_val_score(model2,x_train,y_train,cv=5,scoring='accuracy')
print('the accuracy of the Randomforest model is:',cv2.mean())


# In[51]:


#CHECKING THE PERFORMANCE OF THE MODELS
cv3=cross_val_score(model3,x_train,y_train,cv=5,scoring='accuracy')
print('the accuracy of the Randomforest model is:',cv3.mean())


# In[52]:


print(classification_report(y_train,prediction1))


# In[53]:


print(classification_report(y_train,prediction2))


# In[54]:


print(classification_report(y_train,prediction3))


# TESTING THE MODELS
# 

# In[55]:


model2=RandomForestClassifier()
model2.fit(x_test,y_test)
prediction_test_2=model2.predict(x_test)


# In[56]:


# Create a FactorAnalysis object
fa = FactorAnalysis(n_components=3,rotation='varimax')

# Fit the FactorAnalysis model to the data
transformed_features = fa.fit_transform(loan_data)
transformed_features


# FACTOR 3 SHOWS A GOOD ANALYSIS OF THE FEATURES TO BE USED 
# BECAUSE MOST OF THE  NUMBERS ARE CLOSE TO -1 TO +1

# In[57]:


# Access the factor loadings
factor_loadings = fa.components_

# Print the transformed features
#print(transformed_features)

# Print the factor loadings
print(factor_loadings*100)

# 


# In[58]:


loan_data.columns


# In[59]:


new_x=loan_data[['fasac', 'fascr', 'ppwrk', 'shapi',
       'favex', 'sizpp', 'regfw', 'smicd', 'mordf', 'cat_switdl']]
new_x
new_y=loan_data['switdl']


# In[60]:


#Using Random sample  crossvalidation (train_test _split)
x_train_new,x_test_new,y_train_new,y_test_new=train_test_split(new_x,new_y,test_size=0.2,random_state=42)


# In[61]:


model_new_1=LogisticRegression()
model_new_1.fit(x_train_new,y_train_new)
prediction_new_1=model_new_1.predict(x_train_new)


# In[62]:


model_new_2=RandomForestClassifier()
model_new_2.fit(x_train_new,y_train_new)
prediction_new_2=model_new_2.predict(x_train_new)


# In[63]:


# check the confusion matrix
conf1=confusion_matrix(y_train_new,prediction_new_1)
print(conf1)


# In[64]:


# check the confusion matrix
conf2=confusion_matrix(y_train_new,prediction_new_2)
print(conf2)


# In[65]:


print(classification_report(y_train_new,prediction_new_2)) 


# In[66]:


x_train.columns


# VISUALIZING THE DECISION TREE MODEL UPON THE TRAINING DATA USING RANDOM TRAIN-TEST SPLIT

# In[67]:


fig , ax= plt.subplots(figsize=(20,20))
tree.plot_tree(model3,fontsize=10)
plt.show()


# EXAMINE THE DATA (IS ITS AN IMBALANCE DATA SET)
# SINCE WE HAVE LESS DATA, WE ARE GOING TO USE OVERSAMPLING

# In[68]:


x_train,x_test,y_train,y_test=train_test_split(new_x,new_y,test_size=0.3,random_state=0)
print('the shape of the x_training:',x_train.shape)
print(y_train.shape)
print('the shape of the x_test:',x_test.shape)


# the training data

# In[69]:


print(y_train.value_counts())


# SHOW THE COUNTS OF THE UNBALANCED DATA

# In[70]:


new_y.value_counts()


# THE BARCHART OF THE UNBALANCED DATA

# In[71]:


new_y.value_counts().plot(kind='bar',color='g')
plt.xlabel('the classes for the people who like to lend money  from Traditional to Digital')
plt.show()


# the test data value

# In[72]:


print(y_test.value_counts())


# OVERSAMPLING THE DATA using SMOTE Technique

# In[73]:


smote=SMOTE(random_state=1)
x_train_balanced,y_train_balanced=smote.fit_resample(x_train,y_train)


# Showing the Training shape  of the Oversampling

# In[74]:


print('the shape of the x_training:',x_train_balanced.shape)


# In[75]:


print(y_train_balanced.value_counts())


# THE BARCHART OF THE BALANCED DATA 

# In[76]:


y_train_balanced.value_counts().plot(kind='bar')


# TRAINGING THE BALANCED DATA

# In[77]:


model1=RandomForestClassifier()
model1.fit(x_train_balanced,y_train_balanced)
prediction1=model2.predict(x_train_balanced)


# In[78]:


model2=RandomForestClassifier()
model2.fit(x_train_balanced,y_train_balanced)
prediction2=model2.predict(x_train_balanced)


# In[79]:


model3=DecisionTreeClassifier(max_depth=2)
model3.fit(x_train_balanced,y_train_balanced)
prediction3=model3.predict(x_train_balanced)


# CHECKING THE PERFORMANCE OF THE MODELS UPON THE BALANCED DATA

# In[80]:


#CHECKING THE PERFORMANCE OF THE MODELS
cv1=cross_val_score(model1,x_train_balanced,y_train_balanced,cv=3,scoring='accuracy')
print('the accuracy of the logistic model is:',cv1.mean())


# In[81]:


#CHECKING THE PERFORMANCE OF THE MODELS
cv2=cross_val_score(model2,x_train_balanced,y_train_balanced,cv=3,scoring='accuracy')
print('the accuracy of the Randomforest model is:',cv2.mean())


# In[82]:


#CHECKING THE PERFORMANCE OF THE MODELS
cv3=cross_val_score(model3,x_train_balanced,y_train_balanced,cv=3,scoring='accuracy')
print('the accuracy of the Decision model is:',cv3.mean())


# check the confusion matrix

# In[83]:


conf1=confusion_matrix(y_train_balanced,prediction1)
print(conf1)


# In[84]:


conf2=confusion_matrix(y_train_balanced,prediction2)
print(conf2)


# In[85]:


conf3=confusion_matrix(y_train_balanced,prediction3)
print(conf3)


# TESTING THE MODELS FOR THE RANDOMFOREST(BEST MODEL)

# In[86]:


model2=RandomForestClassifier()
model2.fit(x_test,y_test)
prediction_test_2=model2.predict(x_test)


# In[87]:


# check the confusion matrix
conf_test_2=confusion_matrix(y_test,prediction_test_2)
print(conf_test_2)


# In[88]:


print(classification_report(y_test,prediction_test_2))


# #CHECKING THE PERFORMANCE OF THE RANDOMFOREST  MODELS ON TEST THE DATA

# In[89]:



cv2=cross_val_score(model2,x_test,y_test,cv=5,scoring='accuracy')
print('the accuracy of the Randomforest model is:',cv2.mean())


# IT'S CLEARLY SHOWN THAT RANDOM FOREST PERFORM BETTER WHEN COMPARED TO OTHER MODELS
# AS THE F1-SCORE RESULT TO 1
