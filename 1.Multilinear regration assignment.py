# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 17:30:50 2023

@author: kailas
"""

1]PROBLEM

BUSINESS OBJECTIVE==We are going to predict PROFIT using different attributes..



#Import Liabrary
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pylab as plt
import scipy
from scipy import stats
import pylab

#Dataset
data=pd.read_csv("D:/data science assignment/Assignments/5.Multinomial Regression/50_Startups (1).csv")

#EDA
data.info()
data.describe()
data.drop(['State'],axis=1,inplace=True)
data=data.rename(columns={'R&D Spend':'rdspend','Administration':'admin','Marketing Spend':'marketspend','Profit':'profit'})

#Graphical Visulization(Univariate)

#rdspend
plt.hist(data.rdspend)
plt.boxplot(data.rdspend)
plt.bar(height=data.rdspend,x=np.arange(1,51,1))

#admin
plt.hist(data.admin)
plt.boxplot(data.admin)
plt.bar(height=data.admin,x=np.arange(1,51,1))

#marketspend
plt.hist(data.marketspend)
plt.boxplot(data.marketspend)
plt.bar(height=data.marketspend,x=np.arange(1,51,1))

#profit
plt.hist(data.profit)
plt.boxplot(data.profit)
plt.bar(height=data.profit,x=np.arange(1,51,1))

#There are Outliers in profit.so we replace it by logical value using Winsorizer function 
from feature_engine.outliers import Winsorizer
w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['profit'])
data['profit']=w.fit_transform(data[['profit']])

#Scatter plot along with Histogram
sns.pairplot(data)

#Correlation Matrix
data.corr()
#After Analyzing,there is colineratiy problem exists between input variables such as (marketspend and rdspend)

#Preparing the model
import statsmodels.formula.api as smf
model=smf.ols('profit ~ rdspend+admin+marketspend',data=data).fit()
model.summary()
#P values of admin,marketspend are more than 0.05

#Checking whether any influential values
#Influential Index plot
import statsmodels.api as sm
sm.graphics.influence_plot(model)

#Index 49,48,46,45 have high influence value,so we are going to exclude entire row.
data1=data.drop(data.index[[49,48,46,45]])

#again prepare the model
model=smf.ols('profit ~ rdspend+admin+marketspend',data=data1).fit()
model.summary()

#Checking Colinearity to decide whether which variables we are going yo remove using VIF.
#Assumption::- VIF>10=Colinearity
#Checking Colinearity for individual variables

rdspend_c=smf.ols('rdspend ~ admin+marketspend+profit',data=data1).fit().rsquared
rdsepnd_value=1/(1-rdspend_c)

admin_c=smf.ols('admin ~ rdspend+marketspend+profit',data=data1).fit().rsquared
admin_value=1/(1-admin_c)

marketspend_c=smf.ols('marketspend ~ admin+rdspend+profit',data=data1).fit().rsquared
marketsepnd_value=1/(1-marketspend_c)

#using admin.. 
#R*2 value get reduced,so we exclude it..

fianl_model=smf.ols('(profit) ~ (marketspend+rdspend)',data=data1).fit()
fianl_model.summary()

pred=fianl_model.predict(data1)

#Probplot for normality
stats.probplot(pred,plot=pylab)

#Q-Qplot for normality
res=fianl_model.resid
sm.qqplot(res)

#Fitted vs Residual Plot
sns.residplot(x=pred,y=data1.profit,lowess=True)
plt.xlabel('fitted')
plt.ylabel('residual')
plt.title('fitted vs residual')
plt.show()

#RMSE
error=data1.profit-pred
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse

#LOG Transformations
fianl_model=smf.ols('(profit) ~ np.log(marketspend+rdspend)',data=data1).fit()
fianl_model.summary()

#expontial Transformations
fianl_model=smf.ols('np.log(profit) ~ (marketspend+rdspend)',data=data1).fit()
fianl_model.summary()

#Square Transformations
fianl_model=smf.ols('(profit) ~ (marketspend+rdspend)*(marketspend+rdspend)',data=data1).fit()
fianl_model.summary()

#We tune the model using different transformations..but no one gives best result..so we use '(profit) ~ (marketspend+rdspend)' .these transformastion as Final...



from sklearn.model_selection import train_test_split
train,test=train_test_split(data1,test_size=0.2)

model_train=smf.ols('profit ~ marketspend+rdspend',data=train).fit()
model_train.summary()
#FOR TRAIN DATA
train_pred=model_train.predict(train)

# train residual values 
train_resid  = train_pred - train.profit
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

#FOR TEST DATA
test_pred=model_train.predict(test)
#test residual values
test_resid = test_pred - test.profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse



2]PROBLEM

BUSINESS OBJECTIVE:-Predicting the Price of model using different attributes.



# dataset
data=pd.read_csv("D:/data science assignment/Assignments/5.Multinomial Regression/ToyotaCorolla.csv",encoding=('ISO-8859-1'))

#As per given in the problem,we are excluding some columns which are not relevant.
data=data[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]

#EDA
data.info()
data.describe()
data=data.rename(columns={'Price':'price','Age_08_04':'age','Quarterly_Tax':'tax','Weight':'weight','Doors':'doors','Gears':'gear'})

#Graphical Representation(Univarate Data)
plt.boxplot(data.price)
plt.boxplot(data.age)
plt.boxplot(data.KM)
plt.boxplot(data.HP)
plt.boxplot(data.cc)
plt.boxplot(data.doors)
plt.boxplot(data.gear)
plt.boxplot(data.tax)
plt.boxplot(data.weight)
#all columns have outliers,so we remove it by using Winsorizer
from feature_engine.outliers import Winsorizer
w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['price'])
data['price']=w.fit_transform(data[['price']])
w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['age'])
data['age']=w.fit_transform(data[['age']])
w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['KM'])
data['KM']=w.fit_transform(data[['KM']])
w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['HP'])
data['HP']=w.fit_transform(data[['HP']])
w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['cc'])
data['cc']=w.fit_transform(data[['cc']])
w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['doors'])
data['doors']=w.fit_transform(data[['doors']])
w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['tax'])
data['tax']=w.fit_transform(data[['tax']])
w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['weight'])
data['weight']=w.fit_transform(data[['weight']])

#Histogram
plt.hist(data.price)
plt.hist(data.age)
plt.hist(data.KM)
plt.hist(data.HP)
plt.hist(data.cc)
plt.hist(data.doors)
plt.hist(data.gear)
plt.hist(data.tax)
plt.hist(data.weight)

#Scatter Diagram(Bivariate Data)
sns.pairplot(data)
#Corelation Matrix
corr=data.corr()

#Regression Model
import statsmodels.formula.api as smf
model=smf.ols('price ~ age+KM+HP+cc+doors+gear+tax+weight',data=data).fit()
model.summary()

pred=model.predict(data)

#LOG Transformations
model=smf.ols('price ~ np.log(age+KM+HP+cc+doors+gear+tax+weight)',data=data).fit()
model.summary()
#EXPONTIAL Transformations
model=smf.ols('np.log(price) ~ age+KM+HP+cc+doors+gear+tax+weight',data=data).fit()
model.summary()
#SQUARE Transformations
model=smf.ols('price ~ age+KM+HP+cc+doors+gear+tax+weight * age+KM+HP+cc+doors+gear+tax+weight',data=data).fit()
model.summary()
#SQUARE ROOT Transformations
model=smf.ols('(price) ~ np.sqrt(age+KM+HP+cc+doors+gear+tax+weight)',data=data).fit()
model.summary()

#Finally we choose SQUARE Transformations

f_model=smf.ols('price ~ age+KM+HP+cc+doors+gear+tax+weight * age+KM+HP+cc+doors+gear+tax+weight',data=data).fit()
f_model.summary()

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm
sm.graphics.influence_plot(f_model)

#some Influential values(rows) we are going to drop

data1=data.drop(data.index[[960,523,1109,1073,696]])

#Develop Final model

f_model=smf.ols('price ~ age+KM+HP+cc+doors+gear+tax+weight * age+KM+HP+cc+doors+gear+tax+weight',data=data1).fit()
f_model.summary()

predictt=f_model.predict(data1)

#Q-Q Plot
res=f_model.resid
stats.probplot(res,plot=pylab)

# Residuals vs Fitted plot
sns.residplot(x = predictt, y = data1.price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
train,test = train_test_split(data1, test_size = 0.2) # 20% test data

trainmodel=f_model=smf.ols('price ~ age+KM+HP+cc+doors+gear+tax+weight * age+KM+HP+cc+doors+gear+tax+weight',data=data1).fit()
trainmodel.summary()

test_pred = trainmodel.predict(test)

# test residual values 
test_resid = test_pred - test.price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = trainmodel.predict(train)

# train residual values 
train_resid  = train_pred - train.price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
