#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import math
from sklearn.metrics import explained_variance_score
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data13 = pd.read_csv('C://Users//ASUSNB//Desktop//THesis//Data//Yillik//YillikFiltreli//kiraci13.csv', sep = ";")
data14 = pd.read_csv('C://Users//ASUSNB//Desktop//THesis//Data//Yillik//YillikFiltreli//kiraci14.csv', sep = ";")
data15 = pd.read_csv('C://Users//ASUSNB//Desktop//THesis//Data//Yillik//YillikFiltreli//kiraci15.csv', sep = ";")
data16 = pd.read_csv('C://Users//ASUSNB//Desktop//THesis//Data//Yillik//YillikFiltreli//kiraci16.csv', sep = ";")
data17 = pd.read_csv('C://Users//ASUSNB//Desktop//THesis//Data//Yillik//YillikFiltreli//kiraci17.csv', sep = ";")
data18 = pd.read_csv('C://Users//ASUSNB//Desktop//THesis//Data//Yillik//YillikFiltreli//kiraci18.csv', sep = ";")


# In[3]:


datas = [data13, data14, data15, data16, data17, data18]
datalar = [i for i in datas]
for i in datas:
    i['RealtyPublishID'].fillna(99999, inplace=True)
    i['RealtyResidenceID'].fillna(99999, inplace=True)
    i['RealtyFloorID'].fillna(99999, inplace=True)
    i['RealtyFuelID'].fillna(99999, inplace=True)
    i['RealtyBuildID'].fillna(99999, inplace=True)
    i['RealtyBuildStateID'].fillna(99999, inplace=True)
    i['RealtyUsageID'].fillna(99999, inplace=True)
    i['RealtyIsStudentOrSingle'].fillna(99999, inplace=True)
    i['RealtyPriceShow'].fillna(99999, inplace=True)
    i['RealtyIsHousingComplex'].fillna(99999, inplace=True)
    i['RealtySubCategoryID'].fillna(99999, inplace=True)
    i['RealtyHeatingID'].fillna(99999, inplace=True)
data13.info()

data_1318 = pd.concat([data13, data14, data15, data16, data17, data18])


# In[4]:


data_1318['DistrictMedian'] = data_1318.groupby('DistrictID').SqmPrice.transform('median')
data_1318['MonthMedian'] = data_1318.groupby('Month').SqmPrice.transform('median')


# In[5]:


data13.info()


# In[6]:


data_1318.info()


# In[7]:


X = data_1318.drop(['CountyID', 'RealtySubCategoryID', 'SqmPrice', 'DistrictID', 'CountyName', 'DistrictName',
                    'Residence_Miss', 'FloorID_Miss', 'Fuel_Miss', 'Build_Miss', 'Age_Miss',
                    'BuildState_Miss', 'Usage_Miss', 'Student_Miss', 'FloorCount_Miss',
                    'PriceTL', 'RealtyEndDateTime', 'RealtyID', 'Tarihx', 'RealtyPriceCurrencyID', 'Month'], axis=1)
Y = data_1318['SqmPrice']


# In[8]:


X.info()


# In[9]:


X.RealtyResidenceID = X.RealtyResidenceID.astype('object')
X.RealtyPublishID = X.RealtyPublishID.astype('object')
X.RealtyFloorID = X.RealtyFloorID.astype('object')
X.RealtyHeatingID = X.RealtyHeatingID.astype('object')
X.RealtyBuildID = X.RealtyBuildID.astype('object')
X.RealtyBuildStateID = X.RealtyBuildStateID.astype('object')
X.RealtyUsageID = X.RealtyUsageID.astype('object')
X.RealtyFuelID = X.RealtyFuelID.astype('object')
X.RealtyIsStudentOrSingle = X.RealtyIsStudentOrSingle.astype('object')
X.RealtyCloseID = X.RealtyCloseID.astype('object')
X.RealtyIsHousingComplex = X.RealtyIsHousingComplex.astype('object')
X.Year = X.Year.astype('object')


# In[10]:


X.info()


# In[11]:


X = pd.concat([X, pd.get_dummies(X.select_dtypes(include='object'))], axis=1)
X = X.drop(['RealtyPublishID', 'RealtyResidenceID', 'RealtyFloorID', 'RealtyHeatingID', 'RealtyBuildID', 'RealtyBuildStateID', 'RealtyUsageID', 'RealtyIsStudentOrSingle', 'RealtyCloseID', 'RealtyPriceShow',
           'RealtyIsHousingComplex','Year', 'RealtyFuelID'], axis=1)
X.info()


# In[12]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# In[13]:


xgb_reg = XGBRegressor(colsample_bytree = 0.9683, learning_rate = 0.01, early_stopping_rounds=20, gamma=3.462,
                       subsample = 0.1391, eta = 0.3134, 
                       max_depth = 15, n_estimators = 825)
xgb_reg.fit(X_train, Y_train)

predict_train = xgb_reg.predict(X_train)
predict_test = xgb_reg.predict(X_test)


mae_xgb_test = mean_absolute_error(predict_test, Y_test)
mse_xgb_test = mean_squared_error(predict_test, Y_test)
rmse_xgb_test = np.sqrt(mse_xgb_test)

print('Mean Absolute Error (MAE_xgb_test): %.2f' % mae_xgb_test)
print('Mean Squared Error (MSE_xgb_test): %.2f' % mse_xgb_test)
print('Root Mean Squared Error (RMSE_xgb_test): %.2f' % rmse_xgb_test)

print('++++')

mae_xgb_train = mean_absolute_error(predict_train, Y_train)
mse_xgb_train = mean_squared_error(predict_train, Y_train)
rmse_xgb_train = np.sqrt(mse_xgb_train)

print('Mean Absolute Error (MAE_xgb_test): %.2f' % mae_xgb_train)
print('Mean Squared Error (MSE_xgb_test): %.2f' % mse_xgb_train)
print('Root Mean Squared Error (RMSE_xgb_test): %.2f' % rmse_xgb_train)

print('++++')

print("MAPE_xgb")
print("Train : ",np.mean(np.abs((Y_train - predict_train) / Y_train)) * 100)
print("Test  : ",np.mean(np.abs((Y_test - predict_test) / Y_test)) * 100)


# In[14]:


plt.figure(figsize=(12, 6))

ranking2 = xgb_reg.feature_importances_
features2 = np.argsort(ranking2)[::-1][:11]
columns2 = X.columns

plt.title("Feature importances based on XGB2", y = 1.03, size = 18)
plt.bar(range(len(features2)), ranking2[features2], color="aqua", align="center")
plt.xticks(range(len(features2)), columns2[features2], rotation=80)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




