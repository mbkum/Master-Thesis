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
data13.info()


# In[3]:


datas = [data13, data14, data15, data16, data17, data18]
datalar = [i for i in datas]
for i in datas:
    i['RealtyResidenceID'].fillna(i.RealtyResidenceID.mode()[0], inplace = True)
    i['RealtyPublishID'].fillna(i.RealtyPublishID.mode()[0], inplace = True)
    i['RealtyRoom'].fillna(i.RealtyRoom.mode()[0], inplace = True)
    i['RealtyFloorCount'].fillna(i.RealtyFloorCount.mode()[0], inplace = True)
    i['RealtyFloorID'].fillna(i.RealtyFloorID.mode()[0], inplace = True)
    i['RealtyAge'].fillna(i.RealtyAge.mode()[0], inplace = True)
    i['RealtyFuelID'].fillna(i.RealtyFuelID.mode()[0], inplace = True)
    i['RealtyBuildID'].fillna(i.RealtyBuildID.mode()[0], inplace = True)
    i['RealtyBuildStateID'].fillna(i.RealtyBuildStateID.mode()[0], inplace = True)
    i['RealtyUsageID'].fillna(i.RealtyUsageID.mode()[0], inplace = True)
    i['RealtyIsStudentOrSingle'].fillna(i.RealtyIsStudentOrSingle.mode()[0], inplace = True)
    i['RealtyPriceShow'].fillna(i.RealtyPriceShow.mode()[0], inplace = True)
    i['RealtyIsHousingComplex'].fillna(i.RealtyIsHousingComplex.mode()[0], inplace = True)
    i['RealtySubCategoryID'].fillna(i.RealtySubCategoryID.mode()[0], inplace = True)
    i['RealtyLivingRoom'].fillna(i.RealtyLivingRoom.mode()[0], inplace = True)
    i['RealtyBathroom'].fillna(i.RealtyBathroom.mode()[0], inplace = True)
    i['RealtySqm'].fillna(i.RealtySqm.mode()[0], inplace = True)
    i['RealtyHeatingID'].fillna(i.RealtyHeatingID.mode()[0], inplace=True)
data13.info()


# In[4]:


datas = [data13, data14, data15, data16, data17, data18]
datalar = [i for i in datas]
for i in datas:
    i.RealtyEndDateTime = i.RealtyEndDateTime.astype('object')
    i.PriceTL = i.PriceTL.astype('float64')
    i.RealtySubCategoryID = i.RealtySubCategoryID.astype('object')
    i.RealtyPublishID = i.RealtyPublishID.astype('object')
    i.RealtyResidenceID = i.RealtyResidenceID.astype('object')
    i.RealtyRoom = i.RealtyRoom.astype('int64')
    i.RealtyLivingRoom = i.RealtyLivingRoom.astype('int64')
    i.RealtyBathroom = i.RealtyBathroom.astype('int64')
    i.RealtySqm = i.RealtySqm.astype('float64')
    i.RealtyFloorCount = i.RealtyFloorCount.astype('int64')
    i.RealtyFloorID = i.RealtyFloorID.astype('object')
    i.RealtyAge = i.RealtyAge.astype('int64')
    i.RealtyHeatingID = i.RealtyHeatingID.astype('object')
    i.RealtyFuelID = i.RealtyFuelID.astype('object')
    i.RealtyBuildID = i.RealtyBuildID.astype('object')
    i.RealtyBuildStateID = i.RealtyBuildStateID.astype('object')
    i.RealtyUsageID = i.RealtyUsageID.astype('object')
    i.RealtyIsStudentOrSingle = i.RealtyIsStudentOrSingle.astype('object')
    i.RealtyCloseID = i.RealtyCloseID.astype('object')
    i.RealtyMapLatitude = i.RealtyMapLatitude.astype('float64')
    i.RealtyMapLongtitude = i.RealtyMapLongtitude.astype('float64')
    i.RealtyPriceShow = i.RealtyPriceShow.astype('object')
    i.RealtyCloseID = i.RealtyCloseID.astype('object')
    i.RealtyIsHousingComplex = i.RealtyIsHousingComplex.astype('object')
    i.DistrictID = i.DistrictID.astype('object')
    i.DistrictName = i.DistrictName.astype('object')
    i.RealtyPriceCurrencyID = i.RealtyPriceCurrencyID.astype('object')
    i.CountyID = i.CountyID.astype('object')
    i.CountyName = i.CountyName.astype('object')
    i.Month = i.Month.astype('object')
    i.Year = i.Year.astype('object')
    i.Mortgage = i.Mortgage.astype('float64')
    i.HousePriceIndex = i.HousePriceIndex.astype('float64')
    i.Metrobus = i.Metrobus.astype('float64')
    i.Metro = i.Metro.astype('float64')
    i.UFE = i.UFE.astype('float64')
    i.SqmPrice = i.SqmPrice.astype('float64')


# In[5]:


data_1318 = pd.concat([data13, data14, data15, data16, data17, data18])
data_1318.info()


# In[6]:


data_1318['DistrictMedian'] = data_1318.groupby('DistrictID').SqmPrice.transform('median')
data_1318['MonthMedian'] = data_1318.groupby('Month').SqmPrice.transform('median')


# In[7]:


X = data_1318.drop(['CountyID', 'RealtySubCategoryID', 'SqmPrice', 'DistrictID', 'CountyName', 'DistrictName', 
                    'PriceTL', 'RealtyEndDateTime', 'RealtyID', 'Tarihx', 'Month', 'RealtyPriceCurrencyID'], axis=1)
Y = data_1318['SqmPrice']


# In[9]:


X.info()


# In[10]:


X = pd.concat([X, pd.get_dummies(X.select_dtypes(include='object'))], axis=1)
X = X.drop(['RealtyPublishID', 'RealtyResidenceID', 'RealtyFloorID', 'RealtyHeatingID',
           'RealtyFuelID', 'RealtyBuildID', 'RealtyBuildStateID', 'RealtyUsageID', 'RealtyIsStudentOrSingle', 'RealtyCloseID', 'RealtyPriceShow',
           'RealtyIsHousingComplex', 'Year'], axis=1)
X.info()


# In[11]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# In[ ]:





# In[15]:


rf2 = RandomForestRegressor(n_estimators=600, min_samples_split = 6, min_samples_leaf = 10, max_depth = 100, max_features= 'auto', random_state=42)
rf2.fit(X_train, Y_train)

predict_train = rf2.predict(X_train)
predict_test = rf2.predict(X_test)


mae_test = mean_absolute_error(predict_test, Y_test)
mse_test = mean_squared_error(predict_test, Y_test)
rmse_test = np.sqrt(mse_test)

print('Mean Absolute Error (MAE_test): %.2f' % mae_test)
print('Mean Squared Error (MSE_test): %.2f' % mse_test)
print('Root Mean Squared Error (RMSE_test): %.2f' % rmse_test)

print('----')

mae_train = mean_absolute_error(predict_train, Y_train)
mse_train = mean_squared_error(predict_train, Y_train)
rmse_train = np.sqrt(mse_train)

print('Mean Absolute Error (MAE_train): %.2f' % mae_train)
print('Mean Squared Error (MSE_train): %.2f' % mse_train)
print('Root Mean Squared Error (RMSE_train): %.2f' % rmse_train)

print('++++')

print("MAPE_rf")
print("Train : ",np.mean(np.abs((Y_train - predict_train) / Y_train)) * 100)
print("Test  : ",np.mean(np.abs((Y_test - predict_test) / Y_test)) * 100)



# In[16]:


plt.figure(figsize=(12, 6))

ranking2 = rf2.feature_importances_
features2 = np.argsort(ranking2)[::-1][:11]
columns2 = X.columns

plt.title("Feature importances based on Random Forest Regressor", y = 1.03, size = 18)
plt.bar(range(len(features2)), ranking2[features2], color="aqua", align="center")
plt.xticks(range(len(features2)), columns2[features2], rotation=80)
plt.show()


# In[ ]:


rom sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=10, random_state=7)
scores_rfr = cross_val_score(rf2, X2, Y2, cv=kfold,scoring='explained_variance')
mean_rfrs = []
std_rfrs_upper = []
std_rfrs_lower = []

print("Est. explained variance: %0.2f (+/- %0.2f)" % (scores_rfr.mean(), scores_rfr.std() * 2))
print("")
mean_rfrs.append(scores_rfr.mean())
std_rfrs_upper.append(scores_rfr.mean()+scores_rfr.std()*2)
std_rfrs_lower.append(scores_rfr.mean()-scores_rfr.std()*2)


# In[ ]:





# In[ ]:





# In[ ]:


import hshshshs
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [400, 600, 800, 1000, 1200]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [30, 40, 50, 60, 70, 80, 90, 100, None]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 5, 8]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42)
rf_random.fit(X2_train, Y2_train)

