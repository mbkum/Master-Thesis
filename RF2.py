#!/usr/bin/env python
# coding: utf-8

# In[1]:
#Calling libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
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
# Reading datasets

data13 = pd.read_csv('C://Users//ASUSNB//Desktop//THesis//Data//Yillik//YillikFiltreli//kiraci13.csv', sep = ";")
data14 = pd.read_csv('C://Users//ASUSNB//Desktop//THesis//Data//Yillik//YillikFiltreli//kiraci14.csv', sep = ";")
data15 = pd.read_csv('C://Users//ASUSNB//Desktop//THesis//Data//Yillik//YillikFiltreli//kiraci15.csv', sep = ";")
data16 = pd.read_csv('C://Users//ASUSNB//Desktop//THesis//Data//Yillik//YillikFiltreli//kiraci16.csv', sep = ";")
data17 = pd.read_csv('C://Users//ASUSNB//Desktop//THesis//Data//Yillik//YillikFiltreli//kiraci17.csv', sep = ";")
data18 = pd.read_csv('C://Users//ASUSNB//Desktop//THesis//Data//Yillik//YillikFiltreli//kiraci18.csv', sep = ";")
data13.info()


# In[3]:
# Imputing categorical variables with "999999" (A new subcategory for missing values), numerical variables with group modes for each year

datas = [data13, data14, data15, data16, data17, data18]
datalar = [i for i in datas]
for i in datas:
    i['RealtyResidenceID'].fillna(99999, inplace=True)
    i['RealtyPublishID'].fillna(99999, inplace=True)
    i['RealtyRoom'].fillna(i.RealtyRoom.mode()[0], inplace = True)
    i['RealtyFloorCount'].fillna(i.RealtyFloorCount.mode()[0], inplace = True)
    i['RealtyFloorID'].fillna(99999, inplace=True)
    i['RealtyAge'].fillna(i.RealtyAge.mode()[0], inplace = True)
    i['RealtyFuelID'].fillna(99999, inplace=True)
    i['RealtyBuildID'].fillna(99999, inplace=True)
    i['RealtyBuildStateID'].fillna(99999, inplace=True)
    i['RealtyUsageID'].fillna(99999, inplace=True)
    i['RealtyIsStudentOrSingle'].fillna(99999, inplace=True)
    i['RealtyPriceShow'].fillna(99999, inplace=True)
    i['RealtyIsHousingComplex'].fillna(99999, inplace=True)
    i['RealtyLivingRoom'].fillna(i.RealtyLivingRoom.mode()[0], inplace = True)
    i['RealtyBathroom'].fillna(i.RealtyBathroom.mode()[0], inplace = True)
    i['RealtySqm'].fillna(i.RealtySqm.mode()[0], inplace = True)
    i['RealtyHeatingID'].fillna(99999, inplace=True)


# In[4]:
#Converting data types. This process is necessary for specifying categorical variables that have subcategories in numbers.

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
#Merging datasets.

data_1318 = pd.concat([data13, data14, data15, data16, data17, data18])
data_1318.info()


# In[6]:
#Generating "DistrictMedian" and "MonthMedian" variables.

data_1318['DistrictMedian'] = data_1318.groupby('DistrictID').SqmPrice.transform('median')
data_1318['MonthMedian'] = data_1318.groupby('Month').SqmPrice.transform('median')


# In[7]:
#Specifying dependent and independent variables. Dropped variables are un-used features except "SqmPrice", the target value.

X = data_1318.drop(['CountyID', 'RealtySubCategoryID', 'SqmPrice', 'DistrictID', 'CountyName', 'DistrictName', 
                    'PriceTL', 'RealtyEndDateTime', 'RealtyID', 'Tarihx', 'Month', 'RealtyPriceCurrencyID', 'Residence_Miss',
                    'FloorID_Miss', 'Fuel_Miss', 'Build_Miss', 'BuildState_Miss', 'Usage_Miss', 'Student_Miss'], axis=1)
Y = data_1318['SqmPrice']


# In[8]:
#Checking variable information if everything goes correct until this point.

X.info()


# In[9]:
# Generating dummy variables from each subcategories of categorical variables. Then dropping the original object columns.

X = pd.concat([X, pd.get_dummies(X.select_dtypes(include='object'))], axis=1)
X = X.drop(['RealtyPublishID', 'RealtyResidenceID', 'RealtyFloorID', 'RealtyHeatingID',
           'RealtyFuelID', 'RealtyBuildID', 'RealtyBuildStateID', 'RealtyUsageID', 'RealtyIsStudentOrSingle', 'RealtyCloseID', 'RealtyPriceShow',
           'RealtyIsHousingComplex', 'Year'], axis=1)
X.info()


# In[10]:
# Randomly splitting dataset into training and test sets.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# In[11]:
# Hyperparameter tuning with Random Search. Since I have done this search in LRZ's computer, I implement the best features to the original model by hand

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [200, 400, 600, 800, 1000]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [30, 40, 50, 60, 70, 80, 90, 100]
# Minimum number of samples required to split a node
min_samples_split = [5, 6, 10, 15, 20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 4, 6, 8, 10]
# Specifying features for the grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
#Specifying the model
rf_search = RandomForestRegressor()
#Finally writing random search model with 20 iteration and 5-fold, then fitting the optimization to the training set
rf_random = RandomizedSearchCV(estimator = rf_search, param_distributions = random_grid, n_iter = 20, cv = 5, verbose=2, random_state=42)
rf_random.fit(X_train, Y_train)


# In[12]:
# Hyperparameter tuning with Bayesian optimization

from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
#set of hyperparameter values
pbounds = {
    'n_estimators': (200, 1000),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (5, 20),
    'max_depth': (20, 100)}
#defining hyperparameters for Bayesian optimization
def rf_hyper_param(n_estimators,
                   min_samples_leaf,
			       max_depth,
			       min_samples_split):

    n_estimators = int(n_estimators)
    min_samples_leaf = int(min_samples_leaf)
    min_samples_split = int(min_samples_split)
    #Specifying the model
    model = RandomForestRegressor(
    #Using maximum CPU
	n_jobs=-1,
        max_depth= max_depth,
        min_samples_split=min_samples_split,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf)
    # Take the mean of 5-fold CV results.
    # Same process is repeated for scoring='MSE' by multiplying it with -1, since Bayesian optimization is finding the maximum.
    # return -1.0 * np.mean(cross_val_score(model, X_train, Y_train, cv=5, scoring='mean_squared_error'))
    return np.mean(cross_val_score(model, X_train, Y_train, cv=5, scoring='explained_variance'))
# Writing the optimization problem with set of hyperparameter values
optimizer = BayesianOptimization( f=rf_hyper_param, pbounds=pbounds, random_state=1)

import warnings
warnings.filterwarnings("ignore")
# Finally optimizing the Bayesian with 20 iterations
optimizer.maximize(init_points=3, n_iter=20, acq='ei')


# In[13]:
# Set the best model's parameters to your Random Forest model
# Fit the model to training set

rf2 = RandomForestRegressor(n_estimators=600, min_samples_split = 5, min_samples_leaf = 10, max_depth = 100, max_features= 'auto', random_state=42)
rf2.fit(X_train, Y_train)

# Predict the target values for both training and test sets
predict_train = rf2.predict(X_train)
predict_test = rf2.predict(X_test)

# Compute the metrics for test set
mae_test = mean_absolute_error(predict_test, Y_test)
mse_test = mean_squared_error(predict_test, Y_test)
rmse_test = np.sqrt(mse_test)

print('Mean Absolute Error (MAE_test): %.2f' % mae_test)
print('Mean Squared Error (MSE_test): %.2f' % mse_test)
print('Root Mean Squared Error (RMSE_test): %.2f' % rmse_test)

print('----')

# Compute the metrics for training set
mae_train = mean_absolute_error(predict_train, Y_train)
mse_train = mean_squared_error(predict_train, Y_train)
rmse_train = np.sqrt(mse_train)

print('Mean Absolute Error (MAE_train): %.2f' % mae_train)
print('Mean Squared Error (MSE_train): %.2f' % mse_train)
print('Root Mean Squared Error (RMSE_train): %.2f' % rmse_train)

print('++++')
## MAPE results
print("MAPE_rf")
print("Train : ",np.mean(np.abs((Y_train - predict_train) / Y_train)) * 100)
print("Test  : ",np.mean(np.abs((Y_test - predict_test) / Y_test)) * 100)


# In[14]:

# Visualization of the feature importance
plt.figure(figsize=(12, 6))

ranking2 = rf2.feature_importances_
features2 = np.argsort(ranking2)[::-1][:11]
columns2 = X.columns

plt.title("Feature importances based on RF2", y = 1.03, size = 18)
plt.bar(range(len(features2)), ranking2[features2], color="aqua", align="center")
plt.xticks(range(len(features2)), columns2[features2], rotation=80)
plt.show()

