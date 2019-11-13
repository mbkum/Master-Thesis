#!/usr/bin/env python
# coding: utf-8

# In[1]:
#Calling libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import math
from sklearn.metrics import explained_variance_score
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:
# Reading datasets

data13 = pd.read_csv('C://Users//ASUSNB//Desktop//THesis//Data//Yillik//YillikFiltreli//kiraci13.csv', sep = ";")
data14 = pd.read_csv('C://Users//ASUSNB//Desktop//THesis//Data//Yillik//YillikFiltreli//kiraci14.csv', sep = ";")
data15 = pd.read_csv('C://Users//ASUSNB//Desktop//THesis//Data//Yillik//YillikFiltreli//kiraci15.csv', sep = ";")
data16 = pd.read_csv('C://Users//ASUSNB//Desktop//THesis//Data//Yillik//YillikFiltreli//kiraci16.csv', sep = ";")
data17 = pd.read_csv('C://Users//ASUSNB//Desktop//THesis//Data//Yillik//YillikFiltreli//kiraci17.csv', sep = ";")
data18 = pd.read_csv('C://Users//ASUSNB//Desktop//THesis//Data//Yillik//YillikFiltreli//kiraci18.csv', sep = ";")


# In[3]:
# Imputing categorical variables with "999999" (A new subcategory for missing values), numerical variables with group modes for each year

datas = [data13, data14, data15, data16, data17, data18]
datalar = [i for i in datas]
for i in datas:
    i['RealtyPublishID'].fillna(99999, inplace=True)
    i['RealtyResidenceID'].fillna(99999, inplace=True)
    i['RealtyRoom'].fillna(i.RealtyRoom.mode()[0], inplace = True)
    i['RealtyFloorCount'].fillna(i.RealtyFloorCount.mode()[0], inplace = True)
    i['RealtyAge'].fillna(i.RealtyAge.mode()[0], inplace = True)
    i['RealtyLivingRoom'].fillna(i.RealtyLivingRoom.mode()[0], inplace = True)
    i['RealtyBathroom'].fillna(i.RealtyBathroom.mode()[0], inplace = True)
    i['RealtySqm'].fillna(i.RealtySqm.mode()[0], inplace = True)
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

# Merging datasets
data_1318 = pd.concat([data13, data14, data15, data16, data17, data18])




# In[4]:
#Generating DistrictMedian and MonthMedian variables.

data_1318['DistrictMedian'] = data_1318.groupby('DistrictID').SqmPrice.transform('median')
data_1318['MonthMedian'] = data_1318.groupby('Month').SqmPrice.transform('median')


# In[5]:
#Checking variable information if everything goes correct until this point.

data_1318.info()


# In[6]:
#Specifying dependent and independent variables. Dropped variables are un-used features except "SqmPrice", the target value.

X = data_1318.drop(['CountyID', 'RealtySubCategoryID', 'SqmPrice', 'DistrictID', 'CountyName', 'DistrictName', 
                      'PriceTL', 'RealtyEndDateTime', 'RealtyID', 'Tarihx', 'Month', 'RealtyPriceCurrencyID', 'Residence_Miss',
                    'FloorID_Miss', 'Fuel_Miss', 'Build_Miss', 'BuildState_Miss', 'Usage_Miss', 'Student_Miss'], axis=1)
Y = data_1318['SqmPrice']


# In[7]:
#Converting data types, specifying categorical variables as "object".

X.RealtyResidenceID = X.RealtyResidenceID.astype('object')
X.RealtyPublishID = X.RealtyPublishID.astype('object')
X.RealtyFloorID = X.RealtyFloorID.astype('object')
X.RealtyHeatingID = X.RealtyHeatingID.astype('object')
X.RealtyFuelID = X.RealtyFuelID.astype('object')
X.RealtyBuildID = X.RealtyBuildID.astype('object')
X.RealtyBuildStateID = X.RealtyBuildStateID.astype('object')
X.RealtyUsageID = X.RealtyUsageID.astype('object')
X.RealtyIsStudentOrSingle = X.RealtyIsStudentOrSingle.astype('object')
X.RealtyCloseID = X.RealtyCloseID.astype('object')
X.RealtyIsHousingComplex = X.RealtyIsHousingComplex.astype('object')
X.Year = X.Year.astype('object')


# In[8]:
# Generating dummy variables from each subcategories of categorical variables.

X = pd.concat([X, pd.get_dummies(X.select_dtypes(include='object'))], axis=1)
# Dropping the original categorical variables
X = X.drop(['RealtyPublishID', 'RealtyResidenceID', 'RealtyFloorID', 'RealtyHeatingID',
           'RealtyFuelID', 'RealtyBuildID', 'RealtyBuildStateID', 'RealtyUsageID', 'RealtyIsStudentOrSingle', 'RealtyCloseID', 'RealtyPriceShow',
           'RealtyIsHousingComplex','Year'], axis=1)
X.info()


# In[9]:
# Randomly splitting dataset into training and test sets.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# In[10]:
# converting dataset to matrix

dtrain = xgb.DMatrix(X_train, label=Y_train)
dtest = xgb.DMatrix(X_test, label=Y_test)


# In[11]:
# Hyperparameter tuning with Bayesian Optimization

# Defining set of hyperparameter values
def hyp_xgb(subsample, n_estimators, colsample,eta, gamma, max_depth):
        # fixed parameters
        params = {'nthread':-1, 'early_stopping_round':10, 'learning_rate':0.01,
                  'eval_metric': 'rmse'}
        # Optimized parameters
        params["subsample"] = max(min(subsample, 1), 0)
        params["colsample"] = max(min(colsample, 1), 0)
        params["n_estimators"] = int(round(n_estimators))
        params['eta'] = max(min(eta, 1), 0)
        params['gamma'] = max(min(gamma, 4), 0)
        params['max_depth'] = int(round(max_depth))
        # 5-fold CV
        cv_results = xgb.cv(params, dtrain, nfold=5, num_boost_round=100,  seed=17, stratified=False, verbose_eval=None)
        # Multiplying the mean results with -1, since Bayesian Optimization is maximizing the results.
        return -1.0 * cv_results['test-rmse-mean'].iloc[-1]
# Setting rages for hyperparameters
pds = {'subsample': (0, 1),
       'colsample': (0, 1),
       'gamma': (0, 4),
       'eta': (0, 1),
       'max_depth': (6, 20),
       'n_estimators': (300, 1500),
          }
# Optimization function
optimizer = BayesianOptimization(hyp_xgb,pds,random_state=7)

# Optimizing Bayesian function with 50 iterations
optimizer.maximize(init_points=3, n_iter=50)


# In[12]:
# Hyperparameter tuning with Bayesian Optimization (explained variance)

# Setting ranges for hyperparameters 
pbounds2 = {
    'learning_rate': (0.01, 0.01),
    'n_estimators': (300, 1500),
    'max_depth': (6, 20),
    'subsample': (0, 1.0),
    'colsample': (0, 1.0),
    'eta': (0, 1),
    'gamma': (0, 4)}
#Set of hyperparameters
def xgboost_hyper_param2(learning_rate,
                        n_estimators,
                        max_depth,
                        subsample,
                        eta,
                        colsample,
                        gamma):

    max_depth = int(max_depth)
    n_estimators = int(n_estimators)
    
    # The model with optimized hyperparameters
    clf = XGBRegressor(
        max_depth=max_depth,
        eta=eta,
        max_leaf_nodes=max_leaf_nodes,
        colsample=colsample,
        subsample=subsample,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        gamma=gamma)
    # 5-fold CV with explained variance, returning mean of the results
    return np.mean(cross_val_score(clf, X_train, Y_train, cv=5, scoring='explained_variance'))

#Optimization function
optimizer = BayesianOptimization( f=xgboost_hyper_param2, pbounds=pbounds2, random_state=1)

import warnings
warnings.filterwarnings("ignore")

#Optimizing the function with 50 iterations
optimizer.maximize(init_points=3, n_iter=50, acq='ei')



# In[13]:

# Setting optimized hyperparameters to the XGBoost Regression function
xgb_reg = XGBRegressor(colsample_bytree = 0.9683, learning_rate = 0.01, early_stopping_rounds=20, gamma=3.462,
                       subsample = 0.1391, eta = 0.3134, 
                       max_depth = 15, n_estimators = 820)

# Training the model
xgb_reg.fit(X_train, Y_train)

# Predict the target values for both training and test sets
predict_train = xgb_reg.predict(X_train)
predict_test = xgb_reg.predict(X_test)

# Compute the metrics for test set
mae_xgb_test = mean_absolute_error(predict_test, Y_test)
mse_xgb_test = mean_squared_error(predict_test, Y_test)
rmse_xgb_test = np.sqrt(mse_xgb_test)

print('Mean Absolute Error (MAE_xgb_test): %.2f' % mae_xgb_test)
print('Mean Squared Error (MSE_xgb_test): %.2f' % mse_xgb_test)
print('Root Mean Squared Error (RMSE_xgb_test): %.2f' % rmse_xgb_test)

print('++++')

# Compute the metrics for training set
mae_xgb_train = mean_absolute_error(predict_train, Y_train)
mse_xgb_train = mean_squared_error(predict_train, Y_train)
rmse_xgb_train = np.sqrt(mse_xgb_train)

print('Mean Absolute Error (MAE_xgb_test): %.2f' % mae_xgb_train)
print('Mean Squared Error (MSE_xgb_test): %.2f' % mse_xgb_train)
print('Root Mean Squared Error (RMSE_xgb_test): %.2f' % rmse_xgb_train)

print('++++')

## MAPE results
print("MAPE_xgb")
print("Train : ",np.mean(np.abs((Y_train - predict_train) / Y_train)) * 100)
print("Test  : ",np.mean(np.abs((Y_test - predict_test) / Y_test)) * 100)


# In[16]:

# Visualization of the feature importance
plt.figure(figsize=(12, 6))

ranking2 = xgb_reg.feature_importances_
features2 = np.argsort(ranking2)[::-1][:11]
columns2 = X.columns

plt.title("Feature importances based on XGB1", y = 1.03, size = 18)
plt.bar(range(len(features2)), ranking2[features2], color="aqua", align="center")
plt.xticks(range(len(features2)), columns2[features2], rotation=80)
plt.show()
