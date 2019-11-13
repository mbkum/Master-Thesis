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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
from lightgbm import LGBMRegressor

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
# Imputing categorical variables with "999999" (A new subcategory for missing values)

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

X = data_1318.drop(['CountyID', 'RealtySubCategoryID', 'SqmPrice', 'DistrictID', 'CountyName', 'DistrictName', 'Age_Miss', 'FloorCount_Miss',
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
# Calling explained variance score

from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Defining our own explained variance, since LGBRegressor cannot read "explained_variance"
def lgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', explained_variance_score(labels, preds), True

# In[11]:
# converting train dataset to matrix
dtrain = lgb.Dataset(data=X_train, label=Y_train, params={'verbose': -1}, free_raw_data=False)

# In[12]:

#defining hyperparameter set
def hyp_lgbm(subsample, num_leaves, n_estimators, colsample_bytree, feature_fraction, bagging_fraction, max_depth):
        #fixed hyperparameters
        params = {'application':'regression','subsample_freq':1, 'learning_rate':0.005, 'verbose': -1,
                  'early_stopping_round':50,
                  'metric':'lgb_r2_score'}
        #optimized hyperparameters
        params["subsample"] = max(min(subsample, 1), 0)
        params["num_leaves"] =  int(round(num_leaves))
        params["colsample_bytree"] = max(min(colsample_bytree, 1), 0)
        params["n_estimators"] = int(round(n_estimators))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        #Setting LightGBM's 5-fold CV, metric "l2" is "RMSE".
        cv_results = lgb.cv(params, dtrain, nfold=5, seed=17,categorical_feature=[], stratified=False,
                            verbose_eval =False, metrics = 'l2')
        # Multiplying by -1.0 is necessary, since Bayesian optimization is maximizing the solution
        return  -1 * np.mean(cv_results['l2-mean'])
        # We repeat the same process with explained variance by using the explained variance definition we have described before
        # cv_results = lgb.cv(params, dtrain, nfold=5, seed=17,categorical_feature=[], stratified=False,verbose_eval =None, 
        #feval=lgb_r2_score)
        # return np.max(cv_results['r2-mean'])

# Setting the ranges of parameters for optimization
pds = {'colsample_bytree': (0.0, 1),
       'num_leaves': (100, 12000),
       'subsample': (0.0, 1),
       'feature_fraction': (0.0, 1.0),
       'bagging_fraction': (0.0, 1),
       'max_depth': (6, 20),
       'n_estimators': (1000, 4000)
          }

# Optimization function
optimizer = BayesianOptimization(hyp_lgbm,pds,random_state=7)

# Optimizing the Bayesian function with 80 iterations
optimizer.maximize(init_points=3, n_iter=80)





# In[12]:
# Setting optimized hyperparameters to the LGBMRegressor function

lgb_reg = LGBMRegressor(subsample_freq=1, feature_fraction=0.6969, num_leaves=7867, learning_rate = 0.005, 
                        bagging_fraction=0.3895,colsample_bytree=0.9049,
                       subsample = 0.8073, max_depth = 17, n_estimators = 3340)

# Training the model
lgb_reg.fit(X_train, Y_train)

# Predict the target values for both training and test sets
predict_train = lgb_reg.predict(X_train)
predict_test = lgb_reg.predict(X_test)


# Compute the metrics for test set
mae_lgb_test = mean_absolute_error(predict_test, Y_test)
mse_lgb_test = mean_squared_error(predict_test, Y_test)
rmse_lgb_test = np.sqrt(mse_lgb_test)

print('Mean Absolute Error (MAE_lgb_test): %.2f' % mae_lgb_test)
print('Mean Squared Error (MSE_lgb_test): %.2f' % mse_lgb_test)
print('Root Mean Squared Error (RMSE_lgb_test): %.2f' % rmse_lgb_test)

print('++++')

# Compute the metrics for training set
mae_lgb_train = mean_absolute_error(predict_train, Y_train)
mse_lgb_train = mean_squared_error(predict_train, Y_train)
rmse_lgb_train = np.sqrt(mse_lgb_train)

print('Mean Absolute Error (MAE_lgb_test): %.2f' % mae_lgb_train)
print('Mean Squared Error (MSE_lgb_test): %.2f' % mse_lgb_train)
print('Root Mean Squared Error (RMSE_lgb_test): %.2f' % rmse_lgb_train)

print('++++')

## MAPE results
print("MAPE_lgb")
print("Train : ",np.mean(np.abs((Y_train - predict_train) / Y_train)) * 100)
print("Test  : ",np.mean(np.abs((Y_test - predict_test) / Y_test)) * 100)


# In[13]:
# Visualization of the feature importance

plt.figure(figsize=(12, 6))

ranking2 = lgb_reg.feature_importances_
features2 = np.argsort(ranking2)[::-1][:11]
columns2 = X.columns

plt.title("Feature importances based on LGBM2", y = 1.03, size = 18)
plt.bar(range(len(features2)), ranking2[features2], color="aqua", align="center")
plt.xticks(range(len(features2)), columns2[features2], rotation=80)
plt.show()

