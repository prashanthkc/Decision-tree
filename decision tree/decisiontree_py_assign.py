# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 11:20:37 2021

@author: prashanth
"""

################################# Problem 1 ################################

#loading the data
import pandas as pd
import numpy as np
company = pd.read_csv("F:/assignment/decision tree/Datasets_DTRF/Company_Data.csv")

#dropping column Age
company_data = company.drop(["Age"], axis = 1)

#creating dummy for categorical data
company_data = pd.get_dummies(company_data, columns = ["ShelveLoc", "Urban", "US"])

#converting continous type to categorical
max = company_data['Sales'].max()
company_data['Sales'] = pd.cut(company_data.Sales, bins = [-999 , max/2 , 999] , labels=['low' , 'high'])

#checking for null and na values
company_data.isnull().sum()
company_data.isna().sum()

# Splitting data into training and testing data set
colnames = list(company_data.columns)
predictors = colnames[1:]
target = colnames[0]

from sklearn.model_selection import train_test_split
train, test = train_test_split(company_data, test_size = 0.3)

#building model
from sklearn.tree import DecisionTreeClassifier as DT

model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])

# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy

#model is overfitting
#using pruning technique to overcome this overfitting problem

#dropping column Age
company_data = company.drop(["Age"], axis = 1)

#creating dummy for categorical data
company_data = pd.get_dummies(company_data, columns = ["ShelveLoc", "Urban", "US"])

# Splitting data into training and testing data set
colnames = list(company_data.columns)
predictors = colnames[1:]
target = colnames[0]

from sklearn.model_selection import train_test_split
train, test = train_test_split(company_data, test_size = 0.3)

# Train the Regression DT
from sklearn import tree
regtree = tree.DecisionTreeRegressor(max_depth = 5 , ccp_alpha= 0.05)
regtree.fit(train[predictors], train[target])

# Prediction
test_pred = regtree.predict(test[predictors])
train_pred = regtree.predict(train[predictors])

# Measuring accuracy
from sklearn.metrics import mean_squared_error, r2_score

# Error on test dataset
mean_squared_error(test[target], test_pred)
r2_score(test[target], test_pred)

# Error on train dataset
mean_squared_error(train[target], train_pred)
r2_score(train[target], train_pred)

# Minimum observations at the internal node approach
regtree2 = tree.DecisionTreeRegressor(min_samples_split = 5)
regtree2.fit(train[predictors], train[target])

# Prediction
test_pred2 = regtree2.predict(test[predictors])
train_pred2 = regtree2.predict(train[predictors])

# Error on test dataset
mean_squared_error(test[target], test_pred2)
r2_score(test[target], test_pred2)

# Error on train dataset
mean_squared_error(train[target], train_pred2)
r2_score(train[target], train_pred2)

## Minimum observations at the leaf node approach
regtree3 = tree.DecisionTreeRegressor(min_samples_leaf = 2)
regtree3.fit(train[predictors], train[target])

# Prediction
test_pred3 = regtree3.predict(test[predictors])
train_pred3 = regtree3.predict(train[predictors])

# measure of error on test dataset
mean_squared_error(test[target], test_pred3)
r2_score(test[target], test_pred3)

# measure of error on train dataset
mean_squared_error(train[target], train_pred3)
r2_score(train[target], train_pred3)

#random Forest technique
#dropping column Age
company_data = company.drop(["Age"], axis = 1)

#creating dummy for categorical data
company_data = pd.get_dummies(company_data, columns = ["ShelveLoc", "Urban", "US"])

#converting continous type to categorical
max = company_data['Sales'].max()
company_data['Sales'] = pd.cut(company_data.Sales, bins = [-999 , max/2 , 999] , labels=['low' , 'high'])

#checking for null and na values
company_data.isnull().sum()
company_data.isna().sum()

# Splitting data into training and testing data set
colnames = list(company_data.columns)
predictors = colnames[1:]
target = colnames[0]

from sklearn.model_selection import train_test_split
train, test = train_test_split(company_data, test_size = 0.3)

#building the random forest model
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

rf_clf.fit(train[predictors], train[target])

from sklearn.metrics import accuracy_score, confusion_matrix

#accuracy on test data
confusion_matrix(test[target], rf_clf.predict(test[predictors]))
accuracy_score(test[target], rf_clf.predict(test[predictors]))

#accuracy on train data
confusion_matrix(train[target], rf_clf.predict(train[predictors]))
accuracy_score(train[target], rf_clf.predict(train[predictors]))

# GridSearchCV

from sklearn.model_selection import GridSearchCV

rf_clf_grid = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

param_grid = {"max_features": [4, 5, 6, 7, 8, 9, 10], "min_samples_split": [2, 3, 10]}

grid_search = GridSearchCV(rf_clf_grid, param_grid, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(train[predictors], train[target])

grid_search.best_params_

cv_rf_clf_grid = grid_search.best_estimator_

from sklearn.metrics import accuracy_score, confusion_matrix

#testing accuracy
confusion_matrix(test[target], cv_rf_clf_grid.predict(test[predictors]))
accuracy_score(test[target], cv_rf_clf_grid.predict(test[predictors]))

#training accuracy
confusion_matrix(train[target], cv_rf_clf_grid.predict(train[predictors]))
accuracy_score(train[target], cv_rf_clf_grid.predict(train[predictors]))

#model rightfit

###########################################Problem 2############################################
#loading the data
import pandas as pd
import numpy as np
diabetes = pd.read_csv("F:/assignment/decision tree/Datasets_DTRF/Diabetes.csv")

#checking for null and na values
diabetes.isnull().sum()
diabetes.isna().sum()

# Splitting data into training and testing data set
colnames = list(diabetes.columns)
predictors = colnames[:8]
target = colnames[8]

from sklearn.model_selection import train_test_split
train, test = train_test_split(diabetes, test_size = 0.3)

#building model
from sklearn.tree import DecisionTreeClassifier as DT

model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])

# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy

#model is overfitting
#using pruning technique to overcome this overfitting problem

#creating dummy for categorical data
diabetes = pd.get_dummies(diabetes, columns = [" Classvariable"])

# Splitting data into training and testing data set
colnames = list(diabetes.columns)
predictors = colnames[1:]
target = colnames[0]

from sklearn.model_selection import train_test_split
train, test = train_test_split(diabetes, test_size = 0.3)

# Train the Regression DT
from sklearn import tree
regtree = tree.DecisionTreeRegressor(max_depth = 3 , ccp_alpha= 0)
regtree.fit(train[predictors], train[target])

# Prediction
test_pred = regtree.predict(test[predictors])
train_pred = regtree.predict(train[predictors])

# Measuring accuracy
from sklearn.metrics import mean_squared_error, r2_score

# Error on test dataset
mean_squared_error(test[target], test_pred)
r2_score(test[target], test_pred)

# Error on train dataset
mean_squared_error(train[target], train_pred)
r2_score(train[target], train_pred)

# Minimum observations at the internal node approach
regtree2 = tree.DecisionTreeRegressor(min_samples_split = 5)
regtree2.fit(train[predictors], train[target])

# Prediction
test_pred2 = regtree2.predict(test[predictors])
train_pred2 = regtree2.predict(train[predictors])

# Error on test dataset
mean_squared_error(test[target], test_pred2)
r2_score(test[target], test_pred2)

# Error on train dataset
mean_squared_error(train[target], train_pred2)
r2_score(train[target], train_pred2)

## Minimum observations at the leaf node approach
regtree3 = tree.DecisionTreeRegressor(min_samples_leaf = 2)
regtree3.fit(train[predictors], train[target])

# Prediction
test_pred3 = regtree3.predict(test[predictors])
train_pred3 = regtree3.predict(train[predictors])

# measure of error on test dataset
mean_squared_error(test[target], test_pred3)
r2_score(test[target], test_pred3)

# measure of error on train dataset
mean_squared_error(train[target], train_pred3)
r2_score(train[target], train_pred3)

#random Forest technique
diabetes = pd.read_csv("F:/assignment/decision tree/Datasets_DTRF/Diabetes.csv")

# Splitting data into training and testing data set
colnames = list(diabetes.columns)
predictors = colnames[:8]
target = colnames[8]

from sklearn.model_selection import train_test_split
train, test = train_test_split(diabetes, test_size = 0.3)

#building the random forest model
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

rf_clf.fit(train[predictors], train[target])

from sklearn.metrics import accuracy_score, confusion_matrix

#accuracy on test data
confusion_matrix(test[target], rf_clf.predict(test[predictors]))
accuracy_score(test[target], rf_clf.predict(test[predictors]))

#accuracy on train data
confusion_matrix(train[target], rf_clf.predict(train[predictors]))
accuracy_score(train[target], rf_clf.predict(train[predictors]))

# GridSearchCV

from sklearn.model_selection import GridSearchCV

rf_clf_grid = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

param_grid = {"max_features": [4, 5, 6, 7, 8, 9, 10], "min_samples_split": [2, 3, 10]}

grid_search = GridSearchCV(rf_clf_grid, param_grid, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(train[predictors], train[target])

grid_search.best_params_

cv_rf_clf_grid = grid_search.best_estimator_

from sklearn.metrics import accuracy_score, confusion_matrix

#testing accuracy
confusion_matrix(test[target], cv_rf_clf_grid.predict(test[predictors]))
accuracy_score(test[target], cv_rf_clf_grid.predict(test[predictors]))

#training accuracy
confusion_matrix(train[target], cv_rf_clf_grid.predict(train[predictors]))
accuracy_score(train[target], cv_rf_clf_grid.predict(train[predictors]))

#model rightfit

####################################Problem 3#######################################

#loading the data
import pandas as pd
import numpy as np
fraud_check = pd.read_csv("F:/assignment/decision tree/Datasets_DTRF/Fraud_check.csv")

#creating dummy for categorical data
fraud_check = pd.get_dummies(fraud_check, columns = ["Undergrad", "Marital.Status", "Urban"])

#converting continous type to categorical
fraud_check['Taxable.Income'] = pd.cut(fraud_check["Taxable.Income"], bins = [-999 , 30000 , 99999999] , labels=['Risky' , 'Good'])

#checking for null and na values
fraud_check.isnull().sum()
fraud_check.isna().sum()

# Splitting data into training and testing data set
colnames = list(fraud_check.columns)
predictors = colnames[1:]
target = colnames[0]

from sklearn.model_selection import train_test_split
train, test = train_test_split(fraud_check, test_size = 0.3)

#building model
from sklearn.tree import DecisionTreeClassifier as DT

model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])

# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy

#model is overfitting

#using pruning technique to overcome this overfitting problem

#creating dummy for categorical data
fraud_check = pd.read_csv("C:/Users/hp/Desktop/Decision tree assi/Fraud_check.csv")
fraud_check = pd.get_dummies(fraud_check, columns = ["Undergrad", "Marital.Status", "Urban"])

# Splitting data into training and testing data set
colnames = list(fraud_check.columns)
predictors = colnames[1:]
target = colnames[0]

from sklearn.model_selection import train_test_split
train, test = train_test_split(fraud_check, test_size = 0.3)

# Train the Regression DT
from sklearn import tree
regtree = tree.DecisionTreeRegressor(max_depth = 4)
regtree.fit(train[predictors], train[target])

# Prediction
test_pred = regtree.predict(test[predictors])
train_pred = regtree.predict(train[predictors])

# Measuring accuracy
from sklearn.metrics import mean_squared_error, r2_score

# Error on test dataset
mean_squared_error(test[target], test_pred)
r2_score(test[target], test_pred)

# Error on train dataset
mean_squared_error(train[target], train_pred)
r2_score(train[target], train_pred)

# Minimum observations at the internal node approach
regtree2 = tree.DecisionTreeRegressor(min_samples_split = 5)
regtree2.fit(train[predictors], train[target])

# Prediction
test_pred2 = regtree2.predict(test[predictors])
train_pred2 = regtree2.predict(train[predictors])

# Error on test dataset
mean_squared_error(test[target], test_pred2)
r2_score(test[target], test_pred2)

# Error on train dataset
mean_squared_error(train[target], train_pred2)
r2_score(train[target], train_pred2)

## Minimum observations at the leaf node approach
regtree3 = tree.DecisionTreeRegressor(min_samples_leaf = 2)
regtree3.fit(train[predictors], train[target])

# Prediction
test_pred3 = regtree3.predict(test[predictors])
train_pred3 = regtree3.predict(train[predictors])

# measure of error on test dataset
mean_squared_error(test[target], test_pred3)
r2_score(test[target], test_pred3)

# measure of error on train dataset
mean_squared_error(train[target], train_pred3)
r2_score(train[target], train_pred3)

#random Forest technique

#converting continous type to categorical
fraud_check['Taxable.Income'] = pd.cut(fraud_check["Taxable.Income"], bins = [-999 , 30000 , 99999999] , labels=['Risky' , 'Good'])

# Splitting data into training and testing data set
colnames = list(fraud_check.columns)
predictors = colnames[1:]
target = colnames[0]

from sklearn.model_selection import train_test_split
train, test = train_test_split(fraud_check, test_size = 0.3)

#building the random forest model
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

rf_clf.fit(train[predictors], train[target])

from sklearn.metrics import accuracy_score, confusion_matrix

#accuracy on test data
confusion_matrix(test[target], rf_clf.predict(test[predictors]))
accuracy_score(test[target], rf_clf.predict(test[predictors]))

#accuracy on train data
confusion_matrix(train[target], rf_clf.predict(train[predictors]))
accuracy_score(train[target], rf_clf.predict(train[predictors]))

# GridSearchCV

from sklearn.model_selection import GridSearchCV

rf_clf_grid = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

param_grid = {"max_features": [4, 5, 6, 7, 8, 9, 10], "min_samples_split": [2, 3, 10]}

grid_search = GridSearchCV(rf_clf_grid, param_grid, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(train[predictors], train[target])

grid_search.best_params_

cv_rf_clf_grid = grid_search.best_estimator_

from sklearn.metrics import accuracy_score, confusion_matrix

#testing accuracy
confusion_matrix(test[target], cv_rf_clf_grid.predict(test[predictors]))
accuracy_score(test[target], cv_rf_clf_grid.predict(test[predictors]))

#training accuracy
confusion_matrix(train[target], cv_rf_clf_grid.predict(train[predictors]))
accuracy_score(train[target], cv_rf_clf_grid.predict(train[predictors]))

#model rightfit

########################################Problem 4###########################################
#load the data

import pandas as pd
import numpy as np
HR_data = pd.read_csv("F:/assignment/decision tree/Datasets_DTRF/HR_DT.csv")

#dummy values
HR_data = pd.get_dummies(HR_data, columns = ["Position of the employee"])

#random Forest technique

# Splitting data into training and testing data set
colnames = list(HR_data.columns)
target = colnames[1]
predictors = colnames[:1]+colnames[2:]

from sklearn.model_selection import train_test_split
train, test = train_test_split(HR_data, test_size = 0.3)

#building the random forest model
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=50)

rf_clf.fit(train[predictors], train[target])

from sklearn.metrics import accuracy_score, confusion_matrix

#accuracy on test data
confusion_matrix(test[target], rf_clf.predict(test[predictors]))
accuracy_score(test[target], rf_clf.predict(test[predictors]))

#accuracy on train data
confusion_matrix(train[target], rf_clf.predict(train[predictors]))
accuracy_score(train[target], rf_clf.predict(train[predictors]))

#create a dataframe of customers claim details
#filling Region Manager as 1 anyways we convert it to dummy here by saving computation
customer_claim_list = [[1, 5.0 , 70000]]

customer_claim = pd.DataFrame(customer_claim_list , columns= ["Position of the employee" , "no of Years of Experience of employee", " monthly income of employee"])

#combining and concatenating 2 dataframes
df = [test , customer_claim]
test = pd.concat(df)

#filling all na by 0
test = test.fillna(0)

#predicting using testing data where customer claims present at last row of test
preds = rf_clf.predict(test[predictors])

#storing predicted values in seperate column
test["predicted salary"] = preds

#accessing predicted value
test.iloc[59,13]

# predicted as 67938 , candidate claimed 70000 which is almost close.
#it can be assumed that candidate is genuine

###########################################END#####################################
