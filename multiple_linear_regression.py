# Multiple Linear Regression

import numpy as np
import pandas as pd

# Importing the datasets

datasets = pd.read_csv('50_Startups.csv')
X = datasets.iloc[:, :-1].values
Y = datasets.iloc[:, 4].values

# Encoding categorical data

# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Fitting the Multiple Linear Regression in the Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)

# Predicting the Test set results

Y_Pred = regressor.predict(X_Test)

# Building the optimal model using Backward Elimination

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

X_Optimal = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_Optimal).fit()
regressor_OLS.summary()

X_Optimal = X[:, [0,1,2,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_Optimal).fit()
regressor_OLS.summary()

X_Optimal = X[:, [0,1,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_Optimal).fit()
regressor_OLS.summary()

X_Optimal = X[:, [0,1,4]]
regressor_OLS = sm.OLS(endog = Y, exog = X_Optimal).fit()
regressor_OLS.summary()

# Fitting the Multiple Linear Regression in the Optimal Training set

X_Optimal_Train, X_Optimal_Test = train_test_split(X_Optimal,test_size = 0.2, random_state = 0)
regressor.fit(X_Optimal_Train, Y_Train)

# Predicting the Optimal Test set results

Y_Optimal_Pred = regressor.predict(X_Optimal_Test)
