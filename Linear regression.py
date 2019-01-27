# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Simple Linear Regression :  Linear dependency between Salary & Years of Exeperience 
# Machine is : Simple Linear Regression Model
# Learning is : We trained our machine model on training set that is we train our model.
# It learns the correlations of the training set to be able to some Future Prediction.


# Importing Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Splitting dataset into training and testing sets
dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 1].values

# x is matrix and y is vector
# to have same result we are keeping random_state=0

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)


# in Simple Linear regression Feature scaling is  not required as it is care taken by scikit learn libraries.

# Fitting simple linear regression to training set

# LinearRegression is class
from sklearn.linear_model import LinearRegression

 # regressor is an Object of the class  
regressor =LinearRegression()

# call method as fit
regressor.fit(x_train,y_train)

# fit regressor to training data that is regressor is our Machine learning model

# Predicting Test set result
# y_pred is vector of all dependent variable
y_pred=regressor.predict(x_test)

# y_test is real salary from dataset 
#y_pred is predicted salary by ML Model


#Visualisation the training set'

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vrs Experience(treaining set)')
plt.xlabel('Exeperience')
plt.ylabel('Salary')
plt.show()



plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vrs Experience(treaining set)')
plt.xlabel('Exeperience')
plt.ylabel('Salary')
plt.show()

 