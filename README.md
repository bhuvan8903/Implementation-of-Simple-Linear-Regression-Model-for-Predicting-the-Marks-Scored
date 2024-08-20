# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and read the dataframe.

2.Assign hours to X and scores to Y.

3.Implement training set and test set of the dataframe

4.Plot the required graph both for test data and training data.

5.Find the values of MSE,MAE and RMSE

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Bhuvaneshwaran H
RegisterNumber: 212223240018

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(2)
df.tail(4)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='violet')
plt.plot(x_train,regressor.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='green')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
/*
```

## Output:
DATA SET

![image](https://github.com/user-attachments/assets/6d7154d6-59da-468c-b13d-1e78a1d2d3fa)

HEAD VALUES

![Screenshot 2024-08-16 222912](https://github.com/user-attachments/assets/3ac3a891-a6f3-4451-9a5f-767b5faa7512)

TAIL VALUES

![Screenshot 2024-08-16 223004](https://github.com/user-attachments/assets/7048b66e-afeb-48da-8341-6aaa268b1aee)

X VALUES

![Screenshot 2024-08-16 223053](https://github.com/user-attachments/assets/970963be-e486-4dd3-8bb4-d3b93b226905)

Y VALUES

![Screenshot 2024-08-16 223151](https://github.com/user-attachments/assets/e94241fe-6ad9-45f9-a2e2-7010665fc78c)

PREDICTION VALUES

![Screenshot 2024-08-16 223159](https://github.com/user-attachments/assets/2825fe48-396b-4b3b-a931-f7bf71b2e0e2)
![Screenshot 2024-08-16 223209](https://github.com/user-attachments/assets/b570a5db-d811-48bb-8627-6aee182918a7)


MSE,MAE AND RMSE

![Screenshot 2024-08-16 223225](https://github.com/user-attachments/assets/da849325-29b3-4f21-bc19-1c0d6d148aae)

TRAINING SET

![Screenshot 2024-08-16 223240](https://github.com/user-attachments/assets/8aacd0d7-b7c4-4da4-80d4-fb958f9d672e)

TESTING SET

![Screenshot 2024-08-16 223250](https://github.com/user-attachments/assets/77327811-6692-45fb-b0b7-af7dd2aaf14c)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
