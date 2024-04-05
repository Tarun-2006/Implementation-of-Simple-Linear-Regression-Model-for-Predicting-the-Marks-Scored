# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import pandas, numpy and sklearn.                  
2. Calculate the values for the training data set.             
3. Calculate the values for the test data set.
4. Plot the graph for both the data sets and calculate for MAE, MSE and RMSE

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Tarun S
RegisterNumber:  212223040226
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("CSVs/student_scores.csv")
df.head()
df.tail()
X,Y=df.iloc[:,:-1].values, df.iloc[:,1].values
print(X)
print(Y)
from sklearn.model_selection import train_test_split as tts
Xtrain,Xtest,Ytrain,Ytest=tts(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression as lr
reg=lr()
reg.fit(Xtrain,Ytrain)
Ypred=reg.predict(Xtest)
print(Ypred)
plt.scatter(Xtrain,Ytrain,color="orange")
plt.plot(Xtrain,reg.predict(Xtrain),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(Xtest,Ytest,color="blue")
plt.plot(Xtest,reg.predict(Xtest),color="green")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
print("MSE : ",mean_squared_error(Ytest,Ypred))
print("MAE : ",mean_absolute_error(Ytest,Ypred))
print("RMSE : ",np.sqrt(mse))
```

## Output:

![image](https://github.com/Tarun-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145584190/370cb9dc-952c-43a1-b986-787ff3c535a0)


![image](https://github.com/Tarun-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145584190/24edab83-a06b-409b-b166-f73addc34b88)


![image](https://github.com/Tarun-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145584190/078e6101-29ec-4746-a3a8-a1238515561a)


![image](https://github.com/Tarun-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145584190/95f7c5bc-9a94-4866-9d99-0bbf19af7513)

![image](https://github.com/Tarun-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145584190/a5dce75d-6b40-4b3e-9a3b-67a0b5ce1332)

![image](https://github.com/Tarun-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145584190/73c5437b-2602-4e6f-9d12-35d9b04494f4)


![image](https://github.com/Tarun-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145584190/dafbe1c1-3cda-4247-8da3-ea412ecaf5aa)

![image](https://github.com/Tarun-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145584190/bd7ee5ce-fb4b-478e-87d3-15badd15c1d7)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

