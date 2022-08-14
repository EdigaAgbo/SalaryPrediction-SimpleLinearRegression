# SalaryPrediction-SimpleLinearRegression
Simple linear regression model for salary prediction
# Problem Description 
The HR department of a company set up a project to review the salary of new employees. The task is to predict the salary range that is suitable for new employees based on their years of experience. A simple linear regression model is trained on existing salary data of employees within the company and is used to predict the salary of new employees based on their years of experience.

```python
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,[0]].values
y = dataset.iloc[:,[1]].values
```
![My Image](dataset.png)
```python
# Splitting the data into training and test set
from sklearn.model_selection import train_test_split
X_training_data, X_test_data, y_training_data, y_test_data = train_test_split(X, y, test_size=0.2, random_state = 0)

# Fitting the simple linear regression model on the training data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_training_data, y_training_data)
y_pred = regressor.predict(X_test_data)

# Visualising results of the training data
plt.scatter (X_training_data, y_training_data, color = 'red')
plt.plot (X_training_data, regressor.predict(X_training_data), color = 'blue')
plt.title ('Predicting salary based on years of experience (Training data)')
plt.xlabel ('Years of experience')
plt.ylabel ('Salary')
plt.show()
```
![My Image](training-data.png)

```python 
# Visualising results of the test data
plt.scatter (X_test_data, y_test_data, color = 'red')
plt.plot (X_training_data, regressor.predict(X_training_data), color = 'blue')
plt.title ('Predicting salary based on years of experience (Test data)')
plt.xlabel ('Years of experience')
plt.ylabel ('Salary')
plt.show()
```
![My Image](test-data.png)
# Discussion and recommendations

The regression model trained on the salary data generally showed the closeness of the predicted salaries to the actual salaries of employees - with the blue linear line showing the predicted salaries and the actual salaries represented by the red points. This is model is trained on the test data which shows the expected salaries of new employees. Overall, the model acurately predicts the salary range that new employees are entitled to based on their years of experience.
