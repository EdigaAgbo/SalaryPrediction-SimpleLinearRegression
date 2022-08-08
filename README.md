# SalaryPrediction-SimpleLinearRegression
Simple linear regression model for salary prediction
# Importing libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,[0]].values
y = dataset.iloc[:,[1]].values
```
<img src="/EdigaAgbo/SalaryPrediction-SimpleLinearRegression/dataset.png" alt="Alt text" title="Optional title">
