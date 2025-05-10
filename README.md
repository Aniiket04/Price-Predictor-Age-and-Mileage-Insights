# Price-Predictor-Age-and-Mileage-Insights

## Project Overview 

**Project Title: Price-Predictor-Age-and-Mileage-Insights**

The primary goal of this project is to build a machine learning model that predicts the selling price of a vehicle (e.g., a car) based on two key factors: its age and mileage. The project focuses on leveraging historical data to enable accurate and actionable price predictions for buyers, sellers, and market analysts.

## Objectives

To predict selling price of a vehicle (e.g a car) based on its age and mileage using Linear regression.

## Project Structure

### 1. Importing Libraries
numpy for numerical operations
pandas for data manipulation
sklearn (from scikit-learn) for machine learning tools
```python
import numpy as np
import pandas as pd
from sklearn import linear_model
```

### 2. Loading the Dataset
The given dataset is loaded using pandas.read_csv().This dataset contains data about Mileage, age(input features) and selling prices(target variable) of vehicles
```python
df=pd.read_csv("Data3.csv")
df.columns
df
```

### 3. Data processing
Split the dataset into features(x) and target (y) to prepare for machine learning tasks.
```python
x=df[['Mileage ','Age(yrs)']]
y=df['Sell Price($)']
x
y
```
This creates a new DataFrame, x, containing only the independent variables (features):
Mileage : The mileage of the vehicle.
Age(yrs): The age of the vehicle in years.

### 4. Train/Test Split
Divide the dataset into two parts:
a. Training set : Used to train the model
b. Testing set : Used to evaluate the model's performance on unseen data.
```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
len(x_train)
len(x_test)
```
test_size=0.2 function reserves 20% of the dataset for testing and uses the remaining 80% for training.

### 5.Model Training
Training a Linear Regression model using sklearn.linear_model.LinearRegression
Fitting the model on the training data
```python
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
y_test
```
### 6. Model Prediction
Predictions on the held-out dataset.
```python
model.predict([[60000,4]])
y
x
model.score(x_test,y_test)
```
The model.score() function evaluates the performance of a trained model.

## Conclusion
The project successfully developed a machine learning model capable of predicting the selling price of a vehicle (e.g., a car) based on its age and mileage. This tool can assist stakeholders in making informed pricing decisions with high accuracy.

## Author - Aniket Pal
This project is part of my portfolio, showcasing the machine learning skills essential for data science roles.

-**LinkedIn**: [ www.linkedin.com/in/aniket-pal-098690204 ]
-**Email**: [ aniketspal04@gmail.com ]

