# importing the Dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data collection and data Processing
sonar_data= pd.read_csv("Copy of sonar data.csv", header = None)

#seprating data and lables
X = sonar_data.drop(columns = 60, axis = 1) # dataset without rock or mine column
Y = sonar_data[60]  # dataset sonsist of rock or mine colum

# Spliting the training and testing data
# X_train -> training data    Y_train -> training label
# X_test  -> testing data     Y_test  -> testing data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify = Y, random_state = 1)

# Model ->> Logistic Regression  (Choosing the Correct Model)
model = LogisticRegression()

# Training the Logistic regression model with training data
model.fit(X_train, Y_train)

# Model Evaluation  ->> accuracy of training data  (data on which model is train)
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train,)
# print("accuracy on training data : ", training_data_accuracy)  #-> print accuracy

# Accuracy in test data (data unknown to model)
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test,)
# print("accuracy on testing data : ", testing_data_accuracy)  #-> print accuracy

# Making a predictive System where User enter the data
user_input = input("Enter the data : ")
input_data = tuple(float(x) for x in user_input.split(",")) 

# sample testing without user input
# input_data = (0.0412,0.1135,0.0518,0.0232,0.0646,0.1124,0.1787,0.2407,0.2682,0.2058,0.1546,0.2671,0.3141,0.2904,0.3531,0.5079,0.4639,0.1859,0.4474,0.4079,0.5400,0.4786,0.4332,0.6113,0.5091,0.4606,0.7243,0.8987,0.8826,0.9201,0.8005,0.6033,0.2120,0.2866,0.4033,0.2803,0.3087,0.3550,0.2545,0.1432,0.5869,0.6431,0.5826,0.4286,0.4894,0.5777,0.4315,0.2640,0.1794,0.0772,0.0798,0.0376,0.0143,0.0272,0.0127,0.0166,0.0095,0.0225,0.0098,0.0085)

# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
# print(prediction)

print("----------------------------")
if(prediction[0] == 'R'):
    print("It is a Rock")
else:
    print("It is a Mine")
print("----------------------------")