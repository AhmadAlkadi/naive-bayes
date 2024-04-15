#-------------------------------------------------------------------------
# AUTHOR: Ahmad Alkadi
# FILENAME: naive_bayes
# SPECIFICATION: naive_bayes predection
# FOR: CS 5990- Assignment #3
# TIME SPENT: 30 min
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

#reading the training data
#--> add your Python code here
dataSets = ['weather_training.csv', 'weather_test.csv']
dfOne = pd.read_csv(dataSets[0])
data_training = np.array(dfOne.values)[:,1:].astype('f')
dfTwo = pd.read_csv(dataSets[1])
data_test = np.array(dfTwo.values)[:,1:].astype('f')
x_training = data_training[:,:-1]
y_training = data_training[:,-1]

#update the training class values according to the discretization (11 values only)
#--> add your Python code here
def difference_from_value(x, value):
    return abs(x - value)
def find_nearest_classes(values, classes):
    nearest_classes = []
    for value in values:
        differences = [difference_from_value(x, value) for x in classes]
        nearest_class = classes[differences.index(min(differences))]
        nearest_classes.append(nearest_class)
    return nearest_classes
y_training = find_nearest_classes(y_training, classes)

#reading the test data
#--> add your Python code here
x_test = data_test[:,:-1]
y_test = data_test[:,-1]

#update the test class values according to the discretization (11 values only)
#--> add your Python code here
y_test = find_nearest_classes(y_test, classes)

#fitting the naive_bayes to the data
clf = GaussianNB()
clf = clf.fit(x_training, y_training)

#make the naive_bayes prediction for each test sample and start computing its accuracy
#the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
#to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
#--> add your Python code here
count =0
for (real_value,predicted_value) in zip(clf.predict(x_test),y_test):
    percentage_difference = 100*(abs(predicted_value - real_value)/real_value)
    if(percentage_difference<=15):
        count+=1

#print the naive_bayes accuracyy
#--> add your Python code here
accuracy = count / len(x_test)
print("naive_bayes accuracy: " + str(accuracy))



