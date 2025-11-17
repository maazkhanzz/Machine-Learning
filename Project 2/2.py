import pandas
from sklearn import linear_model
import matplotlib.pyplot as plt

"""
PART 1: basic linear regression
The goal is to predict the profit of a restaurant, based on the number of habitants where the restaurant 
is located. The chain already has several restaurants in different cities. Your goal is to model 
the relationship between the profit and the populations from the cities where they are located.
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html 
"""

# Open the csv file RegressionData.csv in Excel, notepad++ or any other applications to have a 
# rough overview of the data at hand. 
# You will notice that there are several instances (rows), of 2 features (columns). 
# The values to be predicted are reported in the 2nd column.

# Load the data from the file RegressionData.csv in a pandas dataframe. Make sure all the instances 
# are imported properly. Name the first feature 'X' and the second feature 'y' (these are the labels)
data = pandas.read_csv("RegressionData.csv", header = None , names=['X', 'y']) 
# Reshape the data so that it can be processed properly
X = data['X'].values.reshape(-1,1) 
y = data['y'] 
# Plot the data using a scatter plot to visualize the data
plt.scatter(data['X'], data['y']) 

# Linear regression using least squares optimization
reg = linear_model.LinearRegression() 
reg.fit(X,y) 

# Plot the linear fit
fig = plt.figure()
y_pred = reg.predict(X) 
plt.scatter(X,y, c='b') 
plt.plot(X, y_pred, 'r') 
fig.canvas.draw()

# # Complete the following print statement (replace the blanks _____ by using a command, do not hard-code the values):
print("The linear relationship between X and y was modeled according to the equation: y = b_0 + X*b_1, \
where the bias parameter b_0 is equal to ", reg.intercept_, " and the weight b_1 is equal to ", reg.coef_)


# Predict the profit of a restaurant, if this restaurant is located in a city of 18 habitants 
print("the profit/loss in a city with 18 habitants is ", reg.predict([[18]]))

    
"""
PART 2: logistic regression 
You are a recruiter and your goal is to predict whether an applicant is likely to get hired or rejected. 
You have gathered data over the years that you intend to use as a training set. 
Your task is to use logistic regression to build a model that predicts whether an applicant is likely to
be hired or not, based on the results of a first round of interview (which consisted of two technical questions).
The training instances consist of the two exam scores of each applicant, as well as the hiring decision.
"""

# Open the csv file in Excel, notepad++ or any other applications to have a rough overview of the data at hand. 

# Load the data from the file 'LogisticRegressionData.csv' in a pandas dataframe. Make sure all the instances 
# are imported properly. Name the first feature 'Score1', the second feature 'Score2', and the class 'y'
data = pandas.read_csv("LogisticRegressionData.csv", header = None, names=['Score1', 'Score2', 'y'])

# Seperate the data features (score1 and Score2) from the class attribute 
X = data[['Score1', 'Score2']] 
y = data['y'] 

# Plot the data using a scatter plot to visualize the data. 
# Represent the instances with different markers of different colors based on the class labels.
m = ['o', 'x']
c = ['hotpink', '#88c999']
fig = plt.figure()
for i in range(len(data)):
    plt.scatter(data['Score1'][i], data['Score2'][i], marker= m[data['y'][i]], color = c[data['y'][i]])
fig.canvas.draw()

# Train a logistic regression classifier to predict the class labels y using the features X
regS = linear_model.LogisticRegression()
regS.fit(X, y)

# Now, we would like to visualize how well does the trained classifier perform on the training data
# Use the trained classifier on the training data to predict the class labels
y_pred = regS.predict(X)
# To visualize the classification error on the training instances, we will plot again the data. However, this time,
# the markers and colors selected will be determined using the predicted class labels
m = ['o', 'x']
c = ['red', 'blue'] #this time in red and blue
fig = plt.figure()
for i in range(len(data)):
    plt.scatter(data['Score1'][i], data['Score2'][i], marker= m[y_pred[i]], color = c[y_pred[i]])
fig.canvas.draw()
# Notice that some of the training instances are not correctly classified. These are the training errors.

"""
PART 3: Multi-class classification using logistic regression 
Not all classification algorithms can support multi-class classification (classification tasks with more than two classes).
Logistic Regression was designed for binary classification.
One approach to alleviate this shortcoming, is to split the dataset into multiple binary classification datasets 
and fit a binary classification model on each. 
Two different examples of this approach are the One-vs-Rest and One-vs-One strategies.
"""

#  One-vs-Rest method (a.k.a. One-vs-All)

# Explain below how the One-vs-Rest method works for multi-class classification
"""
In the One-vs-Rest approach, we train one model for each class. Each model learns to tell whether
a sample belongs to that class or to any of the others combined. So if there are three possible 
classes, we will have three separate logistic regression models. When we make a prediction for a new 
sample, every model gives a probability that the sample belongs to its class, and we simply pick 
the class with the highest probability.It works well with logistic regression, and does not require 
any major changes to the algorithm. Overall, this is a practical choice when you have a moderate number 
of classes and a dataset large enough to train separate models efficiently.
"""

# Explain below how the One-Vs-One method works for multi-class classification
"""
The One-vs-One approach works a bit differently.Instead of comparing each class against all the others, 
it builds one model for every possible pair of classes.So if there are three classes (A, B, and C), 
we train three small models, one for A vs B, one for A vs C, and one for B vs C. When we predict a new 
sample, each model votes for one of its two classes, and the class that gets the most votes overall 
becomes the final prediction.This method takes longer to train since there are more models, but each 
model is simpler and can sometimes give better results when the classes are hard to separate. 
Thi approach is often preferred when the dataset is not too large or when the classes overlap, since 
smaller pair models can sometimes make cleaner and concise decisions.
"""
