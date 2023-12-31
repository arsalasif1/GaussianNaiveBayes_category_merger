# Naive Bayes Classification with Pishing Dataset
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.model_selection import train_test_split as tts

# load dataset
dataframe = read_csv("phishing.csv", header=0)
dataset = dataframe.values
# print(dataset[0])
# split into input (X) and output (Y) variables
dataset=dataset.astype(float)

################### Categotry merger######## Comment this out to unMerge
dataset[:,1:31][dataset[:,1:31] == -1] = 200 #Malicious
dataset[:,1:31][dataset[:,1:31] == 0] = 200  #Suspicious
dataset[:,1:31][dataset[:,1:31] == 1] = 100  #Ligitimate
preprocessing.normalize(dataset)
##################


X = dataset[:,1:31].astype(float)
# print(X[0]) #Check values
Y = dataset[:,31]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y) # transform -1's into 0
# print(encoded_Y)
# print(Y)

#Splitting Dataset for training
X_train , X_test,  y_train, y_test = tts(X,encoded_Y,test_size=0.25)

# print(X_train.shape)
# print(y_train.shape)


# Create and train the Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)
# print(X_train.shape)
# print(X_test[1].shape)


# Make predictions and calculate accuracy
s=0
for i in range(0,len(X_test)):
        predictions = model.predict(X_test[i].reshape(1, -1))-y_test[i]
        # print('predicted = %s, actual= %s', clf.predict(X_test[i].reshape(1, -1)),y_test[i])
        s=s+predictions
        
        # print(predictions)
print("Error = " + str((abs(s[0])/len(X_test))*100) +"%")