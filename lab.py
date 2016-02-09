import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import tree
from sklearn.naive_bayes import GaussianNB



training = pd.read_csv('training.txt', sep=',', header=None)
testing = pd.read_csv('testing.txt',sep=',', header=None)

features_train = training.iloc[:, :-1]
labels_train = training.iloc[:, -1]

features_test = testing.iloc[:, :-1]
labels_test = testing.iloc[:, -1]

#-------------- Question 1a--------------------

#Support Vector Machine
#We setup the classifiers first
clf = SVC()
#perform the training:
clf.fit(features_train, labels_train)
#perform the actual classifications or predictions
pred = clf.predict(features_test)

print("SVM:")
print (accuracy_score(pred,labels_test)*100)

# Decision Tree

clf  = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
print("Decision Tree:")
print(accuracy_score(pred, labels_test)*100)

# Gaussian Naive Bayes

clf = GaussianNB()
clf.fit(features_train, labels_train)

pred= clf.predict(features_test)
print("Gaussian Naive Bayes:")
print(accuracy_score(pred, labels_test)*100)

#------------Question 1b------------

#50 examples

#First, we get 50 examples of each class separately
featuresC0 = training.loc[training[64] == 0].sample(n=50)
featuresC1 = training.loc[training[64] == 1].sample(n=50)
featuresC2 = training.loc[training[64] == 2].sample(n=50)
featuresC3 = training.loc[training[64] == 3].sample(n=50)
featuresC4 = training.loc[training[64] == 4].sample(n=50)
featuresC5 = training.loc[training[64] == 5].sample(n=50)
featuresC6 = training.loc[training[64] == 6].sample(n=50)
featuresC7 = training.loc[training[64] == 7].sample(n=50)
featuresC8 = training.loc[training[64] == 8].sample(n=50)
featuresC9 = training.loc[training[64] == 9].sample(n=50)

#and get the labels
labelsC0= featuresC0.iloc[:, -1]
labelsC1= featuresC1.iloc[:, -1]
labelsC2= featuresC2.iloc[:, -1]
labelsC3= featuresC3.iloc[:, -1]
labelsC4= featuresC4.iloc[:, -1]
labelsC5= featuresC5.iloc[:, -1]
labelsC6= featuresC6.iloc[:, -1]
labelsC7= featuresC7.iloc[:, -1]
labelsC8= featuresC8.iloc[:, -1]
labelsC9= featuresC9.iloc[:, -1]

labels_train50 =pd.concat([labelsC0,labelsC1,labelsC2,labelsC3,labelsC4,labelsC5,labelsC6,labelsC7,labelsC8,labelsC9])

features_train50 =  pd.concat([featuresC0,featuresC1,featuresC2,featuresC3,featuresC4,featuresC5,featuresC6,featuresC7,
                        featuresC8,featuresC9])

features_train50 = features_train50.iloc[:, :-1]

#And we perform the same classifications as in question 1a but with the new training set

print('\n')
print("with 50 examples from each class:")

#Support Vector Machine
clf = SVC()
clf.fit(features_train50, labels_train50)

pred = clf.predict(features_test)

print("SVM:")
print (accuracy_score(pred,labels_test)*100)

# Decision Tree

clf  = tree.DecisionTreeClassifier()
clf = clf.fit(features_train50, labels_train50)

pred = clf.predict(features_test)
print("Decision Tree:")
print(accuracy_score(pred, labels_test)*100)

# Gaussian Naive Bayes

clf = GaussianNB()
clf.fit(features_train50, labels_train50)

pred= clf.predict(features_test)
print("Gaussian Naive Bayes:")
print(accuracy_score(pred, labels_test)*100)

#-------------------100 examples-------------------

#First, we get 100 examples of each class separately
featuresC0 = training.loc[training[64] == 0].sample(n=100)
featuresC1 = training.loc[training[64] == 1].sample(n=100)
featuresC2 = training.loc[training[64] == 2].sample(n=100)
featuresC3 = training.loc[training[64] == 3].sample(n=100)
featuresC4 = training.loc[training[64] == 4].sample(n=100)
featuresC5 = training.loc[training[64] == 5].sample(n=100)
featuresC6 = training.loc[training[64] == 6].sample(n=100)
featuresC7 = training.loc[training[64] == 7].sample(n=100)
featuresC8 = training.loc[training[64] == 8].sample(n=100)
featuresC9 = training.loc[training[64] == 9].sample(n=100)

labelsC0= featuresC0.iloc[:, -1]
labelsC1= featuresC1.iloc[:, -1]
labelsC2= featuresC2.iloc[:, -1]
labelsC3= featuresC3.iloc[:, -1]
labelsC4= featuresC4.iloc[:, -1]
labelsC5= featuresC5.iloc[:, -1]
labelsC6= featuresC6.iloc[:, -1]
labelsC7= featuresC7.iloc[:, -1]
labelsC8= featuresC8.iloc[:, -1]
labelsC9= featuresC9.iloc[:, -1]

labels_train =pd.concat([labelsC0,labelsC1,labelsC2,labelsC3,labelsC4,labelsC5,labelsC6,labelsC7,labelsC8,labelsC9])

features_train =  pd.concat([featuresC0,featuresC1,featuresC2,featuresC3,featuresC4,featuresC5,featuresC6,featuresC7,
                        featuresC8,featuresC9])

features_train = features_train.iloc[:, :-1]

print('\n')
print("with 100 examples from each class:")

#Support Vector Machine
clf = SVC()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

print("SVM:")
print (accuracy_score(pred,labels_test)*100)

# Decision Tree

clf  = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
print("Decision Tree:")
print(accuracy_score(pred, labels_test)*100)

# Gaussian Naive Bayes

clf = GaussianNB()
clf.fit(features_train, labels_train)

pred= clf.predict(features_test)
print("Gaussian Naive Bayes:")
print(accuracy_score(pred, labels_test)*100)

#-------------------200 examples-------------------

#First, we get 200 examples of each class separately
featuresC0 = training.loc[training[64] == 0].sample(n=200)
featuresC1 = training.loc[training[64] == 1].sample(n=200)
featuresC2 = training.loc[training[64] == 2].sample(n=200)
featuresC3 = training.loc[training[64] == 3].sample(n=200)
featuresC4 = training.loc[training[64] == 4].sample(n=200)
featuresC5 = training.loc[training[64] == 5].sample(n=200)
featuresC6 = training.loc[training[64] == 6].sample(n=200)
featuresC7 = training.loc[training[64] == 7].sample(n=200)
featuresC8 = training.loc[training[64] == 8].sample(n=200)
featuresC9 = training.loc[training[64] == 9].sample(n=200)

labelsC0= featuresC0.iloc[:, -1]
labelsC1= featuresC1.iloc[:, -1]
labelsC2= featuresC2.iloc[:, -1]
labelsC3= featuresC3.iloc[:, -1]
labelsC4= featuresC4.iloc[:, -1]
labelsC5= featuresC5.iloc[:, -1]
labelsC6= featuresC6.iloc[:, -1]
labelsC7= featuresC7.iloc[:, -1]
labelsC8= featuresC8.iloc[:, -1]
labelsC9= featuresC9.iloc[:, -1]

labels_train =pd.concat([labelsC0,labelsC1,labelsC2,labelsC3,labelsC4,labelsC5,labelsC6,labelsC7,labelsC8,labelsC9])

features_train =  pd.concat([featuresC0,featuresC1,featuresC2,featuresC3,featuresC4,featuresC5,featuresC6,featuresC7,
                        featuresC8,featuresC9])

features_train = features_train.iloc[:, :-1]

print('\n')
print("with 200 examples from each class:")

#Support Vector Machine
clf = SVC()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

print("SVM:")
print (accuracy_score(pred,labels_test)*100)

# Decision Tree

clf  = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
print("Decision Tree:")
print(accuracy_score(pred, labels_test)*100)

# Gaussian Naive Bayes

clf = GaussianNB()
clf.fit(features_train, labels_train)

pred= clf.predict(features_test)
print("Gaussian Naive Bayes:")
print(accuracy_score(pred, labels_test)*100)

#-------------------300 examples-------------------

#First, we get 300 examples of each class separately
featuresC0 = training.loc[training[64] == 0].sample(n=300)
featuresC1 = training.loc[training[64] == 1].sample(n=300)
featuresC2 = training.loc[training[64] == 2].sample(n=300)
featuresC3 = training.loc[training[64] == 3].sample(n=300)
featuresC4 = training.loc[training[64] == 4].sample(n=300)
featuresC5 = training.loc[training[64] == 5].sample(n=300)
featuresC6 = training.loc[training[64] == 6].sample(n=300)
featuresC7 = training.loc[training[64] == 7].sample(n=300)
featuresC8 = training.loc[training[64] == 8].sample(n=300)
featuresC9 = training.loc[training[64] == 9].sample(n=300)

labelsC0= featuresC0.iloc[:, -1]
labelsC1= featuresC1.iloc[:, -1]
labelsC2= featuresC2.iloc[:, -1]
labelsC3= featuresC3.iloc[:, -1]
labelsC4= featuresC4.iloc[:, -1]
labelsC5= featuresC5.iloc[:, -1]
labelsC6= featuresC6.iloc[:, -1]
labelsC7= featuresC7.iloc[:, -1]
labelsC8= featuresC8.iloc[:, -1]
labelsC9= featuresC9.iloc[:, -1]

labels_train =pd.concat([labelsC0,labelsC1,labelsC2,labelsC3,labelsC4,labelsC5,labelsC6,labelsC7,labelsC8,labelsC9])

features_train =  pd.concat([featuresC0,featuresC1,featuresC2,featuresC3,featuresC4,featuresC5,featuresC6,featuresC7,
                        featuresC8,featuresC9])

features_train = features_train.iloc[:, :-1]

print('\n')
print("with 300 examples from each class:")

#Support Vector Machine
clf = SVC()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

print("SVM:")
print (accuracy_score(pred,labels_test)*100)

# Decision Tree

clf  = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
print("Decision Tree:")
print(accuracy_score(pred, labels_test)*100)

# Gaussian Naive Bayes

clf = GaussianNB()
clf.fit(features_train, labels_train)

pred= clf.predict(features_test)
print("Gaussian Naive Bayes:")
print(accuracy_score(pred, labels_test)*100)

