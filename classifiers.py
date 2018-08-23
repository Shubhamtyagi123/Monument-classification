import numpy as np
from sklearn import neighbors,linear_model
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB 

imageSet = []
labelSet = []

imageSet = np.load('info.npy')
labelSet = np.load('lb.npy')

sample_i = len(imageSet)
sample_l = len(labelSet)

imageSet_train = imageSet[:int(0.8*sample_i)] 
imageSet_test = imageSet[int(0.8*sample_i):]

labelSet_train = labelSet[:int(0.8*sample_l)]
labelSet_test = labelSet[int(0.8*sample_l):]

kNN = neighbors.KNeighborsClassifier()
Logistic = linear_model.LogisticRegression()

print('KNN : %f' % kNN.fit(imageSet_train, labelSet_train).score(imageSet_test, labelSet_test))
print('LogisticRegression : %f' % Logistic.fit(imageSet_train, labelSet_train).score(imageSet_train, labelSet_train))

#print ("[+] processing...")
classify = svm.SVC(kernel='rbf', C=1.0)
classify.fit(imageSet_train,labelSet_train)
pred = classify.predict(imageSet_test)
acurracy = accuracy_score(labelSet_test, pred)
print "Linear SVM accuracy :",acurracy

classify1 = GaussianNB()
classify1.fit(imageSet_train,labelSet_train)
pred1 = classify1.predict(imageSet_test)
accuracy1 = accuracy_score(labelSet_test, pred1)
print "Naive bayes : ",accuracy1