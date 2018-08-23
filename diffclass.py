import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from nolearn.dbn import DBN
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
# from KNN import Knn

accuracies=[]
labels = []

def result(Y_test,pred):
    correct=0
    for i in range(len(Y_test)):
        if(Y_test[i]==pred[i]):
            correct+=1
    total=len(Y_test)
    return (float(correct)/total*100)

def randomforest(X_train, X_test, y_train,y_test):
    st = "RF"
    print "Random Forest"
    # labels.append(st)
    clf_rf = RandomForestClassifier()
    clf_rf.fit(X_train, y_train)
    y_pred_rf = clf_rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    accuracies.append(acc_rf*100.0)

def neuralnetwork(X_train, X_test, y_train, y_test):
    st = "NN"
    print "Neural Network"
    # labels.append(st)
    clf_nn = DBN(learn_rates=0.3,learn_rate_decays=0.9,epochs=15)
    clf_nn.fit(X_train, y_train)
    acc_nn = clf_nn.score(X_test,y_test)
    accuracies.append(acc_nn*100)

def kNearestNeighbor(X_train,X_test,y_train,y_test):
    st = "kNN"
    print st
    # labels.append(st)
    knn = KNeighborsClassifier()
    knn.fit(X_train,y_train)
    pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test,pred)*100)

def multilayerperceptron(X_train,X_test,Y_train,Y_test):
    st = "SNN"
    print("Supervised Neural Network")
    # labels.append(st)
    MLP=MLPClassifier(activation="relu",alpha=1e-05,momentum=0.12)
    MLP.fit(X_train,Y_train)
    pred=MLP.predict(X_test)
    accuracies.append(result(Y_test,pred))

def quadraticdiscriminant(X_train,X_test,Y_train,Y_test):
    st = "QDA"
    print("Quadratic Discriminant Analysis")
    # labels.append(st)
    qd = QuadraticDiscriminantAnalysis()
    qd.fit(X_train,Y_train)
    prediction = qd.predict(X_test)
    accuracies.append(result(prediction,Y_test))

def naivebayes(X_train,X_test,Y_train,Y_test,mode):
    st = "MNB"
    print("Naive Bayes")
    # labels.append(st)
    st = "GNB"
    # labels.append(st)
    naivebayes = object 
    if mode == 'M':
        naivebayes = MultinomialNB()
    elif mode == 'G':
        naivebayes = GaussianNB()
    naivebayes.fit(X_train,Y_train)
    prediction = naivebayes.predict(X_test)
    accuracies.append(result(prediction,Y_test))

def supportvectormachine(X_train,X_test,Y_train,Y_test):
    print("Support Vector Machine")
    #labels.append("SVM")
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    sVm = svm.SVC(gamma=2,C=0.01,decision_function_shape='ovo',kernel='rbf')
    sVm.fit(X_train,Y_train)
    pred=sVm.predict(X_test)
    accuracies.append(result(Y_test,pred))
    
def main():
    X = np.load('X.npy')
    y = np.load('Y.npy')
    X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.025)

    randomforest(X_train, X_test, y_train,y_test)
    neuralnetwork(X_train, X_test, y_train, y_test)
    kNearestNeighbor(X_train, X_test, y_train,y_test)  
    quadraticdiscriminant(X_train, X_test, y_train,y_test)
    # ans = Knn()
    # accuracies.append(ans)
    multilayerperceptron(X_train, X_test, y_train, y_test)
    naivebayes(X_train, X_test, y_train,y_test,"M")
    naivebayes(X_train, X_test, y_train,y_test,"G")
    supportvectormachine(X_train, X_test, y_train,y_test)
    print(accuracies)

    plt.style.use('ggplot')
    #plt.subplots(figsize(1000,900))
    plt.figure()
    x = [0,1,2,3,4,5,6,7]
    labels = ["RF","NN","kNN","SNN","QDA","MNB","GNB","SVM"]
    plt.xticks(x, labels)
    plt.bar(x,accuracies,width=0.5,align='center')
    plt.title("Accuracies of Classifiers")
    plt.xlabel("Classifiers")
    plt.ylabel("Accuracies")
    plt.savefig("plot-acc.png")
    
if __name__ == "__main__":
    main()

# fig_size = plt.rcParams["figure.figsize"]
 
# # Prints: [8.0, 6.0]
# print "Current size:", fig_size
 
# # Set figure width to 12 and height to 9
# fig_size[0] = 12
# fig_size[1] = 9
# plt.rcParams["figure.figsize"] = fig_size