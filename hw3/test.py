import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs

import hw3_submission as hw3

import numpy as np
import os


seed = 222
np.random.seed(seed)

data_cancer = load_breast_cancer()
X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(data_cancer.data, data_cancer.target, test_size=0.33, random_state=seed)
#print("yTrainCancer" )
#print(y_train_cancer)
#print(np.unique(y_train_cancer))
#print(np.unique(y_train_cancer)[0])

data_digit = load_digits()
X_train_digit, X_test_digit, y_train_digit, y_test_digit = train_test_split(data_digit.data, data_digit.target, test_size=0.33, random_state=seed)
#print("yTrainDigit")
#print(y_train_digit)
#print(np.unique(y_train_digit))
#print(len(np.unique(y_train_digit)))

def test_nn(X_train, X_test, y_train, y_test):
    lr = hw3.Neural_Network()
    lr.fit(X_train, y_train.reshape(-1,1))
    
    acc = np.mean(lr.predict(X_test) == y_test.reshape(-1,1))
    print('test accuracy (yours) : {:.3f}'.format(acc))
    
    LR = LogisticRegression(solver='liblinear')
    LR.fit(X_train, y_train)
    
    y_hat = LR.predict(X_test)
    acc_LR = (np.mean(y_hat == y_test))
    print('test accuracy (sklearn package): {:.3f}'.format(acc_LR))
    
    
def test_bayes(X_train, X_test, y_train, y_test):
    nb = hw3.Gaussian_Naive_Bayes()
    nb.fit(X_train, y_train)
    
    acc = np.mean(nb.predict(X_test).argmax(-1) == y_test)
    print('test accuracy (yours) : {:.3f}'.format(acc))
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)

    acc_BN = (np.mean(y_hat == y_test))
    print('test accuracy (sklearn package): {:.3f}'.format(acc_BN))

    
    
if __name__ == '__main__':
    print(' ######## Test Neural Network (Cancer) ######## ')
    test_nn(X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer)
    
    print(' ######## Test Gaussian Naive Bayes (Cancer) ######## ')
    test_bayes(X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer)
    
    print(' ######## Test Gaussian Naive Bayes (Digits) ######## ')
    test_bayes(X_train_digit, X_test_digit, y_train_digit, y_test_digit)

    
    