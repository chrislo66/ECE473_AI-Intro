import collections
import math
import numpy as np

class Gaussian_Naive_Bayes():
    def fit(self, X_train, y_train):
        """
        fit with training data
        Inputs:
            - X_train: A numpy array of shape (N, D) containing training data; there are N
                training samples each of dimension D.
            - y_train: A numpy array of shape (N,) containing training labels; y[i] = c
                means that X[i] has label 0 <= c < C for C classes.
                
        With the input dataset, function gen_by_class will generate class-wise mean and variance to implement bayes inference.

        Returns:
        None
        
        """
        self.x = X_train
        self.y = y_train

        self.gen_by_class()

    def gen_by_class(self):
        """
        With the given input dataset (self.x, self.y), generate 3 dictionaries to calculate class-wise mean and variance of the data.
        - self.x_by_class : A dictionary of numpy arraies with the keys as each class label and values as data with such label.
        - self.mean_by_class : A dictionary of numpy arraies with the keys as each class label and values as mean of the data with such label.
        - self.std_by_class : A dictionary of numpy arraies with the keys as each class label and values as standard deviation of the data with such label.
        - self.y_prior : A numpy array of shape (C,) containing prior probability of each class
        """
        self.x_by_class = dict()
        self.mean_by_class = dict()
        self.std_by_class = dict()
        self.y_prior = []

        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Generate dictionaries.
        # hint : to see all unique y labels, you might use np.unique function, e.g., np.unique(self.y)

    # Each Class
        for i in range(len(np.unique(self.y))):
            classType = np.unique(self.y)[i]
            if classType not in self.x_by_class:
                self.x_by_class[classType] = []
            if classType not in self.mean_by_class:
                self.mean_by_class[classType] = []
            if classType not in self.std_by_class:
                self.std_by_class[classType] = []

        # Sort Sample to Each Class
            for sampleNum in range(len(self.x)):
                if self.y[sampleNum] == classType:
                    self.x_by_class[classType].append(self.x[sampleNum])
            NumSameClass = len(self.x_by_class[classType])
            # Important
            self.x_by_class[classType] = np.array(self.x_by_class[classType])
        # Calculate self.y_prior for Each Class
            self.y_prior.append(NumSameClass / len(self.y))

        self.y_prior = np.array(self.y_prior)


        for j in range(len(np.unique(self.y))):
            classT = np.unique(self.y)[j]

        # Calculate Mean and Std of Each Feature
            NumFeatures = self.x.shape[1]
            for countFeature in range(NumFeatures):
            # Mean of Each Feature
                mean_a_feature = self.mean(self.x_by_class[classT][:, countFeature])
                self.mean_by_class[classT].append(mean_a_feature)

            # Std of Each Feature
                std_a_feature = self.std(self.x_by_class[classT][:, countFeature])
                self.std_by_class[classT].append(std_a_feature)

            self.mean_by_class[classT] = np.array(self.mean_by_class[classT])
            self.std_by_class[classT] = np.array(self.std_by_class[classT])

        #print(self.x_by_class)
        #print(self.mean_by_class)
        #print(self.std_by_class)
        #print(self.y_prior)

        pass;

        # END_YOUR_CODE
        ############################################################
        ############################################################        

    def mean(self, x):
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate mean of input x
        mean = sum(x) / (len(x))
        pass;

        # END_YOUR_CODE
        ############################################################
        ############################################################
        return mean

    def std(self, x):
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate standard deviation of input x, do not use np.std
        # x: a feature vector of all samples
        mean = sum(x) / (len(x))
        diff = 0
        for num in x:
            diff += (num-mean)**2
        std = 1E-3 + np.sqrt(diff / (len(x) - 1))
        pass;
        # END_YOUR_CODE
        ############################################################
        ############################################################
        return std

    def calc_gaussian_dist(self, x, mean, std):
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # calculate gaussian probability of input x given mean and std
        gaussian = ((1 / (np.sqrt(2 * np.pi) * std + 1E-3)) * np.exp(-(x-mean)**2 / (2 * std**2 + 1E-3)))
        pass;
        # END_YOUR_CODE
        ############################################################
        ############################################################
        return gaussian

    def predict(self, x):
        """
        Use the acquired mean and std for each class to predict class for input x.
        Inputs:

        Returns:
        - prediction: Predicted labels for the data in x. prediction is (N, C) dimensional array, for N samples and C classes.
        """

        n = len(x)
        num_class = len(np.unique(self.y))
        prediction = np.zeros((n, num_class))

        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # calculate naive bayes probability of each class of input x
        for idxClass in range(num_class):
            classType = np.unique(self.y)[idxClass]
            for countSample in range(n):
                prediction[countSample, idxClass] = np.log((1E-7 + self.y_prior[idxClass]))
                for countFeature in range(x.shape[1]):
                    prediction[countSample, idxClass] += np.log((1E-7 + self.calc_gaussian_dist(x[countSample, countFeature], self.mean_by_class[classType][countFeature], self.std_by_class[classType][countFeature])))

        pass;

        # END_YOUR_CODE
        ############################################################
        ############################################################

        return prediction


class Neural_Network():
    def __init__(self, hidden_size = 64, output_size = 1):
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        self.hidden_size = hidden_size
        self.output_size = output_size

    def fit(self, x, y, batch_size = 64, iteration = 2000, learning_rate = 1e-3):
        """
        Train this 2 layered neural network classifier using mini-batch stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - iteration: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        
        Use the given learning_rate, iteration, or batch_size for this homework problem.

        Returns:
        None
        """
        dim = x.shape[1]
        num_train = x.shape[0]

        #initialize W
        if self.W1 == None:
            self.W1 = 0.001 * np.random.randn(dim, self.hidden_size)
            self.b1 = 0

            self.W2 = 0.001 * np.random.randn(self.hidden_size, self.output_size)
            self.b2 = 0


        for it in range(iteration):
            batch_ind = np.random.choice(num_train, batch_size)

            x_batch = x[batch_ind]
            y_batch = y[batch_ind]

            loss, gradient = self.loss(x_batch, y_batch)

            ############################################################
            ############################################################
            # BEGIN_YOUR_CODE
            # Update parameters with mini-batch stochastic gradient descent method
            self.W1 -= gradient['dW1'] * learning_rate
            self.b1 -= gradient['db1'] * learning_rate
            self.W2 -= gradient['dW2'] * learning_rate
            self.b2 -= gradient['db2'] * learning_rate

            pass;

            # END_YOUR_CODE
            ############################################################
            ############################################################

            y_pred = self.predict(x_batch)
            acc = np.mean(y_pred == y_batch)

            if it % 50 == 0:
                print('iteration %d / %d: accuracy : %f: loss : %f' % (it, iteration, acc, loss))

    def loss(self, x_batch, y_batch, reg = 1e-3):
            """
            Implement feed-forward computation to calculate the loss function.
            And then compute corresponding back-propagation to get the derivatives. 

            Inputs:
            - X_batch: A numpy array of shape (N, D) containing a minibatch of N
              data points; each point has dimension D. (64,30)
            - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
            - reg: hyperparameter which is weight of the regularizer. (64,1)

            Returns: A tuple containing:
            - loss as a single float
            - gradient dictionary with four keys : 'dW1', 'db1', 'dW2', and 'db2'
            """
            gradient = {'dW1' : None, 'db1' : None, 'dW2' : None, 'db2' : None}


            ############################################################
            ############################################################
            # BEGIN_YOUR_CODE
            # Calculate y_hat which is probability of the instance is y = 0.
            g1 = x_batch.dot(self.W1) + self.b1
            h1 = self.activation(g1)
            g2 = np.dot(h1, self.W2) + self.b2
            y_hat = self.sigmoid(g2)
            pass;

            # END_YOUR_CODE
            ############################################################
            ############################################################


            ############################################################
            ############################################################
            # BEGIN_YOUR_CODE
            # Calculate loss and gradient
            loss_sum_part = np.dot(np.log((y_hat.T + 1E-7)), y_batch) + np.dot(np.log((1 - y_hat.T + 1E-7)), (1 - y_batch))
            loss = -(loss_sum_part / np.abs(x_batch.shape[0])) + reg * 0.5 * (np.sum(np.multiply(self.W1, self.W1)) + np.sum(np.multiply(self.W2, self.W2)))


            y_hat[range(x_batch.shape[0]), range(y_batch.shape[1])] -= 1
            #dw2 = np.dot(h1.T, y_hat) / x_batch.shape[0] + 2 * reg * self.W2
            #db2 = np.sum(y_hat, axis=0, keepdims=True) / x_batch.shape[0]

            delta = y_hat.dot(self.W2.T)
            delta = delta * (h1 > 0)
            dw1 = np.dot(x_batch.T, delta) / x_batch.shape[0] + 2 * reg * self.W1
            db1 = np.sum(delta, axis=0, keepdims=True) / x_batch.shape[0] + 2 * reg * self.b1

            dw2 = np.dot(h1.T, (y_hat - y_batch)) / x_batch.shape[0] + 2 * reg * self.W2
            db2 = np.sum(y_hat - y_batch) / x_batch.shape[0] + 2 * reg * self.b2
            #summation = np.zeros([1, h1.shape[1]])
            #dw1 = np.dot(np.dot((y_hat - y_batch).T, x_batch).T, summation) + 2 * reg * self.W1
            #db1 = (np.mean(y_hat) - np.mean(y_batch)) * np.sum(summation) + 2 * reg * self.b1

            gradient['dW1'] = dw1
            gradient['dW2'] = dw2
            gradient['db1'] = db1
            gradient['db2'] = db2
            pass;

            # END_YOUR_CODE
            ############################################################
            ############################################################
            return loss, gradient

    def activation(self, z):
        """
        Compute the ReLU output of z
        Inputs:
        z : A scalar or numpy array of any size.
        Return:
        s : output of ReLU(z)
        """
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Implement ReLU 
        if z.any() > 0:
            s = z.any()
        else:
            s = 0
        pass;

        # END_YOUR_CODE
        ############################################################
        ############################################################

        return s

    def sigmoid(self, z):
        """
        Compute the sigmoid of z
        Inputs:
        z : A scalar or numpy array of any size.
        Return:
        s : sigmoid of input
        """
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        s = 1/(1 + np.exp(-z))
        pass;

        # END_YOUR_CODE
        ############################################################
        ############################################################

        return s

    def predict(self, x):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.
        Inputs:

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate predicted y
        g1 = x.dot(self.W1) + self.b1
        h1 = self.activation(g1)
        g2 = np.dot(h1, self.W2) + self.b2
        y_hat = self.sigmoid(g2)
        if y_hat.any() > 0.5:
            y_pred = 1
        else:
            y_pred = 0
        pass;

        # END_YOUR_CODE
        ############################################################
        ############################################################
        return y_pred

