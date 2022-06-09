import collections
import math
import numpy as np


class Logistic_Regression():
    def __init__(self):
        self.W = None
        self.b = None
    
        
    def fit(self, x, y, batch_size = 64, iteration = 2000, learning_rate = 1e-2):
        """
        Train this Logistic Regression classifier using mini-batch stochastic gradient descent.
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
        if self.W == None:
            self.W = 0.001 * np.random.randn(dim, 1)
            self.b = 0


        for it in range(iteration):
            batch_ind = np.random.choice(num_train, batch_size)

            x_batch = x[batch_ind]
            y_batch = y[batch_ind]

            ############################################################
            ############################################################
            # BEGIN_YOUR_CODE
            # Calculate loss and update W, b 

            pass;

            # END_YOUR_CODE
            ############################################################
            ############################################################
            
            if it % 50 == 0:
                print('iteration %d / %d: accuracy : %f: loss : %f' % (it, iteration, acc, loss))

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

        pass;

        # END_YOUR_CODE
        ############################################################
        ############################################################
        return y_pred

    
    def loss(self, x_batch, y_pred, y_batch):
        """
        Compute the loss function and its derivative. 
        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.

        Returns: A tuple containing:
        - loss as a single float
        - gradient dictionary with two keys : 'dW' and 'db'
        """
        gradient = {'dW' : None, 'db' : None}
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate loss and gradient

        pass;

        # END_YOUR_CODE
        ############################################################
        ############################################################
        return loss, gradient

    
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
        # Calculate loss and update W 

        pass;

        # END_YOUR_CODE
        ############################################################
        ############################################################
        
        return s

class Naive_Bayes():
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
        self.y_prior = None
        
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Generate dictionaries.
        # hint : to see all unique y labels, you might use np.unique function, e.g., np.unique(self.y)

        pass

        # END_YOUR_CODE
        ############################################################
        ############################################################        

    def mean(self, x):
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate mean of input x
        pass
        # END_YOUR_CODE
        ############################################################
        ############################################################
        return mean
    
    def std(self, x):
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate standard deviation of input x, do not use np.std
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
        pass;
        # END_YOUR_CODE
        ############################################################
        ############################################################
        
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
        
        pass
        # END_YOUR_CODE
        ############################################################
        ############################################################
        
        return prediction

    
class Spam_Naive_Bayes(object):
    """Implementation of Naive Bayes for Spam detection."""
    def clean(self, s):
        translator = str.maketrans("", "", string.punctuation)
        return s.translate(translator)
 
    def tokenize(self, text):
        text = self.clean(text).lower()
        return re.split("\W+", text)
 
    def get_word_counts(self, words):
        """
        Generate a dictionary 'word_counts' 
        Hint: You can use helper function self.clean and self.toeknize.
              self.tokenize(x) can generate a list of words in an email x.

        Inputs:
            -words : list of words that is used in a data sample
        Output:
            -word_counts : contains each word as a key and number of that word is used from input words.
        """
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # calculate naive bayes probability of each class of input x
        
        pass
        # END_YOUR_CODE
        ############################################################
        ############################################################
            
        return word_counts

    def fit(self, X_train, y_train):
        """
        compute likelihood of all words given a class

        Inputs:
            -X_train : list of emails
            -y_train : list of target label (spam : 1, non-spam : 0)
            
        Variables:
            -self.num_messages : dictionary contains number of data that is spam or not
            -self.word_counts : dictionary counts the number of certain word in class 'spam' and 'ham'.
            -self.class_priors : dictionary of prior probability of class 'spam' and 'ham'.
        Output:
            None
        """
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # calculate naive bayes probability of each class of input x
        
        pass
        # END_YOUR_CODE
        ############################################################
        ############################################################
                
    def predict(self, X):
        """
        predict that input X is spam of not. 
        Given a set of words {x_i}, for x_i in an email(x), if the likelihood 
        
        p(x_0|spam) * p(x_1|spam) * ... * p(x_n|spam) * y(spam) > p(x_0|ham) * p(x_1|ham) * ... * p(x_n|ham) * y(ham),
        
        then, the email would be spam.

        Inputs:
            -X : list of emails

        Output:
            -result : A numpy array of shape (N,). It should tell rather a mail is spam(1) or not(0).
        """
            
        result = []
        for x in X:
            ############################################################
            ############################################################
            # BEGIN_YOUR_CODE
            # calculate naive bayes probability of each class of input x

            pass
            # END_YOUR_CODE
            ############################################################
            ############################################################
            
        result = np.array(result)
        return result
    