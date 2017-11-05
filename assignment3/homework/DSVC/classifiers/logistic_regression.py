import numpy as np
import random
import math

class LogisticRegression(object):

    def __init__(self):
        self.w = None

    def loss(self, X_batch, y_batch):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
        data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """

        #########################################################################
        # TODO:                                                                 #
        # calculate the loss and the derivative                                 #
        #########################################################################
        g = np.dot(X_batch, self.w)
        cost_pos = np.dot(y_batch.T, np.log(self.sigmoid(g)))
        cost_neg = np.dot((1-y_batch).T, np.log(1-self.sigmoid(g)))
        cost = - (cost_pos + cost_neg) / len(y_batch)
        grad = np.dot(X_batch.T, (self.sigmoid(g) - y_batch))
 
        return cost, grad
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def sigmoid(self, x):
    	return 1 / (1+np.exp(-x))

    def train(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose=True):

        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape	
	
        if self.w is None:
            self.w = 0.001 * np.random.randn(dim)

        loss_history = []

        for it in xrange(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            # np.random.seed(1024)
            # X_zero = X[y==0]
            # X_one = X[y==1]
            # y_zero = y[y==0]
            # y_one = y[y==1]
            # index_zero = np.random.choice(len(y_zero), batch_size/2, replace=False)
            # index_one = np.random.choice(len(y_one), batch_size-batch_size/2, replace=False)
            # X_batch = np.vstack((X_zero[index_zero], X_one[index_one]))
            # y_batch = np.vstack((y_zero[index_zero], y_one[index_one])).reshape(batch_size)
            index = np.random.choice(num_train, batch_size, replace=False)
            X_batch = X[index]
            y_batch = y[index]
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            self.w = self.w - grad * learning_rate / len(y_batch)
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: N x D array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
        array of length N, and each element is an integer giving the predicted
        class.
        """
        # y_pred = np.zeros(X.shape[1])?????????
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        if self.w.shape[0] == 10:
            result = self.sigmoid(np.dot(X, self.w.T))
            y_pred = np.argmax(result, axis = 1)
        else:
            result = self.sigmoid(np.dot(X, self.w))
            for i in np.arange(X.shape[0]):
                if result[i] >= 0.5:
                    y_pred[i] = 1
                else:
                    y_pred[i] = 0
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def one_vs_all(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose = True):
        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        """
        # if self.w is None:
        #     self.w = 0.001 * np.random.randn(X.shape[1])
        w_all = np.zeros(X.shape[1])
        loss_history = []

        for i in np.arange(10):
            self.w = 0.001 * np.random.randn(X.shape[1])
            y_train = np.copy(y)
            X_train = np.copy(X)
            for j in xrange(len(y_train)):
                if y_train[j] == i:
                    y_train[j] = 1
                else:
                    y_train[j] = 0
            for it in xrange(num_iters):
                # X_zero = X[y==0]
                # X_one = X[y==1]
                # y_zero = y[y==0]
                # y_one = y[y==1]
                # index_zero = np.random.choice(len(y_zero), batch_size/2, replace=False)
                # index_one = np.random.choice(len(y_one), batch_size-batch_size/2, replace=False)
                # X_batch = np.vstack((X_zero[index_zero], X_one[index_one]))
                # y_batch = np.vstack((y_zero[index_zero], y_one[index_one])).reshape(batch_size)
                index = np.random.choice(len(y), batch_size, replace=False)
                X_batch = X_train[index]
                y_batch = y_train[index]

                loss, grad = self.loss(X_batch, y_batch)
                loss_history.append(loss)
                self.w = self.w - grad * learning_rate / len(y_batch)

                if verbose and it % 100 == 0:
                    print 'iteration %d / %d: loss %f' % (it, num_iters, loss)
            w_all = np.vstack((w_all, self.w))

        w_all = w_all[1:, :]
        self.w = w_all
        
        return loss_history