import numpy as np
import random
import matplotlib.pyplot as plt
from your_code import HingeLoss, SquaredLoss
from your_code import metrics
from your_code import GradientDescent, load_data
from your_code import L1Regularization, L2Regularization


class GradientDescentQ1:
    """
    This is a linear classifier similar to the one you implemented in the
    linear regressor homework. This is the classification via regression
    case. The goal here is to learn some hyperplane, y = w^T x + b, such that
    when features, x, are processed by our model (w and b), the result is
    some value y. If y is in [0.0, +inf), the predicted classification label
    is +1 and if y is in (-inf, 0.0) the predicted classification label is
    -1.

    The catch here is that we will not be using the closed form solution,
    rather, we will be using gradient descent. In your fit function you
    will determine a loss and update your model (w and b) using gradient
    descent. More details below.

    Arguments:
        loss - (string) The loss function to use. Either 'hinge' or 'squared'.
        regularization - (string or None) The type of regularization to use.
            One of 'l1', 'l2', or None. See regularization.py for more details.
        learning_rate - (float) The size of each gradient descent update step.
        reg_param - (float) The hyperparameter that controls the amount of
            regularization to perform. Must be non-negative.
    """

    def __init__(self, loss, regularization=None,
                 learning_rate=0.01, reg_param=0.05, question='1a'):
        self.learning_rate = learning_rate

        # Select regularizer
        if regularization == 'l1':
            regularizer = L1Regularization(reg_param)
        elif regularization == 'l2':
            regularizer = L2Regularization(reg_param)
        elif regularization is None:
            regularizer = None
        else:
            raise ValueError(
                'Regularizer {} is not defined'.format(regularization))

        # Select loss function
        if loss == 'hinge':
            self.loss = HingeLoss(regularizer)
        elif loss == 'squared':
            self.loss = SquaredLoss(regularizer)
        else:
            raise ValueError('Loss function {} is not defined'.format(loss))

        self.model = None
        self.question = question

    def fit(self, features, targets, batch_size=None, max_iter=1000):
        """
        Fits a gradient descent learner to the features and targets. The
        pseudocode for the fitting algorithm is as follow:
          - Initialize the model parameters to uniform random values in the
            interval [-0.1, +0.1].
          - While not converged:
            - Compute the gradient of the loss with respect to the current
              batch.
            - Update the model parameters by moving them in the direction
              opposite to the current gradient. Use the learning rate as the
              step size.
        For the convergence criteria, compute the loss over all examples. If
        this loss changes by less than 1e-4 during an update, assume that the
        model has converged. If this convergence criteria has not been met
        after max_iter iterations, also assume convergence and terminate.

        You should include a bias term by APPENDING a column of 1s to your
        feature matrix. The bias term is then the last value in self.model.

        Arguments:
            features - (np.array) An Nxd array of features, where N is the
                number of examples and d is the number of features.
            targets - (np.array) A 1D array of targets of length N.
            batch_size - (int or None) The number of examples used in each
                iteration. If None, use all of the examples in each update.
            max_iter - (int) The maximum number of updates to perform.
        Modifies:
            self.model - (np.array) A 1D array of model parameters of length
                d+1. The +1 refers to the bias term.
        """
        accuracy_list = []
        loss_list = []
        iteration_list = []
        self.model = np.zeros(features[0].shape)
        for i in range(len(self.model)):
            self.model[i] = random.uniform(-0.1, 0.1)
        iteration = 0
        loss = None
        while iteration < max_iter:
            if batch_size is None:
                sample_features = features
                sample_targets = targets
            else:
                sample_features = random.sample(features, batch_size)
                sample_targets = random.sample(targets, batch_size)

            self.model = self.model - self.learning_rate * \
                self.loss.backward(sample_features, self.model, sample_targets)
            new_loss = self.loss.forward(
                sample_features, self.model, sample_targets)
            if loss is not None and abs(new_loss - loss) < 1e-4:
                break
            loss = new_loss
            loss_list.append(loss)
            accuracy = metrics.accuracy(targets, self.predict(features))
            accuracy_list.append(accuracy)
            iteration_list.append(iteration)
            iteration += 1

        # data = np.c_[features, np.ones((features.shape[0], 1))] #add one column of 1 to features
        # w = np.random.uniform(-0.1, 0.1, data.shape[1])    #initialize parameters

        # batch = data[:]                    #initialize batch of data
        # batch_target = targets[:]          #initialize batch of labels

        # if batch_size != None:

        #     rand = np.random.randint(data.shape[0], size=batch_size)
        #     batch = data[:batch_size]
        #     batch_target = targets[:batch_size]
            
        #     #a good practice of randomly selecting data and corresponding label
        #     for j in range(batch_size):
        #         batch[j] = data[rand[j]]
        #         batch_target[j] = targets[rand[j]]

        # for i in range(max_iter):
            
        #     old = self.loss.forward(batch, w, batch_target)         #The calculated loss due to current parameters
        #     w -= self.loss.backward(batch, w, batch_target) * self.learning_rate    #update parameters
        #     new = self.loss.forward(batch, w, batch_target)         #The calculated loss due to updated parameters

        #     #converged
        #     if np.abs(new - old) < 1e-4:
        #         self.model = w
        #         break
        #     loss_list.append(new)
        #     accuracy = metrics.accuracy(batch_target, self.predict(features))
        #     accuracy_list.append(accuracy)
        #     iteration_list.append(i)
        # #reach preset maximum # of iteration
        
        # self.model = w


        plt.figure()
        plt.plot(iteration_list, loss_list, color='orange', label='Loss')
        plt.plot(iteration_list, accuracy_list, color='blue', label='Accuracy')
        plt.title('Loss & Accuracy Vs. Iteration No.')
        plt.xlabel('Iteration')
        plt.ylabel('Loss & Accuracy')
        plt.legend(loc="best")
        if self.question == '1a':
            plt.savefig("Q1a.png")
        if self.question == '1b':
            plt.savefig("Q1b.png")

    def predict(self, features):
        """
        Predicts the class labels of each example in features. Model output
        values at and above 0 are predicted to have label +1. Non-positive
        output values are predicted to have label -1.

        NOTE: your predict function should make use of your confidence
        function (see below).

        Arguments:
            features - (np.array) A Nxd array of features, where N is the
                number of examples and d is the number of features.
        Returns:
            predictions - (np.array) A 1D array of predictions of length N,
                where index d corresponds to the prediction of row N of
                features.
        """
        confidence = self.confidence(features)
        predictions = np.zeros(confidence.shape)
        for i in range(len(confidence)):
            predictions[i] = np.sign(confidence[i])

        return predictions

    def confidence(self, features):
        """
        Returns the raw model output of the prediction. In other words, rather
        than predicting +1 for values above 0 and -1 for other values, this
        function returns the original, unquantized value.

        Arguments:
            features - (np.array) A Nxd array of features, where N is the
                number of examples and d is the number of features.
        Returns:
            confidence - (np.array) A 1D array of confidence values of length
                N, where index d corresponds to the confidence of row N of
                features.
        """
        confidence = features.dot(self.model)
        return confidence


print('Question 1a')
max_iter = 1000
batch_size = None
fraction = 1
learning_rate = 1e-4
reg_param = 0.05
loss = 'hinge'
regularization = None

train_features, test_features, train_targets, test_targets = \
    load_data('mnist-binary', fraction=fraction)
learner = GradientDescentQ1(loss=loss, regularization=regularization,
                            learning_rate=learning_rate, reg_param=reg_param, question='1a')
learner.fit(train_features, train_targets)
predictions = learner.predict(test_features)

'''
print('Question 1b')
max_iter = 1000000
batch_size = 1
fraction = 1
learning_rate = 1e-4
reg_param = 0.05
loss = 'hinge'
regularization = None

train_features, test_features, train_targets, test_targets = \
    load_data('mnist-binary', fraction=fraction)
learner = GradientDescentQ1(loss=loss, regularization=regularization,
                            learning_rate=learning_rate, reg_param=reg_param, question='1b')
learner.fit(train_features, train_targets)
predictions = learner.predict(test_features)
'''