import numpy as np

class PriorProbability():
    def __init__(self):
        """
        This is a simple classifier that only uses prior probability to classify 
        points. It just looks at the classes for each data point and always predicts
        the most common class.

        """
        self.most_common_class = None

    def fit(self, features, targets):
        """
        Implement a classifier that works by prior probability. Takes in features
        and targets and fits the features to the targets using prior probability.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N 
                examples.
        Output:
            VOID: You should be updating self.most_common_class with the most common class
            found from the prior probability.
        """
        numberoftrue = sum(1 for i in targets if i==1)
        numberoffalse = len(targets)-numberoftrue
        #print(numberoftrue)
        #print(numberoffalse)
        if numberoftrue>numberoffalse:
            self.most_common_class = 1
        else:
            self.most_common_class = 0

        #raise NotImplementedError()

    def predict(self, data):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        Outputs:
            predictions (np.array): numpy array of size N array which has the predicitons 
            for the input data.
        """
        predictions = np.full(len(data),self.most_common_class, dtype=np.int)
        return predictions        
        #raise NotImplementedError()

'''
features = np.array([1,1,1])
targets = np.array([1,0])

data_new = data.load_data("data/candy-data.csv")[0]
#print(data_new)
mypriorprobability = PriorProbability()
mypriorprobability.fit(features,targets)
mypriorprobability.predict(data_new)
'''
