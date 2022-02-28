import numpy as np 
from .distances import euclidean_distances, manhattan_distances
from scipy import stats
import copy


class KNearestNeighbor():    
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator='mode'):
        """
        K-Nearest Neighbor is a straightforward algorithm that can be highly
        effective. Training time is...well...is there any training? At test time, labels for
        new points are predicted by comparing them to the nearest neighbors in the
        training data.

        ```distance_measure``` lets you switch between which distance measure you will
        use to compare data points. The behavior is as follows:

        If 'euclidean', use euclidean_distances, if 'manhattan', use manhattan_distances.

        ```aggregator``` lets you alter how a label is predicted for a data point based 
        on its neighbors. If it's set to `mean`, it is the mean of the labels of the
        neighbors. If it's set to `mode`, it is the mode of the labels of the neighbors.
        If it is set to median, it is the median of the labels of the neighbors. If the
        number of dimensions returned in the label is more than 1, the aggregator is
        applied to each dimension independently. For example, if the labels of 3 
        closest neighbors are:
            [
                [1, 2, 3], 
                [2, 3, 4], 
                [3, 4, 5]
            ] 
        And the aggregator is 'mean', applied along each dimension, this will return for 
        that point:
            [
                [2, 3, 4]
            ]

        Arguments:
            n_neighbors {int} -- Number of neighbors to use for prediction.
            distance_measure {str} -- Which distance measure to use. Can be one of
                'euclidean' or 'manhattan'. This is the distance measure
                that will be used to compare features to produce labels. 
            aggregator {str} -- How to aggregate a label across the `n_neighbors` nearest
                neighbors. Can be one of 'mode', 'mean', or 'median'.
        """
        self.n_neighbors = n_neighbors
        self.distance_measure = distance_measure
        self.aggregator = aggregator
        self.data_features = np.array([])
        self.data_targets = np.array([])

        #raise NotImplementedError()


    def fit(self, features, targets):
        """Fit features, a numpy array of size (n_samples, n_features). For a KNN, this
        function should store the features and corresponding targets in class 
        variables that can be accessed in the `predict` function. Note that targets can
        be multidimensional! 
        
        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            targets {[type]} -- Target labels for each data point, shape of (n_samples, 
                n_dimensions).
        """

        '''
        for row in features:
            self.data_features = np.append(self.data_features,row)
        '''
        self.data_features = copy.deepcopy(features)

        '''
        for row in targets:
            self.data_targets = np.append(self.data_targets,row)
        '''
        self.data_targets = copy.deepcopy(targets)

        #raise NotImplementedError()
        

    def predict(self, features, ignore_first=False):
        """Predict from features, a numpy array of size (n_samples, n_features) Use the
        training data to predict labels on the test features. For each testing sample, compare it
        to the training samples. Look at the self.n_neighbors closest samples to the 
        test sample by comparing their feature vectors. The label for the test sample
        is the determined by aggregating the K nearest neighbors in the training data.

        Note that when using KNN for imputation, the predicted labels are the imputed testing data
        and the shape is (n_samples, n_features).

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            ignore_first {bool} -- If this is True, then we ignore the closest point
                when doing the aggregation. This is used for collaborative
                filtering, where the closest point is itself and thus is not a neighbor. 
                In this case, we would use 1:(n_neighbors + 1).

        Returns:
            labels {np.ndarray} -- Labels for each data point, of shape (n_samples,
                n_dimensions). This n_dimensions should be the same as n_dimensions of targets in fit function.
        """
        n_samples = np.shape(features)[0]
        n_features = np.shape(features)[1]

        label_dimension = np.shape(self.data_targets)[1]

        labels = np.zeros(shape=(n_samples,label_dimension))

        #calculate distance of self.data_features and features
        # distance {np.ndarray}: np.shape(self.data_features)[0] x np.shape(self.features)[0] matrix 
        # with distances between rows of self.data_features and rows of features.
        if self.distance_measure == 'euclidean':
            distance = euclidean_distances(features,self.data_features)
        elif self.distance_measure == 'manhattan':
            distance = manhattan_distances(features,self.data_features)
        
        for row in range(n_samples):

            #get the labels value row by row
            '''index: a 1d list to keep track of the index of most close training_features:
               e.g: self.n_neighbors = 3, then index is a 1*3 array as [5,10,19]       
            '''

            index = distance[row].argsort()[:self.n_neighbors]   # e.g: index = np.array([5,19,10])
            for j in range(label_dimension):

                values = []
                for i in index:
                    values.append(self.data_targets[i][j])
                values = np.array(values)

                if self.aggregator == 'mean':
                    labels[row][j] = np.mean(values)
                    '''
                    sum = 0
                    for i in index:
                        sum += self.data_targets[i][j]
                    labels[row][j] = sum / self.n_neighbors
                    '''
                elif self.aggregator == 'mode':
                    
                    labels[row][j] = stats.mode(values)[0]

                elif self.aggregator == 'median':
                    labels[row][j] = np.median(values)
        

        return labels
        #raise NotImplementedError()


