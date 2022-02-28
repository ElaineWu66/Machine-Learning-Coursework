from types import new_class
import numpy as np

class KMeans():
    def __init__(self, n_clusters):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.n_clusters = n_clusters
        self.means = None
    
    def measure_distance(self,data,mean):
        '''
        Args:
            data (np.ndarray): array containing inputs of size (1, n_features+1).
            mean (np.ndarray): array containing inputs of size (1, n_features).

        Returns:
            distance (int) Euclidean distance between mean and data(except last label column)
        '''
        feature = data[:-1]
        distance = np.linalg.norm(feature - mean)
        return distance

        #raise NotImplementedError()



    def update_assignments(self,data):
        '''
        Args:
            data (np.ndarray): array containing inputs of size (1, n_features+1).

        Returns:
            None. Update the last column of feat as the index of the cluster the data belong to
        '''
        group = 0
        curr_distance = self.measure_distance(data,self.means[0])
        for i in range(1,self.n_clusters):
            if (self.measure_distance(data,self.means[i]) < curr_distance):
                curr_distance = self.measure_distance(data,self.means[i])
                group = i
        data[-1] = group
        #raise NotImplementedError()
        

    def update_means(self,datas):
        '''
        Args:
            data (np.ndarray): array containing inputs of size (some_nonnegative_integer, n_features+1).

        Returns:
            None. Update the self.means
        '''
        # if there's no data, make no modification

        if np.shape(datas)[0] == 0:
            return 
        
        else:
            #print("datas= ", datas)
            group_index = datas[0][-1]
            group_index = int(group_index)
            #print("group_index= ", group_index)
            features = datas[:,:-1] # delete the last column (label column)
            #print("self.means= ", self.means)
            #print("XSHCSDH:", self.means[group_index])
            #print("WJYWJYWJY: ", np.mean(features, axis = 0))
            self.means[group_index] = np.mean(features, axis = 0)
            #print("self.means2= ", self.means)
        #raise NotImplementedError()
    
    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """

        n_samples = np.shape(features)[0]
        n_features = np.shape(features)[1]

        # update shape of means(np.ndarray)
        #self.means = np.zeros(shape=(self.n_clusters,n_features))

        '''
        #initialize self.means with random points
        shape of self.means: (self.n_clusters,n_features)
        '''
        random_indices = np.random.choice(n_samples, size=self.n_clusters, replace=False)
        self.means = features[random_indices, :]

        '''
        copy the features into feat, and append a new column to the numpy array to maintain the group each
        data is in
        '''
        label = np.zeros(shape = (n_samples,1))
        feat = np.append(features,label,axis=1)

        '''
        start to cluster the data
        '''
        last_time_means = np.copy(self.means)
        while True:
            for i in range(n_samples):
                self.update_assignments(feat[i])

            #update means group by group
            for i in range(self.n_clusters):
                selected_rows = feat[np.where(feat[:,-1] == i)]# select the rows with label i into selected rows
                self.update_means(selected_rows)
            if (last_time_means == self.means).all():
                break
            else:
                last_time_means = np.copy(self.means)
        #raise NotImplementedError()

    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """

        n_samples = np.shape(features)[0]
        n_features = np.shape(features)[1]

        '''
        copy the features into feat, and append a new column to the numpy array to maintain the group each
        data is in
        '''
        label = np.zeros(shape = (n_samples,1))
        feat = np.append(features,label,axis=1)

        '''
        start predictions
        '''
        for i in range(n_samples):
            self.update_assignments(feat[i])

        predictions = feat[:,-1]
        predictions.flatten()

        return predictions
        #raise NotImplementedError()