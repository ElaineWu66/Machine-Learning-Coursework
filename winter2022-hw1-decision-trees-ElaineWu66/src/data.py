import numpy as np 
import os
import csv
import copy
import random

def load_data(data_path):
    """
    Associated test case: tests/test_data.py

    Reading and manipulating data is a vital skill for machine learning.

    This function loads the data in data_path csv into two numpy arrays:
    features (of size NxK) and targets (of size Nx1) where N is the number of rows
    and K is the number of features. 
    
    data_path leads to a csv comma-delimited file with each row corresponding to a 
    different example. Each row contains binary features for each example 
    (e.g. chocolate, fruity, caramel, etc.) The last column indicates the label for the
    example how likely it is to win a head-to-head matchup with another candy 
    bar.

    This function reads in the csv file, and reads each row into two numpy arrays.
    The first array contains the features for each row. For example, in candy-data.csv
    the features are:

    chocolate,fruity,caramel,peanutyalmondy,nougat,crispedricewafer,hard,bar,pluribus

    The second array contains the targets for each row. The targets are in the last 
    column of the csv file (labeled 'class'). The first row of the csv file contains 
    the labels for each column and shouldn't be read into an array.

    Example:
    chocolate,fruity,caramel,peanutyalmondy,nougat,crispedricewafer,hard,bar,pluribus,class
    1,0,1,0,0,1,0,1,0,1

    should be turned into:

    [1,0,1,0,0,1,0,1,0] (features) and [1] (targets).

    This should be done for each row in the csv file, making arrays of size NxK and Nx1.

    Args:
        data_path (str): path to csv file containing the data

    Output:
        features (np.array): numpy array of size NxK containing the K features
        targets (np.array): numpy array of size 1xN containing the N targets.
        attribute_names (list): list of strings containing names of each attribute 
            (headers of csv)
    """

    # Implement this function and remove the line that raises the error after.

    file_data = np.genfromtxt(data_path, delimiter = ',')
    #print(file_data)
    
    lenth = len(file_data[0])
    features = copy.deepcopy(file_data)
    features = np.delete(features,(0),axis=0)
    features = np.delete(features,(lenth-1),axis=1)

    targets = file_data[1:][:]
    targets = targets[:,-1]
    #print(len(targets))


    attribute_names = np.genfromtxt(data_path, delimiter=',', dtype=str, max_rows=1)
    attribute_names = attribute_names.tolist()
    attribute_names = attribute_names[:-1]
    #print(attribute_names)    
    '''
    file_data = open(data_path)
    print(type(file_data))
    features = copy.deepcopy(file_data)

    for row in features:
        print(row)
    '''

    return features, targets, attribute_names

    #raise NotImplementedError()

def train_test_split(features, targets, fraction):
    """
    Split features and targets into training and testing, randomly. N points from the data 
    sampled for training and (features.shape[0] - N) points for testing. Where N:

        N = int(features.shape[0] * fraction)
    
    Returns train_features (size NxK), train_targets (Nx1), test_features (size MxK 
    where M is the remaining points in data), and test_targets (Mx1).
    
    Special case: When fraction is 1.0. Training and test splits should be exactly the same. 
    (i.e. Return the entire feature and target arrays for both train and test splits)

    Args:
        features (np.array): numpy array containing features for each example
        targets (np.array): numpy array containing labels corresponding to each example.
        fraction (float between 0.0 and 1.0): fraction of examples to be drawn for training

    Returns
        train_features: subset of features containing N examples to be used for training.
        train_targets: subset of targets corresponding to train_features containing targets.
        test_features: subset of features containing M examples to be used for testing.
        test_targets: subset of targets corresponding to test_features containing targets.
    """
    if (fraction > 1.0):
        raise ValueError('N cannot be bigger than number of examples!')
    if (fraction == 1.0):
        return features,targets,features,targets
    N = int(features.shape[0] * fraction)

    random_index = random.sample(range(0,features.shape[0]),N)  #random_index is an array of N unique numbers within the range of valid index
    complement_index = [i for i in range(0,features.shape[0]) if i not in random_index]

    #print(random_index)
    #print(complement_index)

    train_features = features[random_index]
    train_targets = targets[random_index]

    test_features = features[complement_index]
    test_targets = targets[complement_index]


    #print(len(train_features))
    return train_features,train_targets,test_features,test_targets
    #raise NotImplementedError()


#data = load_data("data/candy-data.csv")
#train_test_split(data[0],data[1],0.25)