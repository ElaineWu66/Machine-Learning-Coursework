import numpy as np 

def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    M = np.shape(X)[0]
    K = np.shape(X)[1]
    N = np.shape(Y)[0]

    D = np.zeros(shape=(M,N))
    for i in range(M):
        for j in range(N):
            for k in range(K):
                D[i][j] += ((X[i][k]-Y[j][k]) ** 2)
    
    D = np.sqrt(D)
    return D

    #raise NotImplementedError()


def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    M = np.shape(X)[0]
    K = np.shape(X)[1]
    N = np.shape(Y)[0]

    D = np.zeros(shape=(M,N))
    for i in range(M):
        for j in range(N):
            for k in range(K):
                D[i][j] += abs((X[i][k]-Y[j][k]))
    
    D = np.absolute(D)
    return D


    #raise NotImplementedError()

'''

X = np.array([[1,2,1,2],
              [0,1,2,1]])

Y = np.array([[1,2,3,4],
              [5,6,7,8],
              [2,2,2,2]])

print(manhattan_distances(X, Y))
'''
