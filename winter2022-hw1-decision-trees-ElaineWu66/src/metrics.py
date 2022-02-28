import numpy as np

def confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the confusion matrix. The confusion 
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    YOU DO NOT NEED TO IMPLEMENT CONFUSION MATRICES THAT ARE FOR MORE THAN TWO 
    CLASSES (binary).
    
    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """
    


    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    '''
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for i in range(0,len(predictions)):
        if(predictions[i] == actual[i] & actual[i]!=0):
            true_positives +=1
    for i in range(0,len(predictions)):
        if(predictions[i] == actual[i] & actual[i]==0):
            true_negatives +=1
    for i in range(0,len(predictions)):
        if(predictions[i] != actual[i] & actual[i]==0):
            false_positives +=1
    for i in range(0,len(predictions)):
        if(predictions[i] != actual[i] & actual[i]!=0):
            false_negatives +=1

    '''
    true_positives = sum(1 for i in range(0,len(predictions)) if ((predictions[i] == actual[i]) & (actual[i]!=0)))
    true_negatives = sum(1 for i in range(0,len(predictions)) if ((predictions[i] == actual[i]) & (actual[i]==0)))
    false_positives = sum(1 for i in range(0,len(predictions)) if ((predictions[i] != actual[i]) & (actual[i]==0)))
    false_negatives = sum(1 for i in range(0,len(predictions)) if ((predictions[i] != actual[i]) & (actual[i]!=0)))
    
    #print(np.array([[true_negatives, false_positives],
    #                 [false_negatives, true_positives]]))
    return np.array([[true_negatives, false_positives],
                     [false_negatives, true_positives]])

    '''
    for i in range(0,N):
        if predictions[i] == actual[i] and actual[i]:
            true_positive_number +=1

    '''
    #raise NotImplementedError()

def accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the accuracy:

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    confusion_Matrix = confusion_matrix(actual, predictions)
    accuracy = (confusion_Matrix[0][0]+confusion_Matrix[1][1]) / np.sum(confusion_Matrix)
    return accuracy
    #raise NotImplementedError()

def precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        precision (float): precision
        recall (float): recall
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    '''
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    '''
    confusion_Matrix = confusion_matrix(actual,predictions)
    precision = confusion_Matrix[1][1]/(confusion_Matrix[1][1]+confusion_Matrix[0][1])
    recall = confusion_Matrix[1][1]/(confusion_Matrix[1][1]+confusion_Matrix[1][0])

    return precision, recall
    #raise NotImplementedError()

def f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the F1-measure:

   https://en.wikipedia.org/wiki/Precision_and_recall#F-measure

    Hint: implement and use the precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and 
        recall)
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    
    precision, recall = precision_and_recall(actual,predictions)
    f1_measure = 2*precision*recall/(precision + recall)
    return f1_measure
    #raise NotImplementedError()

'''
a = np.array([1,0,1,0])
b = np.array([0,0,1,1])
confusion_matrix(a,b)
'''