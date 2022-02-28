from .decision_tree import DecisionTree
from .prior_probability import PriorProbability
from .metrics import precision_and_recall, confusion_matrix, f1_measure, accuracy
from .data import load_data, train_test_split

def run(data_path, learner_type, fraction):
    #print("REACHED RUN")
    """
    This function walks through an entire machine learning workflow as follows:

        1. takes in a path to a dataset
        2. loads it into a numpy array with `load_data`
        3. instantiates the class used for learning from the data using learner_type (e.g
           learner_type is 'decision_tree', 'prior_probability')
        4. splits the data into training and testing with `train_test_split` and `fraction`.
        5. trains a learner using the training split with `fit`
        6. tests the trained learner using the testing split with `predict`
        7. evaluates the trained learner with precision_and_recall, confusion_matrix, and
           f1_measure

    Each run of this function constitutes a trial. Your learner should be pretty
    robust across multiple runs, as long as `fraction` is sufficiently high. See how
    unstable your learner gets when less and less data is used for training by
    playing around with `fraction`.

    IMPORTANT:
    If fraction == 1.0, then your training and testing sets should be exactly the
    same. This is so that the test cases are deterministic. The test case checks if you
    are fitting the training data correctly, rather than checking for generalization to
    a testing set.

    Args:
        data_path (str): path to csv file containing the data
        learner_type (str): either 'decision_tree' or 'prior_probability'. For each of these,
            the associated learner is instantiated and used for the experiment.
        fraction (float between 0.0 and 1.0): fraction of examples to be drawn for training

    Returns:
        confusion_matrix (np.array): Confusion matrix of learner on testing examples
        accuracy (np.float): Accuracy on testing examples using learner
        precision (np.float): Precision on testing examples using learner
        recall (np.float): Recall on testing examples using learner
        f1_measure (np.float): F1 Measure on testing examples using learner
    """
    # 1. takes in a path to a dataset
    '''
    print("data_path= ", data_path)
    print("learner type= ", learner_type)
    print("fraction = ", fraction)
    '''
    # 2. loads it into a numpy array with `load_data`
    features, targets, attribute_names = load_data(data_path)  #features (np.array): numpy array of size NxK containing the K features
                                                               #targets (np.array): numpy array of size 1xN containing the N targets.
                                                               #attribute_names (list): list of strings containing names of each attribute(headers of csv)
    '''
    print("features length: ",len(features))
    print("targets length: ", len(targets))
    print("attributesname length: ", len(attribute_names))
    '''
    # 3. instantiates the class used for learning from the data using learner_type
    if learner_type == 'prior_probability':
        mymodel = PriorProbability()
        #print("original mostcommon class: ", mymodel.most_common_class)
    
    elif learner_type == 'decision_tree':
        mymodel = DecisionTree(attribute_names)

    # 4. split data
    train_features, train_targets, test_features, test_targets = train_test_split(features,targets,fraction)
    '''
    print("train_features length: ", len(train_features))
    print("train_targets length: ", len(train_targets))
    print("test features length: ", len(test_features))
    print("test_targets length: ", len(test_targets))
    '''
    # 5. train a model with the split training data
    mymodel.fit(train_features,train_targets)
    #print("mymodel.most_common_class: ", mymodel.most_common_class)
    
    #print("train_features: ", train_features)
    #print("train_targets: ", train_targets)
    # 6. tests the trained learner using the testing split with `predict`
    mypredictions = mymodel.predict(test_features)
    #print("length of test_features: ", len(test_features))
    #print("mypredictions: ", mypredictions)
    # 7. evaluates the trained learner with precision_and_recall, confusion_matrix, and f1_measure
    myprecision, recall = precision_and_recall(test_targets,mypredictions)
    myconfusion_matrix = confusion_matrix(test_targets,mypredictions)
    my_f1_measure = f1_measure(test_targets,mypredictions)
    my_accuracy = accuracy(test_targets,mypredictions)


    if learner_type == 'decision_tree':
        mymodel.visualize()
    
    #raise NotImplementedError()

    # Order of these returns must be maintained
    return myconfusion_matrix, my_accuracy, myprecision, recall, my_f1_measure
