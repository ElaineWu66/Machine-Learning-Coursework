import numpy as np

class Node():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Node classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

class DecisionTree():
    def __init__(self, attribute_names):
        """
        TODO: Implement this class.

        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Node classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)
        
        """
        self.attribute_names = attribute_names
        self.tree = None

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        Output:
            VOID: It should update self.tree with a built decision tree.
        """
        self._check_input(features)

        #create root node tree
        self.tree = Node()

        unused_attributes_indexes = list(range(len(features[0])))
        self.extend_tree(unused_attributes_indexes,features,targets,self.tree)

        #raise NotImplementedError()
    

    #unused_attributes: (list of integer) a list containing the index of attribute_name that hasn't been used
    #remaining_features: (2d numpy array with binary value) remaining features been devided into current group
    #current_node: the current_node being extended
    def extend_tree(self,unused_attributes_indexes,remaining_features,remaining_targets,current_node):
        #   If number of predicting attributes is empty, then Return the single node tree Root, 
        #   with label = most common value of the target attribute in the examples.
        if  not unused_attributes_indexes:
            numberoftrue = sum(1 for i in remaining_targets if i==1)
            numberoffalse = len(remaining_targets)-numberoftrue
            if numberoftrue>numberoffalse:
                current_node.value = 1
            else:
                current_node.value = 0 

        #   Otherwise Begin
        #       A ← The Attribute that best classifies examples.
        #       Decision Tree attribute for Root = A.
        #       For each possible value, vi, of A,
        #           Add a new tree branch below Root, corresponding to the test A = vi.
        #           Let Examples(vi) be the subset of examples that have the value vi for A
        #               If Examples(vi) is empty
        #                   Then below this new branch add a leaf node with label = most common target value in the examples
        #               Else below this new branch add the subtree ID3 (Examples(vi), Target_Attribute, Attributes – {A})
        #   End
        #   Return Root

        else:
            #   If all examples are positive, Return the single-node tree Root, with label = +.
            if all(remaining_targets):
                current_node.value = 1

            #   If all examples are negative, Return the single-node tree Root, with label = -.
            elif np.all((remaining_targets == 0)):
                current_node.value = 0
    
            else:
                #print("unused_attributes_indexes: ", unused_attributes_indexes)
                bestIndex = unused_attributes_indexes[0]
                #print("bestIndex: ", bestIndex)
                for i in unused_attributes_indexes:
                    if information_gain(remaining_features, i, remaining_targets) > information_gain(remaining_features,bestIndex,remaining_targets):
                        bestIndex = i  #update the bestIndex

                A = self.attribute_names[bestIndex]
                
                #update current_node with the best attribute we found for it just now
                current_node.attribute_name = A
                current_node.attribute_index = bestIndex

                #create 2 child node and save them in current_node.branches
                negative_node = Node()
                positive_node = Node()

                #initialize baseline node.value
                negative_node.value = np.random.choice(np.arange(0, 2), p=[0.5, 0.5])
                positive_node.value = np.random.choice(np.arange(0, 2), p=[0.5, 0.5])

                current_node.branches.append(negative_node)
                current_node.branches.append(positive_node)

                #unused attributes for child nodes
                #print("unused_attributes_indexes= ",unused_attributes_indexes)
                #print("best index =  ", bestIndex)                
                new_unused_attributes_indexes = unused_attributes_indexes.copy()
                #print("new_unused_attributes_indexes", unused_attributes_indexes)
                new_unused_attributes_indexes.remove(bestIndex)

                #split datas for negative_node and positive_node

                negative_remaining_features = remaining_features[np.where(remaining_features[:,bestIndex] == 0)]
                positive_remaining_features = remaining_features[np.where(remaining_features[:,bestIndex] == 1)]
                
                negative_remaining_targets = remaining_targets[np.where(remaining_features[:,bestIndex] == 0)]
                positive_remaining_targets = remaining_targets[np.where(remaining_features[:,bestIndex] == 1)]

                #recursively construct child node
                if len(negative_remaining_features) != 0: 
                    self.extend_tree(new_unused_attributes_indexes,negative_remaining_features,negative_remaining_targets,negative_node)

                if len(positive_remaining_features) != 0:
                    self.extend_tree(new_unused_attributes_indexes,positive_remaining_features,positive_remaining_targets,positive_node)            


    def predict(self, features):
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
        self._check_input(features)

        
        predictions = np.array([])
        for datapoint in features:
            current_node = self.tree

            #hasn't reach a leaf node
            while (len(current_node.branches) != 0):
                #go left
                if(datapoint[current_node.attribute_index] == 0 ):
                    current_node = current_node.branches[0]
                #go right
                else:
                    current_node = current_node.branches[1]

            #now current_node is leaf node, its value is decision
                #print("current_node.value= ", current_node.value)
            predictions = np.append(predictions,current_node.value)
            #predictions.append(current_node.value)
        #print("predictions = ",predictions)
        return predictions
        #raise NotImplementedError()

    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else 0
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)

def information_gain(features, attribute_index, targets):
    """
    TODO: Implement me!

    Information gain is how a decision tree makes decisions on how to create
    split points in the tree. Information gain is measured in terms of entropy.
    The goal of a decision tree is to decrease entropy at each split point as much as
    possible. This function should work perfectly or your decision tree will not work
    properly.

    Information gain is a central concept in many machine learning algorithms. In
    decision trees, it captures how effective splitting the tree on a specific attribute
    will be for the goal of classifying the training data correctly. Consider
    data points S and an attribute A. S is split into two data points given binary A:

        S(A == 0) and S(A == 1)

    Together, the two subsets make up S. If A was an attribute perfectly correlated with
    the class of each data point in S, then all points in a given subset will have the
    same class. Clearly, in this case, we want something that captures that A is a good
    attribute to use in the decision tree. This something is information gain. Formally:

        IG(S,A) = H(S) - H(S|A)

    where H is information entropy. Recall that entropy captures how orderly or chaotic
    a system is. A system that is very chaotic will evenly distribute probabilities to
    all outcomes (e.g. 50% chance of class 0, 50% chance of class 1). Machine learning
    algorithms work to decrease entropy, as that is the only way to make predictions
    that are accurate on testing data. Formally, H is defined as:

        H(S) = sum_{c in (classes in S)} -p(c) * log_2 p(c)

    To elaborate: for each class in S, you compute its prior probability p(c):

        (# of elements of class c in S) / (total # of elements in S)

    Then you compute the term for this class:

        -p(c) * log_2 p(c)

    Then compute the sum across all classes. The final number is the entropy. To gain
    more intution about entropy, consider the following - what does H(S) = 0 tell you
    about S?

    Information gain is an extension of entropy. The equation for information gain
    involves comparing the entropy of the set and the entropy of the set when conditioned
    on selecting for a single attribute (e.g. S(A == 0)).

    For more details: https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics

    Args:
        features (np.array): numpy array containing features for each example.
        attribute_index (int): which column of features to take when computing the
            information gain
        targets (np.array): numpy array containing labels corresponding to each example.

    Output:
        information_gain (float): information gain if the features were split on the
            attribute_index.
    """
    # 1.calculate H(s) (entropy of the whole set before dividing)
    ptrue = sum(i for i in targets if i==1) / len(targets)
    pfalse = 1-ptrue
    H_s = -ptrue * np.log2(ptrue) - pfalse * np.log2(pfalse)
    '''
    print("ptrue= ",ptrue)
    print("pfalse= ",pfalse)
    print("H_s= ",H_s)
    '''
    # 2. calculate weight1, weight2, entropy1, entropy2 after dividing
    subset1_index = []   # a list containg the index of the rows whose value is true of the attribute_index column
    subset0_index = []
    for i in range(0,len(features)):
        if features[i][attribute_index] == 1:
            subset1_index.append(i)
    '''
    print("attribute_index= ",attribute_index)
    print("features: ", features)
    print("subset1_index: ", subset1_index)
    '''
    for i in range(0,len(features)):
        if i not in subset1_index:
            subset0_index.append(i)
    #subset0_index = [i for i in range(len(features)) and i not in subset1_index]  

    #special case: if value under column index is all 0 or 1, then there is no information gain after dividing
    #as there is actually nothing to divide
    if (len(subset0_index)==0) or (len(subset1_index) == 0):
        return 0

    # 2.1 calculate entropy1
    weight1 = len(subset1_index)/len(features)
    n1 = len(subset1_index)
    numberoftrue1 = sum (1 for i in subset1_index if targets[i]==1)
    p1true = numberoftrue1/n1
    p1false = 1-p1true

    entropy1 = -p1true * np.log2(p1true) - p1false * np.log2(p1false)

    # 2.2 calculate entropy0
    weight0 = 1-weight1
    n0 = len(subset0_index)
    numberoftrue0 = sum (1 for i in subset0_index if targets[i]==1)
    p0true = numberoftrue0/n0
    p0false = 1-p0true

    entropy0 = -p0true * np.log2(p0true) - p0false * np.log2(p0false)

    # 3. calculate information gain

    my_information_gain = H_s - weight1*entropy1 - weight0*entropy0

    #print("my_information_gain= ", my_information_gain)

    return my_information_gain
    #raise NotImplementedError()

if __name__ == '__main__':
    # construct a fake tree
    attribute_names = ['larry', 'curly', 'moe']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Node(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Node(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()
