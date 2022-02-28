from metrics import mean_squared_error
from polynomial_regression import PolynomialRegression
from generate_regression_data import generate_regression_data
from load_json_data import load_json_data
import numpy as np 
import random
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


def FRQ_1():
    features, targets = generate_regression_data(4, 100, 0.1)
    x = np.random.uniform(-1, 1, size=(50))
    y = np.random.uniform(-1, 1, size=(50))
    print(targets.shape[0])
    rand = random.sample(range(0, 50), 50)
    counter = 0
    for i in rand:
        x[counter] = features[i]
        y[counter] = targets[i]
        counter += 1
    a = np.delete(features, rand)
    b = np.delete(targets, rand)
    degr = [0 ,1, 2, 3, 4, 5, 6, 7, 8, 9]
    trainMSE = []
    mse = []
    lowestTest = 0
    lowestTestPred = 0
    lowestTestD = 0
    lowestTestError = 100
    lowestTra = 0
    lowestTraPred = 0
    lowestTraD = 0
    lowestTraError = 100
    for t in range(10):
        learner = PolynomialRegression(degree=t)
        
        
        #print("trani: {}".format(t))
        #print('x: {}'.format(x))
        #print('y: {}'.format(y))
        learner.fit(x, y)
        #for i in rand:
            #a = np.delete(features, i, 0)
        predict = learner.predict(a)
        #print(a)
        mse.append(mean_squared_error(predict, b))
        trainPred = learner.predict(x)
        trainMSE.append(mean_squared_error(trainPred, y))
        #learner.visualize(a, b, "que1Actu_" + str(t) + ".png")
        #learner.visualize(a, predict, "que1Pred_" + str(t) + ".png")
        if mean_squared_error(predict, b) < lowestTestError:
            lowestTestError = mean_squared_error(predict, b)
            lowestTestD = t
            lowestTest = a
            lowestTestPred = predict
        if mean_squared_error(trainPred, y) < lowestTraError:
            lowestTraError = mean_squared_error(trainPred, y)
            lowestTraD = t
            lowestTra = x
            lowestTraPred = trainPred
    degr = np.array(degr)
    mse = np.array(mse)
    trainMSE = np.array(trainMSE)
    mse = np.log10(mse)
    trainMSE = np.log10(trainMSE)
    plt.figure(figsize=(6,4))
    plt.xlabel('degree', fontsize=10)
    plt.ylabel('mean_squared_error', fontsize=10)
    
    plt.subplot(211)
    plt.scatter(degr, mse)
    plt.scatter(degr, trainMSE)
    plt.title("data_file")
    plt.plot(degr, mse, label ='np.log10 test MSE')
    plt.plot(degr, trainMSE, label ='np.log10 train MSE')
    plt.legend()
    
    plt.subplot(212)
    plt.scatter(lowestTra, lowestTraPred, label = 'lowestTrain, degree: ' + str(lowestTraD))
    plt.scatter(lowestTest, lowestTestPred, label = 'lowestTest, degree: ' + str(lowestTestD))
    #plt.plot(lowestTra, lowestTraPred, label = 'lowestTrain, degree: ')
    #plt.plot(degr, lowestTest, label = 'lowestTest, degree: ')
    plt.legend()
    plt.savefig("degr_error_mean_50")
FRQ_1()
