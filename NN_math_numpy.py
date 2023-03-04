import math
import numpy as np
from urllib import request
import gzip
import pickle
import os
import random
import matplotlib.pyplot as plt
import mnist
import collections, functools, operator

### THIS CODE CONTAINS DIFFERENT FUNCTIONS FOR EACH EXERCISE, CALLED:
### exercise_3, exercise_4, exercise_5, exercise_6, exercise_7
### ALL THE FUNCTIONS ARE CALLED IN THE END OF THE CODE
### 
### THE NECESSARY PLOTS ARE AUTOMATICALLY SAVED AFTER RUNNING THE CORRESPONDING EXERCISE WITH THE FOLLOWING NAMES:
### ex4.png, ex5.png, ex6.png, ex7_1.png, ex7_2.png
###



def load_synth(num_train=60_000, num_val=10_000, seed=0):
    """
    Load some very basic synthetic data that should be easy to classify. Two features, so that we can plot the
    decision boundary (which is an ellipse in the feature space).
    :param num_train: Number of training instances
    :param num_val: Number of test/validation instances
    :param num_features: Number of features per instance
    :return: Two tuples and an integer: (xtrain, ytrain), (xval, yval), num_cls. The first contains a matrix of training
     data with 2 features as a numpy floating point array, and the corresponding classification labels as a numpy
     integer array. The second contains the test/validation data in the same format. The last integer contains the
     number of classes (this is always 2 for this function).
    """
    np.random.seed(seed)

    THRESHOLD = 0.6
    quad = np.asarray([[1, -0.05], [1, .4]])

    ntotal = num_train + num_val

    x = np.random.randn(ntotal, 2)

    # compute the quadratic form
    q = np.einsum('bf, fk, bk -> b', x, quad, x)
    y = (q > THRESHOLD).astype(np.int)

    return (x[:num_train, :], y[:num_train]), (x[num_train:, :], y[num_train:]), 2

def load_mnist(final=False, flatten=True):
    """
    Load the MNIST data.
    :param final: If true, return the canonical test/train split. If false, split some validation data from the training
       data and keep the test data hidden.
    :param flatten: If true, each instance is flattened into a vector, so that the data is returns as a matrix with 768
        columns. If false, the data is returned as a 3-tensor preserving each image as a matrix.
    :return: Two tuples and an integer: (xtrain, ytrain), (xval, yval), num_cls. The first contains a matrix of training
     data and the corresponding classification labels as a numpy integer array. The second contains the test/validation
     data in the same format. The last integer contains the number of classes (this is always 2 for this function).
     """

    if not os.path.isfile('mnist.pkl'):
        init()

    xtrain, ytrain, xtest, ytest = load()
    xtl, xsl = xtrain.shape[0], xtest.shape[0]

    if flatten:
        xtrain = xtrain.reshape(xtl, -1)
        xtest  = xtest.reshape(xsl, -1)

    if not final: # return the flattened images
        return (xtrain[:-5000], ytrain[:-5000]), (xtrain[-5000:], ytrain[-5000:]), 10

    return (xtrain, ytrain), (xtest, ytest), 10

# Numpy-only MNIST loader. Courtesy of Hyeonseok Jung
# https://github.com/hsjeong5/MNIST-for-Numpy

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    download_mnist()
    save_mnist()

def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]



##################### EXERCISE 3 ##########################


def exercise_3():

    print('RUNNING EXERCISE 3')
    X = [1,-1]
    Y = [1,0]
    W1 = [[1,1,1],[-1,-1,-1]]
    W2 = [[1,1],[-1,-1],[-1,-1]]
    b1 = [0,0,0]
    b2 = [0,0]

    Z1 = [0,0,0]
    A1 = [0,0,0]
    Z2 = [0,0]
    A2 = [0,0]

    dW2 = [[0,0],[0,0],[0,0]]
    dA1 = [0,0,0]
    dZ1 = [0,0,0]
    dW1 = [[0,0,0],[0,0,0]]

    ### ACTIVATION FUNCTIONS ###

    def sigmoid(x):
        return 1/(1 + math.exp(-x))

    def softmax(X, var):
        denom = 0
        for x in X:
            denom += math.exp(x)
        return math.exp(var) / denom


    ### FORWARD PROPAGATION ####

    for i in range(len(Z1)):
        Z1[i] = X[0]*W1[0][i] + X[1]*W1[1][i] + b1[i]
        A1[i] = sigmoid(Z1[i])

    for i in range(len(Z2)):
        Z2[i] = A1[0]*W2[0][i] + A1[1]*W2[1][i] + A1[2]*W2[2][i] + b1[i]

    ix = 0
    for i in Z2:
        A2[ix] = softmax(Z2, i)
        ix += 1

    cross_loss = -math.log(A2[0]) -math.log(A2[0])


    ### BACKWARD PROPAGATION ###

    dA2 = [-1/A2[0] ,- 1/A2[1]]
    dZ2 = [-(1 - A2[0]), A2[1]]

    for i in range(len(A1)):
        for j in range(len(dZ2)):
            dW2[i][j] = A1[i] * dZ2[j]
    db2 = dZ2
    print('dW2 = ', dW2, '\n')
    print('db2 = ', db2, '\n')


    for i in range(len(W2)):
        for j in range(len(W2[i])):
            dA1[i] += dZ2[j] * W2[i][j]



    dZ1 = [A1[i]*(1-A1[i])*dA1[i] for i in range(len(A1))]
    db1 = dZ1

    for i in range(len(X)):
        for j in range(len(dZ1)):
            dW1[i][j] = X[i] * dZ1[j]

    print('dW1 = ', dW1,'\n')
    print('db1 = ', db1, '\n')
    print('END OF EXERCISE 3')

##################### EXERCISE 4 ##########################

def exercise_4():

    print('RUNNING EXERCISE 4')
    def sigmoid(x):
     return 1/(1 + math.exp(-x))

    def softmax(X, var):
        denom = 0
        max_exp = max(X)
        for x in X:
            denom += math.exp(x-max_exp)
        return math.exp(var-max_exp) / denom


    def forward(X, parameters, cache):

        Z1 = cache["Z1"]
        Z2 = cache["Z2"]
        A1 = cache["A1"]
        A2 = cache["A2"]
        
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        for i in range(len(Z1)):
            Z1[i] = X[0]*W1[0][i] + X[1]*W1[1][i] + b1[i]
            A1[i] = sigmoid(Z1[i])

        for i in range(len(Z2)):
            Z2[i] = A1[0]*W2[0][i] + A1[1]*W2[1][i] + A1[2]*W2[2][i] + b1[i]

        ix = 0
        for i in Z2:
            A2[ix] = softmax(Z2, i)
            ix += 1
        
        cache = {"Z1": Z1,
                "A1": A1,
                "Z2": Z2,
                "A2": A2}
        
        return A2, cache

    def compute_cost(A2, Y):
        return -math.log(A2[Y])


    def backward(parameters, cache, grads, X, Y):

        W1 = parameters["W1"]
        W2 = parameters["W2"]
        A1 = cache["A1"]
        A2 = cache["A2"]
        dW2 = grads["dW2"]
        dW1 = grads["dW1"]


        if Y == 0:
            dZ2 = [-(1 - A2[0]), A2[1]]
        elif Y == 1:
            dZ2 = [A2[0], -(1 - A2[1])]


        for i in range(len(A1)):
            for j in range(len(dZ2)):
                dW2[i][j] = A1[i] * dZ2[j]
        db2 = dZ2

        dA1 = [0,0,0]
        for i in range(len(W2)):
            for j in range(len(W2[i])):
                dA1[i] += dZ2[j] * W2[i][j]


        dZ1 = [A1[i]*(1-A1[i])*dA1[i] for i in range(len(A1))]
        db1 = dZ1

        for i in range(len(X)):
            for j in range(len(dZ1)):
                dW1[i][j] = X[i] * dZ1[j]
        
        grads = {"dW1": dW1,
                "db1": db1,
                "dW2": dW2,
                "db2": db2}
        
        return grads

    def update(parameters, grads, learning_rate):
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]

        W1 = [[w1 - learning_rate*dw1 for w1, dw1 in zip(W1[i], dW1[i])] for i in range(len(dW1))]
        W2 = [[w2 - learning_rate*dw2 for w2, dw2 in zip(W2[i], dW2[i])] for i in range(len(dW2))]
        b1 = [b11 - learning_rate*db11 for b11, db11 in zip(b1, db1)]
        b2 = [b22 - learning_rate*db22 for b22, db22 in zip(b2, db2)]
        
        parameters = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2}
        return parameters
    
    (xtrain, ytrain), (xval, yval), num_cls = load_synth()

    print(num_cls)
    random.seed(12)
    parameters = {"W1": [[random.gauss(mu=0, sigma=1) for col in range(3)] for row in range(2)],
                "b1": [0,0,0],
                "W2": [[random.gauss(mu=0, sigma=1) for col in range(2)] for row in range(3)],
                "b2": [0, 0]}

    grads = {"dW1": [[0 for col in range(3)] for row in range(2)],
                "db1": [0,0,0],
                "dW2": [[0 for col in range(2)] for row in range(3)],
                "db2": [0, 0]}

    cache = {"Z1": [0,0,0],
            "A1": [0,0,0],
            "Z2": [0, 0],
            "A2": [0, 0]}

    final_cost = []
    final_mean_cost = []
    for i in range(0, len(xtrain)):

            A2, cache = forward(xtrain[i], parameters, cache)
            cost = compute_cost(A2, ytrain[i])
            
            if i%200 == 0:
                final_mean_cost.append(np.mean(np.array(final_cost[i-199: i])))
            
            final_cost.append(cost)
            grads = backward(parameters, cache, grads, xtrain[i], ytrain[i])
            parameters = update(parameters, grads, 0.03)


    print('Final cost: ', final_mean_cost[-1])
    plt.plot(range(len(final_mean_cost)),final_mean_cost)
    plt.xlabel('Number of iterations')
    plt.ylabel('Training loss')
    plt.savefig('ex4.png')
    plt.close()
    print('END OF EXERCISE 4')


##################### EXERCISE 5 ##########################

def exercise_5():

    print('RUNNING EXERCISE 5\n')
    def initialize_parameters(n_x, n_h, n_y):
        np.random.seed(12)
        W1 = np.random.randn(n_h, n_x)*0.01
        b1 = np.zeros((n_h,1))
        W2 = np.random.randn(n_y,n_h)*0.01
        b2 = np.zeros((n_y,1))

        
        parameters = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2}
        
        return parameters

    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def softmax(X):
        return [np.exp(x)/np.sum(np.exp(X)) for x in X]

    def forward_propagation(X, parameters):

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        Z1 = np.add(np.dot(W1,X[:,np.newaxis]), b1)
        A1 = sigmoid(Z1)
        Z2 = np.dot(W2,A1) + b2
        A2 = softmax(Z2)
        
        
        cache = {"Z1": Z1,
                "A1": A1,
                "Z2": Z2,
                "A2": A2}
        
        return A2, cache

    def binarize_output(Y):
        new_y = np.zeros(10)
        new_y[Y] = 1
        return new_y

    def compute_cost(A2, Y):
        return - Y[list(Y).index(1)]*np.log(A2[list(Y).index(1)])

    def backward_propagation(parameters, cache, X, Y):


        W1 = parameters["W1"]
        W2 = parameters["W2"]

        A1 = cache["A1"]
        A2 = cache["A2"]

        dZ2 = np.add(A2, -Y[:,np.newaxis])  # A2 - Y
        dW2 = np.matmul(dZ2, A1.T)
        dZ1 = np.matmul(W2.T, dZ2) * (A1*(1-A1))
        dW1 = np.matmul(dZ1, X[:,np.newaxis].T)
        db2 = np.sum(dZ2, axis = 1, keepdims = True)
        db1 = np.sum(dZ1, axis = 1, keepdims = True)

        grads = {"dW1": dW1,
                "db1": db1,
                "dW2": dW2,
                "db2": db2}
        
        return grads

    def mean_grads(batch_grads):
        
        grads = {"dW1": sum(item['dW1'] for item in batch_grads),
                "db1": sum(item['db1'] for item in batch_grads),
                "dW2": sum(item['dW2'] for item in batch_grads),
                "db2": sum(item['db2'] for item in batch_grads)}
        
        grads.update((key, value * 0.1) for key, value in grads.items())
        return grads

    def update_parameters(parameters, grads, learning_rate = 0.01):

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]

        W1 = W1 - learning_rate*dW1
        W2 = W2 - learning_rate*dW2
        b1 = b1 - learning_rate*db1
        b2 = b2 - learning_rate*db2
        
        parameters = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2}
        
        return parameters

    #mnist.init()
    x_train, t_train, x_test, t_test = mnist.load()

    n_x = 784
    n_h = 300
    n_y = 10

    parameters = initialize_parameters(n_x, n_h, n_y)
    final_cost = []
    final_mean_cost = []


    for i in range(0, 60000, 100):
        batch_grad = []
        cost = []
        for j in range(100):
            A2, cache = forward_propagation(x_train[i+j], parameters)
            Y = binarize_output(t_train[i+j])
            cost.append(compute_cost(A2, Y))
            batch_grad.append(backward_propagation(parameters, cache, x_train[i+j], Y))
        final_cost.append(np.mean(np.array(cost)))
        print('Batch ', int(i/100), ', cost: ', np.mean(np.array(cost)))
        grads = mean_grads(batch_grad)
        parameters = update_parameters(parameters, grads)

    plt.plot(range(len(final_cost)),final_cost)
    plt.xlabel('Number of iterations')
    plt.ylabel('Training loss')
    plt.savefig('ex5.png')
    plt.close()
    print('END OF EXERCISE 5')


##################### EXERCISE 6 ##########################

def exercise_6():

    print('RUNNING EXERCISE 6\n')

    def initialize_parameters(n_x, n_h, n_y):

        W1 = np.random.rand(n_x, n_h)*0.01
        b1 = np.zeros((1, n_h))
        W2 = np.random.rand(n_h,n_y)*0.01
        b2 = np.zeros((1, n_y))

        
        parameters = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2}
        
        return parameters

    def sigmoid(x):
        return 1/(1 + np.exp(-x))


    def forward_propagation(X, parameters):

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]


        Z1 = np.add(np.dot(X,W1), b1)
        A1 = sigmoid(Z1)
        Z2 = np.matmul(A1, W2) + b2
        from scipy.special import softmax
        A2 = softmax(Z2,axis = 1)
        
        
        cache = {"Z1": Z1,
                "A1": A1,
                "Z2": Z2,
                "A2": A2}
        
        return A2, cache

    def binarize_output(Y):
        encoded_array = np.zeros((Y.size, Y.max()+1), dtype=int)
        encoded_array[np.arange(Y.size),Y] = 1 

        return encoded_array

    def compute_cost(A2, Y):
        return -np.mean(np.sum(Y*np.log(A2), axis = 1))

    def backward_propagation(parameters, cache, X, Y):


        W1 = parameters["W1"]
        W2 = parameters["W2"]

        A1 = cache["A1"]
        A2 = cache["A2"]

        dZ2 = A2 - Y
        dW2 = np.matmul(A1.T, dZ2)
        dZ1 = np.matmul(dZ2, W2.T) * (A1*(1-A1))
        dW1 = np.matmul(X.T, dZ1)
        db2 = np.sum(dZ2, axis = 0, keepdims = True)
        db1 = np.sum(dZ1, axis = 0, keepdims = True)

        grads = {"dW1": dW1,
                "db1": db1,
                "dW2": dW2,
                "db2": db2}
        
        return grads


    def update_parameters(parameters, grads, learning_rate = 0.0003):

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]

        dW2 = np.clip(dW2, -5, 5)
        W1 = W1 - learning_rate*dW1
        W2 = W2 - learning_rate*dW2
        b1 = b1 - learning_rate*db1
        b2 = b2 - learning_rate*db2
        
        parameters = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2}
        
        return parameters

    
    x_train, t_train, x_test, t_test = mnist.load()

    n_x = 784
    n_h = 300
    n_y = 10
    num_epochs = 150
    parameters = initialize_parameters(n_x, n_h, n_y)
    final_cost = []
    final_mean_cost = []


    for i in range(0, num_epochs):
        grads = {}
        A2, cache = forward_propagation(x_train/255, parameters)
        Y = binarize_output(t_train)
        cost = compute_cost(A2, Y)
        final_cost.append(cost)
        grads = backward_propagation(parameters, cache, x_train, Y)
        parameters = update_parameters(parameters, grads)
        print('Cost at epoch ', i, ': ', cost)

    train_pred = [np.argmax(a2) for a2 in A2]
    print('Final training accuracy: ', ((train_pred - t_train) == 0).sum()/len(train_pred))

    test_prediction = [np.argmax(a2) for a2 in forward_propagation(x_test/255, parameters)[0]]
    print('Test Accuracy: ', ((test_prediction - t_test) == 0).sum()/len(test_prediction))

    plt.plot(range(len(final_cost)),final_cost)
    plt.xlabel('Number of epochs')
    plt.ylabel('Training loss')
    plt.savefig('ex6.png')
    plt.close()
    print('END OF EXERCISE 6')

################# EXERCISE 7 ##################

def exercise_7():

    print('RUNNING EXERCISE 7\n')
    def initialize_parameters(n_x, n_h, n_y):

        W1 = np.random.randn(n_h, n_x)*0.01
        b1 = np.zeros((n_h,1))
        W2 = np.random.randn(n_y,n_h)*0.01
        b2 = np.zeros((n_y,1))

        
        parameters = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2}
        
        return parameters

    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def softmax(X):
        return [np.exp(x)/np.sum(np.exp(X)) for x in X]

    def forward_propagation(X, parameters):

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        Z1 = np.add(np.dot(W1,X[:,np.newaxis]), b1)
        A1 = sigmoid(Z1)
        Z2 = np.dot(W2,A1) + b2
        A2 = softmax(Z2)
        
        
        cache = {"Z1": Z1,
                "A1": A1,
                "Z2": Z2,
                "A2": A2}
        
        return A2, cache

    def binarize_output(Y):
        new_y = np.zeros(10)
        new_y[Y] = 1
        return new_y

    def compute_cost(A2, Y):
        return - Y[list(Y).index(1)]*np.log(A2[list(Y).index(1)])

    def backward_propagation(parameters, cache, X, Y):


        W1 = parameters["W1"]
        W2 = parameters["W2"]

        A1 = cache["A1"]
        A2 = cache["A2"]

        dZ2 = np.add(A2, -Y[:,np.newaxis])  # A2 - Y
        dW2 = np.matmul(dZ2, A1.T)
        dZ1 = np.matmul(W2.T, dZ2) * (A1*(1-A1))
        dW1 = np.matmul(dZ1, X[:,np.newaxis].T)
        db2 = np.sum(dZ2, axis = 1, keepdims = True)
        db1 = np.sum(dZ1, axis = 1, keepdims = True)

        grads = {"dW1": dW1,
                "db1": db1,
                "dW2": dW2,
                "db2": db2}
        
        return grads

    def mean_grads(batch_grads):
        
        grads = {"dW1": sum(item['dW1'] for item in batch_grads),
                "db1": sum(item['db1'] for item in batch_grads),
                "dW2": sum(item['dW2'] for item in batch_grads),
                "db2": sum(item['db2'] for item in batch_grads)}
        
        grads.update((key, value * 0.1) for key, value in grads.items())
        return grads

    def update_parameters(parameters, grads, learning_rate = 0.01):

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]

        W1 = W1 - learning_rate*dW1
        W2 = W2 - learning_rate*dW2
        b1 = b1 - learning_rate*db1
        b2 = b2 - learning_rate*db2
        
        parameters = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2}
        
        return parameters

    
    def forward_propagation_val(X, parameters):
        from scipy.special import softmax as softmax_val

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        Z1 = np.add(np.dot(X,W1.T), b1.T)
        A1 = sigmoid(Z1)
        Z2 = np.matmul(A1, W2.T) + b2.T
        A2 = softmax_val(Z2,axis = 1)
        
        
        cache = {"Z1": Z1,
                "A1": A1,
                "Z2": Z2,
                "A2": A2}
        
        return A2, cache

    def binarize_output_val(Y):
        encoded_array = np.zeros((Y.size, Y.max()+1), dtype=int)
        encoded_array[np.arange(Y.size),Y] = 1 

        return encoded_array

    def compute_cost_val(A2, Y):
        return -np.mean(np.sum(Y*np.log(A2), axis = 1))

    epoch_cost = []
    val_cost = []

    #mnist.init()
    x_train, t_train, x_test, t_test = mnist.load()

    n_x = 784
    n_h = 300
    n_y = 10
    parameters = initialize_parameters(n_x, n_h, n_y)
    final_cost = []

    losses = {}
    for n in range(3):
        parameters = initialize_parameters(n_x, n_h, n_y)
        
        epoch_cost = []
        for m in range(5):
            batch_cost = []
            
            for i in range(0, 60000, 100):
                batch_grad = []
                cost = []

                for j in range(100):
                    A2, cache = forward_propagation(x_train[i+j], parameters)
                    
                    Y = binarize_output(t_train[i+j])
                    
                    cost.append(compute_cost(A2, Y))

                    
                    batch_grad.append(backward_propagation(parameters, cache, x_train[i+j], Y))

                batch_cost.append(np.mean(np.array(cost)))
                grads = mean_grads(batch_grad)
                parameters = update_parameters(parameters, grads)
            
            epoch_cost.append(np.mean(np.array(batch_cost)))
            print('Init', n, 'Epoch', m, ' cost: ', epoch_cost[-1])
            if n == 0:
                A2_val, cache_val = forward_propagation_val(x_test, parameters)
                Y_val = binarize_output_val(t_test)
                val_cost.append(compute_cost_val(A2_val, Y_val))
        

        name = 'l' + str(n)
        losses[name] = epoch_cost

    means = (np.array(losses['l0']) + np.array(losses['l1']) + np.array(losses['l2']))/3
    stds = []
    for i in range(len(losses['l0'])):
        stds.append(np.std(np.array([losses['l0'][i],losses['l1'][i],losses['l2'][i]])))


    plt.plot(range(len(epoch_cost)),epoch_cost, label = 'Training loss', color = 'blue')
    plt.plot(range(len(val_cost)),val_cost, label = 'Validation loss', color = 'red')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")
    plt.savefig('ex7_1.png')
    plt.close()


    fig, ax = plt.subplots(1)
    ax.plot(np.arange(len(means)),means, lw=2, color='blue')
    ax.fill_between(np.arange(len(means)), means + stds, means - stds, facecolor='blue', alpha=0.5)
    plt.xlabel('Number of epochs')
    plt.ylabel('Training loss')

    ax.grid()
    plt.savefig('ex7_2.png')
    plt.close()
    print('END OF EXERCISE 7')


exercise_3()
exercise_4()
exercise_5()
exercise_6()
exercise_7()