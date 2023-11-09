from loadDataset import load_dataset
import lasagne
import theano
import theano.tensor as T
import numpy as np
from random import randint

# Number of training steps. More = Better (Ideally few 100)
TRAINING_STEPS = 45

class NeuralNet:

    def __init__(self):
        self.trainingImages = np.empty([30000, 1])
        self.trainingLabels = np.empty([30000])
        self.testingImages = np.empty([30000, 1])
        self.testingLabels = np.empty([30000])

        self.inputVariable = T.tensor4('inputs')  # empty 4 dimensional array
        self.targetVariable = T.ivector('targets')  # empty 1 dimensional array to represent the labels

        self.network = self.build_NN()  # Build an empty neural network with random weights

    def loadDatasetImages(self):
        self.trainingImages, self.trainingLabels, self.testingImages, self.testingLabels = load_dataset()

    def build_NN(self, input_var=None):

        # ------- Input Layer -------
        # Input layer: 1 x 28 x 28 = 784 (for 1 image. 1 channel x size)
        # Link the input layer to the inputVar, which is the array of images
        inputLayer = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)

        # Will be a 20% dropout between the inputs and the next layer (avoids overfitting)
        inputLayerDropout = lasagne.layers.DropoutLayer(inputLayer, p=0.2)

        # ------- Hidden Layer #1 -------
        # Hidden Layer #1 has 800 nodes - each edge is possible
        # It is given random initial weights using the GlorotUniform which is faster to train from
        hiddenLayer1 = lasagne.layers.DenseLayer(inputLayerDropout, num_units=800,
                                                 nonlinearity=lasagne.nonlinearities.rectify,
                                                 W=lasagne.init.GlorotUniform())

        # Add a 50% dropout to the hidden layer number 1
        hiddenLayer1Dropout = lasagne.layers.DropoutLayer(hiddenLayer1, p=0.5)

        # ------- Hidden Layer #2 -------
        # Hidden Layer #2 has another 800 nodes
        # Again, the layer is given random weights using the GlorotUniform option, hopefully boosting performance.
        hiddenLayer2 = lasagne.layers.DenseLayer(hiddenLayer1Dropout, num_units=800,
                                                 nonlinearity=lasagne.nonlinearities.rectify,
                                                 W=lasagne.init.GlorotUniform())

        # Add another 50% dropout to the hidden layer number 2
        hiddenLayer2Dropout = lasagne.layers.DropoutLayer(hiddenLayer2, p=0.5)

        # ------- Output Layer -------
        # output layer has 10 nodes 0-9. Softmax specifies each output is between 0-1
        # and the max of these will be the final output
        outputLayer = lasagne.layers.DenseLayer(hiddenLayer2Dropout, num_units=10,
                                                nonlinearity=lasagne.nonlinearities.softmax)

        return outputLayer  # Return the output layer, giving access to the entire neural net

    def trainNeuralNetwork(self):
        self.loadDatasetImages()  # Load in the dataset

        # Train Neural Net
        self.network = self.build_NN(self.inputVariable)

        # ---- Training instructions -----
        # 1. Compute an error function
        prediction = lasagne.layers.get_output(self.network)

        # crossentropy is one of the error functions with classification problems
        loss = lasagne.objectives.categorical_crossentropy(prediction, self.targetVariable)
        loss = loss.mean()

        # 2. Update the weights based on the error function
        # get_all_params gets all the values of the current weights
        params = lasagne.layers.get_all_params(self.network, trainable=True)

        # nesterov_momentum is one of the options Lasagne provides for updating the weights
        # It's based on Stochastic Gradient Descent
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

        # Use theano to represent a training step: compute the error, find the current weights, update the weights
        # params are incrementally changed by a certain value
        train_fn = theano.function([self.inputVariable, self.targetVariable], loss, updates=updates)

        for step in range(TRAINING_STEPS):
            train_err = train_fn(self.trainingImages, self.trainingLabels)
            print(
                "Current epoch: " + str(step) + "/" + str(TRAINING_STEPS) + "    Training error: " + str(train_err))

        print("Training Complete")

    def testNeuralNetwork(self):
        self.loadDatasetImages()  # Load in the dataset

        # Deterministic keeps the previously dropped nodes from the network
        test_prediction = lasagne.layers.get_output(self.network, deterministic=True)

        # T.argmax gives us the max value of the output nodes
        # This value is compared with the actual value of the label (.eq)
        accuracyTest = T.mean(T.eq(T.argmax(test_prediction, axis=1), self.targetVariable), dtype=theano.config.floatX)

        accuracyFunction = theano.function([self.inputVariable, self.targetVariable], accuracyTest)

        accuracy = accuracyFunction(self.testingImages, self.testingLabels)

        accuracy = accuracy * 100  # 0 - 1 float as 0 - 100
        accuracy = round(accuracy, 2)  # 0 - 100 float as 2dp

        return accuracy

    def testSingularNeuralNetwork(self):
        self.loadDatasetImages()

        #Random index in testing Images
        testImageIndex = randint(0, len(self.testingImages)-1)
        # To check the prediction on 1 image, we need another function
        test_prediction = lasagne.layers.get_output(self.network)
        val_fn = theano.function([self.inputVariable], test_prediction)

        outputNodes = val_fn([self.testingImages[testImageIndex]])
        # Apply the function to the testImage at the desired index

        # Array currently holds each output node weight (10 nodes)
        # The index of the biggest node is the predicted output
        # Known as winner-takes-all

        # Find the index of the maximum value in the array
        predictedValue = np.argmax(outputNodes, axis=1)

        actualValue = self.testingLabels[testImageIndex]

        return testImageIndex, predictedValue, actualValue

    def getNumberofTrainingImages(self):
        return len(self.trainingImages)

    def getNumberofTestingImages(self):
        return len(self.testingImages)

    def getNumberofTrainingSteps(selfs):
        return TRAINING_STEPS










