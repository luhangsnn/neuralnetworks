'''softmax_layer.py
Constructs, trains, tests single layer neural network with softmax activation function.
Roujia Zhong & Luhang Sun
CS343: Neural Networks
Project 2: Multilayer Perceptrons
'''
import numpy as np


class SoftmaxLayer():
    '''
    SoftmaxLayer is a class for single layer networks with softmax activation and cross-entropy loss
    in the output layer.
    '''
    def __init__(self, num_output_units):
        '''SoftmaxLayer constructor

        Parameters:
        -----------
        num_output_units: int. Num output units. Equal to # data classes.
        '''
        # Network weights
        self.wts = None
        # Bias
        self.b = None
        # Number of data classes C
        self.num_output_units = num_output_units

    def accuracy(self, y, y_pred):
        '''Computes the accuracy of classified samples. Proportion correct

        Parameters:
        -----------
        y: ndarray. int-coded true classes. shape=(Num samps,)
        y_pred: ndarray. int-coded predicted classes by the network. shape=(Num samps,)

        Returns:
        -----------
        float. accuracy in range [0, 1]
        '''
        return (1 - np.mean(y_pred != y))

    def net_in(self, features):
        '''Computes the net input (net weighted sum)
        Parameters:
        -----------
        features: ndarray. input data. shape=(num images (in mini-batch), num features)
        i.e. shape=(N, M)

        Note: shape of self.wts = (M, C), for C output neurons

        Returns:
        -----------
        net_input: ndarray. shape=(N, C)
        '''
        #compute the number of samples
        num_samps = len(features)
        C = len(self.wts[0])
        #create a column for bias
        bias_col = np.ones((num_samps,C))
        net_input = features @ self.wts + bias_col* self.b
        return net_input

    def one_hot(self, y, num_classes):
        '''One-hot codes the output classes for a mini-batch

        Parameters:
        -----------
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,C-1
        num_classes: int. Number of unique output classes total

        Returns:
        -----------
        y_one_hot: One-hot coded class assignments.
            e.g. if y = [0, 2, 1] and num_classes (C) = 4 we have:
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]
        '''
        #initialize a matrix of zeros with shape (len(y), num_classes)
        y_one_hot = np.zeros((len(y), num_classes))
        index0 = np.arange(len(y))
        index1 = y
        y_one_hot[index0, index1] = 1
        return y_one_hot

    def fit(self, features, y, n_epochs=10000, lr=0.0001, mini_batch_sz=256, reg=0, verbose=2, replacement = True):
        '''Trains the network to data in `features` belonging to the int-coded classes `y`.
        Implements stochastic mini-batch gradient descent

        Parameters:
        -----------
        features: ndarray. shape=(Num samples N, num features M)
        y: ndarray. int-coded class assignments of training samples. 0,...,numClasses-1
        n_epochs: int. Number of training epochs
        lr: float. Learning rate
        mini_batch_sz: int. Batch size per training iteration.
            i.e. Chunk this many data samples together to process with the model on each training
            iteration. Then we do gradient descent and update the wts. NOT the same thing as an epoch.
        reg: float. Regularization strength used when computing the loss and gradient.
        verbose: int. 0 means no print outs. Any value > 0 prints Current iteration number and
            training loss every 100 iterations.

        Returns:
        -----------
        loss_history: Python list of floats. Recorded training loss on every mini-batch / training
            iteration.

        NOTE:
        Recall: training epoch is not the same thing as training iteration with mini-batch.
        If we have mini_batch_sz = 100 and N = 1000, then we have 10 iterations per epoch. Epoch
        still means entire pass through the training data "on average". Print this information out
        if verbose > 0.

        TODO:
        -----------
        1) Initialize the wts/bias to small Gaussian numbers:
            mean 0, std 0.01, Wts shape=(num_feat M, num_classes C), b shape=(num_classes C,)
        2) Implement mini-batch support: On every iter draw from our input samples (with replacement)
        a batch of samples equal in size to `mini_batch_sz`. Also keep track of the associated labels.
        THEY MUST MATCH UP!!
            - Keep in mind that mini-batch wt updates are different than epochs. There is a parameter
              for E (num epochs), not number of iterations.
            - Handle this edge case: we do SGD and mini_batch_sz = 1. Add a singleton dimension
              so that the "N"/sample_size dimension is still defined.
        4) Our labels are int coded (0,1,2,3...) but this representation doesnt work well for piping
        signals to the C output neurons (C = num classes). Transform the mini-batch labels to one-hot
        coding from int coding (see function above to write this code).
        5) Compute the "net in".
        6) Compute the activation values for the output neurons (you can defer the actual function
        implementation of this for later).
        7) Compute the cross-entropy loss (again, you can defer the details for now)
        8) Do backprop:
            a) Compute the error gradient for the mini-batch sample,
            b) update weights using gradient descent.

        HINTS:
        -----------
        2) Work in indices, not data elements.
        '''
        num_samps, num_features = features.shape
        num_classes = np.max(y)+1
        #initialize weights and biases
        mu, sigma = 0, 0.01
        self.wts = np.random.normal(mu, sigma, (num_features, num_classes))
        self.b = np.random.normal(mu, sigma, (num_classes,))
        iteration = round(len(features) / mini_batch_sz)
        loss_history = []
        total_iteration = 0
        for i in range(n_epochs):
            for j in range(iteration):
                #generate a set of random index
                series = np.arange(num_samps)

                if replacement == True: 
                    index = np.random.choice(series, size = (mini_batch_sz,), replace=True)
                else:
                    #for extension
                    index = np.random.choice(series, size = (mini_batch_sz,), replace=False)
                samples = features[index]
                labels = y[index]

                #special case when sz = 1
                #Not sure about this condition
                if (mini_batch_sz == 1):
                    samples = np.array(samples.reshape(1, num_classes))
                    labels = np.array([labels].reshape(1,))

                #transform the minibatch labels to one-hot-coding
                num_uniqueclass = np.unique(labels).shape[0]
                # print(num_uniqueclass, labels)
                one_hot_coding = self.one_hot(labels, num_classes)
                #compute net_in
                net_in = self.net_in(samples)
                #compute activation value
                net_act = self.activation(net_in)
                #compute loss
                loss = self.loss(net_act, labels, reg)
                loss_history.append(loss)
                #compute the gradient
                # print(samples.shape, net_act.shape, one_hot_coding.shape)
                grad_wts, grad_b = self.gradient(samples, net_act, one_hot_coding, reg)
                #update weights using gradient descent 
                self.wts = self.wts - lr * grad_wts
                self.b = self.b - lr * grad_b

                total_iteration = total_iteration + 1
        return loss_history


    def predict(self, features):
        '''Predicts the int-coded class value for network inputs ('features').

        Parameters:
        -----------
        features: ndarray. shape=(mini-batch size, num features)

        Returns:
        -----------
        y_pred: ndarray. shape=(mini-batch size,).
            This is the int-coded predicted class values for the inputs passed in.
            Note: You can figure out the predicted class assignments from net_in (i.e. you dont
            need to apply the net activation function — it will not affect the most active neuron).
        '''
        net_input = self.net_in(features)
        y_pred = np.argmax(net_input, axis = 1)
        return y_pred

    def activation(self, net_in):
        '''Applies the softmax activation function on the net_in.

        Parameters:
        -----------
        net_in: ndarray. net in. shape=(mini-batch size, num output neurons)
        i.e. shape=(N, C)

        Returns:
        -----------
        f_z: ndarray. net_act transformed by softmax function. shape=(N, C)

        Tips:
        -----------
        - Remember the adjust-by-the-max trick (for each input samp) to prevent numeric overflow!
        This will make the max net_in value for a given input 0.
        - np.sum and np.max have a keepdims optional parameter that might be useful for avoiding
        going from shape=(X, Y) -> (X,). keepdims ensures the result has shape (X, 1).
        '''
        #fine the maximum net_in
        N = -np.max(net_in, axis = 1, keepdims = True)
        numerator = np.exp(net_in + N)
        denominator = np.sum(numerator, axis = 1, keepdims = True)
        f_z = numerator/denominator
        return f_z

    def loss(self, net_act, y, reg=0):
        '''Computes the cross-entropy loss

        Parameters:
        -----------
        net_act: ndarray. softmax net activation. shape=(mini-batch size, num output neurons)
        i.e. shape=(N, C)
        y: ndarray. correct class values, int-coded. shape=(mini-batch size,)
        reg: float. Regularization strength

        Returns:
        -----------
        loss: float. Regularized (!!!!) average loss over the mini batch

        Tips:
        -----------
        - Remember that the loss is the negative of the average softmax activation values of neurons
        coding the correct classes only.
        - It is handy to use arange indexing to sel ect only the net_act values coded by the correct
          output neurons.
        - NO FOR LOOPS!
        - Remember to add on the regularization term, which has a 1/2 in front of it.
        '''
        #pick the softmax activation values of neurons coding the correct classes
        index0 = np.arange(len(y))
        index1 = y
        loss = -np.sum(np.log(net_act[index0, index1]))/len(y) 
        loss = loss + 1/2*np.sum(np.square(self.wts))*reg
        return loss 

    def gradient(self, features, net_act, y, reg=0):
        '''Computes the gradient of the softmax version of the net

        Parameters:
        -----------
        features: ndarray. net inputs. shape=(;mini-batch-size, Num features)
        net_act: ndarray. net outputs. shape=(mini-batch-size, C)
            In the softmax network, net_act for each input has the interpretation that
            it is a probability that the input belongs to each of the C output classes.
        y: ndarray. one-hot coded class labels. shape=(mini-batch-size, Num output neurons)
        reg: float. regularization strength.

        Returns:
        -----------
        grad_wts: ndarray. Weight gradient. shape=(Num features, C)
        grad_b: ndarray. Bias gradient. shape=(C,)

        NOTE:
        - Gradient is the same as ADALINE, except we average over mini-batch in both wts and bias.
        - NO FOR LOOPS!
        - Don't forget regularization!!!! (Weights only, not for bias)
        '''
        index0 = np.arange(len(y))
        index1 = y
        grad_wts = 1/(len(y))*(features.T @ (net_act - y)) + reg * self.wts
        grad_b = 1/(len(y)) * np.sum(net_act - y, axis = 0)
        #add regularization strength
        return grad_wts, grad_b
        

    def test_loss(self, wts, b, features, labels):
        ''' Tester method for net_in and loss
        '''
        self.wts = wts
        self.b = b

        net_in = self.net_in(features)
        print(f'net in shape={net_in.shape}, min={net_in.min()}, max={net_in.max()}')
        print('Should be\nnet in shape=(15, 10), min=0.7160773059462714, max=1.4072103751494884\n')

        net_act = self.activation(net_in)
        print(f'net act shape={net_act.shape}, min={net_act.min()}, max={net_act.max()}')
        print('Should be\nnet act shape=(15, 10), min=0.0732240641262733, max=0.1433135816597887\n')
        return self.loss(net_act, labels, 0), self.loss(net_act, labels, 0.5)

    def test_gradient(self, wts, b, features, labels, num_unique_classes, reg=0):
        ''' Tester method for gradient
        '''
        self.wts = wts
        self.b = b

        net_in = self.net_in(features)
        print(f'net in: {net_in.shape}, {net_in.min()}, {net_in.max()}')
        print(f'net in 1st few values of 1st input are: {net_in[0, :5]}')

        net_act = self.activation(net_in)
        print(f'net act 1st few values of 1st input are: {net_act[0, :5]}')

        labels_one_hot = self.one_hot(labels, num_unique_classes)
        print(f'y one hot: {labels_one_hot.shape}, sum is {np.sum(labels_one_hot)}')

        return self.gradient(features, net_act, labels_one_hot, reg=reg)

    def fit_batch(self, features, y, n_epochs=10000, lr=0.0001, reg=0, verbose=2):
        '''
        Extension: Using batch gradient descent; Each iteration is consists of running all the samples. 
        '''
        num_samps, num_features = features.shape
        num_classes = np.max(y)+1
        #initialize weights and biases
        mu, sigma = 0, 0.01
        self.wts = np.random.normal(mu, sigma, (num_features, num_classes))
        self.b = np.random.normal(mu, sigma, (num_classes,))
        loss_history = []
        total_iteration = 0
        for i in range(n_epochs):
            #generate a set of random index
            series = np.arange(num_samps)
            index = np.arange(len(features))
            
            # np.random.choice(series, size = (len(features),), replace=True)

            samples = features[index]
            labels = y[index]

            #transform the minibatch labels to one-hot-coding
            num_uniqueclass = np.unique(labels).shape[0]
            # print(num_uniqueclass, labels)
            one_hot_coding = self.one_hot(labels, num_classes)
            #compute net_in
            net_in = self.net_in(samples)
            #compute activation value
            net_act = self.activation(net_in)
            #compute loss
            loss = self.loss(net_act, labels, reg)
            loss_history.append(loss)
            #compute the gradient
            # print(samples.shape, net_act.shape, one_hot_coding.shape)
            grad_wts, grad_b = self.gradient(samples, net_act, one_hot_coding, reg)
            #update weights using gradient descent 
            self.wts = self.wts - lr * grad_wts
            self.b = self.b - lr * grad_b

        return loss_history

    def fit_stochastic(self, features, y, n_epochs=10000, lr=0.0001, reg=0, verbose=2):
        '''
        Extension: Using stochastic gradient descent; Each iteration is consists of running one samples. 
        '''
        num_samps, num_features = features.shape
        num_classes = np.max(y)+1
        #initialize weights and biases
        mu, sigma = 0, 0.01
        self.wts = np.random.normal(mu, sigma, (num_features, num_classes))
        self.b = np.random.normal(mu, sigma, (num_classes,))
        loss_history = []
        total_iteration = 0
        iteration = num_samps

        for i in range(n_epochs):
            for j in range(iteration):
                sample = features[j]
                label = y[j]
                #transform the minibatch labels to one-hot-coding
                num_uniqueclass = np.unique(label).shape[0]
                # print(num_uniqueclass, labels)
                one_hot_coding = self.one_hot([label], num_classes)
                #compute net_in
                net_in = self.net_in(sample)
                #compute activation value
                net_act = self.activation(net_in)
                #compute the gradient
                # print(samples.shape, net_act.shape, one_hot_coding.shape)
                grad_wts, grad_b = self.gradient(sample, net_act, one_hot_coding, reg)
                #update weights using gradient descent 
                self.wts = self.wts - lr * grad_wts
                self.b = self.b - lr * grad_b
                total_iteration = total_iteration + 1

                #compute loss when finish one epoch
                loss = self.loss(net_act, [label], reg)
                loss_history.append(loss)
        # print(total_iteration, loss_history[-1])
        return loss_history
