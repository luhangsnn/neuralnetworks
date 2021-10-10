'''adaline.py
Roujia Zhong & Luhang Sun
CS343: Neural Networks
Project 1: Single Layer Networks
ADALINE (ADaptive LInear NEuron) neural network for classification and regression
'''
import numpy as np


class Adaline():
    ''' Single-layer neural network

    Network weights are organized [bias, wt1, wt2, wt3, ..., wtM] for a net with M input neurons.
    '''
    def __init__(self):
        # Network weights: Bias is stored in self.wts[0], wt for neuron 1 is at self.wts[1],
        # wt for neuron 2 is at self.wts[2], ...
        self.wts = None
        # Record of training loss. Will be a list. Value at index i corresponds to loss on epoch i.
        self.loss_history = None
        # Record of training accuracy. Will be a list. Value at index i corresponds to acc. on epoch i.
        self.accuracy_history = None
        self.y_pred = None

    def get_wts(self):
        ''' Returns a copy of the network weight array'''
        return self.wts.copy()

    def net_input(self, features):
        ''' Computes the net_input (weighted sum of input features,  wts, bias)

        NOTE: bias is the 1st element of self.wts. Wts for input neurons 1, 2, 3, ..., M occupy
        the remaining positions.

        Parameters:
        ----------
        features: ndarray. Shape = [Num samples N, Num features M]
            Collection of input vectors.

        Returns:
        ----------
        The net_input. Shape = [Num samples,]
        '''
        #compute the number of samples
        num_samps = len(features)
        #stack features with a column of ones on the left
        bias_col = np.ones((num_samps,1))
        data = np.hstack((bias_col, features))
        #compute net input
        net_input = data @ self.wts
        return net_input

    def activation(self, net_in):
        '''
        Applies the activation function to the net input and returns the output neuron's activation.
        It is simply the identify function for vanilla ADALINE: f(x) = x

        Parameters:
        ----------
        net_in: ndarray. Shape = [Num samples N,]

        Returns:
        ----------
        net_act. ndarray. Shape = [Num samples N,]
        '''
        net_act = net_in
        return net_act

    def compute_loss(self, y, net_act):
        ''' Computes the Sum of Squared Error (SSE) loss (over a single training epoch)

        Parameters:
        ----------
        y: ndarray. Shape = [Num samples N,]
            True classes corresponding to each input sample in a training epoch (coded as -1 or +1).
        net_act: ndarray. Shape = [Num samples N,]
            Output neuron's activation value (after activation function is applied)

        Returns:
        ----------
        float. The SSE loss (across a single training epoch).
        '''
        squared_error = np.square(y - net_act)
        SSE = 1/2 * np.sum(squared_error)
        return SSE

    def compute_accuracy(self, y, y_pred):
        ''' Computes accuracy (proportion correct) (across a single training epoch)

        Parameters:
        ----------
        y: ndarray. Shape = [Num samples N,]
            True classes corresponding to each input sample in a training epoch  (coded as -1 or +1).
        y_pred: ndarray. Shape = [Num samples N,]
            Predicted classes corresponding to each input sample (coded as -1 or +1).

        Returns:
        ----------
        float. The accuracy for each input sample in the epoch. ndarray.
            Expressed as proportions in [0.0, 1.0]
        '''
        return (1 - np.mean(y_pred != y))

    def gradient(self, errors, features):
        ''' Computes the error gradient of the loss function (for a single epoch).
        Used for backpropogation.

        Parameters:
        ----------
        errors: ndarray. Shape = [Num samples N,]
            Difference between class and output neuron's activation value
        features: ndarray. Shape = [Num samples N, Num features M]
            Collection of input vectors.

        Returns:
        ----------
        grad_bias: float.
            Gradient with respect to the bias term
        grad_wts: ndarray. shape=(Num features N,).
            Gradient with respect to the neuron weights in the input feature layer
        '''
        grad_bias = (-1) * np.sum(errors)
        grad_wts = (-1) * errors.reshape((1,errors.shape[0])) @ features
        
        return grad_bias, grad_wts[0]

    def predict(self, features):
        '''Predicts the class of each test input sample

        Parameters:
        ----------
        features: ndarray. Shape = [Num samples N, Num features M]
            Collection of input vectors.

        Returns:
        ----------
        The predicted classes (-1 or +1) for each input feature vector. Shape = [Num samples N,]

        NOTE: Remember to apply the activation function!
        '''
        netIn = self.net_input(features)
        netAct = self.activation(netIn)
        pred = np.zeros(netAct.shape)
        for i in range (netAct.shape[0]):
            if netAct[i] >= 0:
                pred[i] = 1
            else:
                pred[i] = -1 
        return pred.astype(int)

    def fit(self, features, y, n_epochs=1000, lr=0.001):
        ''' Trains the network on the input features for self.n_epochs number of epochs

        Parameters:
        ----------
        features: ndarray. Shape = [Num samples N, Num features M]
            Collection of input vectors.
        y: ndarray. Shape = [Num samples N,]
            Classes corresponding to each input sample (coded -1 or +1).
        n_epochs: int.
            Number of epochs to use for training the network
        lr: float.
            Learning rate used in weight updates during training

        Returns:
        ----------
        self.loss_history: Python list of network loss values for each epoch of training.
            Each loss value is the loss over a training epoch.
        self.acc_history: Python list of network accuracy values for each epoch of training
            Each accuracy value is the accuracy over a training epoch.

        TODO:
        1. Initialize the weights according to a Gaussian distribution centered
            at 0 with standard deviation of 0.01. Remember to initialize the bias in the same way.
        2. Write the main training loop where you:
            - Pass the inputs in each training epoch through the net.
            - Compute the error, loss, and accuracy (across the entire epoch).
            - Do backprop to update the weights and bias.
        '''
        #initialize weights
        mu, sigma = 0, 0.01
        self.wts = np.random.normal(mu, sigma, len(features[0])+1)

        self.loss_history = []
        self.acc_history = []
        #main loop
        for i in range(n_epochs):
            #pass the inputs
            net_in = self.net_input(features)
            net_act = self.activation(net_in)

            #compute errors and losses
            errors = y - net_act
            loss = self.compute_loss(y, net_act)

            #make prediction and compute accuracy
            self.y_pred = self.predict(features)
            accuracy = self.compute_accuracy(y, self.y_pred)

            self.loss_history.append(loss)
            self.acc_history.append(accuracy)

            #do backprop to update weights 
            grad_bias, grad_wts = self.gradient(errors, features)
            grad_wts = np.concatenate(([grad_bias], grad_wts))
            self.wts = self.wts - lr * grad_wts
            
            ### do we need to update bias as well? - oh we only update its weight

        return self.loss_history, self.acc_history

    def fit_early_stopping(self, features, y, n_epochs=1000, early_stopping=False, lr=0.001, loss_tol=0.1):
        
        #initialize weights
        mu, sigma = 0, 0.01
        self.wts = np.random.normal(mu, sigma, len(features[0])+1)

        self.loss_history = []
        self.acc_history = []
        n = 0

        while n < n_epochs:
            #pass the inputs
            net_in = self.net_input(features)
            net_act = self.activation(net_in)

            #compute errors and losses
            errors = y - net_act
            loss = self.compute_loss(y, net_act)
            
            #store previous loss
            # prev_loss = loss

            #make prediction and compute accuracy
            y_pred = self.predict(features)
            accuracy = self.compute_accuracy(y, y_pred)

            self.loss_history.append(loss)
            self.acc_history.append(accuracy)

            if n > 0 and early_stopping == True and (self.loss_history[-2] - loss < loss_tol):
                break

            #do backprop to update weights
            grad_bias, grad_wts = self.gradient(errors, features)
            grad_wts = np.concatenate(([grad_bias], grad_wts))
            self.wts = self.wts - lr * grad_wts
            n += 1
            
        print(f"number of epochs in early stopping: {n}")
        return self.loss_history, self.acc_history

    def fit_early_stopping_fancy(self, features, y, n_epochs=1000, lr=0.001, loss_tol=0.1, fancy_stop=False):
        mu, sigma = 0, 0.01
        self.wts = np.random.normal(mu, sigma, len(features[0])+1)
        self.loss_history = []
        self.acc_history = []
        difference = [] # keeps track of the difference
        n = 0

        while n < n_epochs:
            net_in = self.net_input(features)
            net_act = self.activation(net_in)

            errors = y - net_act
            loss = self.compute_loss(y, net_act)

            y_pred = self.predict(features)
            accuracy = self.compute_accuracy(y, y_pred)

            self.loss_history.append(loss)
            self.acc_history.append(accuracy)
            
            if n > 0:
                difference.append(self.loss_history[-2] - loss)
                # only terminates back propagation if the change relative to the average loss over the most recent 5 epochs is less than the tolerance
                if fancy_stop == True and n > 5:                    
                    if (sum(difference[-5:])/5) < loss_tol:
                        break
                    
            grad_bias, grad_wts = self.gradient(errors, features)
            grad_wts = np.concatenate(([grad_bias], grad_wts))
            self.wts = self.wts - lr * grad_wts
            n += 1
            
        print(f"number of epochs in fancy stopping: {n}")
        return self.loss_history, self.acc_history

#create a new class named Perceptron
class Perceptron(Adaline):
    def __init__(self):
        Adaline.__init__(self)

    #override the netAct method: 
    def activation(self, net_in):
        '''
        Applies the activation function to the net input and returns the output neuron's activation.
        netAct = f(netIn) = 1 if netIn >= 0
        netAct = f(netIn) = -1 if netIn < 0
        Parameters:
        ----------
        net_in: ndarray. Shape = [Num samples N,]
        Returns:
        ----------
        net_act. ndarray. Shape = [Num samples N,]
        '''        
        #create a net_act array in the same size as net_in
        net_act = np.ones((len(net_in),))
        #find the indices of netIN >= 0:
        idx_1 = np.where(net_in >= 0)[0]
        #set the rows with netIn >= 0 to 1
        net_act[idx_1] = 1 

        #find the indices of netIN <0 0:
        idx_2 = np.where(net_in < 0)[0]
        #set the rows with netIn < 0 to 0
        net_act[idx_2] = -1
        return net_act