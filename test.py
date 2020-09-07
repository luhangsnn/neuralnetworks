print("你好！")

print("test new commit")

print("test v3")

print("TESTING BRANCH CHANGES ON LOCAL 9/7")

print("testing sync - client 2")
print("testing sync issues - client 1")


print("testing branch changes on local")

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
            y_pred = self.predict(features)
            accuracy = self.compute_accuracy(y, y_pred)


            self.loss_history.append(loss)
            self.acc_history.append(accuracy)

            #do backprop to update weights and bias
            grad_bias, grad_wts = self.gradient(errors, features)
            grad_wts = np.concatenate(([grad_bias], grad_wts))
            self.wts = self.wts - lr * grad_wts
            
            ### do we need to update bias as well? - oh we only update its weight

        return self.loss_history, self.acc_history

        

print("changes on master branch remote")

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

