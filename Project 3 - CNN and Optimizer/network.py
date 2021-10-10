'''network.py
Represents  a neural network (collection of layers)
Roujia Zhong & Luhang Sun
CS343: Neural Networks
Project 3: Convolutional Neural Networks
'''
import numpy as np

import layer
import optimizer
import accelerated_layer
np.random.seed(0)


class Network():
    '''Represents a neural network with some number of layers of various types.
    To create a specific network, create a subclass (e.g. ConvNet4) then
    add layers to it. For this project, the focus will be on building the
    ConvNet4 network.
    '''
    def __init__(self, reg=0, verbose=False):
        '''This method is pre-filled for you (shouldn't require modification).
        '''
        # Python list of Layer object references that make up out network
        self.layers = []
        # Regularization strength
        self.reg = reg
        # Whether we want network-related debug/info print outs
        self.verbose = verbose

        # Python list of ints. These are the indices of layers in `self.layers`
        # that have network weights. This should be all types of layers
        # EXCEPT MaxPool2D
        self.wt_layer_inds = []

        # As in former projects, Python list of loss, training/validation
        # accuracy during training recorded at some frequency (e.g. every epoch)
        self.loss_history = []
        self.train_acc_history = []
        self.validation_acc_history = []

    def compile(self, optimizer_name, **kwargs):
        '''Tells each network layer how weights should be updated during backprop
        during training (e.g. stochastic gradient descent, adam, etc.)

        This method is pre-filled for you (shouldn't require modification).

        NOTE: This NEEDS to be called AFTER creating your ConvNet4 object,
        but BEFORE you call `fit()` to train the net (otherwise, how does your
        net know how to update the weights?).

        Parameters:
        -----------
        optimizer_name: string. Name of optimizer class to use to update wts.
            See optimizer::create_optimizer for specific ones supported.
        **kwargs: Any number of optional parameters that get passed to the
            optimizer of your choice. e.g. learning rate.
        '''
        # Only set an optimizer for each layer with weights
        for l in [self.layers[i] for i in self.wt_layer_inds]:
            l.compile(optimizer_name, **kwargs)

    def fit(self, x_train, y_train, x_validate, y_validate, mini_batch_sz=100, n_epochs=10,
            acc_freq=10, print_every=50, verbose=True):
        '''Trains the neural network on data

        Parameters:
        -----------
        x_train: ndarray. shape=(num training samples, n_chans, img_y, img_x).
            Training data.
        y_train: ndarray. shape=(num training samples,).
            Training data classes, int coded.
        x_validate: ndarray. shape=(num validation samples, n_chans, img_y, img_x).
            Every so often during training (see acc_freq param), we compute
            the accuracy of the network in classifying the validation set
            (out-of-training-set generalization). This is the data we use.
        y_validate: ndarray. shape=(num validation samples,).
            Validation data classes, int coded.
        mini_batch_sz: int. Mini-batch training size.
        n_epochs: int. Number of training epochs.
        print_every: int.
            Controls the frequency (in iterations) with which to wait before printing out the loss
            and iteration number.
            NOTE: Previously, you used number of epochs rather than iterations to measure the frequency
            of print-outs. Use the simpler-to-implement units of iterations here because CNNs are
            more computationally intensive and you may want print-outs during an epoch.
        acc_freq: int. Should be equal to or a multiple of `print_every`.
            How many training iterations (weight updates) we wait before computing accuracy on the
            full training and validation sets?
            NOTE: This is is a computationally intensive process for the big network so make sure
            that you only COMPUTE training and validation accuracies this often
            (i.e DON'T compute them every iteration).

        TODO: Complete this method's implementation.
        1. In the main training loop, randomly sample to get a mini-batch.
        2. Do forward pass through network using the mini-batch.
        3. Do backward pass through network using the mini-batch.
        4. Compute the loss on the mini-batch, add it to our loss history list
        5. Call each layer's update wt method.
        6. Add support for `print_every` and `acc_freq`.
        7. Use the Python time module to print out the runtime (in minutes) for iteration 0 only.
            Also printout the projected time for completing ALL training iterations.
            (For simplicity, you don't need to consider the time taken for computing
            train and validation accuracy).
        '''
        num_samps = x_train.shape[0]
        num_classes = 10 # HARD CODED???
        iter_per_epoch = int(num_samps/mini_batch_sz)
        n_iter = iter_per_epoch * n_epochs

        print('Starting to train...')
        print(f'{n_iter} iterations. {iter_per_epoch} iter/epoch.')

        loss_history = []
        val_acc_history = []
        train_acc_history = []
        total_iteration = 0

        import time 
        start = time.time()

        for i in range(n_epochs):
            for j in range(iter_per_epoch):
                #generate a set of random index without replacement
                index = np.random.choice(np.arange(num_samps), size = (mini_batch_sz,), replace=False)
                samples = x_train[index]
                labels = y_train[index]

                #special case when sz = 1, Not sure about this condition
                if (mini_batch_sz == 1):
                    samples = np.array(samples).reshape(1, num_classes+1)
                    labels = np.array([labels]).reshape(1,)

                loss = self.forward(samples, labels)
                self.backward(labels)
                loss_history.append(loss)
                total_iteration += 1

                # call each layer's update weight method every iteration
                for layer in self.layers:
                    layer.update_weights()

                if total_iteration == 1:
                    end = time.time()
                    time = end - start
                    print(f"iteration 1: {round(time, 3)}s\ntotal predicted time: {round(time*n_iter, 3)}\n--------------------")
                
                if total_iteration % print_every == 0: # compute accuracy on validation set
                    print(f"iteration # {total_iteration}, loss: {loss}")

                if total_iteration % acc_freq == 0:
                    val_acc = self.accuracy(x_validate, y_validate)
                    val_acc_history.append(val_acc)
                    train_acc = self.accuracy(x_train, y_train)
                    train_acc_history.append(train_acc)
                    if verbose:
                        print(f"iternation # {total_iteration}, train_acc: {train_acc}, val_acc: {val_acc}")
        
        print(f"final acc: ", val_acc_history[-1])
        return loss_history, val_acc_history, train_acc_history
        

    def predict(self, inputs):
        '''Classifies novel inputs presented to the network using the current
        weights.

        Parameters:
        -----------
        inputs: ndarray. shape=shape=(num test samples, n_chans, img_y, img_x)
            This is test data.

        Returns:
        -----------
        pred_classes: ndarray. shape=shape=(num test samples)
            Predicted classes (int coded) derived from the network.

        Hints:
        -----------
        - The most active net_in values in the output layer gives us the predictions.
        (We don't need to check net_act).
        '''
        # Do a forward sweep through the network
        for layer in self.layers: 
            net_act = layer.forward(inputs)
            inputs = net_act
        y_pred = np.argmax(net_act, axis = 1)
        return y_pred

    def accuracy(self, inputs, y, samp_sz=500, mini_batch_sz=15):
        '''Computes accuracy using current net on the inputs `inputs` with classes `y`.

        This method is pre-filled for you (shouldn't require modification).

        Parameters:
        -----------
        inputs: ndarray. shape=shape=(num samples, n_chans, img_y, img_x)
            We are testing the classification accuracy on these data.
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
            shape=(N,) for mini-batch size N.
        samp_sz: int. If the number of samples is bigger than this number,
            we take a random sample from `inputs` of this size. We do this to
            keep performance of this method reasonable.
        mini_batch_sz: Because it might be tricky to hold all the training
            instances in memory at once, process and evaluate the accuracy of
            samples from `input` in mini-batches. We merge the accuracy scores
            across batches so the result is no different than processing all at
            once.
        '''
        n_samps = len(inputs)

        # Do we subsample the input?
        if n_samps > samp_sz:
            subsamp_inds = np.random.choice(n_samps, samp_sz)
            n_samps = samp_sz
            inputs = inputs[subsamp_inds]
            y = y[subsamp_inds]

        # How many mini-batches do we split the data into to test accuracy?
        n_batches = int(np.ceil(n_samps / mini_batch_sz))
        # Placeholder for our predicted class ints
        y_pred = np.zeros(len(inputs), dtype=np.int32)

        # Compute the accuracy through the `predict` method in batches.
        # Strategy is to use a 1D cursor `b` to extract a range of inputs to
        # process (a mini-batch)
        for b in range(n_batches):
            low = b*mini_batch_sz
            high = b*mini_batch_sz+mini_batch_sz
            # Tolerate out-of-bounds as we reach the end of the num samples
            if high > n_samps:
                high = n_samps

            # Get the network predicted classes and put them in the right place
            y_pred[low:high] = self.predict(inputs[low:high])

        # Accuracy is the average proportion that the prediction matchs the true
        # int class
        acc = np.mean(y_pred == y)

        return acc

    def forward(self, inputs, y):
        '''Do forward pass through whole network

        Parameters:
        -----------
        inputs: ndarray. Inputs coming into the input layer of the net. shape=(B, n_chans, img_y, img_x)
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
            shape=(B,) for mini-batch size B.

        Returns:
        -----------
        loss: float. REGULARIZED loss.

        TODO:
        1. Call the forward method of each layer in the network.
            Make the output of the previous layer the input to the next.
        2. Compute and get the loss from the LAST network layer.
        2. Compute and get the weight regularization via `self.wt_reg_reduce()` (implement this next)
        4. Return the sum of the loss and the regularization term.
        '''
        for layer in self.layers: 
            #call the forward method of each layer
            net_act = layer.forward(inputs)
            inputs = net_act
            # print(layer.name)
            # print(inputs.shape)
        #compute the loss of the last network layer
        loss = layer.loss(y)
        #compute and get the weight regularization
        reg = self.wt_reg_reduce()
        loss = loss + reg
        return loss


    def wt_reg_reduce(self):
        '''Computes the loss weight regularization for all network layers that have weights

        Returns:
        -----------
        wt_reg: float. Regularization for weights from all layers across the network.

        NOTE: You only can compute regularization for layers with wts!
        Layer indicies with weights are maintained in `self.wt_layer_inds`.
        The network regularization `wt_reg` is simply the sum of all the regularization terms
        for each individual layer.
        '''
        wt_reg = 0
        for ind in self.wt_layer_inds:
            wt_reg = wt_reg + 1/2 * self.reg * np.sum(self.layers[ind].get_wts() * self.layers[ind].get_wts())
        return wt_reg

    def backward(self, y):
        '''Initiates the backward pass through all the layers of the network.

        Parameters:
        -----------
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
            shape=(B,) for mini-batch size B.

        Returns:
        -----------
        None

        TODO:
        1. Initialize d_upstream, d_wts, d_b to None.
        2. Loop through the network layers in REVERSE ORDER, calling the `Layer` backward method.
            Remember that the output of layer.backward() becomes the d_upstream to the next layer down.
            We don't care about d_wts, d_b in this method (computed/stored in Layer).
        '''
        d_upstream = None
        d_wts = None
        d_b = None

        # loop through the list in reversed order
        for layer in reversed(self.layers): 
            #call the backward method of each layer
            d_upstream_updated, d_wts, d_b = layer.backward(d_upstream, y)
            d_upstream = d_upstream_updated # use the output as next layer's upstream gradient


class ConvNet4(Network):
    '''
    Makes a ConvNet4 network with the following layers: Conv2D -> MaxPooling2D -> Dense -> Dense

    1. Convolution (net-in), Relu (net-act).
    2. Max pool 2D (net-in), linear (net-act).
    3. Dense (net-in), Relu (net-act).
    4. Dense (net-in), soft-max (net-act).
    '''
    def __init__(self, input_shape=(3, 32, 32), n_kers=(32,), ker_sz=(7,), dense_interior_units=(100,),
                 pooling_sizes=(2,), pooling_strides=(2,), n_classes=10, wt_scale=1e-3, reg=0, verbose=False, additionalLayer = False):
        '''
        Parameters:
        -----------
        input_shape: tuple. Shape of a SINGLE input sample (no mini-batch). By default: (n_chans, img_y, img_x)
        n_kers: tuple. Number of kernels/units in the 1st convolution layer. Format is (32,), which is a tuple
            rather than just an int. The reasoning is that if you wanted to create another Conv2D layer, say with 16
            units, n_kers would then be (32, 16). Thus, this format easily allows us to make the net deeper.
        ker_sz: tuple. x/y size of each convolution filter. Format is (7,), which means make 7x7 filters in the FIRST
            Conv2D layer. If we had another Conv2D layer with filters size 5x5, it would be ker_sz=(7,5)
        dense_interior_units: tuple. Number of hidden units in each dense layer. Same format as above.
            NOTE: Does NOT include the output layer, which has # units = # classes.
        pooling_sizes: tuple. Pooling extent in the i-th MaxPooling2D layer.  Same format as above.
        pooling_strides: tuple. Pooling stride in the i-th MaxPooling2D layer.  Same format as above.
        n_classes: int. Number of classes in the input. This will become the number of units in the Output Dense layer.
        wt_scale: float. Global weight scaling to use for all layers with weights
        reg: float. Regularization strength
        verbose: bool. Do we want to term network-related debug print outs on?
            NOTE: This is different than per-layer verbose settings, which are turned manually on below.

        TODO:
        1. Assemble the layers of the network and add them (in order) to `self.layers`.
        2. Remember to define self.wt_layer_inds as the list indicies in self.layers that have weights.
        '''
        super().__init__(reg, verbose)

        n_chans, h, w = input_shape
        ker_list = list(n_kers)
        strides, = pooling_strides

        if additionalLayer == False:
            # 1) Input convolutional layer
            conv_layer = layer.Conv2D(0, 'conv2', n_kers=ker_list[0], ker_sz=ker_sz[0], n_chans=n_chans, wt_scale = wt_scale, activation = "relu", reg = reg, verbose = verbose)

            # 2) 2x2 max pooling layer
            pool_layer = layer.MaxPooling2D(1, 'pool', pool_size=pooling_sizes[0], strides=strides, activation = "linear", reg = reg, verbose = verbose)

            # 3) Dense layer
            hidden_layer1 = layer.Dense(2, 'hidden1', units=dense_interior_units[0], n_units_prev_layer=np.prod(ker_list)*int(h/strides)*int(w/strides), wt_scale = wt_scale, reg = reg, activation='relu', verbose=verbose)

            # 4) Dense softmax output layer
            hidden_layer2 = layer.Dense(3, 'hidden2', units=n_classes, n_units_prev_layer=dense_interior_units[0], activation='softmax', reg = reg, wt_scale = wt_scale, verbose=verbose)
        
            self.wt_layer_inds = [0, 2, 3]
            self.layers = [conv_layer, pool_layer, hidden_layer1, hidden_layer2]

        #Extension: Add another layer of conv2, maxpooling, and hidden layer
        if additionalLayer == True:
            ker_list = list(n_kers)

            # 1) Input convolutional layer
            conv_layer = layer.Conv2D(0, 'conv2', n_kers=ker_list[0], ker_sz=ker_sz[0], n_chans=n_chans, wt_scale = wt_scale, activation = "relu", reg = reg, verbose = verbose)

            # 2) 2x2 max pooling layer
            pool_layer = layer.MaxPooling2D(1, 'pool', pool_size=pooling_sizes[0], strides=strides, activation = "linear", reg = reg, verbose = verbose)

            # 3) Input convolutional layer #2
            conv_layer2 = layer.Conv2D(2, 'conv2', n_kers=ker_list[1], ker_sz=ker_sz[1], n_chans=ker_list[0], wt_scale = wt_scale, activation = "relu", reg = reg, verbose = verbose)

            # 4) 2x2 max pooling layer #2
            pool_layer2 = layer.MaxPooling2D(3, 'pool', pool_size=pooling_sizes[1], strides=strides, activation = "linear", reg = reg, verbose = verbose)

            # 5) Dense layer
            # hidden_layer1 = layer.Dense(4, 'hidden1', units=dense_interior_units[0], n_units_prev_layer=ker_list[1]*int(h/strides/strides)*int(w/strides/strides), wt_scale = wt_scale, reg = reg, activation='relu', verbose=verbose)
            hidden_layer1 = layer.Dense(4, 'hidden1', units=dense_interior_units[0], n_units_prev_layer=ker_list[1]*int(h/strides/strides)*int(w/strides/strides), wt_scale = wt_scale, reg = reg, activation='relu', verbose=verbose)

            # 6) Dense softmax layer (original output layer)
            hidden_layer2 = layer.Dense(5, 'hidden2', units=dense_interior_units[1], n_units_prev_layer=dense_interior_units[0], activation='softmax', reg = reg, wt_scale = wt_scale, verbose=verbose)
        
            # 7) Additional Dense layer as new output layer
            hidden_layer3 = layer.Dense(6, 'hidden3', units=n_classes, n_units_prev_layer=dense_interior_units[1], activation='softmax', reg = reg, wt_scale = wt_scale, verbose=verbose)
        
            self.wt_layer_inds = [0, 2, 4, 5, 6]
            self.layers = [conv_layer, pool_layer, conv_layer2, pool_layer2, hidden_layer1, hidden_layer2, hidden_layer3]
            # self.wt_layer_inds = [0, 2, 4, 5]
            # self.layers = [conv_layer, pool_layer, conv_layer2, pool_layer2, hidden_layer1, hidden_layer2]


class ConvNet4Accel(Network):
    def __init__(self, input_shape=(3, 32, 32), n_kers=(32,), ker_sz=(7,), dense_interior_units=(100,),
                 pooling_sizes=(2,), pooling_strides=(2,), n_classes=10, wt_scale=1e-3, reg=0, verbose=False, additionalLayer = False):
        super().__init__(reg, verbose)

        n_chans, h, w = input_shape
        ker_list = list(n_kers)
        strides, = pooling_strides

        # 1) Input convolutional layer
        conv_layer = accelerated_layer.Conv2DAccel(0, 'conv2', n_kers=np.prod(ker_list), ker_sz=ker_sz[0], n_chans=n_chans, wt_scale = wt_scale, activation = "relu", reg = reg, verbose = verbose)

        # 2) 2x2 max pooling layer
        pool_layer = accelerated_layer.MaxPooling2DAccel(1, 'pool', pool_size=pooling_sizes[0], strides=strides, activation = "linear", reg = reg, verbose = verbose)

        # 3) Dense layer
        hidden_layer1 = layer.Dense(2, 'hidden1', units=dense_interior_units[0], n_units_prev_layer=np.prod(ker_list)*int(h/strides)*int(w/strides), wt_scale = wt_scale, reg = reg, activation='relu', verbose=verbose)

        # 4) Dense softmax output layer
        hidden_layer2 = layer.Dense(3, 'hidden2', units=n_classes, n_units_prev_layer=dense_interior_units[0], activation='softmax', reg = reg, wt_scale = wt_scale, verbose=verbose)
        
        self.wt_layer_inds = [0, 2, 3]
        self.layers = [conv_layer, pool_layer, hidden_layer1, hidden_layer2]

        #Extension: Add another layer of conv2, maxpooling, and hidden layer
        if additionalLayer == True:
            ker_list = list(n_kers)

            # 1) Input convolutional layer
            conv_layer = accelerated_layer.Conv2DAccel(0, 'conv2', n_kers=ker_list[0], ker_sz=ker_sz[0], n_chans=n_chans, wt_scale = wt_scale, activation = "relu", reg = reg, verbose = verbose)

            # 2) 2x2 max pooling layer
            pool_layer = accelerated_layer.MaxPooling2DAccel(1, 'pool', pool_size=pooling_sizes[0], strides=strides, activation = "linear", reg = reg, verbose = verbose)

            # 3) Input convolutional layer #2
            conv_layer2 = accelerated_layer.Conv2DAccel(2, 'conv2', n_kers=ker_list[1], ker_sz=ker_sz[1], n_chans=ker_list[0], wt_scale = wt_scale, activation = "relu", reg = reg, verbose = verbose)

            # 4) 2x2 max pooling layer #2
            pool_layer2 = accelerated_layer.MaxPooling2DAccel(3, 'pool', pool_size=pooling_sizes[1], strides=strides, activation = "linear", reg = reg, verbose = verbose)

            # 5) Dense layer
            # hidden_layer1 = layer.Dense(4, 'hidden1', units=dense_interior_units[0], n_units_prev_layer=ker_list[1]*int(h/strides/strides)*int(w/strides/strides), wt_scale = wt_scale, reg = reg, activation='relu', verbose=verbose)
            hidden_layer1 = layer.Dense(4, 'hidden1', units=dense_interior_units[0], n_units_prev_layer=ker_list[1]*int(h/strides/strides)*int(w/strides/strides), wt_scale = wt_scale, reg = reg, activation='relu', verbose=verbose)

            # 6) Dense softmax output layer
            hidden_layer2 = layer.Dense(5, 'hidden2', units=dense_interior_units[1], n_units_prev_layer=dense_interior_units[0], activation='softmax', reg = reg, wt_scale = wt_scale, verbose=verbose)
        
            # 7) Additional Dense layer
            hidden_layer3 = layer.Dense(6, 'hidden3', units=n_classes, n_units_prev_layer=dense_interior_units[1], activation='softmax', reg = reg, wt_scale = wt_scale, verbose=verbose)
        
            self.wt_layer_inds = [0, 2, 4, 5, 6]
            self.layers = [conv_layer, pool_layer, conv_layer2, pool_layer2, hidden_layer1, hidden_layer2, hidden_layer3]