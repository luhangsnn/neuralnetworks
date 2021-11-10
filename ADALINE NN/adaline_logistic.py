import adaline
import numpy as np 

class AdalineLogistic(adaline.Adaline):
    def __init__(self):
        super().__init__()

    def activation(self, net_in):
        '''
        sigmoid activation function
        '''
        net_act = 1/(1 + np.exp(-net_in))
        return net_act

    def predict(self, features):
        '''
        The predicted classes are = or +1 for each input feature vector. 
        '''
        netIn = self.net_input(features)
        netAct = self.activation(netIn)
        pred = np.zeros(netAct.shape)
        for i in range (netAct.shape[0]):
            if netAct[i] >= 0.5:
                pred[i] = 1
        return pred.astype(int)

    def compute_loss(self, y, net_act):
        ''' 
        cross-entropy loss function
        -----------------------------------
        y: ndarray. Shape = [Num samples N,]
            True classes corresponding to each input sample in a training epoch (coded as -1 or +1).
        net_act: ndarray. Shape = [Num samples N,]
            Output neuron's activation value (after activation function is applied)
        '''
        entropy = -y * np.log(net_act) - (1-y) * np.log(1-net_act)
        sum = entropy.sum()
        return sum
