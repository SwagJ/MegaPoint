import tensorflow as tf
import numpy as numpy

class ConstantWeightsInitializer(object):
    def __init__(self, weightsDictionary):
        """ Initialize weights in the Hourglass model
        Arguments:
            weightsDictionary {dictionary} -- dictionary of numpy arrays holding the weights of each
        """
        self.weightsDictionary = weightsDictionary
            
    def conv2d_kernel(self, path, index=0):
        if index > 0:
            conv2dChoice = 'conv2d_{}/kernel'.format(index)
        else:
            conv2dChoice = 'conv2d/kernel'
        return tf.constant_initializer(self.weightsDictionary[path + conv2dChoice])            
    
    def conv2d_bias(self, path, index=0):
        if index > 0:
            conv2dChoice = 'conv2d_{}/bias'.format(index)
        else:
            conv2dChoice = 'conv2d/bias'
        return tf.constant_initializer(self.weightsDictionary[path +conv2dChoice])
    
    def BN_mean(self, path, index=0):
        if index > 0:
            bnChoice = 'batch_normalization_{}/moving_mean'.format(index)
        else:
            bnChoice = 'batch_normalization/moving_mean'
        return tf.constant_initializer(self.weightsDictionary[path + bnChoice])
    
    def BN_variance(self, path, index=0):
        if index > 0:
            bnChoice = 'batch_normalization_{}/moving_variance'.format(index)
        else:
            bnChoice = 'batch_normalization/moving_variance' 
        return tf.constant_initializer(self.weightsDictionary[path + bnChoice])
    
    def BN_gamma(self, path, index=0):
        if index > 0:
            bnChoice = 'batch_normalization_{}/gamma'.format(index)
        else:
            bnChoice = 'batch_normalization/gamma'
        return tf.constant_initializer(self.weightsDictionary[path + bnChoice])
    
    def BN_beta(self, path, index=0):
        if index > 0:
            bnChoice = 'batch_normalization_{}/beta'.format(index)
        else:
            bnChoice = 'batch_normalization/beta'
        return tf.constant_initializer(self.weightsDictionary[path + bnChoice])
