import tensorflow as tf
import numpy as numpy

class ConstantWeightsInitializer(object):
    def __init__(self, weights_dictionary, debug_mode=False):
        """ Initialize weights in the Hourglass model
        Arguments:
            weights_dictionary {dictionary} -- dictionary of numpy arrays holding the weights of each
        """
        self.weights_dictionary = weights_dictionary
        self.debug_mode = debug_mode
        if debug_mode:
            self.all_used_paths = []
    def conv2d(self, path):
        return {
            'kernel_initializer' : self.conv2d_kernel(path),
            'bias_initializer'   : self.conv2d_bias(path)
        }
    def BN(self, path):
        return {
                'beta_initializer' : self.BN_beta(path),
                'gamma_initializer' : self.BN_gamma(path),
                'moving_mean_initializer' : self.BN_mean(path),
                'moving_variance_initializer' : self.BN_variance(path)
        }
    def conv2d_kernel(self, path):
        conv2dChoice = 'conv/kernel'
        if self.debug_mode:
            self.all_used_paths.append(path + conv2dChoice)
        return tf.constant_initializer(self.weights_dictionary[path + conv2dChoice])
    
    def conv2d_bias(self, path):
        conv2dChoice = 'conv/bias'
        if self.debug_mode:
            self.all_used_paths.append(path + conv2dChoice)
        return tf.constant_initializer(self.weights_dictionary[path +conv2dChoice])
    
    def BN_mean(self, path):
        bnChoice = 'bn/moving_mean'
        if self.debug_mode:
            self.all_used_paths.append(path + bnChoice)
        return tf.constant_initializer(self.weights_dictionary[path + bnChoice])
    
    def BN_variance(self, path):
        bnChoice = 'bn/moving_variance'
        if self.debug_mode:
            self.all_used_paths.append(path + bnChoice)
        return tf.constant_initializer(self.weights_dictionary[path + bnChoice])
    
    def BN_gamma(self, path):
        bnChoice = 'bn/gamma'
        if self.debug_mode:
            self.all_used_paths.append(path + bnChoice)
        return tf.constant_initializer(self.weights_dictionary[path + bnChoice])
    
    def BN_beta(self, path):
        bnChoice = 'bn/beta'
        if self.debug_mode:
            self.all_used_paths.append(path + bnChoice)
        return tf.constant_initializer(self.weights_dictionary[path + bnChoice])
