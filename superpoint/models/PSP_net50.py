# taken from https://github.com/wwwfan628/PSPNet-tensorflow
import tensorflow as tf
import numpy as np

DEFAULT_DATAFORMAT = 'channels_last'

class ConstantWeightsInitializer(object):
    def __init__(self, path):
        self.weights_dict = np.load(path,allow_pickle=True).item()
    def conv2D_init(self, name, bias=False):
        kernel_init = tf.constant_initializer(self.weights_dict[name+'/kernel'])
        if bias:
            bias_init = tf.constant_initializer(self.weights_dict[name+'/biases'])
            return {'kernel_initializer' : kernel_init,
                    'bias_initializer'   : bias_init}
        else:
            return {'kernel_initializer' : kernel_init}

    def bn_init(self, name):
        gamma_init = tf.constant_initializer(self.weights_dict[name+'/gamma'])
        beta_init = tf.constant_initializer(self.weights_dict[name+'/beta'])
        moving_mean_init = tf.constant_initializer(self.weights_dict[name+'/moving_mean'])
        moving_variance_init = tf.constant_initializer(self.weights_dict[name+'/moving_variance'])
        return {
            'gamma_initializer' :gamma_init,
            'beta_initializer' :beta_init,
            'moving_mean_initializer' :moving_mean_init,
            'moving_variance_initializer' :moving_variance_init}

class PSPNet50(tf.keras.Model):
    def __init__(self, num_classes=150, checkpoint_npy_path=None):
        super(PSPNet50, self).__init__()
        if checkpoint_npy_path: # initializa with checkpoint
            initializer = ConstantWeightsInitializer(checkpoint_npy_path)
            # PSPNet layers with initializer
            name = 'conv1_1_3x3_s2'
            
            self.conv1_1_3x3_s2 = tf.keras.layers.Conv2D(64, [3, 3], [2, 2], padding='SAME', use_bias=False,name=name,**initializer.conv2D_init(name=name))
            name = 'conv1_1_3x3_s2_bn'

            self.conv1_1_3x3_s2_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name))
            name = 'conv1_2_3x3'
            
            self.conv1_2_3x3 = tf.keras.layers.Conv2D(64, [3, 3], [1, 1], padding='SAME', use_bias=False,name=name,**initializer.conv2D_init(name=name))
            name = 'conv1_2_3x3_bn'

            self.conv1_2_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name=name,**initializer.bn_init(name=name))
            name = 'conv1_3_3x3'
            
            self.conv1_3_3x3 = tf.keras.layers.Conv2D(128, [3, 3], [1, 1], padding='SAME', use_bias=False,name=name,**initializer.conv2D_init(name=name))
            name = 'conv1_3_3x3_bn'

            self.conv1_3_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name=name, **initializer.bn_init(name=name))
            name = 'conv2_1_1x1_proj'
            
            self.conv2_1_1x1_proj = tf.keras.layers.Conv2D(256, [1, 1], [1, 1], use_bias=False, name=name,**initializer.conv2D_init(name=name))
            name = 'conv2_1_1x1_proj_bn'

            self.conv2_1_1x1_proj_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))

            name = 'conv2_1_1x1_reduce'
            
            self.conv2_1_1x1_reduce = tf.keras.layers.Conv2D(64, [1, 1], [1, 1], use_bias=False,name=name,**initializer.conv2D_init(name=name))
            name = 'conv2_1_1x1_reduce_bn'

            self.conv2_1_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))
            name = 'conv2_1_3x3'
            
            self.conv2_1_3x3 = tf.keras.layers.Conv2D(64, [3, 3], [1, 1], use_bias=False, name=name,**initializer.conv2D_init(name=name))
            name = 'conv2_1_3x3_bn'

            self.conv2_1_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name=name, **initializer.bn_init(name=name))
            name = 'conv2_1_1x1_increase'
            
            self.conv2_1_1x1_increase = tf.keras.layers.Conv2D(256, [1, 1], [1, 1], use_bias=False,name=name,**initializer.conv2D_init(name=name))
            name = 'conv2_1_1x1_increase_bn'

            self.conv2_1_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))

            name = 'conv2_2_1x1_reduce'
            
            self.conv2_2_1x1_reduce = tf.keras.layers.Conv2D(64, [1, 1], [1, 1], use_bias=False,name=name,**initializer.conv2D_init(name=name))
            name = 'conv2_2_1x1_reduce_bn'

            self.conv2_2_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))
            name = 'conv2_2_3x3'

            self.conv2_2_3x3 = tf.keras.layers.Conv2D(64, [3, 3], [1, 1], use_bias=False, name=name, **initializer.conv2D_init(name=name))
            name = 'conv2_2_3x3_bn'

            self.conv2_2_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))
            name = 'conv2_2_1x1_increase'

            self.conv2_2_1x1_increase = tf.keras.layers.Conv2D(256, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv2_2_1x1_increase_bn'

            self.conv2_2_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))

            name = 'conv2_3_1x1_reduce'

            self.conv2_3_1x1_reduce = tf.keras.layers.Conv2D(64, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv2_3_1x1_reduce_bn'

            self.conv2_3_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))
            name = 'conv2_3_3x3'

            self.conv2_3_3x3 = tf.keras.layers.Conv2D(64, [3, 3], [1, 1], use_bias=False, name=name, **initializer.conv2D_init(name=name))
            name = 'conv2_3_3x3_bn'

            self.conv2_3_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name=name, **initializer.bn_init(name=name))
            name = 'conv2_3_1x1_increase'

            self.conv2_3_1x1_increase = tf.keras.layers.Conv2D(256, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv2_3_1x1_increase_bn'

            self.conv2_3_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))

            name = 'conv3_1_1x1_proj'

            self.conv3_1_1x1_proj = tf.keras.layers.Conv2D(512, [1, 1], [2, 2], use_bias=False, name=name, **initializer.conv2D_init(name=name))
            name = 'conv3_1_1x1_proj_bn'

            self.conv3_1_1x1_proj_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))

            name = 'conv3_1_1x1_reduce'

            self.conv3_1_1x1_reduce = tf.keras.layers.Conv2D(128, [1, 1], [2, 2], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv3_1_1x1_reduce_bn'

            self.conv3_1_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))
            name = 'conv3_1_3x3'

            self.conv3_1_3x3 = tf.keras.layers.Conv2D(128, [3, 3], [1, 1], use_bias=False, name=name, **initializer.conv2D_init(name=name))
            name = 'conv3_1_3x3_bn'

            self.conv3_1_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name=name, **initializer.bn_init(name=name))
            name = 'conv3_1_1x1_increase'

            self.conv3_1_1x1_increase = tf.keras.layers.Conv2D(512, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv3_1_1x1_increase_bn'

            self.conv3_1_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))

            name = 'conv3_2_1x1_reduce'

            self.conv3_2_1x1_reduce = tf.keras.layers.Conv2D(128, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv3_2_1x1_reduce_bn'

            self.conv3_2_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))
            name = 'conv3_2_3x3'

            self.conv3_2_3x3 = tf.keras.layers.Conv2D(128, [3, 3], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv3_2_3x3_bn'

            self.conv3_2_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name=name, **initializer.bn_init(name=name))
            name = 'conv3_2_1x1_increase'

            self.conv3_2_1x1_increase = tf.keras.layers.Conv2D(512, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv3_2_1x1_increase_bn'

            self.conv3_2_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))

            name = 'conv3_3_1x1_reduce'

            self.conv3_3_1x1_reduce = tf.keras.layers.Conv2D(128, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv3_3_1x1_reduce_bn'

            self.conv3_3_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))
            name = 'conv3_3_3x3'

            self.conv3_3_3x3 = tf.keras.layers.Conv2D(128, [3, 3], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv3_3_3x3_bn'

            self.conv3_3_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name=name, **initializer.bn_init(name=name))
            name = 'conv3_3_1x1_increase'

            self.conv3_3_1x1_increase = tf.keras.layers.Conv2D(512, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv3_3_1x1_increase_bn'

            self.conv3_3_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))

            name = 'conv3_4_1x1_reduce'

            self.conv3_4_1x1_reduce = tf.keras.layers.Conv2D(128, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv3_4_1x1_reduce_bn'

            self.conv3_4_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))
            name = 'conv3_4_3x3'

            self.conv3_4_3x3 = tf.keras.layers.Conv2D(128, [3, 3], [1, 1], use_bias=False, name=name, **initializer.conv2D_init(name=name))
            name = 'conv3_4_3x3_bn'

            self.conv3_4_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name=name, **initializer.bn_init(name=name))
            name = 'conv3_4_1x1_increase'

            self.conv3_4_1x1_increase = tf.keras.layers.Conv2D(512, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv3_4_1x1_increase_bn'

            self.conv3_4_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))

            name = 'conv4_1_1x1_proj'

            self.conv4_1_1x1_proj = tf.keras.layers.Conv2D(1024, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv4_1_1x1_proj_bn'

            self.conv4_1_1x1_proj_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))

            name = 'conv4_1_1x1_reduce'

            self.conv4_1_1x1_reduce = tf.keras.layers.Conv2D(256, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv4_1_1x1_reduce_bn'

            self.conv4_1_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))
            name = 'conv4_1_3x3'

            self.conv4_1_3x3 = tf.keras.layers.Conv2D(256, [3, 3], dilation_rate=2, use_bias=False, name=name, **initializer.conv2D_init(name=name))
            name = 'conv4_1_3x3_bn'

            self.conv4_1_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name=name, **initializer.bn_init(name=name))
            name = 'conv4_1_1x1_increase'

            self.conv4_1_1x1_increase = tf.keras.layers.Conv2D(1024, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv4_1_1x1_increase_bn'

            self.conv4_1_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))

            name = 'conv4_2_1x1_reduce'

            self.conv4_2_1x1_reduce = tf.keras.layers.Conv2D(256, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv4_2_1x1_reduce_bn'

            self.conv4_2_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))
            name = 'conv4_2_3x3'

            self.conv4_2_3x3 = tf.keras.layers.Conv2D(256, [3, 3], dilation_rate=2, use_bias=False, name=name, **initializer.conv2D_init(name=name))
            name = 'conv4_2_3x3_bn'

            self.conv4_2_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name=name, **initializer.bn_init(name=name))
            name = 'conv4_2_1x1_increase'

            self.conv4_2_1x1_increase = tf.keras.layers.Conv2D(1024, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv4_2_1x1_increase_bn'

            self.conv4_2_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))

            name = 'conv4_3_1x1_reduce'

            self.conv4_3_1x1_reduce = tf.keras.layers.Conv2D(256, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv4_3_1x1_reduce_bn'

            self.conv4_3_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))
            name = 'conv4_3_3x3'

            self.conv4_3_3x3 = tf.keras.layers.Conv2D(256, [3, 3], dilation_rate=2, use_bias=False, name=name, **initializer.conv2D_init(name=name))
            name = 'conv4_3_3x3_bn'

            self.conv4_3_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name=name, **initializer.bn_init(name=name))
            name = 'conv4_3_1x1_increase'

            self.conv4_3_1x1_increase = tf.keras.layers.Conv2D(1024, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv4_3_1x1_increase_bn'

            self.conv4_3_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))

            name = 'conv4_4_1x1_reduce'

            self.conv4_4_1x1_reduce = tf.keras.layers.Conv2D(256, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv4_4_1x1_reduce_bn'

            self.conv4_4_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))
            name = 'conv4_4_3x3'

            self.conv4_4_3x3 = tf.keras.layers.Conv2D(256, [3, 3], dilation_rate=2, use_bias=False, name=name, **initializer.conv2D_init(name=name))
            name = 'conv4_4_3x3_bn'

            self.conv4_4_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name=name, **initializer.bn_init(name=name))
            name = 'conv4_4_1x1_increase'

            self.conv4_4_1x1_increase = tf.keras.layers.Conv2D(1024, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv4_4_1x1_increase_bn'

            self.conv4_4_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))

            name = 'conv4_5_1x1_reduce'

            self.conv4_5_1x1_reduce = tf.keras.layers.Conv2D(256, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv4_5_1x1_reduce_bn'

            self.conv4_5_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))
            name = 'conv4_5_3x3'

            self.conv4_5_3x3 = tf.keras.layers.Conv2D(256, [3, 3], dilation_rate=2, use_bias=False, name=name, **initializer.conv2D_init(name=name))
            name = 'conv4_5_3x3_bn'

            self.conv4_5_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name=name, **initializer.bn_init(name=name))
            name = 'conv4_5_1x1_increase'

            self.conv4_5_1x1_increase = tf.keras.layers.Conv2D(1024, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv4_5_1x1_increase_bn'

            self.conv4_5_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))

            name = 'conv4_6_1x1_reduce'

            self.conv4_6_1x1_reduce = tf.keras.layers.Conv2D(256, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv4_6_1x1_reduce_bn'

            self.conv4_6_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))
            name = 'conv4_6_3x3'

            self.conv4_6_3x3 = tf.keras.layers.Conv2D(256, [3, 3], dilation_rate=2, use_bias=False, name=name, **initializer.conv2D_init(name=name))
            name = 'conv4_6_3x3_bn'

            self.conv4_6_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name=name, **initializer.bn_init(name=name))
            name = 'conv4_6_1x1_increase'

            self.conv4_6_1x1_increase = tf.keras.layers.Conv2D(1024, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv4_6_1x1_increase_bn'

            self.conv4_6_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))

            name = 'conv5_1_1x1_proj'

            self.conv5_1_1x1_proj = tf.keras.layers.Conv2D(2048, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv5_1_1x1_proj_bn'

            self.conv5_1_1x1_proj_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))

            name = 'conv5_1_1x1_reduce'

            self.conv5_1_1x1_reduce = tf.keras.layers.Conv2D(512, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv5_1_1x1_reduce_bn'

            self.conv5_1_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))
            name = 'conv5_1_3x3'

            self.conv5_1_3x3 = tf.keras.layers.Conv2D(512, [3, 3], dilation_rate=4, use_bias=False, name=name, **initializer.conv2D_init(name=name))
            name = 'conv5_1_3x3_bn'

            self.conv5_1_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))
            name = 'conv5_1_1x1_increase'

            self.conv5_1_1x1_increase = tf.keras.layers.Conv2D(2048, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv5_1_1x1_increase_bn'

            self.conv5_1_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))

            name = 'conv5_2_1x1_reduce'

            self.conv5_2_1x1_reduce = tf.keras.layers.Conv2D(512, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv5_2_1x1_reduce_bn'

            self.conv5_2_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))
            name = 'conv5_2_3x3'

            self.conv5_2_3x3 = tf.keras.layers.Conv2D(512, [3, 3], dilation_rate=4, use_bias=False, name=name, **initializer.conv2D_init(name=name))
            name = 'conv5_2_3x3_bn'

            self.conv5_2_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name=name, **initializer.bn_init(name=name))
            name = 'conv5_2_1x1_increase'

            self.conv5_2_1x1_increase = tf.keras.layers.Conv2D(2048, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv5_2_1x1_increase_bn'

            self.conv5_2_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))

            name = 'conv5_3_1x1_reduce'

            self.conv5_3_1x1_reduce = tf.keras.layers.Conv2D(512, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv5_3_1x1_reduce_bn'

            self.conv5_3_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))
            name = 'conv5_3_3x3'

            self.conv5_3_3x3 = tf.keras.layers.Conv2D(512, [3, 3], dilation_rate=4, use_bias=False, name=name, **initializer.conv2D_init(name=name))
            name = 'conv5_3_3x3_bn'

            self.conv5_3_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name=name, **initializer.bn_init(name=name))
            name = 'conv5_3_1x1_increase'

            self.conv5_3_1x1_increase = tf.keras.layers.Conv2D(2048, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv5_3_1x1_increase_bn'

            self.conv5_3_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))

            name = 'conv5_3_pool1_conv'

            self.conv5_3_pool1_conv = tf.keras.layers.Conv2D(512, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv5_3_pool1_conv_bn'

            self.conv5_3_pool1_conv_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))

            name = 'conv5_3_pool2_conv'

            self.conv5_3_pool2_conv = tf.keras.layers.Conv2D(512, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv5_3_pool2_conv_bn'

            self.conv5_3_pool2_conv_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))

            name = 'conv5_3_pool3_conv'

            self.conv5_3_pool3_conv = tf.keras.layers.Conv2D(512, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv5_3_pool3_conv_bn'

            self.conv5_3_pool3_conv_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))

            name = 'conv5_3_pool6_conv'

            self.conv5_3_pool6_conv = tf.keras.layers.Conv2D(512, [1, 1], [1, 1], use_bias=False,name=name, **initializer.conv2D_init(name=name))
            name = 'conv5_3_pool6_conv_bn'

            self.conv5_3_pool6_conv_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name=name, **initializer.bn_init(name=name))

            name = 'conv5_4'

            self.conv5_4 = tf.keras.layers.Conv2D(512, [3, 3], [1, 1], padding='SAME', use_bias=False, name=name, **initializer.conv2D_init(name=name))
            name = 'conv5_4_bn'

            self.conv5_4_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name=name, **initializer.bn_init(name=name))
            name = 'conv6'

            self.conv6 = tf.keras.layers.Conv2D(num_classes, [1, 1], [1, 1], use_bias=True,name=name, **initializer.conv2D_init(name=name,bias=True))
        else:
            self.conv1_1_3x3_s2 = tf.keras.layers.Conv2D(64, [3, 3], [2, 2], padding='SAME', use_bias=False, name='conv1_1_3x3_s2')
            self.conv1_1_3x3_s2_bn = tf.keras.layers.BatchNormalization(momentum=0.95,epsilon=1e-5,name='conv1_1_3x3_s2_bn')
            self.conv1_2_3x3 = tf.keras.layers.Conv2D(64, [3, 3], [1, 1], padding='SAME', use_bias=False, name='conv1_2_3x3')
            self.conv1_2_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95,epsilon=1e-5,name='conv1_2_3x3_bn')
            self.conv1_3_3x3 = tf.keras.layers.Conv2D(128, [3, 3], [1, 1], padding='SAME', use_bias=False,name='conv1_3_3x3')
            self.conv1_3_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95,epsilon=1e-5,name='conv1_3_3x3_bn')
            self.conv2_1_1x1_proj = tf.keras.layers.Conv2D(256, [1, 1], [1, 1], use_bias=False,name='conv2_1_1x1_proj')
            self.conv2_1_1x1_proj_bn = tf.keras.layers.BatchNormalization(momentum=0.95,epsilon=1e-5,name='conv2_1_1x1_proj_bn')

            self.conv2_1_1x1_reduce = tf.keras.layers.Conv2D(64, [1,1], [1,1], use_bias=False,name='conv2_1_1x1_reduce')
            self.conv2_1_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv2_1_1x1_reduce_bn')
            self.conv2_1_3x3 = tf.keras.layers.Conv2D(64, [3,3], [1,1], use_bias=False,name='conv2_1_3x3')
            self.conv2_1_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv2_1_3x3_bn')
            self.conv2_1_1x1_increase = tf.keras.layers.Conv2D(256, [1,1], [1,1], use_bias=False,name='conv2_1_1x1_increase')
            self.conv2_1_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv2_1_1x1_increase_bn')

            self.conv2_2_1x1_reduce = tf.keras.layers.Conv2D(64, [1,1], [1,1], use_bias=False,name='conv2_2_1x1_reduce')
            self.conv2_2_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv2_2_1x1_reduce_bn')
            self.conv2_2_3x3 = tf.keras.layers.Conv2D(64, [3,3], [1,1], use_bias=False,name='conv2_2_3x3')
            self.conv2_2_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv2_2_3x3_bn')
            self.conv2_2_1x1_increase = tf.keras.layers.Conv2D(256, [1,1], [1,1], use_bias=False,name='conv2_2_1x1_increase')
            self.conv2_2_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv2_2_1x1_increase_bn')

            self.conv2_3_1x1_reduce = tf.keras.layers.Conv2D(64, [1, 1], [1, 1], use_bias=False,name='conv2_3_1x1_reduce')
            self.conv2_3_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv2_3_1x1_reduce_bn')
            self.conv2_3_3x3 = tf.keras.layers.Conv2D(64, [3, 3], [1, 1], use_bias=False,name='conv2_3_3x3')
            self.conv2_3_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name='conv2_3_3x3_bn')
            self.conv2_3_1x1_increase = tf.keras.layers.Conv2D(256, [1, 1], [1, 1], use_bias=False,name='conv2_3_1x1_increase')
            self.conv2_3_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv2_3_1x1_increase_bn')

            self.conv3_1_1x1_proj = tf.keras.layers.Conv2D(512, [1, 1], [2, 2], use_bias=False,name='conv3_1_1x1_proj')
            self.conv3_1_1x1_proj_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv3_1_1x1_proj_bn')

            self.conv3_1_1x1_reduce = tf.keras.layers.Conv2D(128, [1, 1], [2, 2], use_bias=False,name='conv3_1_1x1_reduce')
            self.conv3_1_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv3_1_1x1_reduce_bn')
            self.conv3_1_3x3 = tf.keras.layers.Conv2D(128, [3, 3], [1, 1], use_bias=False,name='conv3_1_3x3')
            self.conv3_1_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name='conv3_1_3x3_bn')
            self.conv3_1_1x1_increase = tf.keras.layers.Conv2D(512, [1, 1], [1, 1], use_bias=False,name='conv3_1_1x1_increase')
            self.conv3_1_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv3_1_1x1_increase_bn')

            self.conv3_2_1x1_reduce = tf.keras.layers.Conv2D(128, [1, 1], [1, 1], use_bias=False,name='conv3_2_1x1_reduce')
            self.conv3_2_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv3_2_1x1_reduce_bn')
            self.conv3_2_3x3 = tf.keras.layers.Conv2D(128, [3, 3], [1, 1], use_bias=False,name='conv3_2_3x3')
            self.conv3_2_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name='conv3_2_3x3_bn')
            self.conv3_2_1x1_increase = tf.keras.layers.Conv2D(512, [1, 1], [1, 1], use_bias=False, name='conv3_2_1x1_increase')
            self.conv3_2_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv3_2_1x1_increase_bn')

            self.conv3_3_1x1_reduce = tf.keras.layers.Conv2D(128, [1, 1], [1, 1], use_bias=False,name='conv3_3_1x1_reduce')
            self.conv3_3_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv3_3_1x1_reduce_bn')
            self.conv3_3_3x3 = tf.keras.layers.Conv2D(128, [3, 3], [1, 1], use_bias=False,name='conv3_3_3x3')
            self.conv3_3_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name='conv3_3_3x3_bn')
            self.conv3_3_1x1_increase = tf.keras.layers.Conv2D(512, [1, 1], [1, 1], use_bias=False,name='conv3_3_1x1_increase')
            self.conv3_3_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv3_3_1x1_increase_bn')

            self.conv3_4_1x1_reduce = tf.keras.layers.Conv2D(128, [1, 1], [1, 1], use_bias=False,name='conv3_4_1x1_reduce')
            self.conv3_4_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv3_4_1x1_reduce_bn')
            self.conv3_4_3x3 = tf.keras.layers.Conv2D(128, [3, 3], [1, 1], use_bias=False,name='conv3_4_3x3')
            self.conv3_4_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name='conv3_4_3x3_bn')
            self.conv3_4_1x1_increase = tf.keras.layers.Conv2D(512, [1, 1], [1, 1], use_bias=False,name='conv3_4_1x1_increase')
            self.conv3_4_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv3_4_1x1_increase_bn')

            self.conv4_1_1x1_proj = tf.keras.layers.Conv2D(1024, [1, 1], [1, 1], use_bias=False,name='conv4_1_1x1_proj')
            self.conv4_1_1x1_proj_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv4_1_1x1_proj_bn')

            self.conv4_1_1x1_reduce = tf.keras.layers.Conv2D(256, [1, 1], [1, 1], use_bias=False,name='conv4_1_1x1_reduce')
            self.conv4_1_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv4_1_1x1_reduce_bn')
            self.conv4_1_3x3 = tf.keras.layers.Conv2D(256, [3, 3], dilation_rate=2, use_bias=False,name='conv4_1_3x3')
            self.conv4_1_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name='conv4_1_3x3_bn')
            self.conv4_1_1x1_increase = tf.keras.layers.Conv2D(1024, [1, 1], [1, 1], use_bias=False,name='conv4_1_1x1_increase')
            self.conv4_1_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv4_1_1x1_increase_bn')

            self.conv4_2_1x1_reduce = tf.keras.layers.Conv2D(256, [1, 1], [1, 1], use_bias=False,name='conv4_2_1x1_reduce')
            self.conv4_2_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv4_2_1x1_reduce_bn')
            self.conv4_2_3x3 = tf.keras.layers.Conv2D(256, [3, 3], dilation_rate=2, use_bias=False,name='conv4_2_3x3')
            self.conv4_2_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name='conv4_2_3x3_bn')
            self.conv4_2_1x1_increase = tf.keras.layers.Conv2D(1024, [1, 1], [1, 1], use_bias=False,name='conv4_2_1x1_increase')
            self.conv4_2_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv4_2_1x1_increase_bn')

            self.conv4_3_1x1_reduce = tf.keras.layers.Conv2D(256, [1, 1], [1, 1], use_bias=False,name='conv4_3_1x1_reduce')
            self.conv4_3_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv4_3_1x1_reduce_bn')
            self.conv4_3_3x3 = tf.keras.layers.Conv2D(256, [3, 3], dilation_rate=2, use_bias=False,name='conv4_3_3x3')
            self.conv4_3_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name='conv4_3_3x3_bn')
            self.conv4_3_1x1_increase = tf.keras.layers.Conv2D(1024, [1, 1], [1, 1], use_bias=False,name='conv4_3_1x1_increase')
            self.conv4_3_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv4_3_1x1_increase_bn')

            self.conv4_4_1x1_reduce = tf.keras.layers.Conv2D(256, [1, 1], [1, 1], use_bias=False,name='conv4_4_1x1_reduce')
            self.conv4_4_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv4_4_1x1_reduce_bn')
            self.conv4_4_3x3 = tf.keras.layers.Conv2D(256, [3, 3], dilation_rate=2, use_bias=False,name='conv4_4_3x3')
            self.conv4_4_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name='conv4_4_3x3_bn')
            self.conv4_4_1x1_increase = tf.keras.layers.Conv2D(1024, [1, 1], [1, 1], use_bias=False,name='conv4_4_1x1_increase')
            self.conv4_4_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv4_4_1x1_increase_bn')

            self.conv4_5_1x1_reduce = tf.keras.layers.Conv2D(256, [1, 1], [1, 1], use_bias=False,name='conv4_5_1x1_reduce')
            self.conv4_5_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv4_5_1x1_reduce_bn')
            self.conv4_5_3x3 = tf.keras.layers.Conv2D(256, [3, 3], dilation_rate=2, use_bias=False,name='conv4_5_3x3')
            self.conv4_5_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name='conv4_5_3x3_bn')
            self.conv4_5_1x1_increase = tf.keras.layers.Conv2D(1024, [1, 1], [1, 1], use_bias=False,name='conv4_5_1x1_increase')
            self.conv4_5_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv4_5_1x1_increase_bn')

            self.conv4_6_1x1_reduce = tf.keras.layers.Conv2D(256, [1, 1], [1, 1], use_bias=False,name='conv4_6_1x1_reduce')
            self.conv4_6_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv4_6_1x1_reduce_bn')
            self.conv4_6_3x3 = tf.keras.layers.Conv2D(256, [3, 3], dilation_rate=2, use_bias=False,name='conv4_6_3x3')
            self.conv4_6_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name='conv4_6_3x3_bn')
            self.conv4_6_1x1_increase = tf.keras.layers.Conv2D(1024, [1, 1], [1, 1], use_bias=False,name='conv4_6_1x1_increase')
            self.conv4_6_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv4_6_1x1_increase_bn')

            self.conv5_1_1x1_proj = tf.keras.layers.Conv2D(2048, [1, 1], [1, 1], use_bias=False,name='conv5_1_1x1_proj')
            self.conv5_1_1x1_proj_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name='conv5_1_1x1_proj_bn')

            self.conv5_1_1x1_reduce = tf.keras.layers.Conv2D(512, [1, 1], [1, 1], use_bias=False,name='conv5_1_1x1_reduce')
            self.conv5_1_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv5_1_1x1_reduce_bn')
            self.conv5_1_3x3 = tf.keras.layers.Conv2D(512, [3, 3], dilation_rate=4, use_bias=False,name='conv5_1_3x3')
            self.conv5_1_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name='conv5_1_3x3_bn')
            self.conv5_1_1x1_increase = tf.keras.layers.Conv2D(2048, [1, 1], [1, 1], use_bias=False,name='conv5_1_1x1_increase')
            self.conv5_1_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv5_1_1x1_increase_bn')

            self.conv5_2_1x1_reduce = tf.keras.layers.Conv2D(512, [1, 1], [1, 1], use_bias=False,name='conv5_2_1x1_reduce')
            self.conv5_2_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv5_2_1x1_reduce_bn')
            self.conv5_2_3x3 = tf.keras.layers.Conv2D(512, [3, 3], dilation_rate=4, use_bias=False,name='conv5_2_3x3')
            self.conv5_2_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name='conv5_2_3x3_bn')
            self.conv5_2_1x1_increase = tf.keras.layers.Conv2D(2048, [1, 1], [1, 1], use_bias=False,name='conv5_2_1x1_increase')
            self.conv5_2_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv5_2_1x1_increase_bn')

            self.conv5_3_1x1_reduce = tf.keras.layers.Conv2D(512, [1, 1], [1, 1], use_bias=False,name='conv5_3_1x1_reduce')
            self.conv5_3_1x1_reduce_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv5_3_1x1_reduce_bn')
            self.conv5_3_3x3 = tf.keras.layers.Conv2D(512, [3, 3], dilation_rate=4, use_bias=False,name='conv5_3_3x3')
            self.conv5_3_3x3_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name='conv5_3_3x3_bn')
            self.conv5_3_1x1_increase = tf.keras.layers.Conv2D(2048, [1, 1], [1, 1], use_bias=False,name='conv5_3_1x1_increase')
            self.conv5_3_1x1_increase_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv5_3_1x1_increase_bn')

            self.conv5_3_pool1_conv = tf.keras.layers.Conv2D(512, [1, 1], [1, 1], use_bias=False,name='conv5_3_pool1_conv')
            self.conv5_3_pool1_conv_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name='conv5_3_pool1_conv_bn')

            self.conv5_3_pool2_conv = tf.keras.layers.Conv2D(512, [1, 1], [1, 1], use_bias=False,name='conv5_3_pool2_conv')
            self.conv5_3_pool2_conv_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5, name='conv5_3_pool2_conv_bn')

            self.conv5_3_pool3_conv = tf.keras.layers.Conv2D(512, [1, 1], [1, 1], use_bias=False,name='conv5_3_pool3_conv')
            self.conv5_3_pool3_conv_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv5_3_pool3_conv_bn')

            self.conv5_3_pool6_conv = tf.keras.layers.Conv2D(512, [1, 1], [1, 1], use_bias=False,name='conv5_3_pool6_conv')
            self.conv5_3_pool6_conv_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv5_3_pool6_conv_bn')

            self.conv5_4 = tf.keras.layers.Conv2D(512, [3, 3], [1, 1], padding='SAME', use_bias=False,name='conv5_4')
            self.conv5_4_bn = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5,name='conv5_4_bn')
            self.conv6 = tf.keras.layers.Conv2D(num_classes, [1, 1], [1, 1], use_bias=True,name='conv6')

    def call(self, input, is_training=False):
        output_conv1_1_3x3_s2 = self.conv1_1_3x3_s2(input)
        output_conv1_1_3x3_s2_bn = tf.nn.relu(self.conv1_1_3x3_s2_bn(output_conv1_1_3x3_s2, is_training))
        output_conv1_2_3x3 = self.conv1_2_3x3(output_conv1_1_3x3_s2_bn)
        output_conv1_2_3x3_bn = tf.nn.relu(self.conv1_2_3x3_bn(output_conv1_2_3x3, is_training))
        output_conv1_3_3x3 = self.conv1_3_3x3(output_conv1_2_3x3_bn)
        output_conv1_3_3x3_bn = tf.nn.relu(self.conv1_3_3x3_bn(output_conv1_3_3x3,is_training))
        output_pool1_3x3_s2 = tf.nn.max_pool(output_conv1_3_3x3_bn, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', data_format='NHWC', name='pool1_3x3_s2')
        output_conv2_1_1x1_proj = self.conv2_1_1x1_proj(output_pool1_3x3_s2)
        output_conv2_1_1x1_proj_bn = self.conv2_1_1x1_proj_bn(output_conv2_1_1x1_proj, is_training)

        output_conv2_1_1x1_reduce = self.conv2_1_1x1_reduce(output_pool1_3x3_s2)
        output_conv2_1_1x1_reduce_bn = tf.nn.relu(self.conv2_1_1x1_reduce_bn(output_conv2_1_1x1_reduce, is_training))
        pad_mat = np.array([[0, 0], [1, 1], [1, 1], [0, 0]])
        output_padding1 = tf.pad(output_conv2_1_1x1_reduce_bn, paddings=pad_mat, name='padding1')
        output_conv2_1_3x3 = self.conv2_1_3x3(output_padding1)
        output_conv2_1_3x3_bn = tf.nn.relu(self.conv2_1_3x3_bn(output_conv2_1_3x3,is_training))
        output_conv2_1_1x1_increase = self.conv2_1_1x1_increase(output_conv2_1_3x3_bn)
        output_conv2_1_1x1_increase_bn = self.conv2_1_1x1_increase_bn(output_conv2_1_1x1_increase, is_training)

        output_conv2_1 = tf.add_n([output_conv2_1_1x1_proj_bn,output_conv2_1_1x1_increase_bn], name='conv2_1')
        output_conv2_1_relu = tf.nn.relu(output_conv2_1, name='conv2_1/relu')
        output_conv2_2_1x1_reduce = self.conv2_2_1x1_reduce(output_conv2_1_relu)
        output_conv2_2_1x1_reduce_bn = tf.nn.relu(self.conv2_2_1x1_reduce_bn(output_conv2_2_1x1_reduce,is_training))
        pad_mat = np.array([[0, 0], [1, 1], [1, 1], [0, 0]])
        output_padding2 = tf.pad(output_conv2_2_1x1_reduce_bn, paddings=pad_mat, name='padding2')
        output_conv2_2_3x3 = self.conv2_2_3x3(output_padding2)
        output_conv2_2_3x3_bn = tf.nn.relu(self.conv2_2_3x3_bn(output_conv2_2_3x3,is_training))
        output_conv2_2_1x1_increase = self.conv2_2_1x1_increase(output_conv2_2_3x3_bn)
        output_conv2_2_1x1_increase_bn = self.conv2_2_1x1_increase_bn(output_conv2_2_1x1_increase,is_training)

        output_conv2_2 = tf.add_n([output_conv2_1_relu,output_conv2_2_1x1_increase_bn],name='conv2_2')
        output_conv2_2_relu = tf.nn.relu(output_conv2_2, name='conv2_2/relu')
        output_conv2_3_1x1_reduce = self.conv2_3_1x1_reduce(output_conv2_2_relu)
        output_conv2_3_1x1_reduce_bn = tf.nn.relu(self.conv2_3_1x1_reduce_bn(output_conv2_3_1x1_reduce, is_training))
        pad_mat = np.array([[0, 0], [1, 1], [1, 1], [0, 0]])
        output_padding3 = tf.pad(output_conv2_3_1x1_reduce_bn, paddings=pad_mat, name='padding3')
        output_conv2_3_3x3 = self.conv2_3_3x3(output_padding3)
        output_conv2_3_3x3_bn = tf.nn.relu(self.conv2_3_3x3_bn(output_conv2_3_3x3, is_training))
        output_conv2_3_1x1_increase = self.conv2_3_1x1_increase(output_conv2_3_3x3_bn)
        output_conv2_3_1x1_increase_bn = self.conv2_3_1x1_increase_bn(output_conv2_3_1x1_increase, is_training)

        output_conv2_3 = tf.add_n([output_conv2_2_relu, output_conv2_3_1x1_increase_bn], name='conv2_3')
        output_conv2_3_relu = tf.nn.relu(output_conv2_3, name='conv2_3/relu')
        output_conv3_1_1x1_proj = self.conv3_1_1x1_proj(output_conv2_3_relu)
        output_conv3_1_1x1_proj_bn = self.conv3_1_1x1_proj_bn(output_conv3_1_1x1_proj,is_training)

        output_conv3_1_1x1_reduce = self.conv3_1_1x1_reduce(output_conv2_3_relu)
        output_conv3_1_1x1_reduce_bn = tf.nn.relu(self.conv3_1_1x1_reduce_bn(output_conv3_1_1x1_reduce, is_training))
        pad_mat = np.array([[0, 0], [1, 1], [1, 1], [0, 0]])
        output_padding4 = tf.pad(output_conv3_1_1x1_reduce_bn, paddings=pad_mat, name='padding4')
        output_conv3_1_3x3 = self.conv3_1_3x3(output_padding4)
        output_conv3_1_3x3_bn = tf.nn.relu(self.conv3_1_3x3_bn(output_conv3_1_3x3,is_training))
        output_conv3_1_1x1_increase = self.conv3_1_1x1_increase(output_conv3_1_3x3_bn)
        output_conv3_1_1x1_increase_bn = self.conv3_1_1x1_increase_bn(output_conv3_1_1x1_increase,is_training)

        output_conv3_1 = tf.add_n([output_conv3_1_1x1_proj_bn,output_conv3_1_1x1_increase_bn],name='conv3_1')
        output_conv3_1_relu = tf.nn.relu(output_conv3_1, name='conv3_1/relu')
        output_conv3_2_1x1_reduce = self.conv3_2_1x1_reduce(output_conv3_1_relu)
        output_conv3_2_1x1_reduce_bn = tf.nn.relu(self.conv3_2_1x1_reduce_bn(output_conv3_2_1x1_reduce,is_training))
        pad_mat = np.array([[0, 0], [1, 1], [1, 1], [0, 0]])
        output_padding5 = tf.pad(output_conv3_2_1x1_reduce_bn, paddings=pad_mat, name='padding5')
        output_conv3_2_3x3 = self.conv3_2_3x3(output_padding5)
        output_conv3_2_3x3_bn = tf.nn.relu(self.conv3_2_3x3_bn(output_conv3_2_3x3,is_training))
        output_conv3_2_1x1_increase = self.conv3_2_1x1_increase(output_conv3_2_3x3_bn)
        output_conv3_2_1x1_increase_bn = self.conv3_2_1x1_increase_bn(output_conv3_2_1x1_increase,is_training)

        output_conv3_2 = tf.add_n([output_conv3_1_relu, output_conv3_2_1x1_increase_bn], name='conv3_2')
        output_conv3_2_relu = tf.nn.relu(output_conv3_2, name='conv3_2/relu')
        output_conv3_3_1x1_reduce = self.conv3_3_1x1_reduce(output_conv3_2_relu)
        output_conv3_3_1x1_reduce_bn = tf.nn.relu(self.conv3_3_1x1_reduce_bn(output_conv3_3_1x1_reduce, is_training))
        pad_mat = np.array([[0, 0], [1, 1], [1, 1], [0, 0]])
        output_padding6 = tf.pad(output_conv3_3_1x1_reduce_bn, paddings=pad_mat, name='padding6')
        output_conv3_3_3x3 = self.conv3_3_3x3(output_padding6)
        output_conv3_3_3x3_bn = tf.nn.relu(self.conv3_3_3x3_bn(output_conv3_3_3x3, is_training))
        output_conv3_3_1x1_increase = self.conv3_3_1x1_increase(output_conv3_3_3x3_bn)
        output_conv3_3_1x1_increase_bn = self.conv3_3_1x1_increase_bn(output_conv3_3_1x1_increase, is_training)

        output_conv3_3 = tf.add_n([output_conv3_2_relu, output_conv3_3_1x1_increase_bn], name='conv3_3')
        output_conv3_3_relu = tf.nn.relu(output_conv3_3, name='conv3_3/relu')
        output_conv3_4_1x1_reduce = self.conv3_4_1x1_reduce(output_conv3_3_relu)
        output_conv3_4_1x1_reduce_bn = tf.nn.relu(self.conv3_4_1x1_reduce_bn(output_conv3_4_1x1_reduce, is_training))
        pad_mat = np.array([[0, 0], [1, 1], [1, 1], [0, 0]])
        output_padding7 = tf.pad(output_conv3_4_1x1_reduce_bn, paddings=pad_mat, name='padding7')
        output_conv3_4_3x3 = self.conv3_4_3x3(output_padding7)
        output_conv3_4_3x3_bn = tf.nn.relu(self.conv3_4_3x3_bn(output_conv3_4_3x3, is_training))
        output_conv3_4_1x1_increase = self.conv3_4_1x1_increase(output_conv3_4_3x3_bn)
        output_conv3_4_1x1_increase_bn = self.conv3_4_1x1_increase_bn(output_conv3_4_1x1_increase, is_training)

        output_conv3_4 = tf.add_n([output_conv3_3_relu, output_conv3_4_1x1_increase_bn], name='conv3_4')
        output_conv3_4_relu = tf.nn.relu(output_conv3_4, name='conv3_4/relu')
        output_conv4_1_1x1_proj = self.conv4_1_1x1_proj(output_conv3_4_relu)
        output_conv4_1_1x1_proj_bn = self.conv4_1_1x1_proj_bn(output_conv4_1_1x1_proj, is_training)

        output_conv4_1_1x1_reduce = self.conv4_1_1x1_reduce(output_conv3_4_relu)
        output_conv4_1_1x1_reduce_bn = tf.nn.relu(self.conv4_1_1x1_reduce_bn(output_conv4_1_1x1_reduce))
        pad_mat = np.array([[0, 0], [2, 2], [2, 2], [0, 0]])
        output_padding8 = tf.pad(output_conv4_1_1x1_reduce_bn, paddings=pad_mat, name='padding8')
        output_conv4_1_3x3 = self.conv4_1_3x3(output_padding8)
        output_conv4_1_3x3_bn = tf.nn.relu(self.conv4_1_3x3_bn(output_conv4_1_3x3, is_training))
        output_conv4_1_1x1_increase = self.conv4_1_1x1_increase(output_conv4_1_3x3_bn)
        output_conv4_1_1x1_increase_bn = self.conv4_1_1x1_increase_bn(output_conv4_1_1x1_increase, is_training)

        output_conv4_1 = tf.add_n([output_conv4_1_1x1_proj_bn, output_conv4_1_1x1_increase_bn], name='conv4_1')
        output_conv4_1_relu = tf.nn.relu(output_conv4_1, name='conv4_1/relu')
        output_conv4_2_1x1_reduce = self.conv4_2_1x1_reduce(output_conv4_1_relu)
        output_conv4_2_1x1_reduce_bn = tf.nn.relu(self.conv4_2_1x1_reduce_bn(output_conv4_2_1x1_reduce, is_training))
        pad_mat = np.array([[0, 0], [2, 2], [2, 2], [0, 0]])
        output_padding9 = tf.pad(output_conv4_2_1x1_reduce_bn, paddings=pad_mat, name='padding9')
        output_conv4_2_3x3 = self.conv4_2_3x3(output_padding9)
        output_conv4_2_3x3_bn = tf.nn.relu(self.conv4_2_3x3_bn(output_conv4_2_3x3, is_training))
        output_conv4_2_1x1_increase = self.conv4_2_1x1_increase(output_conv4_2_3x3_bn)
        output_conv4_2_1x1_increase_bn = self.conv4_2_1x1_increase_bn(output_conv4_2_1x1_increase,is_training)

        output_conv4_2 = tf.add_n([output_conv4_1_relu,output_conv4_2_1x1_increase_bn], name='conv4_2')
        output_conv4_2_relu = tf.nn.relu(output_conv4_2, name='conv4_2/relu')
        output_conv4_3_1x1_reduce = self.conv4_3_1x1_reduce(output_conv4_2_relu)
        output_conv4_3_1x1_reduce_bn = tf.nn.relu(self.conv4_3_1x1_reduce_bn(output_conv4_3_1x1_reduce,is_training))
        pad_mat = np.array([[0, 0], [2, 2], [2, 2], [0, 0]])
        output_padding10 = tf.pad(output_conv4_3_1x1_reduce_bn, paddings=pad_mat, name='padding10')
        output_conv4_3_3x3 = self.conv4_3_3x3(output_padding10)
        output_conv4_3_3x3_bn = tf.nn.relu(self.conv4_3_3x3_bn(output_conv4_3_3x3,is_training))
        output_conv4_3_1x1_increase = self.conv4_3_1x1_increase(output_conv4_3_3x3_bn)
        output_conv4_3_1x1_increase_bn = self.conv4_3_1x1_increase_bn(output_conv4_3_1x1_increase,is_training)

        output_conv4_3 = tf.add_n([output_conv4_2_relu, output_conv4_3_1x1_increase_bn], name='conv4_3')
        output_conv4_3_relu = tf.nn.relu(output_conv4_3, name='conv4_3/relu')
        output_conv4_4_1x1_reduce = self.conv4_4_1x1_reduce(output_conv4_3_relu)
        output_conv4_4_1x1_reduce_bn = tf.nn.relu(self.conv4_4_1x1_reduce_bn(output_conv4_4_1x1_reduce, is_training))
        pad_mat = np.array([[0, 0], [2, 2], [2, 2], [0, 0]])
        output_padding11 = tf.pad(output_conv4_4_1x1_reduce_bn, paddings=pad_mat, name='padding11')
        output_conv4_4_3x3 = self.conv4_4_3x3(output_padding11)
        output_conv4_4_3x3_bn = tf.nn.relu(self.conv4_4_3x3_bn(output_conv4_4_3x3, is_training))
        output_conv4_4_1x1_increase = self.conv4_4_1x1_increase(output_conv4_4_3x3_bn)
        output_conv4_4_1x1_increase_bn = self.conv4_4_1x1_increase_bn(output_conv4_4_1x1_increase, is_training)

        output_conv4_4 = tf.add_n([output_conv4_3_relu, output_conv4_4_1x1_increase_bn], name='conv4_4')
        output_conv4_4_relu = tf.nn.relu(output_conv4_4, name='conv4_4/relu')
        output_conv4_5_1x1_reduce = self.conv4_5_1x1_reduce(output_conv4_4_relu)
        output_conv4_5_1x1_reduce_bn = tf.nn.relu(self.conv4_5_1x1_reduce_bn(output_conv4_5_1x1_reduce, is_training))
        pad_mat = np.array([[0, 0], [2, 2], [2, 2], [0, 0]])
        output_padding12 = tf.pad(output_conv4_5_1x1_reduce_bn, paddings=pad_mat, name='padding12')
        output_conv4_5_3x3 = self.conv4_5_3x3(output_padding12)
        output_conv4_5_3x3_bn = tf.nn.relu(self.conv4_5_3x3_bn(output_conv4_5_3x3, is_training))
        output_conv4_5_1x1_increase = self.conv4_5_1x1_increase(output_conv4_5_3x3_bn)
        output_conv4_5_1x1_increase_bn = self.conv4_5_1x1_increase_bn(output_conv4_5_1x1_increase, is_training)

        output_conv4_5 = tf.add_n([output_conv4_4_relu, output_conv4_5_1x1_increase_bn], name='conv4_5')
        output_conv4_5_relu = tf.nn.relu(output_conv4_5, name='conv4_5/relu')
        output_conv4_6_1x1_reduce = self.conv4_6_1x1_reduce(output_conv4_5_relu)
        output_conv4_6_1x1_reduce_bn = tf.nn.relu(self.conv4_6_1x1_reduce_bn(output_conv4_6_1x1_reduce, is_training))
        pad_mat = np.array([[0, 0], [2, 2], [2, 2], [0, 0]])
        output_padding13 = tf.pad(output_conv4_6_1x1_reduce_bn, paddings=pad_mat, name='padding13')
        output_conv4_6_3x3 = self.conv4_6_3x3(output_padding13)
        output_conv4_6_3x3_bn = tf.nn.relu(self.conv4_6_3x3_bn(output_conv4_6_3x3, is_training))
        output_conv4_6_1x1_increase = self.conv4_6_1x1_increase(output_conv4_6_3x3_bn)
        output_conv4_6_1x1_increase_bn = self.conv4_6_1x1_increase_bn(output_conv4_6_1x1_increase, is_training)

        output_conv4_6 = tf.add_n([output_conv4_5_relu,output_conv4_6_1x1_increase_bn], name='conv4_6')
        output_conv4_6_relu = tf.nn.relu(output_conv4_6,name='conv4_6/relu')
        output_conv5_1_1x1_proj = self.conv5_1_1x1_proj(output_conv4_6_relu)
        output_conv5_1_1x1_proj_bn = self.conv5_1_1x1_proj_bn(output_conv5_1_1x1_proj)

        output_conv5_1_1x1_reduce = self.conv5_1_1x1_reduce(output_conv4_6_relu)
        output_conv5_1_1x1_reduce_bn = tf.nn.relu(self.conv5_1_1x1_reduce_bn(output_conv5_1_1x1_reduce,is_training))
        pad_mat = np.array([[0, 0], [4, 4], [4, 4], [0, 0]])
        output_padding31 = tf.pad(output_conv5_1_1x1_reduce_bn, paddings=pad_mat, name='padding31')
        output_conv5_1_3x3 = self.conv5_1_3x3(output_padding31)
        output_conv5_1_3x3_bn = tf.nn.relu(self.conv5_1_3x3_bn(output_conv5_1_3x3,is_training))
        output_conv5_1_1x1_increase = self.conv5_1_1x1_increase(output_conv5_1_3x3_bn)
        output_conv5_1_1x1_increase_bn = self.conv5_1_1x1_increase_bn(output_conv5_1_1x1_increase)

        output_conv5_1 = tf.add_n([output_conv5_1_1x1_proj_bn,output_conv5_1_1x1_increase_bn], name='conv5_1')
        output_conv5_1_relu = tf.nn.relu(output_conv5_1,name='conv5_1/relu')
        output_conv5_2_1x1_reduce = self.conv5_2_1x1_reduce(output_conv5_1_relu)
        output_conv5_2_1x1_reduce_bn = tf.nn.relu(self.conv5_2_1x1_reduce_bn(output_conv5_2_1x1_reduce, is_training))
        pad_mat = np.array([[0, 0], [4, 4], [4, 4], [0, 0]])
        output_padding32 = tf.pad(output_conv5_2_1x1_reduce_bn, paddings=pad_mat, name='padding32')
        output_conv5_2_3x3 = self.conv5_2_3x3(output_padding32)
        output_conv5_2_3x3_bn = tf.nn.relu(self.conv5_2_3x3_bn(output_conv5_2_3x3, is_training))
        output_conv5_2_1x1_increase = self.conv5_2_1x1_increase(output_conv5_2_3x3_bn)
        output_conv5_2_1x1_increase_bn = self.conv5_2_1x1_increase_bn(output_conv5_2_1x1_increase)

        output_conv5_2 = tf.add_n([output_conv5_1_relu,output_conv5_2_1x1_increase_bn], name='conv5_2')
        output_conv5_2_relu = tf.nn.relu(output_conv5_2, name='conv5_2/relu')
        output_conv5_3_1x1_reduce = self.conv5_3_1x1_reduce(output_conv5_2_relu)
        output_conv5_3_1x1_reduce_bn = tf.nn.relu(self.conv5_3_1x1_reduce_bn(output_conv5_3_1x1_reduce,is_training))
        pad_mat = np.array([[0, 0], [4, 4], [4, 4], [0, 0]])
        output_padding33 = tf.pad(output_conv5_3_1x1_reduce_bn, paddings=pad_mat, name='padding33')
        output_conv5_3_3x3 = self.conv5_3_3x3(output_padding33)
        output_conv5_3_3x3_bn = tf.nn.relu(self.conv5_3_3x3_bn(output_conv5_3_3x3,is_training))
        output_conv5_3_1x1_increase = self.conv5_3_1x1_increase(output_conv5_3_3x3_bn)
        output_conv5_3_1x1_increase_bn = tf.nn.relu(self.conv5_3_1x1_increase_bn(output_conv5_3_1x1_increase,is_training))

        output_conv5_3 = tf.add_n([output_conv5_2_relu, output_conv5_3_1x1_increase_bn], name='conv5_3')
        output_conv5_3_relu = tf.nn.relu(output_conv5_3, name='conv5_3_relu')

        shape = tf.shape(output_conv5_3_relu)[1:3]

        output_conv5_3_pool1 = tf.nn.avg_pool(output_conv5_3_relu, [60, 60], [60, 60], padding='VALID', name='conv5_3_pool1')
        output_conv5_3_pool1_conv = self.conv5_3_pool1_conv(output_conv5_3_pool1)
        output_conv5_3_pool1_conv_bn = tf.nn.relu(self.conv5_3_pool1_conv_bn(output_conv5_3_pool1_conv, is_training))
        output_conv5_3_pool1_interp = tf.image.resize(output_conv5_3_pool1_conv_bn,size=shape,name='conv5_3_pool1_interp')

        output_conv5_3_pool2 = tf.nn.avg_pool(output_conv5_3_relu, [30,30], [30,30], padding='VALID', name='conv5_3_pool2')
        output_conv5_3_pool2_conv = self.conv5_3_pool2_conv(output_conv5_3_pool2)
        output_conv5_3_pool2_conv_bn = tf.nn.relu(self.conv5_3_pool2_conv_bn(output_conv5_3_pool2_conv, is_training))
        output_conv5_3_pool2_interp = tf.image.resize(output_conv5_3_pool2_conv_bn, size=shape, name='conv5_3_pool2_interp')

        output_conv5_3_pool3 = tf.nn.avg_pool(output_conv5_3_relu, [20, 20], [20, 20], padding='VALID',name='conv5_3_pool3')
        output_conv5_3_pool3_conv = self.conv5_3_pool3_conv(output_conv5_3_pool3)
        output_conv5_3_pool3_conv_bn = tf.nn.relu(self.conv5_3_pool3_conv_bn(output_conv5_3_pool3_conv, is_training))
        output_conv5_3_pool3_interp = tf.image.resize(output_conv5_3_pool3_conv_bn, size=shape,name='conv5_3_pool3_interp')

        output_conv5_3_pool6 = tf.nn.avg_pool(output_conv5_3_relu, [10, 10], [10, 10], padding='VALID',name='conv5_3_pool6')
        output_conv5_3_pool6_conv = self.conv5_3_pool6_conv(output_conv5_3_pool6)
        output_conv5_3_pool6_conv_bn = tf.nn.relu(self.conv5_3_pool6_conv_bn(output_conv5_3_pool6_conv, is_training))
        output_conv5_3_pool6_interp = tf.image.resize(output_conv5_3_pool6_conv_bn, size=shape,name='conv5_3_pool6_interp')

        output_conv5_3_concat = tf.concat(axis=-1, values=[output_conv5_3_relu,output_conv5_3_pool6_interp,
                                                           output_conv5_3_pool3_interp,output_conv5_3_pool2_interp,
                                                           output_conv5_3_pool1_interp], name='conv5_3_concat')
        output_conv5_4 = self.conv5_4(output_conv5_3_concat)
        output_conv5_4_bn = tf.nn.relu(self.conv5_4_bn(output_conv5_4,is_training))
        output_conv6 = self.conv6(output_conv5_4_bn)

        return output_conv6



























