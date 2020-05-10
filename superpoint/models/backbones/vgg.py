import tensorflow as tf

class VGGBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, name,
                 data_format, training=False,
                 batch_normalization=True, kernel_reg=0., initializer=None, path='', **params):
        super(VGGBlock, self).__init__()
        self.batch_normalization = batch_normalization
        if initializer:
            self.conv = tf.keras.layers.Conv2D(filters, kernel_size, kernel_regularizer=tf.keras.regularizers.l2(kernel_reg), data_format=data_format,
                                               kernel_initializer=initializer.conv2d_kernel(path),
                                               bias_initializer=initializer.conv2d_bias(path)) # 'conv'
            if batch_normalization:
                self.bn = tf.keras.layers.BatchNormalization(trainable=training, fused=True, axis=1 if data_format == 'channels_first' else -1,
                                                        beta_initializer=initializer.BN_beta(path),
                                                        gamma_initializer=initializer.BN_gamma(path),
                                                        moving_mean_initializer=initializer.BN_mean(path),
                                                        moving_variance_initializer=initializer.BN_variance(path)) # 'bn'
        else:
            self.conv = tf.keras.layers.Conv2D(filters, kernel_size, kernel_regularizer=tf.keras.regularizers.l2(kernel_reg), data_format=data_format) # 'conv'
            if batch_normalization:
                self.bn = tf.keras.layers.BatchNormalization(trainable=training, fused=True, axis=1 if data_format == 'channels_first' else -1) # 'bn'
    def call(self, inputs):
        out = self.conv(inputs)
        if self.batch_normalization:
            out = self.bn(out)
        return out

class VGGBackbone(tf.keras.Model):
    def __init__(self, config, initializer=None, path='', name=''):
        """[summary]
            self.pool1/2/3 can be just one pool since they use the same aruments
        Arguments:
            tf {[type]} -- [description]
            config {[type]} -- [description]

        Keyword Arguments:
            initializer {[type]} -- [description] (default: {None})
            path {str} -- [description] (default: {''})
            name {str} -- [description] (default: {''})
        """
        super(VGGBackbone, self).__init__()
        _path = path + 'vgg/'
        params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                       'activation': tf.nn.relu, 'batch_normalization': True,
                       'training': config['training'],
                       'kernel_reg': config.get('kernel_reg', 0.)}
        self.params_pool = {'padding': 'SAME', 'data_format': config['data_format']}

        self.conv1_1 = VGGBlock(64, 3, 'conv1_1', initializer=initializer, path=_path+'conv1_1/', **params_conv)
        self.conv1_2 = VGGBlock(64, 3, 'conv1_2', initializer=initializer, path=_path+'conv1_2/', **params_conv)
        self.pool1 = tf.keras.layers.MaxPool2D(2, 2, name='pool1', **self.params_pool)

        self.conv2_1 = VGGBlock(64, 3, 'conv2_1', initializer=initializer, path=_path+'conv2_1/', **params_conv)
        self.conv2_2 = VGGBlock(64, 3, 'conv2_2', initializer=initializer, path=_path+'conv2_2/', **params_conv)
        self.pool2 = tf.keras.layers.MaxPool2D(2, 2, name='pool1', **self.params_pool)
        
        self.conv3_1 = VGGBlock(128, 3, 'conv3_1', initializer=initializer, path=_path+'conv3_1/', **params_conv)
        self.conv3_2 = VGGBlock(128, 3, 'conv3_2', initializer=initializer, path=_path+'conv3_2/', **params_conv)
        self.pool3 = tf.keras.layers.MaxPool2D(2, 2, name='pool1', **self.params_pool)
        
        self.conv4_1 = VGGBlock(128, 3, 'conv4_1', initializer=initializer, path=_path+'conv4_1/', **params_conv)
        self.conv4_2 = VGGBlock(128, 3, 'conv4_2', initializer=initializer, path=_path+'conv4_2/', **params_conv)
    
    def call(self, x):
        _x = self.conv1_1(x)
        _x = self.conv1_2(_x)
        _x = self.pool1(_x)
        
        _x = self.conv2_1(_x)
        _x = self.conv2_2(_x)
        _x = self.pool2(_x)
        
        _x = self.conv3_1(_x)
        _x = self.conv3_2(_x)
        _x = self.pool2(_x)
                
        _x = self.conv4_1(_x)
        _x = self.conv4_2(_x)
        return _x