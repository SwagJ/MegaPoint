import tensorflow as tf

class VGGBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, name,
                 data_format, training=False, 
                 batch_normalization=True, kernel_reg=0., initializer=None, path='', **params):
        super(VGGBlock, self).__init__()
        self.batch_normalization = batch_normalization
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, kernel_regularizer=tf.keras.regularizers.l2(kernel_reg), data_format=data_format) # 'conv'
        if batch_normalization:
            self.bn = tf.keras.layers.BatchNormalization(training=training, fused=True, axis=1 if data_format == 'channels_first' else -1) # 'bn'

    def call(self, inputs):
        out = self.conv(inputs)
        if self.batch_normalization:
            out = self.bn(out)
        return out

class VGGBackbone(tf.keras.Model):
    def __init__(self, config, initializer=None, path=''):
        super(VGGBackbone, self).__init__()
        params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                       'activation': tf.nn.relu, 'batch_normalization': True,
                       'training': config['training'],
                       'kernel_reg': config.get('kernel_reg', 0.)}
        self.params_pool = {'padding': 'SAME', 'data_format': config['data_format']}

        self.conv1_1 = VGGBlock(64, 3, 'conv1_1', initializer, path, **params_conv)
        self.conv1_2 = VGGBlock(64, 3, 'conv1_2', initializer, path, **params_conv)

        self.conv2_1 = VGGBlock(64, 3, 'conv2_1', initializer, path, **params_conv)
        self.conv2_2 = VGGBlock(64, 3, 'conv2_2', initializer, path, **params_conv)

        self.conv3_1 = VGGBlock(128, 3, 'conv3_1', initializer, path, **params_conv)
        self.conv3_2 = VGGBlock(128, 3, 'conv3_2', initializer, path, **params_conv)

        self.conv4_1 = VGGBlock(128, 3, 'conv4_1', initializer, path, **params_conv)
        self.conv4_2 = VGGBlock(128, 3, 'conv4_2', initializer, path, **params_conv)
    
    def call(self, x):
        _x = self.conv1_1(x)
        _x = self.conv1_2(_x)
        _x = tf.keras.layers.MaxPool2D(_x, 2, 2, name='pool1', **self.params_pool)
        
        _x = self.conv2_1(_x)
        _x = self.conv2_2(_x)
        _x = tf.keras.layers.MaxPool2D(_x, 2, 2, name='pool2', **self.params_pool)
        
        _x = self.conv3_1(_x)
        _x = self.conv3_2(_x)
        _x = tf.keras.layers.MaxPool2D(_x, 2, 2, name='pool3', **self.params_pool)
                
        _x = self.conv4_1(_x)
        _x = self.conv4_2(_x)
        return _x