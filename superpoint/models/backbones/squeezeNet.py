import tensorflow as tf
from tensorflow.keras import layers as tfl


def squeezeNet_backbone(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'activation': tf.nn.relu, 'batch_normalization': True,
                   'training': config['training'],
                   'kernel_reg': config.get('kernel_reg', 0.),
                   'FREEZE_LAYERS': {
                                'conv1': False,
                                'fire2': False,
                                'fire3': False,
                                'fire4': False,
                                'fire5': False,
                                'fire6': False,
                                'fire7': False,
                                'fire8': False,
                                'fire9': False,
                                'fire10': False,
                                'fire11': False,
                                'conv12': False},
                   'keep_prob': 0.6,
                   'num_output': 128}
    params_pool = {'padding': 'SAME', 'data_format': config['data_format']}
    """NN architecture."""

    conv1 = tfl.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='SAME', activation='relu',
        name='conv1') (inputs)
    # , freeze=mc.FREEZE_LAYERS["conv1"])

    pool1 = tfl.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='SAME') (conv1)
    
    fire2 = fire_layer(
        'fire2', pool1, s1x1=16, e1x1=64, e3x3=64, freeze=params_conv["FREEZE_LAYERS"]["fire2"])
    fire3 = fire_layer(
        'fire3', fire2, s1x1=16, e1x1=64, e3x3=64, freeze=params_conv["FREEZE_LAYERS"]["fire3"])
    pool3 = tfl.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='SAME', name='pool3') (fire3)

    fire4 = fire_layer(
        'fire4', pool3, s1x1=32, e1x1=128, e3x3=128, freeze=params_conv["FREEZE_LAYERS"]["fire4"])
    fire5 = fire_layer(
        'fire5', fire4, s1x1=32, e1x1=128, e3x3=128, freeze=params_conv["FREEZE_LAYERS"]["fire5"])
    pool5 = tfl.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='SAME', name='pool5') (fire5)

    fire6 = fire_layer(
        'fire6', pool5, s1x1=48, e1x1=192, e3x3=192, freeze=params_conv["FREEZE_LAYERS"]["fire6"])
    fire7 = fire_layer(
        'fire7', fire6, s1x1=48, e1x1=192, e3x3=192, freeze=params_conv["FREEZE_LAYERS"]["fire7"])
    fire8 = fire_layer(
        'fire8', fire7, s1x1=64, e1x1=256, e3x3=256, freeze=params_conv["FREEZE_LAYERS"]["fire8"])
    fire9 = fire_layer(
        'fire9', fire8, s1x1=64, e1x1=256, e3x3=256, freeze=params_conv["FREEZE_LAYERS"]["fire9"])

    # Two extra fire modules that are not trained before
    fire10 = fire_layer(
        'fire10', fire9, s1x1=96, e1x1=384, e3x3=384, freeze=params_conv["FREEZE_LAYERS"]["fire10"])

    fire11 = fire_layer(
        'fire11', fire10, s1x1=96, e1x1=384, e3x3=384, freeze=params_conv["FREEZE_LAYERS"]["fire11"])

    dropout11 = tfl.Dropout(1.0-params_conv["keep_prob"], name='drop11') (fire11, training=params_conv['training'])
    
    preds = tfl.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='SAME', activation=None,
                        name='conv1') (dropout11)
    
    return preds


def fire_layer(layer_name, inputs, s1x1, e1x1, e3x3, stddev=0.01, freeze=False):
    """Fire layer constructor.
    Args:
        layer_name: layer name
        inputs: input tensor
        s1x1: number of 1x1 filters in squeeze layer.
        e1x1: number of 1x1 filters in expand layer.
        e3x3: number of 3x3 filters in expand layer.
        freeze: if true, do not train parameters in this layer.
    Returns:
        fire layer operation.
    """
    sq1x1 = tfl.Conv2D(filters=s1x1, kernel_size=(1,1), strides=(1,1), padding='SAME', activation='relu',
                       kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                stddev=stddev),
                       bias_initializer=tf.constant_initializer(0.0),
                       name=layer_name+'/squeeze1x1') (inputs)

    ex1x1 = tfl.Conv2D(filters=e1x1, kernel_size=(1,1), strides=(1,1), padding='SAME', activation='relu',
                       kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                stddev=stddev),
                       bias_initializer=tf.constant_initializer(0.0),
                       name=layer_name+'/expand1x1') (sq1x1)

    ex3x3 = tfl.Conv2D(filters=e3x3, kernel_size=(3,3), strides=(1,1), padding='SAME', activation='relu',
                       kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                stddev=stddev),
                       bias_initializer=tf.constant_initializer(0.0),
                       name=layer_name+'/expand3x3') (sq1x1)

    return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')

