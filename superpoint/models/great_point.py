import tensorflow as tf
import numpy as np
from .backbones.vgg import VGGBackbone
from . import utils
from . import super_point
from . import PSP_net50
from . import hourglass

class GreatPoint(tf.keras.Model):
    defaultInitializerPaths = {
        'hourglass' : None,
        'PSP_net50' : None,
        'super_point' : None
    }
    def __init__(self, config, training, initializerPaths=None, path='', name='GreatPoint'):
        super(GreatPoint, self).__init__(name=name)
        if initializerPaths == None:
            initializerPaths = GreatPoint.defaultInitializerPaths
        
        self.IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
        
        self.training = training
        self.config = config
        self.depth_net = hourglass.Hourglass(weightsPath=initializerPaths['hourglass'], training=False, normalize=False)
        self.depth_net.trainable = False
        self.psp_net50 = PSP_net50.PSPNet50(num_classes=3, checkpoint_npy_path=initializerPaths['PSP_net50'])
        self.psp_net50.trainable = False
        self.super_point = super_point.SuperPoint(config, training=training, npyWeightsPath=initializerPaths['super_point']) 
        self.super_point.trainable = training

    def call(self, input):
        image = input['input_1']
        
        depth = self.depth_net(image)
        channels = tf.unstack(image, axis=-1)
        bgr_image = tf.stack([channels[2], channels[1], channels[0]], axis=-1)
        semantics = self.psp_net50(bgr_image*255 - self.IMG_MEAN)

        image0, image1, image2 = utils.layer_predictor(depth, semantics, image, batch_size=self.config['batch_size'])
        
        if self.training:
            warped_image = input['input_2']
            warped_depth = self.depth_net(warped_image)
            warped_semantics = self.psp_net50(warped_image)
            warped_image0, warped_image1, warped_image2 = utils.layer_predictor(warped_depth, warped_semantics, warped_image,
                                                           batch_size=self.config['batch_size'])
            pair1 = {'input_1': image0, 'input_2': warped_image0}
            # pair2 = {'input_1': image0, 'input_2': warped_image1}
            # pair3 = {'input_1': image0, 'input_2': warped_image2}

            # pair4 = {'input_1': image1, 'input_2': warped_image0}
            pair5 = {'input_1': image1, 'input_2': warped_image1}
            # pair6 = {'input_1': image1, 'input_2': warped_image2}

            # pair7 = {'input_1': image2, 'input_2': warped_image0}
            # pair8 = {'input_1': image2, 'input_2': warped_image1}
            pair9 = {'input_1': image2, 'input_2': warped_image2}

            s1 = self.super_point(pair1)
            s2 = self.super_point(pair5)
            s3 = self.super_point(pair9)
            # s4 = self.super_point(pair1)
            # s5 = self.super_point(pair1)
            # s6 = self.super_point(pair1)
            # s7 = self.super_point(pair1)
            # s8 = self.super_point(pair1)
            # s9 = self.super_point(pair1)
        else:
            s1 = self.super_point({'input_1' : image0})
            s2 = self.super_point({'input_1' : image1})
            s3 = self.super_point({'input_1' : image2})
        
        ret_list = {}
        keys = s1.keys()
        for k in keys:
            ret_list[k] = s1[k] + s2[k] + s3[k]
        
        ret_list['output_2'] = ret_list['output_2'] / 3
        if self.training:
            ret_list['output_6'] = ret_list['output_6'] / 3
        
        return ret_list

    def set_compiled_loss(self):
        """That is a highly risky function, since it bypasses the container architecture of
           the tf keras loss.
        """
        self.compiled_loss = super_point.SuperPointLoss(self.config, hasWarped=True)
        self.compiled_loss.metrics = [super_point.SuperPointMetrics()]
    
    def comppileWrapper(self):
        """This function call keras compile with arguments comming from the configuration file and
            adds a loss function by calling set_compiled_loss
        """
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss=super_point.SuperPointLoss(self.config))
        self.set_compiled_loss()
