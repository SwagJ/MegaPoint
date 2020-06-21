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
        if initializerPaths == None and not 'initializerPaths' in config.keys() :
            initializerPaths = GreatPoint.defaultInitializerPaths
        elif 'initializerPaths' in config.keys():
            initializerPaths = config['initializerPaths']
            for p in GreatPoint.defaultInitializerPaths.keys():
                if not p in initializerPaths:
                    initializerPaths[p] = None
        self.IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
        self.CROP_SIZE = [2*s for s in config['input_size']]
        
        self.training = training
        self.config = config
        self.depth_net = hourglass.Hourglass(weightsPath=initializerPaths['hourglass'], training=False, normalize=False)
        self.depth_net.trainable = False
        self.psp_net50 = PSP_net50.PSPNet50(num_classes=150, checkpoint_npy_path=initializerPaths['PSP_net50'])
        self.psp_net50.trainable = False
        self.super_point = super_point.SuperPoint(config, training=training, npyWeightsPath=initializerPaths['super_point'], name='superpoint') 
        self.super_point.trainable = training

    def call(self, input):
        image = input['input_1']

        
        depth = self.depth_net(image)

        

        img_shape = tf.cast(tf.shape(image),dtype=tf.int32)

        pad_image = self._psp_img_preprocess(image, self.CROP_SIZE[0], self.CROP_SIZE[1])
        raw_output = self.psp_net50(pad_image)

        raw_output = tf.image.resize(raw_output, size=[image.shape[1],image.shape[2]])
        semantics = tf.argmax(raw_output, axis=3)
        

        if self.config['batch_size'] == 1 or self.config['batch_size'] == 0:
            depth = tf.squeeze(depth,0)
            semantics = tf.squeeze(semantics,0)
            grayImage = tf.image.rgb_to_grayscale(image)
            layerShape = tf.shape(grayImage)
            image = tf.squeeze(image,0)
            image0, image1, image2 = utils.layer_predictor(depth, semantics, grayImage)

        else:
            image0_list,image1_list,image2_list = [],[],[]
            for k in range(image.shape[0]):
                depth_i = tf.gather_nd(depth,k)
                semantics_i = tf.gather_nd(semantics,k)
                image_i = tf.gather_nd(image,k)
                image0_i, image1_i, image2_i = utils.layer_predictor(depth_i, semantics_i, image_i)
                image0_list.append(image0_i)
                image1_list.append(image1_i)
                image2_list.append(image2_i)

            image0 = tf.concat(image0_list,axis=0)
            image1 = tf.concat(image1_list,axis=0)
            image2 = tf.concat(image2_list,axis=0)
        
            
        if self.training:
            warped_image = input['input_2']
            warped_depth = self.depth_net(warped_image)

            warped_img_shape = tf.cast(tf.shape(warped_image),dtype=tf.int32)

            warped_pad_image = self._psp_img_preprocess(warped_image, self.CROP_SIZE[0], self.CROP_SIZE[1])
            warped_raw_output = self.psp_net50(warped_pad_image)

            warped_raw_output = tf.image.resize(warped_raw_output, size=[warped_image.shape[1],warped_image.shape[2]])
            warped_semantics = tf.argmax(warped_raw_output, axis=3)


            if self.config['batch_size'] == 1 or self.config['batch_size'] == 0:
                warped_depth = tf.squeeze(warped_depth,0)
                warped_semantics = tf.squeeze(warped_semantics,0)
                warped_grayImage = tf.image.rgb_to_grayscale(warped_image)
                warped_image = tf.squeeze(warped_image,0)
                warped_image0, warped_image1, warped_image2 = utils.layer_predictor(warped_depth, warped_semantics, warped_grayImage)
            else:
                warped_image0_list,warped_image1_list,warped_image2_list = [],[],[]
                for k in range(image.shape[0]):
                    warped_depth_i = tf.gather_nd(warped_depth,k)
                    warped_semantics_i = tf.gather_nd(warped_semantics,k)
                    warped_image_i = tf.gather_nd(warped_image,k)
                    warped_image0_i, warped_image1_i, warped_image2_i = utils.layer_predictor(warped_depth_i, warped_semantics_i, warped_image_i)
                    warped_image0_list.append(warped_image0_i)
                    warped_image1_list.append(warped_image1_i)
                    warped_image2_list.append(warped_image2_i)

                warped_image0 = tf.concat(warped_image0_list,axis=0)
                warped_image1 = tf.concat(warped_image1_list,axis=0)
                warped_image2 = tf.concat(warped_image2_list,axis=0)

            pair1 = {'input_1': image0, 'input_2': warped_image0}
            # pair2 = {'input_1': image0, 'input_2': warped_image1}
            # pair3 = {'input_1': image0, 'input_2': warped_image2}

            # pair4 = {'input_1': image1, 'input_2': warped_image0}
            pair5 = {'input_1': image1, 'input_2': warped_image1}
            # pair6 = {'input_1': image1, 'input_2': warped_image2}

            # pair7 = {'input_1': image2, 'input_2': warped_image0}
            # pair8 = {'input_1': image2, 'input_2': warped_image1}
            pair9 = {'input_1': image2, 'input_2': warped_image2}
            # s1 = self.super_point(pair1)
            # s2 = self.super_point(pair5)
            # s3 = self.super_point(pair9)
            # s4 = self.super_point(pair1)
            # s5 = self.super_point(pair1)
            # s6 = self.super_point(pair1)
            # s7 = self.super_point(pair1)
            # s8 = self.super_point(pair1)
            # s9 = self.super_point(pair1)
        else:
            pair1 = {'input_1': image0}
            pair5 = {'input_1': image1}
            pair9 = {'input_1': image2}


        s1 = self.super_point(pair1)
        s2 = self.super_point(pair5)
        s3 = self.super_point(pair9)
        
        ret_list = {}
        keys = s1.keys()
        for k in keys:
            ret_list[k] = s1[k] + s2[k] + s3[k]
        
        ret_list['output_2'] = ret_list['output_2'] * 0.5
        if self.training:
            ret_list['output_6'] = ret_list['output_6'] * 0.5
        
        # ret_list['image0'] = image0
        # ret_list['image1'] = image1
        # ret_list['image2'] = image2
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

    def _psp_img_preprocess(self, img, h, w):
        """Preprocess image for PSPNet
        """
        # Convert RGB to BGR
        img_r, img_g, img_b = tf.split(axis=3, num_or_size_splits=3, value=img)
        img = tf.cast(tf.concat(axis=3, values=[img_b, img_g, img_r]), dtype=tf.float32)
        # Extract mean.
        img -= self.IMG_MEAN

        pad_img = tf.image.pad_to_bounding_box(img, 0, 0, h, w)
        #pad_img = tf.expand_dims(pad_img, 0)

        return pad_img
