import tensorflow as tf

from .backbones.vgg import VGGBackbone
from . import utils



class NetBackend(tf.keras.Model):
    def __init__(self, config={}, initializer=None, path=''):
        super(NetBackend, self).__init__()
        self.features = VGGBackbone(config, initializer, path=path)
        self.detector = utils.DetectorHead(config, initializer, path=path)
        self.descriptor = utils.DescriptorHead(config, initializer, path=path)
        
    def call(self, image):
        _features = self.features(image)
        _detections = self.detector(_features)
        _logits = _detections[0]
        _prob = _detections[1]
        
        _descriptors = self.descriptor(_features)
        _descriptors_raw = _descriptors[0]
        _desc_processed = _descriptors[1]

        return _logits, _prob, _descriptors_raw, _desc_processed


class NetBackendLoss(tf.keras.losses.Loss):
    def __init__(self, config, is_warped=False):
        super(tf.keras.losses.Loss, self).__init__()
        self.config = config
        self.is_warped = is_warped
        self.detector_loss = utils.DetectorHeadLoss(config)
        self.descriptor_loss = utils.DescriptorHeadLoss(config)

    def call(self, y_true, y_pred):
        logits = y_pred[0]    
        descriptors = y_pred[2]
        keypoint_map = y_true[0]
        valid_mask = y_true[1]
        # Compute the loss for the detector head
        # detector_loss = utils.detector_loss(
        #         inputs['keypoint_map'], logits,
        #         valid_mask=inputs['valid_mask'], **config)
        detector_loss = self.detector_loss([keypoint_map, valid_mask], logits)
        if(self.is_warped):
            homography = y_pred[4]
            # Compute the loss for the descriptor head
            # descriptor_loss = utils.descriptor_loss(
            #         descriptors, warped_descriptors, outputs['homography'],
            #         valid_mask=inputs['warped']['valid_mask'], **config)
            descriptor_loss = self.descriptor_loss(valid_mask, [descriptors, homography])

class SuperPoint(tf.keras.Model):
    # input_spec = {
    #     'image': {'shape': [None, None, None, 1], 'dtype': tf.float32},
    #     'warped/image' : {'shape': [None, None, None, 1], 'dtype': tf.float32}
    # }
    def __init__(self, config={}, training=False, initializer=None, path='', name='superpoint'):
        super(SuperPoint, self).__init__(name=name)
        self.input_names = ['input_1', 'input_2']
        self.output_names = ['output_1', 'output_2', 'output_3','output_4', 'output_5', 'output_6', 'output_7',
                             'output_8', 'output_9', 'output_10']
        default_config = {
            'data_format': 'channels_last',
            'grid_size': 8,
            'detection_threshold': 0.4,
            'descriptor_size': 256,
            'batch_size': 32,
            'learning_rate': 0.001,
            'lambda_d': 250,
            'positive_margin': 1,
            'negative_margin': 0.2,
            'lambda_loss': 0.0001,
            'nms': 0,
            'top_k': 0,
        }
        self.training = training
        self.config=utils._extend_dict(config, default_config)
        
        # for image input
        self._net_image = NetBackend(self.config, initializer, path + 'superpoint/')

    def call(self, x):
        # x = image, warped image, homography
        image = x['input_1'] # ['image']
        ret_list = []
        _logits, _prob, _descriptors_raw, _desc_processed = self._net_image(image)
        ret_list = {
            'output_1' : _logits,
            'output_2' : _prob,
            'output_3' : _descriptors_raw,
            'output_4' : _desc_processed}
        
        if self.training:
            warped_image = x['input_2'] # ['warped/image']
            # homography = x[2]
            # warped_results = net(inputs['warped']['image'])
            _logits_warped, _prob_warped, _descriptors_raw_warped, _desc_processed_warped = self._net_image(warped_image)
            ret_list = { **ret_list,
                        'output_5': _logits_warped,
                        'output_6': _prob_warped,
                        'output_7': _descriptors_raw_warped,
                        'output_8': _desc_processed_warped}
        
        if self.config['nms']:
            _results_prob_nms = tf.map_fn(lambda p: utils.box_nms(
                                          p, self.config['nms'], keep_top_k=self.config['top_k']), _prob)
            pred = tf.cast(tf.greater_equal(
                                        _results_prob_nms, self.config['detection_threshold']), dtype=tf.int32)
        else:
            _results_prob_nms = None
            pred = tf.cast(tf.greater_equal(
                                        _prob, self.config['detection_threshold']), dtype=tf.int32)

        ret_list['output_9'] = pred
        ret_list['output_10'] = _results_prob_nms
        
        
        # [_logits, _prob, _descriptors_raw, _desc_processed,  # 0 1 2 3
        # _logits_warped, _prob_warped, _descriptors_raw_warped, _desc_processed_warped, # 4 5 6 7 
        # pred, _results_prob_nms] # 8 9
        return ret_list 
    def set_compiled_loss(self):
        self.compiled_loss = SuperPointLoss(self.config, hasWarped=True)
        self.compiled_loss.metrics = [SuperPointMetrics()]


class SuperPointLoss(object):
    def __init__(self, config, hasWarped=True): # hasWarped=(training ==True) in this version 
        # super(SuperPointLoss, self).__init__()
        self.lambda_loss = config['lambda_loss']
        self.hasWarped = hasWarped   
        self.detector_loss = utils.DetectorHeadLoss(config)
        if self.hasWarped:
            self.warped_detector_loss = utils.DetectorHeadLoss(config)
        self.descriptor_loss = utils.DescriptorHeadLoss(config)
    
    def __call__(self, y_true, y_pred, sample_weight, regularization_losses=None):
        keypoint_map = y_true['output_1']
        valid_mask = y_true['output_2']
        warped_keypoint_map = y_true['output_3']
        warped_valid_mask = y_true['output_4']
        homography = y_true['output_5']

        
        logits = y_pred['output_1']
        descriptors_raw = y_pred['output_3'] # outputs['descriptors_raw']
        warped_logits = y_pred['output_5'] # outputs['warped_results']['logits']
        warped_descriptors_raw = y_pred['output_7'] # outputs['warped_results']['descriptors_raw']

        # Compute the loss for the detector head
        # detector_loss = utils.detector_loss(
        #         inputs['keypoint_map'], logits,
        #         valid_mask=inputs['valid_mask'], **config)
        _detector_loss = self.detector_loss([keypoint_map, valid_mask], [logits])
        # warped_detector_loss = utils.detector_loss(
        #         inputs['warped']['keypoint_map'], warped_logits,
        #         valid_mask=inputs['warped']['valid_mask'], **config)
        if self.hasWarped:
            _warped_detector_loss = self.warped_detector_loss([warped_keypoint_map, warped_valid_mask],[warped_logits])
            tf.summary.scalar('detector_loss2', _warped_detector_loss)
            L1 = _detector_loss + _warped_detector_loss
        else:
            L1 = _detector_loss
        
        # Compute the loss for the descriptor head
        # descriptor_loss = utils.descriptor_loss(
        #         descriptors, warped_descriptors, outputs['homography'],
        #         valid_mask=inputs['warped']['valid_mask'], **config)
        _descriptor_loss = self.descriptor_loss([warped_descriptors_raw, warped_valid_mask, homography],
                                                [descriptors_raw, None])

        L2 = self.lambda_loss * _descriptor_loss
        
        tf.summary.scalar('detector_loss1', _detector_loss)
        tf.summary.scalar('detector_loss_full', L1)
        tf.summary.scalar('descriptor_loss', L2)

        loss = L1 + L2

        return loss

class SuperPointMetrics(tf.keras.metrics.Metric):
    def __init__(self, name='SuperPoint_metrics', **kwargs):
        super(SuperPointMetrics, self).__init__(name, **kwargs)
        self.precisions = self.add_weight(name='precision', initializer='zeros')
        self.recalls = self.add_weight(name='recall', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):

        keypoint_map = y_true['output_1']
        valid_mask = y_true['output_2']
        warped_keypoint_map = y_true['output_3']
        warped_valid_mask = y_true['output_4']

        
        pred = y_pred['output_9']
        
        _pred = valid_mask * pred
        labels = keypoint_map
        
        precision = tf.cast(tf.reduce_sum(pred * labels) / tf.reduce_sum(pred), dtype=tf.float32)
        recall = tf.cast(tf.reduce_sum(pred * labels) / tf.reduce_sum(labels), dtype=tf.float32)

        self.precisions.assign_add(precision)
        self.recalls.assign_add(recall)
        
    def result(self):
        return [self.precisions, self.recalls]
