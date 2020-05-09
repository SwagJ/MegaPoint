import tensorflow as tf

from .backbones.vgg import vgg_backbone
from . import utils



class NetBackend(tf.keras.Model):
    def __init__(self, config={}, initializer=None, path=''):
        super(NetBackend, self).__init__()
        self.features = vgg_backbone(initializer, path=path, **config)
        self.detections = utils.detector_head(initializer, path=path, **config)
        self.descriptors = utils.descriptor_head(initializer, path=path, **config)
        
    def call(self, image):
        _features = self.features(image)
        _detections = self.detections(_features)
        _logits = _detections[0]
        _prob = _detections[1]        
        
        _descriptors = self.descriptors(_features)        
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
    input_spec = {
        'image': {'shape': [None, None, None, 1], 'type': tf.float32}
    }
    required_config_keys = []
    default_config = {
        'data_format': 'channels_first',
        'grid_size': 8,
        'detection_threshold': 0.4,
        'descriptor_size': 256,
        'batch_size': 32,
        'learning_rate': 0.001,
        'lambda_d': 250,
        'descriptor_size': 256,
        'positive_margin': 1,
        'negative_margin': 0.2,
        'lambda_loss': 0.0001,
        'nms': 0,
        'top_k': 0,
    }
    def __init__(self, config=SuperPoint.default_config, initializer=None,  training=True, path=''):
        super(SuperPoint, self).__init__()
        self.config=config
        self.training = training
        
        # for image input
        self._net_image = NetBackend(config, initializer, path)

        # for warped image input
        if self.training:
            self._net_warped_image = NetBackend(config, initializer, path)


    def call(self, x):
        # x = image, warped image, homography
        image = x[0]
        ret_list = []
        _logits, _prob, _descriptors_raw, _desc_processed = self._net_image(image)
        ret_list = ret_list.extend([_logits, _prob, _descriptors_raw, _desc_processed])
        
        if self.training:
            warped_image = x[1]
            homography = x[2]
            # warped_results = net(inputs['warped']['image'])
            _logits_warped, _prob_warped, _descriptors_raw_warped, _desc_processed_warped = self._net_warped_image(warped_image)
            ret_list = ret_list.extend([_logits_warped, _prob_warped, _descriptors_raw_warped, _desc_processed_warped])
        
        if self.config['nms']:
            _results_prob_nms = tf.map_fn(lambda p: utils.box_nms(
                                            p, config['nms'], keep_top_k=config['top_k']), _prob)
            _results_pred = tf.cast(tf.greater_equal(
                                        _results_prob_nms, self.config['detection_threshold']), dtype=tf.int32)
        else:
            _results_prob_nms = None
            _results_pred = tf.cast(tf.greater_equal(
                                        _prob, self.config['detection_threshold']), dtype=tf.int32)

        ret_list.append(_results_pred)
        ret_list.append(_results_prob_nms)
        # [_logits, _prob, _descriptors_raw, _desc_processed,  # 0 1 2 3
        # _logits_warped, _prob_warped, _descriptors_raw_warped, _desc_processed_warped, # 4 5 6 7 
        # _results_pred, _results_prob_nms] # 8 9
        return ret_list 

class SuperPointLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(SuperPointLoss, self).__init__()
    def call(self, y_true, y_pred):
        pass





























    def _loss(self, outputs, inputs, **config):
        logits = outputs['logits']
        warped_logits = outputs['warped_results']['logits']
        descriptors = outputs['descriptors_raw']
        warped_descriptors = outputs['warped_results']['descriptors_raw']

        # Switch to 'channels last' once and for all
        if config['data_format'] == 'channels_first':
            logits = tf.transpose(logits, [0, 2, 3, 1])
            warped_logits = tf.transpose(warped_logits, [0, 2, 3, 1])
            descriptors = tf.transpose(descriptors, [0, 2, 3, 1])
            warped_descriptors = tf.transpose(warped_descriptors, [0, 2, 3, 1])

        # Compute the loss for the detector head
        detector_loss = utils.detector_loss(
                inputs['keypoint_map'], logits,
                valid_mask=inputs['valid_mask'], **config)
        warped_detector_loss = utils.detector_loss(
                inputs['warped']['keypoint_map'], warped_logits,
                valid_mask=inputs['warped']['valid_mask'], **config)

        # Compute the loss for the descriptor head
        descriptor_loss = utils.descriptor_loss(
                descriptors, warped_descriptors, outputs['homography'],
                valid_mask=inputs['warped']['valid_mask'], **config)

        tf.summary.scalar('detector_loss1', detector_loss)
        tf.summary.scalar('detector_loss2', warped_detector_loss)
        tf.summary.scalar('detector_loss_full', detector_loss + warped_detector_loss)
        tf.summary.scalar('descriptor_loss', config['lambda_loss'] * descriptor_loss)

        loss = (detector_loss + warped_detector_loss
                + config['lambda_loss'] * descriptor_loss)
        return loss

    def _metrics(self, outputs, inputs, **config):
        pred = inputs['valid_mask'] * outputs['pred']
        labels = inputs['keypoint_map']

        precision = tf.reduce_sum(pred * labels) / tf.reduce_sum(pred)
        recall = tf.reduce_sum(pred * labels) / tf.reduce_sum(labels)

        return {'precision': precision, 'recall': recall}
