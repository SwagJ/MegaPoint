import tensorflow as tf

from .base_model import BaseModel, Mode
from .backbones.vgg import vgg_backbone
from .utils import detector_head, detector_loss, box_nms,layer_predictor_ioannis 
from .homographies import homography_adaptation


class GreatPoint(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 1], 'type': tf.float32}
    }
    required_config_keys = []
    default_config = {
            'data_format': 'channels_first',
            'kernel_reg': 0.,
            'grid_size': 8,
            'detection_threshold': 0.4,
            'homography_adaptation': {'num': 0},
            'nms': 0,
            'top_k': 0
    }

    def _model(self, inputs, mode, **config):
        config['training'] = (mode == Mode.TRAIN)
        

        def net(image):
            if config['data_format'] == 'channels_first':
                image = tf.transpose(image, [0, 3, 1, 2])
            features = vgg_backbone(image, **config)
            outputs = detector_head(features, **config)
            return outputs

        def train_net(image,depth,semantic):
            if config['data_format'] == 'channels_first':
                image = tf.transpose(image, [0, 3, 1, 2])
                depth = tf.transpose(depth, [0, 3, 1, 2])
                semantic = tf.transpose(semantic,[0, 3, 1, 2])
                print(f"\n OG SET: {image.shape},{depth.shape},{semantic.shape}")
                # depth = tf.squeeze(depth,1)
                # semantic = tf.squeeze(semantic,1)
                # image = tf.squeeze(image,0)
                # chlast_img = tf.transpose(image,[1,2,0])
            print(f"\nDEPTH AND ORIGINAL IMAGE:{image.shape}, {depth.shape}, {semantic.shape}\n")
            mask,masked_image = layer_predictor_ioannis(image,depth,semantic)
            print(f"\nmask shape:{mask.shape}, {masked_image.shape}\n")
                

            features = vgg_backbone(masked_image, **config)
            outputs = detector_head(features, **config)

            return outputs

        if (mode == Mode.PRED) and config['homography_adaptation']['num']:
            outputs = homography_adaptation(inputs['image'], net, config['homography_adaptation'])
        elif (mode == Mode.TRAIN):
            outputs = train_net(inputs['image'],inputs['depth'],inputs['semantic'])
        else:
            outputs = net(inputs['image'])

        prob = outputs['prob']
        if config['nms']:
            prob = tf.map_fn(lambda p: box_nms(p, config['nms'],
                                               min_prob=config['detection_threshold'],
                                               keep_top_k=config['top_k']), prob)
            outputs['prob_nms'] = prob
        pred = tf.compat.v1.to_int32(tf.greater_equal(prob, config['detection_threshold']))
        outputs['pred'] = pred

        return outputs

    def _loss(self, outputs, inputs, **config):
        if config['data_format'] == 'channels_first':
            outputs['logits'] = tf.transpose(outputs['logits'], [0, 2, 3, 1])
        return detector_loss(inputs['keypoint_map'], outputs['logits'],
                             valid_mask=inputs['valid_mask'], **config)

    def _metrics(self, outputs, inputs, **config):
        pred = inputs['valid_mask'] * outputs['pred']
        labels = inputs['keypoint_map']

        precision = tf.reduce_sum(pred * labels) / tf.reduce_sum(pred)
        recall = tf.reduce_sum(pred * labels) / tf.reduce_sum(labels)

        return {'precision': precision, 'recall': recall}
