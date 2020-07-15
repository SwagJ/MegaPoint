import tensorflow as tf

from .base_model import BaseModel, Mode
from .backbones.vgg import vgg_backbone
from . import utils


class MegaPoint(BaseModel):
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

    def _model(self, inputs, mode, **config):
        config['training'] = (mode == Mode.TRAIN)

        def net(image):
            if config['data_format'] == 'channels_first':
                image = tf.transpose(image, [0, 3, 1, 2])
            print(f"\nRun Time Image Size: {image.shape}")
            features = vgg_backbone(image, **config)
            detections = utils.detector_head(features, **config)
            descriptors = utils.descriptor_head(features, **config)
            return {**detections, **descriptors}

        def train_net(image,depth,semantic,warped=None):
            if config['data_format'] == 'channels_first':
                image = tf.transpose(image, [0, 3, 1, 2])
                depth = tf.transpose(depth, [0, 3, 1, 2])
                semantic = tf.transpose(semantic,[0, 3, 1, 2])
                print(f"\n OG SET: {image.shape},{depth.shape},{semantic.shape}")
            if warped == None:
                # depth = tf.squeeze(depth,1)
                # semantic = tf.squeeze(semantic,1)
                # image = tf.squeeze(image,0)
                # chlast_img = tf.transpose(image,[1,2,0])
                print(f"\nDEPTH AND ORIGINAL IMAGE:{image.shape}, {depth.shape}, {semantic.shape}\n")
                mask,masked_image = utils.layer_predictor_ioannis(image,depth,semantic)
                print(f"\nmask shape:{mask.shape}, {masked_image.shape}\n")
                mask = {'mask':mask}

                features = vgg_backbone(masked_image, **config)
                detections = utils.detector_head(features, **config)
                descriptors = utils.descriptor_head(features, **config)

                return {**detections, **descriptors,**mask}
            else:
                masked_image = utils.mask_warped(image,warped)

                features = vgg_backbone(masked_image, **config)
                detections = utils.detector_head(features, **config)
                descriptors = utils.descriptor_head(features, **config)

                return {**detections, **descriptors}
        

        if config['training']:
            results = train_net(inputs['image'],inputs['depth'],inputs['semantic'])
        else:
            results = net(inputs['image'])



        if config['training']:
            warped_results = train_net(inputs['warped']['image'],
                                        inputs['warped']['depth'],
                                        inputs['warped']['semantic'],
                                        results['mask'])
            results = {**results, 'warped_results': warped_results,
                       'homography': inputs['warped']['homography']}

        # Apply NMS and get the final prediction
        prob = results['prob']
        if config['nms']:
            prob = tf.map_fn(lambda p: utils.box_nms(
                p, config['nms'], keep_top_k=config['top_k']), prob)
            results['prob_nms'] = prob
        results['pred'] = tf.compat.v1.to_int32(tf.greater_equal(
            prob, config['detection_threshold']))

        return results

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
        # remaped_kpt = utils.remap_keypoint(inputs['keypoint_map'],outputs['mask'])
        new_mask = inputs['warped']['valid_mask']*tf.squeeze(tf.compat.v1.to_int32(outputs['mask']),axis=1)
        print(f"\nmasks shape:{new_mask.shape}, {outputs['mask'].shape}, {inputs['warped']['valid_mask'].shape}\n")
        print(f"\nkeypoint map shape: {inputs['keypoint_map'].shape}\n")
        detector_loss = utils.detector_loss(
                inputs['keypoint_map'], logits,
                valid_mask=new_mask,
                **config)
        warped_detector_loss = utils.detector_loss(
                inputs['warped']['keypoint_map'], warped_logits,
                valid_mask=new_mask,
                **config)

        # Compute the loss for the descriptor head
        descriptor_loss = utils.descriptor_loss(
                descriptors, warped_descriptors, outputs['homography'],
                valid_mask=new_mask,
                **config)

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
