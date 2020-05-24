import tensorflow as tf
import numpy as np
from .homographies import warp_points
from .backbones.vgg import VGGBlock



def _extend_dict(dict_to_extend, other_dict):
    for conf_key in other_dict.keys():
        if(conf_key not in dict_to_extend.keys()):
            dict_to_extend[conf_key] = other_dict[conf_key]
    return dict_to_extend

def _correct_dict(dict_to_correct, other_dict):
    for conf_key in other_dict.keys():
        if(conf_key in dict_to_correct.keys()):
            dict_to_correct[conf_key] = other_dict[conf_key]
    return dict_to_correct

class DetectorHead(tf.keras.Model):
    params_conv = {'padding': 'SAME', 'data_format': '',
                   'batch_normalization': True,
                   'training': False,
                   'kernel_reg': 0.}
    def __init__(self, config={}, initializer=None, path=''):
        super(DetectorHead, self).__init__()
        self.config = _correct_dict(DetectorHead.params_conv, config)
        self.config['grid_size'] = config['grid_size']
        print('padding=', self.config['padding'])
        self.cfirst = (config['data_format'] == 'channels_first')
        self.cindex = 1 if self.cfirst else -1
        _path = path + 'detector/'
        # with tf.compat.v1.variable_scope('detector', reuse=tf.compat.v1.AUTO_REUSE):
        self.conv1 = VGGBlock(256, 3, 'conv1',
                      activation=tf.nn.relu, path=(_path+'conv1/'), initializer=initializer, **self.config)
        self.conv2 = VGGBlock(1+pow(self.config['grid_size'], 2), 1, 'conv2',
                      activation=None, path=(_path+'conv2/'), initializer=initializer, **self.config)
    
    def call(self, features):
        _x = self.conv1(features)
        logits = self.conv2(_x)
        
        prob = tf.nn.softmax(logits, axis=self.cindex)
        # Strip the extra “no interest point” dustbin
        prob = prob[:, :-1, :, :] if self.cfirst else prob[:, :, :, :-1]
        prob = tf.nn.depth_to_space(
                prob, self.config['grid_size'], data_format='NCHW' if self.cfirst else 'NHWC')
        prob = tf.squeeze(prob, axis=self.cindex)

        return logits, prob

class DetectorHeadLoss(tf.keras.losses.Loss):
    def __init__(self, config):
        super(DetectorHeadLoss, self).__init__()
        self.config = {'grid_size': config['grid_size']}
        
    def call(self, y_true, y_pred):
        keypoint_map = y_true[0]
        valid_mask = y_true[1]
        logits = y_pred[0]
        # Convert the boolean labels to indices including the "no interest point" dustbin
        labels = tf.cast(tf.expand_dims(keypoint_map, -1), tf.float32)  # for GPU
        labels = tf.nn.space_to_depth(labels, self.config['grid_size'])
        shape = tf.concat([tf.shape(labels)[:3], [1]], axis=0)
        labels = tf.concat([2*labels, tf.ones(shape)], 3)
        # multiply by a small random matrix to randomly break ties in argmax
        labels = tf.argmax(labels * tf.compat.v1.random_uniform(tf.shape(labels), 0, 0.1),
                            axis=3)

        # Mask the pixels if bordering artifacts appear
        valid_mask = tf.ones_like(keypoint_map) if valid_mask is None else valid_mask
        valid_mask = tf.cast(tf.expand_dims(valid_mask, -1), tf.float32)  # for GPU
        valid_mask = tf.nn.space_to_depth(valid_mask, self.config['grid_size'])
        valid_mask = tf.reduce_prod(valid_mask, axis=3)  # AND along the channel dim
        
        # labels = tf.compat.v1.Print(labels,[tf.shape(labels), tf.shape(logits), tf.shape(logits)[-1], tf.rank(logits), tf.shape(valid_mask)], "labels, logits, valid_mask =")
        #  output_stream="file:///media/terabyte/projects/SUPERPOINT/SuperPointIoannis/superpoint.info")
        loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits, weights=valid_mask)
        
        return loss

class DescriptorHead(tf.keras.Model):
    params_conv = {'padding': 'SAME', 'data_format': '',
                   'batch_normalization': True,
                   'training': False,
                   'kernel_reg': 0.0}
    def __init__(self, config={}, initializer=None, path=''):
        super(DescriptorHead, self).__init__()
        self.config = _correct_dict(DetectorHead.params_conv, config)
        self.cfirst = (config['data_format'] == 'channels_first')
        self.cindex = 1 if self.cfirst else -1
        _path = path + 'descriptor/'
        # with tf.compat.v1.variable_scope('detector', reuse=tf.compat.v1.AUTO_REUSE):
        # just use a model instance twice or more times and it will have the same result 
        self.conv1 = VGGBlock(256, 3, 'conv1',
                      activation=tf.nn.relu, path=(_path+'conv1/'), initializer=initializer, **self.config)
        self.conv2 = VGGBlock(config['descriptor_size'], 1, 'conv2',
                      activation=None, path=(_path+'conv2/'), initializer=initializer, **self.config)
    def call(self, features):
        _x = self.conv1(features)
        descriptors_raw = self.conv2(_x)
        
        desc = tf.transpose(descriptors_raw, [0, 2, 3, 1]) if self.cfirst else descriptors_raw
        desc = tf.image.resize(
            desc, self.config['grid_size'] * tf.shape(desc)[1:3])
        desc = tf.transpose(desc, [0, 3, 1, 2]) if self.cfirst else desc
        desc = tf.nn.l2_normalize(desc, self.cindex)
        
        return descriptors_raw, desc


class DescriptorHeadLoss(tf.keras.losses.Loss):
    def __init__(self, config):
        super(DescriptorHeadLoss, self).__init__()
        self.config = {
            'grid_size': config['grid_size'],
            'positive_margin': config['positive_margin'],
            'negative_margin': config['negative_margin'],
            'lambda_d': config['lambda_d']
        }
    def call(self, y_true, y_pred):
        warped_descriptors = y_true[0]
        valid_mask = y_true[1]
        homographies = y_true[2]
        
        descriptors_raw = y_pred[0]
        descriptors = y_pred[1] # desc
        
        _grid_size = self.config['grid_size']
        
        # Compute the position of the center pixel of every cell in the image
        (batch_size, Hc, Wc) = tf.unstack(tf.cast(tf.shape(descriptors_raw)[:3], dtype=tf.int32))

        coord_cells = tf.stack(tf.meshgrid(
            tf.range(Hc), tf.range(Wc), indexing='ij'), axis=-1)
        
        coord_cells = coord_cells * _grid_size + _grid_size // 2  # (Hc, Wc, 2)
        # coord_cells is now a grid containing the coordinates of the Hc x Wc
        # center pixels of the 8x8 cells of the image

        # Compute the position of the warped center pixels
        warped_coord_cells = warp_points(tf.reshape(coord_cells, [-1, 2]), homographies)
        # warped_coord_cells is now a list of the warped coordinates of all the center
        # pixels of the 8x8 cells of the image, shape (N, Hc x Wc, 2)

        # Compute the pairwise distances and filter the ones less than a threshold
        # The distance is just the pairwise norm of the difference of the two grids
        # Using shape broadcasting, cell_distances has shape (N, Hc, Wc, Hc, Wc)
        coord_cells = tf.cast(tf.reshape(coord_cells, [1, 1, 1, Hc, Wc, 2]), tf.float32)
        warped_coord_cells = tf.reshape(warped_coord_cells,
                                        [batch_size, Hc, Wc, 1, 1, 2])
        cell_distances = tf.norm(coord_cells - warped_coord_cells, axis=-1)
        s = tf.cast(tf.less_equal(cell_distances, _grid_size - 0.5), tf.float32)

        # Normalize the descriptors and
        # compute the pairwise dot product between descriptors: d^t * d'
        descriptors_raw = tf.reshape(descriptors_raw, [batch_size, Hc, Wc, 1, 1, -1])
        descriptors_raw = tf.nn.l2_normalize(descriptors_raw, -1)
        warped_descriptors = tf.reshape(warped_descriptors,
                                        [batch_size, 1, 1, Hc, Wc, -1])
        warped_descriptors = tf.nn.l2_normalize(warped_descriptors, -1)
        dot_product_desc = tf.reduce_sum(descriptors_raw * warped_descriptors, -1)
        dot_product_desc = tf.nn.relu(dot_product_desc)
        dot_product_desc = tf.reshape(tf.nn.l2_normalize(
            tf.reshape(dot_product_desc, [batch_size, Hc, Wc, Hc * Wc]),
            3), [batch_size, Hc, Wc, Hc, Wc])
        dot_product_desc = tf.reshape(tf.nn.l2_normalize(
            tf.reshape(dot_product_desc, [batch_size, Hc * Wc, Hc, Wc]),
            1), [batch_size, Hc, Wc, Hc, Wc])
        # dot_product_desc[id_batch, h, w, h', w'] is the dot product between the
        # descriptor at position (h, w) in the original descriptors map and the
        # descriptor at position (h', w') in the warped image

        # Compute the loss
        positive_dist = tf.maximum(0., self.config['positive_margin'] - dot_product_desc)
        negative_dist = tf.maximum(0., dot_product_desc - self.config['negative_margin'])
        loss = self.config['lambda_d'] * s * positive_dist + (1 - s) * negative_dist

        # Mask the pixels if bordering artifacts appear
        valid_mask = tf.ones([batch_size,
                            Hc * _grid_size,
                            Wc * _grid_size], tf.float32)\
            if valid_mask is None else valid_mask
        valid_mask = tf.cast(valid_mask[..., tf.newaxis], tf.float32)  # for GPU
        valid_mask = tf.nn.space_to_depth(valid_mask, _grid_size)
        valid_mask = tf.reduce_prod(valid_mask, axis=3)  # AND along the channel dim
        # print(valid_mask, [batch_size, 1, 1, Hc, Wc], config, file=open('valid_mask.info', 'w'))
        vms = tf.shape(valid_mask)
        vmsH = vms[1]
        vmsW = vms[2]
        valid_mask = tf.reshape(valid_mask, [batch_size, 1, 1, vmsH, vmsW])

        normalization = tf.reduce_sum(valid_mask) * tf.cast(Hc * Wc, tf.float32)
        # Summaries for debugging
        # tf.summary.scalar('nb_positive', tf.reduce_sum(valid_mask * s) / normalization)
        # tf.summary.scalar('nb_negative', tf.reduce_sum(valid_mask * (1 - s)) / normalization)
        tf.summary.scalar('positive_dist', tf.reduce_sum(valid_mask * self.config['lambda_d'] *
                                                        s * positive_dist) / normalization)
        tf.summary.scalar('negative_dist', tf.reduce_sum(valid_mask * (1 - s) *
                                                        negative_dist) / normalization)
        # loss = tf.compat.v1.Print(loss, [tf.shape(loss), tf.shape(loss)[-1],
        #                                  tf.shape(valid_mask), tf.shape(valid_mask)[-1],
        #                                  tf.rank(loss), tf.rank(valid_mask)],
        #                                 "loss, valid_mask = ")

        loss = tf.reduce_sum(valid_mask * loss) / normalization
        return loss


def spatial_nms(prob, size):
    """Performs non maximum suppression on the heatmap using max-pooling. This method is
    faster than box_nms, but does not suppress contiguous that have the same probability
    value.

    Arguments:
        prob: the probability heatmap, with shape `[H, W]`.
        size: a scalar, the size of the pooling window.
    """

    with tf.name_scope('spatial_nms'):
        prob = tf.expand_dims(tf.expand_dims(prob, axis=0), axis=-1)
        pooled = tf.nn.max_pool(
                prob, ksize=[1, size, size, 1], strides=[1, 1, 1, 1], padding='SAME')
        prob = tf.where(tf.equal(prob, pooled), prob, tf.zeros_like(prob))
        return tf.squeeze(prob)


def box_nms(prob, size, iou=0.1, min_prob=0.01, keep_top_k=0):
    """Performs non maximum suppression on the heatmap by considering hypothetical
    bounding boxes centered at each pixel's location (e.g. corresponding to the receptive
    field). Optionally only keeps the top k detections.

    Arguments:
        prob: the probability heatmap, with shape `[H, W]`.
        size: a scalar, the size of the bouding boxes.
        iou: a scalar, the IoU overlap threshold.
        min_prob: a threshold under which all probabilities are discarded before NMS.
        keep_top_k: an integer, the number of top scores to keep.
    """
    with tf.name_scope('box_nms'):
        pts = tf.cast(tf.where(tf.greater_equal(prob, min_prob)), tf.float32)
        size = tf.constant(size/2.0, dtype=tf.float32)
        boxes = tf.concat([pts-size, pts+size], axis=1)
        scores = tf.gather_nd(prob, tf.cast(pts, dtype=tf.int32))

        indices = tf.image.non_max_suppression(
                    boxes, scores, tf.shape(boxes)[0], iou)
        pts = tf.gather(pts, indices)
        scores = tf.gather(scores, indices)
        if keep_top_k:
            k = tf.minimum(tf.shape(scores)[0], tf.constant(keep_top_k))  # when fewer
            scores, indices = tf.nn.top_k(scores, k)
            pts = tf.gather(pts, indices)
        prob = tf.scatter_nd(tf.cast(pts, dtype=tf.int32), scores, tf.shape(prob))
    return prob


# def layer_predictor(depth, semantic, og, batch_size=0):
#     """ The layer predictor function takes input a depth field of an image og
#         and a semantic desciription for the image.

#     Arguments:
#         depth {tf tensor} -- A 2D Tensor where each element is the depth of the corresponding pixel in the original image
#         semantic {tf tensor} -- A 2D Tensor where each element is the class of the corresponding pixel in the original image
#         og {tf tensor} -- The original image tensor 3D.
    
#     Note:
#         batched version is not suppported yet

#     Returns:
#         [type] -- [description]
#     """
#     # print("Semantic shape:",np.shape(semantic))
#     # print("OG shape:",np.shape(og))
#     # print("Depth shape:",np.shape(depth))
#     output_shape = tf.shape(og)
#     print(semantic)
#     print(depth)
#     if batch_size == 0:
#         semantic_flat = tf.reshape(semantic, [-1])
#         unique_label, _, _ = tf.unique_with_counts(semantic_flat)
#     elif batch_size == 1:
#         depth = tf.squeeze(depth)
#         semantic = tf.squeeze(semantic)
#         og = tf.squeeze(og)        
#         semantic_flat = tf.reshape(semantic, [-1])
#         unique_label, _, _ = tf.unique_with_counts(semantic_flat)
#     else:
#         # semantic_flat = tf.reshape(semantic, [batch_size, -1])
#         # semantic_flat_unstacked = tf.unstack(semantic_flat, axis=0)
#         # for k in range(batch_size):
#         #     _unique_label, _ = tf.unique(semantic_flat_unstacked[k])
#         raise NotImplementedError("layer predictor is not implemented for batches > 1")
    
#     #calculate class average
#     num_class = tf.size(unique_label)
#     unique_labels = tf.expand_dims(unique_label,axis=1)
#     unique_labels = tf.expand_dims(unique_labels,axis=2)
#     # unique_labels = tf.expand_dims(unique_labels,axis=-1)
#     # unique_labels = tf.reshape(unique_label, 
#     #                            tf.concat([tf.shape(semantic)[:-1],[1]], axis=-1))
#     mask = tf.cast(tf.equal(semantic, unique_labels),dtype=tf.float32)



#     total = tf.math.count_nonzero((tf.reshape(mask,(num_class,-1))),axis=1)
#     total = tf.cast(total,dtype=tf.float32)
    
#     enlarged_depth = tf.concat(num_class*[tf.expand_dims(depth,0)],axis=0)
#     enlarged_depth = tf.cast(enlarged_depth, dtype=tf.float32)

#     masked_depth = tf.math.multiply(mask,enlarged_depth)

#     mask_stack = tf.stack([mask, mask, mask],axis=3)
    
#     enlarged_og = tf.stack([og]*num_class,axis=0)
#     enlarged_og = tf.cast(enlarged_og,dtype=tf.float32)
#     classed_pixel = tf.math.multiply(mask_stack,enlarged_og)
#     depth_avg = tf.math.divide(tf.math.reduce_sum(tf.reshape(masked_depth,(num_class,-1)),axis=1),total)

#     #print(len(classed_pixel))
#     index = tf.argsort(depth_avg)
#     #print(index)
#     num_class = tf.size(unique_label)
#     start_idx = num_class // 4
#     end_idx = num_class - num_class // 4
#     foreground_idx = tf.squeeze(index[0:start_idx])
#     midground_idx = tf.squeeze(index[start_idx:end_idx])
#     background_idx = tf.squeeze(index[end_idx:num_class])
#     #print(len(foreground_idx))

#     fore_class = tf.gather(classed_pixel, foreground_idx, axis=0)
#     mid_class = tf.gather(classed_pixel, midground_idx, axis=0)
#     back_class = tf.gather(classed_pixel, background_idx, axis=0)

#     foreground = tf.reduce_sum(fore_class, axis=0)
#     midground = tf.reduce_sum(mid_class, axis=0)
#     background = tf.reduce_sum(back_class, axis=0)

#     layer0 = foreground + midground
#     layer1 = foreground + background
#     layer2 = midground + background
    
#     fore_mask = tf.gather(mask_stack,foreground_idx, axis=0)
#     mid_mask = tf.gather(mask_stack,midground_idx, axis=0)
#     back_mask = tf.gather(mask_stack, background_idx, axis=0)

#     foreground_mask = tf.reduce_sum(fore_mask, axis=0)
#     midground_mask = tf.reduce_sum(mid_mask, axis=0)
#     background_mask = tf.reduce_sum(back_mask, axis=0)
#     # = original image input shape
    

#     layer0 = tf.reshape(layer0, output_shape)
#     layer1 = tf.reshape(layer1, output_shape)
#     layer2 = tf.reshape(layer2, output_shape)
    
#     return layer0, layer1, layer2

def layer_predictor(depth, semantic, og, batch_size=0):
    """ The layer predictor function takes input a depth field of an image og
        and a semantic desciription for the image.
    Arguments:
        depth {tf tensor} -- A 2D Tensor where each element is the depth of the corresponding pixel in the original image
        semantic {tf tensor} -- A 2D Tensor where each element is the class of the corresponding pixel in the original image
        og {tf tensor} -- The original image tensor 3D.
    
    Note:
        batched version is not suppported yet
    Returns:
        [type] -- [description]
    """
    # print("Semantic shape:",np.shape(semantic))
    # print("OG shape:",np.shape(og))
    # print("Depth shape:",np.shape(depth))
    output_shape = tf.shape(og)
    # print("SHAPES")
    # print(semantic)
    # print(tf.shape(depth))
    # print(depth.shape)
    if batch_size == 0:
        semantic_flat = tf.reshape(semantic, [-1])
        unique_label, _, _ = tf.unique_with_counts(semantic_flat)
    elif batch_size == 1:
        depth = tf.squeeze(depth)
        semantic = tf.squeeze(semantic)
        og = tf.squeeze(og)        
        semantic_flat = tf.reshape(semantic, [-1])
        unique_label, _, _ = tf.unique_with_counts(semantic_flat)
    else:
        # semantic_flat = tf.reshape(semantic, [batch_size, -1])
        # semantic_flat_unstacked = tf.unstack(semantic_flat, axis=0)
        # for k in range(batch_size):
        #     _unique_label, _ = tf.unique(semantic_flat_unstacked[k])
        raise NotImplementedError("layer predictor is not implemented for batches > 1")
    
    #calculate class average
    num_class = tf.size(unique_label)
    unique_labels = tf.expand_dims(unique_label,axis=1)
    unique_labels = tf.expand_dims(unique_labels,axis=2)
    # unique_labels = tf.expand_dims(unique_labels,axis=-1)
    # unique_labels = tf.reshape(unique_label, 
    #                            tf.concat([tf.shape(semantic)[:-1],[1]], axis=-1))
    mask = tf.cast((semantic == unique_labels),dtype=tf.float32)



    total = tf.math.count_nonzero((tf.reshape(mask,(num_class,-1))),axis=1)
    total = tf.cast(total,dtype=tf.float32)
    
    enlarged_depth = tf.concat(num_class*[tf.expand_dims(depth,0)],axis=0)
    enlarged_depth = tf.cast(enlarged_depth, dtype=tf.float32)

    masked_depth = tf.math.multiply(mask,enlarged_depth)

    # mask_stack = tf.stack([mask, mask, mask],axis=3)
    
    enlarged_og = tf.stack([og]*num_class,axis=0)
    enlarged_og = tf.cast(enlarged_og,dtype=tf.float32)
    # classed_pixel = tf.math.multiply(mask_stack,enlarged_og)
    classed_pixel = tf.math.multiply(mask, enlarged_og)
    depth_avg = tf.math.divide(tf.math.reduce_sum(tf.reshape(masked_depth,(num_class,-1)),axis=1),total)

    #print(len(classed_pixel))
    index = tf.argsort(depth_avg)
    #print(index)
    num_class = tf.size(unique_label)
    start_idx = num_class // 4
    end_idx = num_class - num_class // 4
    foreground_idx = tf.squeeze(index[0:start_idx])
    midground_idx = tf.squeeze(index[start_idx:end_idx])
    background_idx = tf.squeeze(index[end_idx:num_class])
    #print(len(foreground_idx))

    fore_class = tf.gather(classed_pixel, foreground_idx, axis=0)
    mid_class = tf.gather(classed_pixel, midground_idx, axis=0)
    back_class = tf.gather(classed_pixel, background_idx, axis=0)

    foreground = tf.reduce_sum(fore_class, axis=0)
    midground = tf.reduce_sum(mid_class, axis=0)
    background = tf.reduce_sum(back_class, axis=0)

    layer0 = foreground + midground
    layer1 = foreground + background
    layer2 = midground + background
    
    # fore_mask = tf.gather(mask_stack,foreground_idx, axis=0)
    # mid_mask = tf.gather(mask_stack,midground_idx, axis=0)
    # back_mask = tf.gather(mask_stack, background_idx, axis=0)
    fore_mask = tf.gather(mask, foreground_idx, axis=0)
    mid_mask = tf.gather(mask, midground_idx, axis=0)
    back_mask = tf.gather(mask, background_idx, axis=0)

    foreground_mask = tf.reduce_sum(fore_mask, axis=0)
    midground_mask = tf.reduce_sum(mid_mask, axis=0)
    background_mask = tf.reduce_sum(back_mask, axis=0)
    # = original image input shape
    

    layer0 = tf.reshape(layer0, output_shape)
    layer1 = tf.reshape(layer1, output_shape)
    layer2 = tf.reshape(layer2, output_shape)
    
    return layer0, layer1, layer2