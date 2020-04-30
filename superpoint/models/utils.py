import tensorflow as tf

from .homographies import warp_points
from .backbones.vgg import vgg_block
from .backbones.squeezeNet import fire_layer

def detector_head(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'batch_normalization': True,
                   'training': config['training'],
                   'kernel_reg': config.get('kernel_reg', 0.)}
    cfirst = config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel

    with tf.compat.v1.variable_scope('detector', reuse=tf.compat.v1.AUTO_REUSE):
        if(config['name'] == "squeeze_point"):
            # x = fire_layer('firelayer11', inputs, 128, 64, 64)
            x = vgg_block(inputs, 256, 3, 'conv1',
                          activation=tf.nn.relu, **params_conv)
            
            # x = tf.concat([tf.keras.layers.Dense(pow(config['grid_size'],2) + 1)(x),
            #                tf.keras.layers.Dense(pow(config['grid_size'],2) + 1)(x)],
            #                                          3, name='firelayer1/concat')
            x0 = tf.concat([vgg_block(x, 1+pow(config['grid_size'], 2), 1, 'conv2',
                          activation=None, **params_conv),
                           vgg_block(x, 1+pow(config['grid_size'], 2), 1, 'conv2',
                          activation=None, **params_conv)], axis=2)
            x1 = tf.concat([vgg_block(x, 1+pow(config['grid_size'], 2), 1, 'conv2',
                          activation=None, **params_conv),
                           vgg_block(x, 1+pow(config['grid_size'], 2), 1, 'conv2',
                          activation=None, **params_conv)], axis=2)                          
            x = tf.concat([x0,x1], axis=1)
            # if params_conv['batch_normalization']:
            #     x = tf.keras.layers.BatchNormalization(
            #         trainable=True if params_conv['training'] else False, name='bn',
            #         fused=True, axis=1 if cfirst else -1)(x)
            y = x
            # x = fire_layer('firelayer3', x, int(pow(config['grid_size'], 2)),
            #                int(pow(config['grid_size'], 2)/2), 1+int(pow(config['grid_size'], 2)/2))
            # x = tf.transpose(x, perm=[0,3,2,1])
            # x = tf.reduce_max(x, axis=-1)
            # S = tf.shape(x)
            # x = tf.reshape(x[:,:,:,:4], [S[0],S[1]*2,S[2]*2])
        else:
            x = vgg_block(inputs, 256, 3, 'conv1',
                          activation=tf.nn.relu, **params_conv)
            x = vgg_block(x, 1+pow(config['grid_size'], 2), 1, 'conv2',
                          activation=None, **params_conv)
            y = x

        # x = tf.compat.v1.Print(x, [tf.shape(x), tf.shape(x)[-1], tf.rank(x)], "x = ")
        prob = tf.nn.softmax(y, axis=cindex)
        # Strip the extra “no interest point” dustbin
        prob = prob[:, :-1, :, :] if cfirst else prob[:, :, :, :-1]

        # prob = tf.compat.v1.Print(prob, [tf.shape(prob), tf.shape(prob)[3], tf.rank(prob)], "prob = ")

        prob = tf.nn.depth_to_space(
                prob, config['grid_size'], data_format='NCHW' if cfirst else 'NHWC')

        # prob = tf.compat.v1.Print(prob, [cindex, tf.shape(prob), tf.shape(prob)[3], tf.rank(prob)], "prob = ")

        prob = tf.squeeze(prob, axis=cindex)

        # prob = tf.compat.v1.Print(prob, [tf.shape(prob), tf.shape(prob)[3], tf.rank(prob)], "prob = ")

    return {'logits': x, 'prob': prob}


def descriptor_head(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'batch_normalization': True,
                   'training': config['training'],
                   'kernel_reg': config.get('kernel_reg', 0.)}
    cfirst = config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel

    with tf.compat.v1.variable_scope('descriptor', reuse=tf.compat.v1.AUTO_REUSE):
        
        if(config['name'] == "squeeze_point"):
            file = open('temp_squeeze.info', 'w')
            # print(inputs, file=file)
            x = fire_layer('firelayer1', inputs, 128, 64, 64)
            # print(x, file=file)
            # x = fire_layer('firelayer2', x, config['descriptor_size'], 
            #                int(config['descriptor_size']), int(config['descriptor_size']))
            # x = tf.keras.layers.Dense(config['descriptor_size'])(x)
            # print(x, file=file)
            x0 = tf.concat([vgg_block(x, 1+pow(config['grid_size'], 2), 1, 'conv2',
                          activation=None, **params_conv),
                           vgg_block(x, 1+pow(config['grid_size'], 2), 1, 'conv2',
                          activation=None, **params_conv)], axis=2)
            x1 = tf.concat([vgg_block(x, 1+pow(config['grid_size'], 2), 1, 'conv2',
                          activation=None, **params_conv),
                           vgg_block(x, 1+pow(config['grid_size'], 2), 1, 'conv2',
                          activation=None, **params_conv)], axis=2)                          
            x = tf.concat([x0,x1], axis=1)
            # x = tf.transpose(__x, perm=[3,1,2,0])
        else:
            file = open('temp_vgg.info', 'w')
            # print(inputs, file=file)
            x = vgg_block(inputs, 256, 3, 'conv1',
                          activation=tf.nn.relu, **params_conv)
            # print(x, file=file)
            x = vgg_block(x, config['descriptor_size'], 1, 'conv2',
                          activation=None, **params_conv)
            # print(x, file=file)
        
        desc = tf.transpose(x, [0, 2, 3, 1]) if cfirst else x
        print(desc, file=file)

        
        desc = tf.image.resize(
            desc, config['grid_size'] * tf.shape(desc)[1:3])
        desc = tf.transpose(desc, [0, 3, 1, 2]) if cfirst else desc
        desc = tf.nn.l2_normalize(desc, cindex)
        # print(desc, file=file)
        file.close()
    return {'descriptors_raw': x, 'descriptors': desc}


def detector_loss(keypoint_map, logits, valid_mask=None, **config):
    # Convert the boolean labels to indices including the "no interest point" dustbin
    labels = tf.cast(keypoint_map[..., tf.newaxis], tf.float32)  # for GPU
    labels = tf.nn.space_to_depth(labels, config['grid_size'])
    shape = tf.concat([tf.shape(labels)[:3], [1]], axis=0)
    labels = tf.concat([2*labels, tf.ones(shape)], 3)
    # multiply by a small random matrix to randomly break ties in argmax
    labels = tf.argmax(labels * tf.compat.v1.random_uniform(tf.shape(labels), 0, 0.1),
                       axis=3)

    # Mask the pixels if bordering artifacts appear
    valid_mask = tf.ones_like(keypoint_map) if valid_mask is None else valid_mask
    valid_mask = tf.cast(valid_mask[..., tf.newaxis], tf.float32)  # for GPU
    valid_mask = tf.nn.space_to_depth(valid_mask, config['grid_size'])
    valid_mask = tf.reduce_prod(valid_mask, axis=3)  # AND along the channel dim

    # labels = tf.compat.v1.Print(labels,[tf.shape(labels), tf.shape(logits), tf.shape(logits)[-1], tf.rank(logits), tf.shape(valid_mask)], "labels, logits, valid_mask =")
    #  output_stream="file:///media/terabyte/projects/SUPERPOINT/SuperPointIoannis/superpoint.info")
    
    loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits, weights=valid_mask)
    return loss


def descriptor_loss(descriptors, warped_descriptors, homographies,
                    valid_mask=None, **config):
    # Compute the position of the center pixel of every cell in the image
    (batch_size, Hc, Wc) = tf.unstack(tf.cast(tf.shape(descriptors)[:3], dtype=tf.int32))
    # print((batch_size, Hc, Wc), file=open("3plet.info", "w"))
    coord_cells = tf.stack(tf.meshgrid(
        tf.range(Hc), tf.range(Wc), indexing='ij'), axis=-1)
    coord_cells = coord_cells * config['grid_size'] + config['grid_size'] // 2  # (Hc, Wc, 2)
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
    s = tf.cast(tf.less_equal(cell_distances, config['grid_size'] - 0.5), tf.float32)
    # s[id_batch, h, w, h', w'] == 1 if the point of coordinates (h, w) warped by the
    # homography is at a distance from (h', w') less than config['grid_size']
    # and 0 otherwise

    # Normalize the descriptors and
    # compute the pairwise dot product between descriptors: d^t * d'
    descriptors = tf.reshape(descriptors, [batch_size, Hc, Wc, 1, 1, -1])
    descriptors = tf.nn.l2_normalize(descriptors, -1)
    warped_descriptors = tf.reshape(warped_descriptors,
                                    [batch_size, 1, 1, Hc, Wc, -1])
    warped_descriptors = tf.nn.l2_normalize(warped_descriptors, -1)
    dot_product_desc = tf.reduce_sum(descriptors * warped_descriptors, -1)
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
    positive_dist = tf.maximum(0., config['positive_margin'] - dot_product_desc)
    negative_dist = tf.maximum(0., dot_product_desc - config['negative_margin'])
    loss = config['lambda_d'] * s * positive_dist + (1 - s) * negative_dist

    # Mask the pixels if bordering artifacts appear
    valid_mask = tf.ones([batch_size,
                          Hc * config['grid_size'],
                          Wc * config['grid_size']], tf.float32)\
        if valid_mask is None else valid_mask
    valid_mask = tf.cast(valid_mask[..., tf.newaxis], tf.float32)  # for GPU
    valid_mask = tf.nn.space_to_depth(valid_mask, config['grid_size'])
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
    tf.summary.scalar('positive_dist', tf.reduce_sum(valid_mask * config['lambda_d'] *
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
        size = tf.constant(size/2.)
        boxes = tf.concat([pts-size, pts+size], axis=1)
        scores = tf.gather_nd(prob, tf.cast(pts, dtype=tf.int32))
        with tf.device('/cpu:0'):
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
