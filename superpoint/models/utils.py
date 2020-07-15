import tensorflow as tf

from .homographies import warp_points
from .backbones.vgg import vgg_block


def detector_head(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'batch_normalization': True,
                   'training': config['training'],
                   'kernel_reg': config.get('kernel_reg', 0.)}
    cfirst = config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel

    with tf.compat.v1.variable_scope('detector', reuse=tf.compat.v1.AUTO_REUSE):
        x = vgg_block(inputs, 256, 3, 'conv1',
                      activation=tf.nn.relu, **params_conv)
        x = vgg_block(x, 1+pow(config['grid_size'], 2), 1, 'conv2',
                      activation=None, **params_conv)

        prob = tf.nn.softmax(x, axis=cindex)
        # Strip the extra “no interest point” dustbin
        prob = prob[:, :-1, :, :] if cfirst else prob[:, :, :, :-1]
        prob = tf.nn.depth_to_space(
                prob, config['grid_size'], data_format='NCHW' if cfirst else 'NHWC')
        prob = tf.squeeze(prob, axis=cindex)

    return {'logits': x, 'prob': prob}


def descriptor_head(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'batch_normalization': True,
                   'training': config['training'],
                   'kernel_reg': config.get('kernel_reg', 0.)}
    cfirst = config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel

    with tf.compat.v1.variable_scope('descriptor', reuse=tf.compat.v1.AUTO_REUSE):
        x = vgg_block(inputs, 256, 3, 'conv1',
                      activation=tf.nn.relu, **params_conv)
        x = vgg_block(x, config['descriptor_size'], 1, 'conv2',
                      activation=None, **params_conv)

        desc = tf.transpose(x, [0, 2, 3, 1]) if cfirst else x
        desc = tf.image.resize(
            desc, config['grid_size'] * tf.shape(desc)[1:3])
        desc = tf.transpose(desc, [0, 3, 1, 2]) if cfirst else desc
        desc = tf.nn.l2_normalize(desc, cindex)

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

    loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits, weights=valid_mask)
    return loss


def descriptor_loss(descriptors, warped_descriptors, homographies,
                    valid_mask=None, **config):
    # Compute the position of the center pixel of every cell in the image
    (batch_size, Hc, Wc) = tf.unstack(tf.compat.v1.to_int32(tf.shape(descriptors)[:3]))
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
    valid_mask = tf.reshape(valid_mask, [batch_size, 1, 1, Hc, Wc])

    normalization = tf.reduce_sum(valid_mask) * tf.cast(Hc * Wc, tf.float32)
    # Summaries for debugging
    # tf.summary.scalar('nb_positive', tf.reduce_sum(valid_mask * s) / normalization)
    # tf.summary.scalar('nb_negative', tf.reduce_sum(valid_mask * (1 - s)) / normalization)
    tf.summary.scalar('positive_dist', tf.reduce_sum(valid_mask * config['lambda_d'] *
                                                     s * positive_dist) / normalization)
    tf.summary.scalar('negative_dist', tf.reduce_sum(valid_mask * (1 - s) *
                                                     negative_dist) / normalization)
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
        scores = tf.gather_nd(prob, tf.compat.v1.to_int32(pts))
        with tf.device('/cpu:0'):
            indices = tf.image.non_max_suppression(
                    boxes, scores, tf.shape(boxes)[0], iou)
        pts = tf.gather(pts, indices)
        scores = tf.gather(scores, indices)
        if keep_top_k:
            k = tf.minimum(tf.shape(scores)[0], tf.constant(keep_top_k))  # when fewer
            scores, indices = tf.nn.top_k(scores, k)
            pts = tf.gather(pts, indices)
        prob = tf.scatter_nd(tf.compat.v1.to_int32(pts), scores, tf.shape(prob))
    return prob

def layer_predictor_jingyuan(image,depth,semantic):
    with tf.name_scope('layer_predictor'):
        #depth = tf.squeeze(depth,0)
        #semantic = tf.squeeze(semantic,0)
        indicator = tf.random.uniform([],maxval=30,dtype=tf.int32) % 3
        height = tf.constant(240,dtype=tf.float32)
        width = tf.constant(320,dtype=tf.float32)
        semantic_flat = tf.reshape(semantic,(tf.size(),))
        #semantic_flat = tf.compat.v1.layers.flatten(semantic)
        print(f"\nsemantic_flat shape:{semantic_flat.shape}\n")
        unique_label, idx, counts = tf.unique_with_counts(semantic_flat)

        num_class = unique_label.shape[0]
        unique_labels = tf.expand_dims(unique_label,axis=1)
        unique_labels = tf.expand_dims(unique_labels,axis=2)

        mask = tf.cast((semantic == unique_labels),dtype=tf.float32)
        print(f"\nmask shape:{mask.shape}\n")

        total = tf.math.count_nonzero((tf.reshape(mask,(num_class,height*width))),axis=1)
        total = tf.cast(total,dtype=tf.float32)
        enlarged_depth = tf.concat(num_class*[tf.expand_dims(depth,0)],axis=0)

        masked_depth = tf.math.multiply(mask,enlarged_depth)
        mask_stack = tf.expand_dims(mask,axis=1)
        enlarged_og = tf.stack([image]*num_class,axis=0)
        classed_pixel = tf.math.multiply(mask_stack,enlarged_og)

        depth_avg = tf.math.divide(tf.math.reduce_sum(tf.reshape(masked_depth,
                                    (num_class,height*width)),axis=1),total)

        index = tf.argsort(depth_avg)
        start_idx = num_class // 4
        end_idx = num_class - num_class // 4
        foreground_idx = (index[0:start_idx])
        midground_idx = (index[start_idx:end_idx])
        background_idx = (index[end_idx:num_class])
        
        fore_class = tf.gather(classed_pixel,foreground_idx,axis=0)
        mid_class = tf.gather(classed_pixel,midground_idx,axis=0)
        back_class = tf.gather(classed_pixel,background_idx,axis=0)

        foreground = tf.reduce_sum(fore_class,axis=0)
        midground = tf.reduce_sum(mid_class,axis=0)
        background = tf.reduce_sum(back_class,axis=0)

        if indicator == 0:
            masked_image = foreground + midground
            mask = tf.add(tf.cast(masked_image==image,dtype=tf.float32),tf.constant(1e-8,dtype=tf.float32))
        elif indicator == 1:
            mask = foreground + background
            mask = tf.add(tf.cast(masked_image==image,dtype=tf.float32),tf.constant(1e-8,dtype=tf.float32))
        else: 
            mask = midground + background
            mask = tf.add(tf.cast(masked_image==image,dtype=tf.float32),tf.constant(1e-8,dtype=tf.float32))
        


        print(f"\n Indictor: {indicator.shape}, foreground:{foreground.shape}\n")
        print(f"\n midground: {midground.shape}, background:{background.shape}\n")
        

    return mask,masked_image

def mask_warped(image,mask):
    #with tf.name_scope('mask_warped'):
    return tf.multiply(image,mask)


def layer_predictor_ioannis(og, depth, semantic):
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
    #depth = tf.squeeze(depth,0)
    #$semantic = tf.squeeze(semantic,0)
    output_shape = tf.shape(og)
    print(og.shape)
    flat_og = tf.reshape(og, [-1])
    flat_semantic = tf.reshape(semantic, [-1])
    flat_depth = tf.reshape(depth, [-1])
    indicator = tf.random.uniform([],maxval=30,dtype=tf.int32) % 3
    
    # find all unique labels of the segmented image
    # also find their indices
    # unique_label has the values of the unique labels
    # un_label_idx holds the label index of each element in flat_semantic
    # unique_label[un_label_idx[i]] = flat_semantic[i]
    unique_label, un_label_idx = tf.unique(flat_semantic)
    
    # The number of unique labels
    num_unique_labels = tf.size(unique_label)

    # un_label_idx[i] point the label index in unique_label that the i-th element has
    # shape(depth) = shape(semantic) --> size(depth) = size(semantic) and 
    # also flat_depth[i] corresponds to flat_semantic[i]
    # un_label_idx[i] is the index of the label index for each depth in flat_depth also 
    # unsorted_segment_mean gets for k = 0,..., num_unique_labels-1 the 
    # mean_j(flat_depth[j], for un_label_idx[j] = k), j=0,...,size(flat_depth)
    mean_depth_per_label = tf.math.unsorted_segment_mean(flat_depth, un_label_idx, num_unique_labels)
    
    # get the indices for sorting the mean_depth_per_label
    index = tf.argsort(mean_depth_per_label)

    # This seperates the sorted indices to 3 pieces
    # in order to extract objects in image 
    # with these pieces
    start_idx = num_unique_labels // 4
    end_idx = num_unique_labels - (num_unique_labels // 4)

    foreground_idx = tf.reshape(index[0:start_idx], [-1,1])
    midground_idx = tf.reshape(index[start_idx:end_idx], [-1,1])
    background_idx = tf.reshape(index[end_idx:num_unique_labels], [-1, 1])

    
    # In case a label does not match to the `layer` idx
    # replaces it with zeros, see tf.where below
    z = tf.zeros_like(flat_og)
    
    # This checks if un_label_idx is one of `layer`_idx values.
    # `layer` can be one of foreground, midground, background
    # fore_mask[i] = un_label_idx[i] in foreground_idx 
    # = any_j(un_label_idx[i] == foreground_idx[j])
    # Note: j is on axis 0
    fore_mask = tf.math.reduce_any(tf.equal(un_label_idx, foreground_idx) ,axis=0 )
    mid_mask  = tf.math.reduce_any(tf.equal(un_label_idx, midground_idx) ,axis=0 )
    back_mask = tf.math.reduce_any(tf.equal(un_label_idx, background_idx) ,axis=0 )    


    foreground_flat = tf.where(fore_mask, flat_og, z)
    midground_flat = tf.where(mid_mask, flat_og, z)
    background_flat = tf.where(back_mask, flat_og, z)

    if indicator == 0:
        masked_image_flat = foreground_flat + midground_flat
        tf.print(f"\nmasked_image shape:{masked_image_flat.shape}\n")
        mask_flat = tf.add(tf.cast(tf.equal(masked_image_flat,flat_og),dtype=tf.float32),
                            tf.constant(1e-6,dtype=tf.float32))
        mask = tf.reshape(mask_flat, output_shape)
        masked_image = tf.reshape(masked_image_flat, output_shape) + 1e-6
    elif indicator == 1:
        masked_image_flat = foreground_flat + background_flat
        tf.print(f"\nmasked_image shape:{masked_image_flat.shape}\n")
        #mask_flat = tf.add(tf.cast(tf.equal(masked_image_flat,flat_og),dtype=tf.float32),
        #                    tf.constant(1e-6,dtype=tf.float32))
        mask_flat = tf.cast(tf.equal(masked_image_flat,flat_og),dtype=tf.float32)
        mask = tf.reshape(mask_flat, output_shape)
        masked_image = tf.reshape(masked_image_flat, output_shape) + 1e-6
    else: 
        masked_image_flat = midground_flat + background_flat
        tf.print(f"\nmasked_image shape:{masked_image_flat.shape}\n")
        mask_flat = tf.add(tf.cast(tf.equal(masked_image_flat,flat_og),dtype=tf.float32),
                            tf.constant(1e-6,dtype=tf.float32))
        mask = tf.reshape(mask_flat, output_shape)
        masked_image = tf.reshape(masked_image_flat, output_shape) + 1e-6
    
    #masked_image = tf.expand_dims(masked_image,0)
    #mask = tf.expand_dims(mask,0)
    print(f"After expand_dims:{mask.shape},{masked_image.shape}")
    #masked_image = tf.transpose(masked_image,[0,3,1,2])
    #mask = tf.transpose(mask,[0,3,1,2])
    # layer0_flat = foreground_flat + midground_flat
    # layer1_flat = foreground_flat + background_flat
    # layer2_flat = midground_flat + background_flat

    # layer0 = tf.reshape(layer0_flat, output_shape)
    # layer1 = tf.reshape(layer1_flat, output_shape)
    # layer2 = tf.reshape(layer2_flat, output_shape)

    return mask, masked_image


