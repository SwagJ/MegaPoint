import os, sys
import numpy as np
import tensorflow as tf
import yaml
import super_point
import initializer

tf.compat.v1.disable_eager_execution()

CONFIG_FILEPATH = '../configs/superpoint_coco.yaml'
WEIGHTS_FILEPATH= ''

if __name__ == "__main__":
    with open(CONFIG_FILEPATH, 'r') as fr:
        conf = dict(yaml.safe_load(fr)['model'])
    conf['training'] = True
    # print(conf)
    # _config = {
    #     'name': 'super_point',
    #     'batch_size': 2,
    #     'eval_batch_size': 2,
    #     'learning_rate': 0.0001,
    #     'lambda_d': 0.05,
    #     'positive_margin': 1,
    #     'negative_margin': 0.2,
    #     'lambda_loss': 10000,
    #     'detection_threshold': 0.001,
    #     'nms': 4,
    #     'top_k': 600,
    #     'training': True
    # }
    # weightsDictionary = np.load(WEIGHTS_FILEPATH)
    # cw_init = initializer.ConstantWeightsInitializer(weightsDictionary)
    cw_init=None
    sp = super_point.SuperPoint(config=conf, training=True, initializer=cw_init, name='superpoint')
    sp.trainable = True
    x = tf.zeros((1,256,256,1))
    wx = tf.zeros((1,256,256,1))
    out = sp.call({'image':x, 'warped': {'image': wx}})
    for var in tf.compat.v1.global_variables():
        print(var.name)
    
    print("FINISHED")