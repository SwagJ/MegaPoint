import os, sys
import numpy as np
import tensorflow as tf
import yaml

from datasets import coco
from models import super_point

CONFIG_FILEPATH = 'configs/superpoint_coco.yaml'


with open(CONFIG_FILEPATH, 'r') as fr:
    conf = dict(yaml.safe_load(fr))

# conf['training'] = True
dataset = coco.Coco(**conf['data'])
d_train = tf.compat.v1.data.make_one_shot_iterator(
    dataset.get_tf_datasets()['training']) # cannot add larger batch
# d_val = dataset.get_tf_datasets()['validation'].batch(1) # inputs are not the same due to keypoints

#def custom_fit(x,batch_size,steps)
#for el in d_train:
#    

conf['model']['training'] = True
sp = super_point.SuperPoint(conf['model'])
sp.trainable = True
sp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=conf['model']['learning_rate']),
           loss=super_point.SuperPointLoss(conf['model']))
        #    metrics=[super_point.SuperPointMetrics()])

sp.fit(x=d_train,
       steps_per_epoch=10)
