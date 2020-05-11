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
d_train = dataset.get_tf_datasets()['training'].batch(1) # cannot add larger batch
d_val = dataset.get_tf_datasets()['validation'].batch(1) # inputs are not the same due to keypoints

conf['model']['training'] = True
sp = super_point.SuperPoint(conf['model'])
sp.trainable = True
sp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=conf['model']['learning_rate']),
           loss=super_point.SuperPointLoss(conf['model']),
           metrics=super_point.SuperPointMetrics())

sp.fit(x=d_train, y=None, validation_data=d_val,
       steps_per_epoch=10)