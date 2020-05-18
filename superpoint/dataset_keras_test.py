import os, sys
import numpy as np
import tensorflow as tf
import yaml

from datasets import coco as coco
from datasets import megadepth as megadepth
from models import super_point

# CONFIG_FILEPATH = 'configs/superpoint_coco.yaml'
CONFIG_FILEPATH = 'configs/superpoint_coco.yaml'

# tf.compat.v1.disable_eager_execution()

with open(CONFIG_FILEPATH, 'r') as fr:
    conf = dict(yaml.safe_load(fr))


# with tf.device('/cpu:0'):
# conf['training'] = True
# dataset = coco.Coco(**conf['data'])
dataset = megadepth.Megadepth(**conf['data'])
# d_train = dataset.data
d_train = \
    dataset.get_tf_datasets()['training'].batch(conf['model']['batch_size'])
# d_val = dataset.get_tf_datasets()['validation'].batch(1) # inputs are not the same due to keypoints

#def custom_fit(x,batch_size,steps)
#for el in d_train:
#    
# d_train_sequence = coco.CocoSequence(d_train, 1) 
i = 0
for d in d_train:
    # print(d)
    i = i + 1
    if i > 2:
        break

# class MyModel(tf.keras.Model):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.d = (tf.keras.layers.Dense(1))
    
#     def call(self, x):
#         out = self.d(x['input_1'])
        
#         return {'output_1': out, 'output_2': out}

# model = MyModel()
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(8, input_shape=[240, 320, 3]))
# model.add()
# model.build((2, 240, 320, 3))

# model.compile(optimizer='sgd', loss='binary_crossentropy')
# model.fit(d_train, steps_per_epoch=10)

# conf['model']['training'] = True
# sp = super_point.SuperPoint(conf['model'], training=True)
# sp.trainable = True
# sp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=conf['model']['learning_rate']),
#             loss=super_point.SuperPointLoss(conf['model']))
        #    metrics=[super_point.SuperPointMetrics()])
# sp.set_compiled_loss()

# sp.fit(x=d_train, steps_per_epoch=10,
#         max_queue_size=20,
#         workers=4)
# workers = 0


