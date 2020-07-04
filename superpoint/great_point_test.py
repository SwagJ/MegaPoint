import os, sys
import numpy as np
import tensorflow as tf
import yaml

from datasets import coco as coco
# from datasets import megadepth as megadepth
from models import great_point
from models import super_point

CONFIG_FILEPATH = 'configs/greatpoint_coco.yaml'
WEIGHTS_FILEPATH=None# '../pretrained_models/weights.pkl.npy'
with open(CONFIG_FILEPATH, 'r') as fr:
    conf = dict(yaml.safe_load(fr))

dataset = coco.Coco(**conf['data'])
d_train = dataset.get_tf_datasets()['training'].batch(conf['model']['batch_size'])

# X = {
#     'input_1' : tf.zeros([1, 240, 320, 3],dtype =tf.float32)
# }
for x in d_train:
    X = x[0]
    break
# print(X)
# sp = super_point.SuperPoint(config=conf['model'], training=True, npyWeightsPath=WEIGHTS_FILEPATH, name='superpoint')
# sp.trainable=True
# sp.compileWrapper()
# sp.fit(x=d_train, steps_per_epoch=10,
    # max_queue_size=20)
gp = great_point.GreatPoint(conf['model'], training=True)
# print(gp.predict(X))
gp.comppileWrapper()
# sp.predict(X)
y = gp.predict(x=X)
# print(y)
# gp.set_compiled_loss()

gp.fit(x=d_train, steps_per_epoch=10,
       max_queue_size=20)