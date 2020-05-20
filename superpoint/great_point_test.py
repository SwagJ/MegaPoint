import os, sys
import numpy as np
import tensorflow as tf
import yaml

from datasets import coco as coco
# from datasets import megadepth as megadepth
from models import great_point

# CONFIG_FILEPATH = 'configs/superpoint_coco.yaml'
CONFIG_FILEPATH = 'configs/superpoint_coco.yaml'

with open(CONFIG_FILEPATH, 'r') as fr:
    conf = dict(yaml.safe_load(fr))



# dataset = coco.Coco(**conf['data'])
# d_train = dataset.get_tf_datasets()['training'].batch(conf['model']['batch_size'])
    
X = {
    'input_1' : tf.zeros([1, 240, 320, 3],dtype =tf.float32)
}
    
gp = great_point.GreatPoint(conf['model'], training=False)

gp.predict(x=X)
# gp.set_compiled_loss()

# gp.fit(x=d_train, steps_per_epoch=10,
#         max_queue_size=20)