import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import tensorflow as tf
import models
from models import magic_point


def main():
    pass

class baseModelTemp(models.base_model.BaseModel):
    def _loss(self, outputs, inputs, **config):
        return tf.Tensor(1000.0)
    def _metrics(self, outputs, inputs, **config):
        return {}
    def _model(self, inputs, mode, **config):
        return {}
        



if __name__ == "__main__":
    # d = magic_point.MagicPoint.default_config
    d = {}
    config = {
            "batch_size" : 2,
            "learning_rate" : 1.0,
            "dataset_name" : "eval"}
    print(config)
    b = baseModelTemp(d, 1, None, batch_size=2, learning_rate=1.0, dataset_name='eval')