import tensorflow as tf
import numpy as np
import os
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

import superpoint.experiment as experiment
from superpoint.settings import EXPER_PATH

# tf.compat.v1.disable_eager_execution()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--export_name', type=str, default=None)
    args = parser.parse_args()

    experiment_name = args.experiment_name
    export_name = args.export_name if args.export_name else experiment_name
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    assert 'eval_iter' in config

    output_dir = Path(EXPER_PATH, 'outputs/{}/'.format(export_name))
    if not output_dir.exists():
        os.makedirs(output_dir)
    checkpoint = Path(EXPER_PATH, experiment_name)
    if 'checkpoint' in config:
        checkpoint = Path(checkpoint, config['checkpoint'])

    with experiment._init_graph(config, with_dataset=True) as (net, dataset):
        if net.trainable:
            net.load_weights(str(checkpoint))
        test_set = dataset.get_tf_datasets()['test']

        pbar = tqdm(total=config['eval_iter'] if config['eval_iter'] > 0 else None)
        i = 0
        for data in test_set:
            data1 = {'image': data['image']}
            data2 = {'image': data['warped/image']}
            out1 = net.predict(data1)
            pred1 = {'prob_nms' : out1['output_2'], 'descriptors' : out1['output_3']}
            out2 = net.predict(data2)
            pred2 = {'prob_nms' : out2['output_2'], 'descriptors' : out2['output_3']}

            pred = {'prob': pred1['prob_nms'],
                    'warped_prob': pred2['prob_nms'],
                    'desc': pred1['descriptors'],
                    'warped_desc': pred2['descriptors'],
                    'homography': data['homography']}

            if not ('name' in data):
                pred.update(data)
            filename = data['name'].decode('utf-8') if 'name' in data else str(i)
            filepath = Path(output_dir, '{}.npz'.format(filename))
            np.savez_compressed(filepath, **pred)
            i += 1
            pbar.update(1)
            if i == config['eval_iter']:
                break
