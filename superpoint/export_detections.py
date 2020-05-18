import tensorflow as tf
import numpy as np
import os
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

import experiment
from settings import EXPER_PATH

# tf.compat.v1.disable_eager_execution()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--export_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pred_only', action='store_true')
    args = parser.parse_args()

    experiment_name = args.experiment_name
    export_name = args.export_name if args.export_name else experiment_name
    batch_size = args.batch_size
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    assert 'eval_iter' in config

    output_dir = Path(EXPER_PATH, 'outputs/{}/'.format(export_name))
    if not output_dir.exists():
        os.makedirs(output_dir)
    checkpoint = Path(EXPER_PATH, experiment_name)
    if 'checkpoint' in config:
        checkpoint = Path(checkpoint, config['checkpoint'])

    config['model']['pred_batch_size'] = batch_size
    batch_size *= experiment.get_num_gpus()

    gpus= tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    with experiment._init_graph(config, with_dataset=True) as (net, dataset):
        if net.trainable:
            net.load_weights(str(checkpoint))
        test_set = dataset.get_tf_datasets()['test'].batch(batch_size)
        
        i = 0
        for _ in test_set:
            if (i == config.get('skip', 0)):
                break


        pbar = tqdm(total=config['eval_iter'] if config['eval_iter'] > 0 else None)
        i = 0
        for _data in test_set:
            # Gather dataset
            data = dict(zip(_data[0], zip(*[d.values() for d in _data])))

            # Predict
            out = net.predict(data)
            if args.pred_only:
                p = {'pred' : out['output_9']}
                pred = {'points': [np.array(np.where(e)).T for e in p['pred']]}
            else:
                pred = out

            # Export
            d2l = lambda d: [dict(zip(d, e)) for e in zip(*d.values())]  # noqa: E731
            for p, d in zip(d2l(pred), d2l(data)):
                if not ('name' in d):
                    p.update(d)  # Can't get the data back from the filename --> dump
                filename = d['name'].decode('utf-8') if 'name' in d else str(i)
                filepath = Path(output_dir, '{}.npz'.format(filename))
                np.savez_compressed(filepath, **p)
                i += 1
                pbar.update(1)

            if config['eval_iter'] > 0 and i >= config['eval_iter']:
                break
