import logging
import yaml
import os
import argparse
import numpy as np
from contextlib import contextmanager
from json import dumps as pprint

from datasets import get_dataset
from models import get_model
from utils.stdout_capturing import capture_outputs
from settings import EXPER_PATH

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
import tensorflow as tf  # noqa: E402

# tf.compat.v1.disable_eager_execution()

outputs_to_names = {
    'output_1'  : '',
    'output_2'  : '',
    'output_3'  : '',
    'output_4'  : '',
    'output_5'  : '',
    'output_6'  : '',
    'output_7'  : '',
    'output_8'  : '',
    'output_9'  : '',
    'output_10' : ''
}


def train(config, n_iter, output_dir, checkpoint_name='model.ckpt', numpyWeightsPaths=None):
    gpus= tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    checkpoint_path = os.path.join(output_dir, checkpoint_name)
    with _init_graph(config, with_dataset=True) as (net, dataset):
        try:
            dtrain = dataset.get_tf_datasets()['training'].batch(config['model']['batch_size'])
            dval = dataset.get_tf_datasets()['validation'].batch(config['model']['batch_size'])
            validation_freq = config.get('validation_interval', 500)
            num_epochs = n_iter // validation_freq
            steps_per_poch = n_iter // num_epochs
            if numpyWeightsPaths:
                pass
            elif os.path.exists(checkpoint_path):
                net.load_weights(checkpoint_path)
            net.fit(x=dtrain, validation_data=dval, steps_per_epoch=steps_per_poch, 
                    validation_freq=validation_freq, validation_steps=100,
                    steps_per_poch=steps_per_poch, max_queue_size=20,
                    callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path),
                               tf.keras.callbacks.TensorBoard(log_dir=output_dir, 
                                                              histogram_freq=2,
                                                              write_images=True)])
        except KeyboardInterrupt:
            logging.info('Got Keyboard Interrupt, saving model and closing.')
        net.save(checkpoint_path)


def evaluate(config, output_dir, n_iter=None):
    with _init_graph(config, with_dataset=True) as (net, dataset):
        dval = dataset.get_tf_datasets()['validation'].batch(config['model']['batch_size'])
        checkpoint_path = os.path.join(output_dir, 'model.ckpt')
        net.load_weights(checkpoint_path)
        results = net.evaluate(x=dval, steps=n_iter, return_dict=True)
    return results


def predict(config, output_dir, n_iter):
    pred = []
    data = []
    with _init_graph(config, with_dataset=True) as (net, dataset):
        if net.trainable:
            checkpoint_path = os.path.join(output_dir, 'model.ckpt')
            net.load_weights(checkpoint_path)
        test_set = dataset.get_tf_datasets()['test'].batch(config['model']['batch_size'])
        for d in test_set:
            data.append(d)
            pred.append(net.predict(x=d))
    return pred, data


def set_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)


def get_num_gpus():
    if( 'CUDA_VISIBLE_DEVICES' in os.environ.keys()):
        return len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    else:
        return 0

@contextmanager
def _init_graph(config, with_dataset=False, numpyWeightsPaths=None):
    set_seed(config.get('seed', int.from_bytes(os.urandom(4), byteorder='big')))
    n_gpus = get_num_gpus()
    logging.info('Number of GPUs detected: {}'.format(n_gpus))

    dataset = get_dataset(config['data']['name'])(**config['data'])
    model = get_model(config['model']['name'])(
            config['model'], training = config['model']['training'])
    model.trainable = config['model']['training']
    model.compileWrapper()

    return dataset, model


def _cli_train(config, output_dir, args):
    assert 'train_iter' in config

    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    train(config, config['train_iter'], output_dir)

    if args.eval:
        _cli_eval(config, output_dir, args)


def _cli_eval(config, output_dir, args):
    # Load model config from previous experiment
    with open(os.path.join(output_dir, 'config.yml'), 'r') as f:
        model_config = yaml.load(f)['model']
    model_config.update(config.get('model', {}))
    config['model'] = model_config

    results = evaluate(config, output_dir, n_iter=config.get('eval_iter'))

    # Print and export results
    logging.info('Evaluation results: \n{}'.format(
        pprint(results, indent=2, default=str)))
    with open(os.path.join(output_dir, 'eval.txt'), 'a') as f:
        f.write('Evaluation for {} dataset:\n'.format(config['data']['name']))
        for r, v in results.items():
            f.write('\t{}:\n\t\t{}\n'.format(r, v))
        f.write('\n')


# TODO
def _cli_pred(config, args):
    raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Training command
    p_train = subparsers.add_parser('train')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.add_argument('--eval', action='store_true')
    p_train.set_defaults(func=_cli_train)

    # Evaluation command
    p_train = subparsers.add_parser('evaluate')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.set_defaults(func=_cli_eval)

    # Inference command
    p_train = subparsers.add_parser('predict')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.set_defaults(func=_cli_pred)

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with capture_outputs(os.path.join(output_dir, 'log')):
        logging.info('Running command {}'.format(args.command.upper()))
        args.func(config, output_dir, args)
