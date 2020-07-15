import numpy as np
import tensorflow as tf
from pathlib import Path
import os

from .base_dataset import BaseDataset
from .utils import pipeline
from superpoint.settings import DATA_PATH, EXPER_PATH


class MegadepthMegapoint(BaseDataset):
    default_config = {
        'labels': None,
        'cache_in_memory': False,
        'validation_size': 100,
        'truncate': None,
        'preprocessing': {
            'resize': [240, 320]
        },
        'num_parallel_calls': 10,
        'augmentation': {
            'photometric': {
                'enable': False,
                'primitives': 'all',
                'params': {},
                'random_order': True,
            },
            'homographic': {
                'enable': False,
                'params': {},
                'valid_border_margin': 0,
            },
        },
        'warped_pair': {
            'enable': False,
            'params': {},
            'valid_border_margin': 0,
        },
    }

    def _init_dataset(self, **config):
        image_paths = []

        
        base_path = Path(DATA_PATH, 'MegaDepth_v1/')
        for sub_dir in list(base_path.iterdir()):
            num_dir = base_path / sub_dir
            for sub_dir2 in list(num_dir.iterdir()):
                dense_dir = num_dir / sub_dir2
                imgs_path = dense_dir / 'imgs'
                for p in list(imgs_path.iterdir()):
                    image_paths.append(p)

        if config['truncate']:
            image_paths = image_paths[:config['truncate']]

        names = [p.stem for p in image_paths]
        depth_paths = []
        semantic_paths = []

        for p in image_paths:
            semantic_path = os.fspath(p.parent.parent) + '/semantics/' + os.fspath(p.stem) + '.jpg'
            #assert semantic_path.exists(), 'Image {} has no corresponding semantic {}'.format(p.stem, semantic)
            depth_path = os.fspath(p.parent.parent) + '/depth/' + os.fspath(p.stem) + '.jpg'
            #assert semantic_path.exists(), 'Image {} has no corresponding depth {}'.format(p.stem, depth)
            depth_paths.append(depth_path)
            semantic_paths.append(semantic_path)
            #print(depth_paths)
        image_paths = [str(p) for p in image_paths]
        files = {'image_paths': image_paths, 'names': names, 
                    'depth_paths': depth_paths, 'semantic_paths': semantic_paths}


        if config['labels']:
            label_paths = []
            for n in names:
                p = Path(EXPER_PATH, config['labels'], '{}.npz'.format(n))
                assert p.exists(), 'Image {} has no corresponding label {}'.format(n, p)
                label_paths.append(str(p))
            files['label_paths'] = label_paths

        tf.data.Dataset.map_parallel = lambda self, fn: self.map(
                fn, num_parallel_calls=config['num_parallel_calls'])

        return files


    def _get_data(self, files, split_name, **config):
        has_keypoints = 'label_paths' in files
        is_training = split_name == 'training'

        def _read_image(path):
            image = tf.io.read_file(path)
            image = tf.image.decode_png(image, channels=3)
            return tf.cast(image, tf.float32)

        def _read_depth_semantic(path):
            image = tf.io.read_file(path)
            image = tf.image.decode_png(image, channels=1)
            return tf.cast(image, tf.float32)

        def _preprocess(image):
            image = tf.image.rgb_to_grayscale(image)
            if config['preprocessing']['resize']:
                image = pipeline.ratio_preserving_resize(image,
                                                         **config['preprocessing'])
            return image

        def _preprocess_depth_semantic(image):
            #image = tf.image.rgb_to_grayscale(image)
            if config['preprocessing']['resize']:
                image = pipeline.ratio_preserving_resize(image,
                                                         **config['preprocessing'])
            return image

        # Python function
        def _read_points(filename):
            return np.load(filename.decode('utf-8'))['points'].astype(np.float32)


        names = tf.data.Dataset.from_tensor_slices(files['names'])

        images = tf.data.Dataset.from_tensor_slices(files['image_paths'])
        depth = tf.data.Dataset.from_tensor_slices(files['depth_paths'])
        semantic = tf.data.Dataset.from_tensor_slices(files['semantic_paths'])

        images = images.map(_read_image)
        images = images.map(_preprocess)

        depth = depth.map(_read_depth_semantic)
        depth = depth.map(_preprocess_depth_semantic)

        semantic = semantic.map(_read_depth_semantic)
        semantic = semantic.map(_preprocess_depth_semantic)


        data = tf.data.Dataset.zip({'image': images, 'depth': depth, 
                                    'semantic':semantic,'name':names})

        #tf.print(whole)
        #mask = data.map_parallel(pipeline.layer_predictor)

        #masked_image = tf.data.Dataset.zip({images,mask})
        #masked_image = masked_image.map(lambda a,b: tf.compat.v1.py_func(pipeline.mask_image,
        #                                [a,b],Tout=tf.float32))
        #stf.print(images)
        #data = tf.data.Dataset.zip({'image': images, 'name': names})
        # Add keypoints
        if has_keypoints:
            kp = tf.data.Dataset.from_tensor_slices(files['label_paths'])
            kp = kp.map(lambda path: tf.compat.v1.py_func(_read_points, [path], tf.float32))
            kp = kp.map(lambda points: tf.reshape(points, [-1, 2]))
            data = tf.data.Dataset.zip((data, kp)).map(
                    lambda d, k: {**d, 'keypoints': k})
            data = data.map(pipeline.add_dummy_valid_mask)

        # Keep only the first elements for validation
        if split_name == 'validation':
            data = data.take(config['validation_size'])

        # Cache to avoid always reading from disk
        if config['cache_in_memory']:
            tf.compat.v1.logging.info('Caching data, fist access will take some time.')
            data = data.cache()

        # Generate the warped pair
        if config['warped_pair']['enable']:
            assert has_keypoints
            warped = data.map_parallel(lambda d: pipeline.homographic_augmentation(
                d, add_homography=True, **config['warped_pair']))
            if is_training and config['augmentation']['photometric']['enable']:
                warped = warped.map_parallel(lambda d: pipeline.photometric_augmentation(
                    d, **config['augmentation']['photometric']))
            warped = warped.map_parallel(pipeline.add_keypoint_map)
            # Merge with the original data
            data = tf.data.Dataset.zip((data, warped))
            data = data.map(lambda d, w: {**d, 'warped': w})

        # Data augmentation
        if has_keypoints and is_training:
            if config['augmentation']['photometric']['enable']:
                data = data.map_parallel(lambda d: pipeline.photometric_augmentation(
                    d, **config['augmentation']['photometric']))
            if config['augmentation']['homographic']['enable']:
                assert not config['warped_pair']['enable']  # doesn't support hom. aug.
                data = data.map_parallel(lambda d: pipeline.homographic_augmentation(
                    d, **config['augmentation']['homographic']))

        # Generate the keypoint map
        if has_keypoints:
            data = data.map_parallel(pipeline.add_keypoint_map)
        data = data.map_parallel(
            lambda d: {**d, 'image': tf.cast(d['image'], tf.float32) / 255.})
        if config['warped_pair']['enable']:
            data = data.map_parallel(
                lambda d: {
                    **d, 'warped': {**d['warped'],
                                    'image': tf.cast(d['warped']['image'], tf.float32) / 255.}})

        return data
