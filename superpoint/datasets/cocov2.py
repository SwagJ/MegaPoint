import numpy as np
import tensorflow as tf
from pathlib import Path

# from base_dataset import BaseDataset
from .utils import pipeline
from superpoint.settings import DATA_PATH, EXPER_PATH
from superpoint.utils.tools import dict_update
# DATA_PATH = '/media/terabyte/projects/datasets/'
# EXPER_PATH = '/media/terabyte/projects/SUPERPOINT/outputs/'


class Coco(object):
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
    def __init__(self, config):
        self.config = dict_update(getattr(self, 'default_config', {}), config)
        self.files = self._getFileSystemFiles()
        self.data = self._get_data()
        
    def _getFileSystemFiles(self):
        """ gets all file paths of images and keypoints
            of the COCO dataset.
        """
        base_path = Path(DATA_PATH, 'COCO/train2014/')
        image_paths = list(base_path.iterdir())
        if self.config['truncate']:
            image_paths = image_paths[:self.config['truncate']]
        names = [p.stem for p in image_paths]
        image_paths = [str(p) for p in image_paths]
                
        files = {}
        if self.config['labels']:
            indicesToRemove = []
            label_paths = []
            for i,n in enumerate(names):
                p = Path(EXPER_PATH, self.config['labels'], '{}.npz'.format(n))
                
                assert p.exists(), 'Image {} has no corresponding label {}'.format(n, p)
                label_paths.append(str(p))
            files['label_paths'] = label_paths

        files['image_paths'] = image_paths
        files['names'] = names
        
        return files
    
    def _get_data(self, split_name='training'):
        has_keypoints = 'label_paths' in self.files.keys()
        is_training = split_name == 'training'
        
        def _read_image(path):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            return tf.cast(image, tf.float32)
        
        def _preprocess(image):
            image = tf.image.rgb_to_grayscale(image)
            target_size = self.config['preprocessing']['resize']
            image = tf.image.resize(image, target_size,
                                        method=tf.image.ResizeMethod.GAUSSIAN,
                                        preserve_aspect_ratio=True)
            image = tf.image.resize_with_crop_or_pad(image, target_size[0], target_size[1])
            return image
        
        def _read_points(filename):
            return np.load(filename.decode('utf-8'))['points'].astype(np.float32)
        
        image_names = tf.data.Dataset.from_tensor_slices(self.files['image_paths'])
        images = image_names.map(_read_image)
        if self.config['preprocessing']['resize']:
            preprocessed_images = images.map(_preprocess)
        else:
            preprocessed_images = images
        # stacked_images = preprocessed_images.map(lambda im: tf.stack((im, im), axis=0))
        
        # Add keypoints
        if has_keypoints:
            kp = tf.data.Dataset.from_tensor_slices(self.files['label_paths'])
            kp = kp.map(lambda path: tf.numpy_function(_read_points, [path], tf.float32))
            kp = kp.map(lambda points: tf.reshape(points, [-1, 2]))
            # data = tf.data.Dataset.zip((data, kp)).map(
            #         lambda d, k: {**d, 'keypoints': k})
            if self.config['preprocessing']['resize']:
                valid_masks = preprocessed_images.map(lambda im: {
                    'output_1': im, 'output_2': tf.ones(self.config['preprocessing']['resize'], dtype=tf.int32)})
            else:
                valid_masks = preprocessed_images.map(lambda im: tf.ones(tf.shape(im)[:2], dtype=tf.int32))
            imagesKeypoints = tf.data.Dataset.zip((preprocessed_images, kp))
        # if split_name == 'validation':
        #     data = data.take(config['validation_size'])

        # Generate the warped pair
        if self.config['warped_pair']['enable']:
            assert has_keypoints
            warped = imagesKeypoints.map(lambda img, keypoint: pipeline.homographic_augmentation(
                                img, keypoint, warped_pair_enable=True, 
                                add_homography=True, **self.config['warped_pair']))
            warpedImage = warped.map(lambda w: w['warped/image'])
            # data = tf.data.Dataset.zip((preprocessed_images, kp, valid_masks, warped))
        # else:
            data = tf.data.Dataset.zip((preprocessed_images, kp, valid_masks, warpedImage))
        # targets = tf.data.Dataset.from_tensor_slices(self.files['names'])
        return tf.data.Dataset.zip(
                (data.map(lambda d0,d1,d2,d3: ({'input_1': d0,
                                             'input_2': d1,
                                             'input_3': d3})),
                                            {'output_1': valid_masks}))