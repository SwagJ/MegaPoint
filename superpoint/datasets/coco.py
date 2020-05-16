import numpy as np
import tensorflow as tf
from pathlib import Path

from .base_dataset import BaseDataset
from .utils import pipeline
from superpoint.settings import DATA_PATH, EXPER_PATH


class Coco(BaseDataset):
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
        base_path = Path(DATA_PATH, 'COCO/train2014/')
        image_paths = list(base_path.iterdir())
        if config['truncate']:
            image_paths = image_paths[:config['truncate']]
        names = [p.stem for p in image_paths]
        image_paths = [str(p) for p in image_paths]
        
        files = {}
        if config['labels']:
            indicesToRemove = []
            label_paths = []
            for i,n in enumerate(names):
                p = Path(EXPER_PATH, config['labels'], '{}.npz'.format(n))
                
                assert p.exists(), 'Image {} has no corresponding label {}'.format(n, p)
                label_paths.append(str(p))
            files['label_paths'] = label_paths

        files['image_paths'] = image_paths
        files['names'] = names

        tf.data.Dataset.map_parallel = lambda self, fn: self.map(
                fn, num_parallel_calls=config['num_parallel_calls'])

        return files

    def _get_data(self, files, split_name, **config):
        has_keypoints = 'label_paths' in files
        is_training = split_name == 'training'
        
        def _read_image(path):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            return tf.cast(image, tf.float32)

        def _preprocess(image):
            image = tf.image.rgb_to_grayscale(image)
            target_size = self.config['preprocessing']['resize']
            if self.config['preprocessing']['resize']:
                image = tf.image.resize(image, target_size,
                                        method=tf.image.ResizeMethod.GAUSSIAN,
                                        preserve_aspect_ratio=True)
                image = tf.image.resize_with_crop_or_pad(image, target_size[0], target_size[1])
            return image
        
        # Python function
        def _read_points(filename):
            return np.load(filename.decode('utf-8'))['points'].astype(np.float32)

        names = tf.data.Dataset.from_tensor_slices(files['names'])
        images = tf.data.Dataset.from_tensor_slices(files['image_paths'])
        images = images.map(_read_image)
        images = images.map(lambda im: tf.image.resize(im, self.config['preprocessing']['resize'],
                                method=tf.image.ResizeMethod.GAUSSIAN))
        
        data = tf.data.Dataset.zip({'image': images, 'name': names})
        
        # Add keypoints
        if has_keypoints:
            kp = tf.data.Dataset.from_tensor_slices(files['label_paths'])
            kp = kp.map(lambda path: tf.numpy_function(_read_points, [path], tf.float32))
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
            warped = data.map(lambda d: pipeline.homographic_augmentation(
                d['image'], d['keypoints'], warped_pair_enable=True, add_homography=True, **config['warped_pair']))
          
            if is_training and config['augmentation']['photometric']['enable']:
                warped = warped.map(lambda w: {**w, 
                            'warped/image': pipeline.photometric_augmentation(
                                w['warped/image'], **config['augmentation']['photometric'])})
            
            warped = warped.map(lambda w: {**w, 
                                            'warped/keypoint_map': pipeline.add_keypoint_map(w['warped/image'],
                                                                                             w['warped/keypoints'])})
            
            # Merge with the original data
            # for key in warped.keys():
                # data['warped/'+key] = warped[key]
            data = tf.data.Dataset.zip((data, warped))
            data = data.map(lambda d, w: {**d, **w})

        # Data augmentation
        if has_keypoints and is_training:
            if config['augmentation']['photometric']['enable']:
                data = data.map(lambda d: {**d, 'image' : pipeline.photometric_augmentation(
                    d['image'], **config['augmentation']['photometric'])})
            if config['augmentation']['homographic']['enable']:
                assert not config['warped_pair']['enable']  # doesn't support hom. aug.
                data = data.map(lambda d: pipeline.homographic_augmentation(
                    d, warped_pair_enable=False, **config['augmentation']['homographic']))
        
        # Generate the keypoint map
        if has_keypoints:
            data = data.map(lambda d: {**d, 
                                        'keypoint_map' : pipeline.add_keypoint_map(d['image'],
                                                                                   d['keypoints'])})
        data = data.map(
            lambda d: {**d, 'image': tf.cast(d['image'], tf.float32) / 255.})
        
        if config['warped_pair']['enable']:
            data = data.map_parallel(
                lambda d: {
                    **d, 'warped/image': tf.cast(d['warped/image'], tf.float32) / 255.})
        
        # inputs = data.map_parallel(lambda d: {'image' : d['image'], 
        #                                       'warped/image' : d['warped/image']})
        # inputs = data.map_parallel(lambda d: [d['image'], 
        #                                       d['warped/image']])
        # targets = data.map_parallel(lambda d : {
        #     'keypoint_map' : d['keypoint_map'],
        #     'valid_mask' : d['valid_mask'],
        #     'warped/keypoint_map' : d['warped/keypoint_map'],
        #     'warped/valid_mask' : d['warped/valid_mask'],
        #     'homography' : d['warped/homography']
        # })
        # targets = data.map_parallel(lambda d : [
        #     d['keypoint_map'],
        #     d['valid_mask'],
        #     d['warped/keypoint_map'],
        #     d['warped/valid_mask'],
        #     d['warped/homography'] ])
        
        # tf.data.Dataset.zip causes segmentation fault
        # if not config['warped_pair']['enable']:
        #     return data.map(lambda d :
        #         ([d['image']],
        #          [d['keypoint_map'], d['valid_mask']]))
        # else:
        #     return data.map(lambda d :
        #         ({'image' : d['image'], 'warped/image' : d['warped/image']},
        #          {'keypoint_map' : d['keypoint_map'],
        #           'valid_mask' : d['valid_mask'],
        #           'warped/keypoint_map' : d['warped/keypoint_map'],
        #           'warped/valid_mask' : d['warped/valid_mask'],
        #           'homography' : d['warped/homography']}))
        # This is a really bad hack, it puts an insane number of unessary zeros to homography
        sizes_to_pad = [[0, self.config['preprocessing']['resize'][0]-1],
                        [0, self.config['preprocessing']['resize'][1]-8],
                        [0,2]]
        # d1 = data.map(lambda d:
        #         tf.stack([d['image'], d['warped/image'],
        #                    tf.pad(tf.expand_dims(
        #                        tf.expand_dims(d['warped/homography'],axis=0), axis=[-1]),
        #                        sizes_to_pad)],
        #                   axis=0)
        # )    
        # d2 = data.map(lambda d: tf.stack([d['keypoint_map'], d['valid_mask'],
        #                    d['warped/keypoint_map'], d['warped/valid_mask']
        #                    ], axis=0)
        #     )
        # return tf.data.Dataset.zip((d1,d2))
        dataIn = data.map(lambda d: ({
                'input_1': d['image'],
                'input_2': d['warped/image']}))
        dataOut = data.map(lambda d: ({
                'output_1': d['keypoint_map'],
                'output_2': d['valid_mask'],
                'output_3': d['warped/keypoint_map'],
                'output_4': d['warped/valid_mask'],
                'output_5': d['warped/homography']}))
        
        data = tf.data.Dataset.zip((dataIn, dataOut))
        # data = data.map(lambda d:
        #         (tf.stack([d['image'], d['warped/image'],
        #                    tf.pad(tf.expand_dims(
        #                        tf.expand_dims(d['warped/homography'],axis=0), axis=[-1]),
        #                        sizes_to_pad)],
        #                   axis=0),
        #          tf.stack([d['keypoint_map'], d['valid_mask'],
        #                    d['warped/keypoint_map'], d['warped/valid_mask']
        #                    ], axis=0))
        # )
        return data       
        
        # return data
        #return data.map(lambda d :
        #         {'input':[d['image'], d['warped/image']],

        #          'output':[d['keypoint_map'], d['valid_mask'],
        #           d['warped/keypoint_map'], d['warped/valid_mask'], d['warped/homography']]} ))


class CocoSequence(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
    def __len__(self):
        return self.batch_size
    def __getitem__(self, idx):
        assert idx == 0
        for d in self.data:
            return d
    def __iter__(self):
        for d in self.data:
            yield d