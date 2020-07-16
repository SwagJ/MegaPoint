<<<<<<< HEAD
# MegaPoint
=======
# SuperPoint
>>>>>>> 983456ced29b798a35a11984225985db00dad67b


## Installation

Python 3.6.1 is required. You will be asked to provide a path to an experiment directory (containing the training and prediction outputs, referred as `$EXPER_DIR`) and a dataset directory (referred as `$DATA_DIR`). Create them wherever you wish and make sure to provide their absolute paths. Path Setup should be modified in `train.sh`.
To Submit Job on Leonhard Cluster, use the following command. For Details about command, please refer to [Leonhard Cluster Tutorial](https://scicomp.ethz.ch/wiki/Tutorials)
```
bsub -W 24:00 -n 8 -R "rusage[mem=4500,scratch=10000,ngpus_excl_p=1]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" ./train.sh bash
```

[MS-COCO 2014](http://cocodataset.org/#download) and [HPatches](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz) should be downloaded into `$DATA_DIR`. The Synthetic Shapes dataset will also be generated there. The folder structure should look like after semantic and depth generated:
```
$DATA_DIR
|-- COCO
|   |-- semantic
|   |-- depth
|   |-- train2014
|   |   |-- file1.jpg
|   |   `-- ...
|   `-- val2014
|       |-- file1.jpg
|       `-- ...
`-- HPatches
|   |-- i_ajuntament
|   `-- ...
`-- synthetic_shapes  # will be automatically created
`-- MegaDepth_V1
|   |-- 0000
|   |   |-- denseX |--imgs
|   |   |          |--semantic
|   |   |          |--depth
|   |   |
|       `-- ...
```

## Usage
All commands should be executed within the `superpoint/` subfolder. When training a model or exporting its predictions, you will often have to change the relevant configuration file in `superpoint/configs/`. Both multi-GPU training and export are supported. Note that MagicPoint and SuperPoint only work on images with dimensions divisible by 8 and the user is responsible for resizing them to a valid dimension

## To prepare depth and semantic for COCO and MegaDepth
For semantics, run
```
cd MegaDepth_tf2_0
python inference_mega_dataset.py --data_path DATA_PATH --dataset [coco/megadepth]
```

For depth, run 
```
cd PSPNet_tf2_0
python inference_eager_dataset.py --data_path DATA_PATH --dataset [coco/megadepth]
```

### 1) Training MagicPoint on Synthetic Shapes
```
python experiment.py train configs/magic-point_shapes.yaml magic-point_synth
```
where `magic-point_synth` is the experiment name, which may be changed to anything. The training can be interrupted at any time using `Ctrl+C` and the weights will be saved in `$EXPER_DIR/magic-point_synth/`. The Tensorboard summaries are also dumped there. When training for the first time, the Synthetic Shapes dataset will be generated.

### 2) Exporting detections on MS-COCO

```
python export_detections.py configs/magic-point_coco_export.yaml magic-point_synth --pred_only --batch_size=5 --export_name=magic-point_coco-export1
```
This will save the pseudo-ground truth interest point labels to `$EXPER_DIR/outputs/magic-point_coco-export1/`. You might enable or disable the Homographic Adaptation in the configuration file.

### 3) Training GreatPoint on MS-COCO
```
python experiment.py train configs/great-point_coco_train.yaml great-point_coco
```
You will need to indicate the paths to the interest point labels in `magic-point_coco_train.yaml` by setting the entry `data/labels`, for example to `outputs/magic-point_coco-export1`. You might repeat steps 2) and 3) several times.

### 4) Evaluating the repeatability on HPatches
```
python export_detections_repeatability.py configs/mega-point_repeatability.yaml mega-point_coco --export_name=mega-point_hpatches-repeatability-v
```
You will need to decide whether you want to evaluate for viewpoint or illumination by setting the entry `data/alteration` in the configuration file. The predictions of the image pairs will be saved in `$EXPER_DIR/outputs/mega-point_hpatches-repeatability-v/`. To proceed to the evaluation, head over to `notebooks/detector_repeatability_hpatches.ipynb`. You can also evaluate the repeatability of the classical detectors using the configuration file `classical-detectors_repeatability.yaml`.


### 6) Training of MegaPoint on MS-COCO
Once you have trained Great with several rounds of homographic adaptation (one or two should be enough), you can export again the detections on MS-COCO as in step 2) and use these detections to train SuperPoint by setting the entry `data/labels`:
```
python experiment.py train configs/megapoint_coco.yaml megapoint_coco
```

### 7) Evaluation of the descriptors with homography estimation on HPatches
```
python export_descriptors.py configs/megapoint_hpatches.yaml megapoint_coco --export_name=megapoint_hpatches-v
```
You will need to decide again whether you want to evaluate for viewpoint or illumination by setting the entry `data/alteration` in the configuration file. The predictions of the image pairs will be saved in `$EXPER_PATH/outputs/superpoint_hpatches-v/`. To proceed to the evaluation, head over to `notebooks/descriptors_evaluation_on_hpatches.ipynb`. You can also evaluate the repeatability of the classical detectors using the configuration file `classical-descriptors.yaml`.


## Credits
This implementation was based on SuperPoint implemented by [Rémi Pautrat](https://github.com/rpautrat) and [Paul-Edouard Sarlin](https://github.com/Skydes).
