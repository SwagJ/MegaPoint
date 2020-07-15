#! /bin/bash

# On cluster environment
#export TMPDIR=/tmp/
#export USERNAME=majing
#export DATA_PATH=/cluster/scratch/${USERNAME}/superpoint
#export EXPER_PATH=/cluster/scratch/${USERNAME}/SuperPoint/outputs

#On local setup
export TMPDIR=/disk_ssd/tmp/
export EXPER_PATH=/disk_ssd/SuperPoint
export DATA_PATH=/disk_hdd/superpoint
export CUDA_VISIBLE_DEVICES=0


# load modules
# module load eth_proxy gcc/6.3.0 python_gpu/3.7.4 cuda/10.1.243 openblas/0.2.19
#install requirement
#pip install -r requirements.txt --upgrade
pip install -e .

#initial setup
echo "DATA_PATH = '$DATA_PATH'" >> ./superpoint/settings.py
echo "EXPER_PATH = '$EXPER_PATH'" >> ./superpoint/settings.py

cd superpoint

#list of commands: uncomment the one you need. For details about command, please refer to 
#our project github webpage: https://github.com/SwagJ/SuperPoint


#python experiment.py train configs/great-point_shapes.yaml magic-point_synth
#python export_detections.py configs/magic-point_megadepth_export.yaml superpoint_megadepth \
#	   --pred_only --batch_size=1 --export_name=superpoint_megadepth-export
#python experiment.py train configs/great-point_coco_train.yaml magic-point_coco
#python export_detections_repeatability.py configs/magic-point_repeatability.yaml \
#	   magic-point_coco --export_name=magic-point_hpatches-repeatability-v
#python experiment.py train configs/megapoint_megadepth.yaml megapoint_megadepth
python export_descriptors.py configs/superpoint_hpatches.yaml \
	   superpoint_megadepth --export_name=superpoint_megadepth_hpatches_descriptor-i
#python export_descriptors.py configs/superpoint_hpatches.yaml \
#	   superpoint_coco --export_name=superpoint_test_export

#python export_detections_repeatability.py configs/megapoint_repeatability.yaml\
#		megapoint_megadepth --export_name=megapoint_megadepth-repeatability-i
#python experiment.py train configs/superpoint_coco.yaml superpoint_coco
