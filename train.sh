#! /bin/bash

# On cluster environment
export TMPDIR=/tmp/
export USERNAME=$USER
export DATA_PATH=/cluster/scratch/${USERNAME}/
export EXPER_PATH=/cluster/scratch/${USERNAME}/outputs

#On local setup
#export EXPER_PATH=$HOME/Desktop/SuperPoint/outputs
#export DATA_PATH=$HOME/Desktop/superpoint
#export CUDA_VISIBLE_DEVICES=0


# load modules
module load eth_proxy gcc/6.3.0 python_gpu/3.7.4 cuda/10.1.243 openblas/0.2.19
#install requirement
pip install -r requirements.txt
pip install -e .

#initial setup
echo "DATA_PATH = '$DATA_PATH'" > ./superpoint/settings.py
echo "EXPER_PATH = '$EXPER_PATH'" >> ./superpoint/settings.py

cd superpoint

#list of commands: uncomment the one you need. For details about command, please refer to 
#our project github webpage: https://github.com/SwagJ/SuperPoint


#python experiment.py train configs/magic-point_shapes.yaml magic-point_synth
python export_detections.py configs/magic-point_coco_export.yaml magic-point_synth \
	   --pred_only --batch_size=5 --export_name=magic-point_coco-export1
#python experiment.py train configs/magic-point_coco_train.yaml magic-point_coco
#python export_detections_repeatability.py configs/magic-point_repeatability.yaml \
#	   magic-point_coco --export_name=magic-point_hpatches-repeatability-v
#python experiment.py train configs/superpoint_coco.yaml superpoint_coco
#python export_descriptors.py configs/superpoint_hpatches.yaml \
#	   superpoint_coco --export_name=superpoint_hpatches-v
#python export_descriptors.py configs/superpoint_hpatches.yaml \
#	   superpoint_coco --export_name=superpoint_hpatches-v

