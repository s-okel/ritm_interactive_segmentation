#!/bin/bash
#
#SBATCH -o /home/014118_emtic_oncology/Pancreas/interactivity/slurm/logs/%j.%x.%N.out
#SBATCH -e /home/014118_emtic_oncology/Pancreas/interactivity/slurm/logs/%j.%x.%N.err
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=sanne.okel@philips.com

python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/scripts/evaluate_model.py NoBRS --exp-path=iter_mask/common_bile_duct_hrnet64_iter/003_hrnet64_common_bile_duct_radius_2/checkpoints/epoch-29 --datasets=Panc --structure=common_bile_duct --n-clicks=50 --save-ious --print-ious --model-name=hrnet-64-iter-m-2 --logs-path=./experiments/iter_mask/common_bile_duct_hrnet64_iter/003_hrnet64_common_bile_duct_radius_2/evaluation_logs --iou-analysis --vis-preds
python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/scripts/evaluate_model.py NoBRS --exp-path=iter_mask/common_bile_duct_hrnet64_iter/003_hrnet64_common_bile_duct_radius_2/checkpoints/epoch-69 --datasets=Panc --structure=common_bile_duct --n-clicks=50 --save-ious --print-ious --model-name=hrnet-64-iter-m-2 --logs-path=./experiments/iter_mask/common_bile_duct_hrnet64_iter/003_hrnet64_common_bile_duct_radius_2/evaluation_logs --iou-analysis --vis-preds
python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/scripts/evaluate_model.py NoBRS --exp-path=iter_mask/common_bile_duct_hrnet64_iter/003_hrnet64_common_bile_duct_radius_2/checkpoints/epoch-109 --datasets=Panc --structure=common_bile_duct --n-clicks=50 --save-ious --print-ious --model-name=hrnet-64-iter-m-2 --logs-path=./experiments/iter_mask/common_bile_duct_hrnet64_iter/003_hrnet64_common_bile_duct_radius_2/evaluation_logs --iou-analysis --vis-preds
python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/scripts/evaluate_model.py NoBRS --exp-path=iter_mask/common_bile_duct_hrnet64_iter/003_hrnet64_common_bile_duct_radius_2/checkpoints/epoch-189 --datasets=Panc --structure=common_bile_duct --n-clicks=50 --save-ious --print-ious --model-name=hrnet-64-iter-m-2 --logs-path=./experiments/iter_mask/common_bile_duct_hrnet64_iter/003_hrnet64_common_bile_duct_radius_2/evaluation_logs --iou-analysis --vis-preds

