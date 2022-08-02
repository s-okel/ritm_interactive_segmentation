#!/bin/bash
#
#SBATCH -o /home/014118_emtic_oncology/Pancreas/interactivity/slurm/logs/%j.%x.%N.out
#SBATCH -e /home/014118_emtic_oncology/Pancreas/interactivity/slurm/logs/%j.%x.%N.err
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=sanne.okel@philips.com

python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/scripts/evaluate_model.py NoBRS --checkpoint=./experiments/iter_mask/common_bile_duct_hrnet64_iter/001_hrnet64_common_bile_duct/checkpoints/epoch-40-val-loss-0.35.pth --datasets=Panc --structure=common_bile_duct --n-clicks=50 --save-ious --print-ious --model-name=hrnet-64-iter-m --logs-path=./experiments/iter_mask/common_bile_duct_hrnet64_iter/001_hrnet64_common_bile_duct/evaluation_logs --iou-analysis --vis-preds
python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/scripts/evaluate_model.py NoBRS --checkpoint=./experiments/iter_mask/gastroduodenalis_hrnet64_iter/000_hrnet64_gastroduodenalis/checkpoints/epoch-104-val-loss-0.30.pth --datasets=Panc --structure=gastroduodenalis --n-clicks=50 --save-ious --print-ious --model-name=hrnet-64-iter-m --logs-path=./experiments/iter_mask/gastroduodenalis_hrnet64_iter/000_hrnet64_gastroduodenalis/evaluation_logs --iou-analysis --vis-preds
python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/scripts/evaluate_model.py NoBRS --checkpoint=./experiments/iter_mask/pancreas_hrnet64_iter/000_hrnet64_pancreas/checkpoints/epoch-10-val-loss-0.41.pth --datasets=Panc --structure=pancreas --n-clicks=50 --save-ious --print-ious --model-name=hrnet-64-iter-m --logs-path=./experiments/iter_mask/pancreas_hrnet64_iter/000_hrnet64_pancreas/evaluation_logs --iou-analysis --vis-preds
