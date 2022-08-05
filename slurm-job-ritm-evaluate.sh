#!/bin/bash
#
#SBATCH -o /home/014118_emtic_oncology/Pancreas/interactivity/slurm/logs/%j.%x.%N.out
#SBATCH -e /home/014118_emtic_oncology/Pancreas/interactivity/slurm/logs/%j.%x.%N.err
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=sanne.okel@philips.com

aorta, sma, pd

python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/scripts/evaluate_model.py NoBRS --checkpoint=./experiments/iter_mask/gastroduodenalis_hrnet64_iter/001_hrnet64_gastroduodenalis/checkpoints/epoch-39-val-loss-0.34.pth --datasets=Panc --structure=gastroduodenalis --n-clicks=50 --save-ious --print-ious --model-name=hrnet-64-iter-m --logs-path=./experiments/iter_mask/gastroduodenalis_hrnet64_iter/001_hrnet64_gastroduodenalis/evaluation_logs --iou-analysis --vis-preds
python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/scripts/evaluate_model.py NoBRS --checkpoint=./experiments/iter_mask/gastroduodenalis_hrnet64_iter/001_hrnet64_gastroduodenalis/checkpoints/epoch-109-val-loss-0.32.pth --datasets=Panc --structure=gastroduodenalis --n-clicks=50 --save-ious --print-ious --model-name=hrnet-64-iter-m --logs-path=./experiments/iter_mask/gastroduodenalis_hrnet64_iter/001_hrnet64_gastroduodenalis/evaluation_logs --iou-analysis --vis-preds
python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/scripts/evaluate_model.py NoBRS --checkpoint=./experiments/iter_mask/pancreas_hrnet64_iter/002_hrnet64_pancreas/checkpoints/epoch-139-val-loss-0.52.pth --datasets=Panc --structure=pancreas --n-clicks=50 --save-ious --print-ious --model-name=hrnet-64-iter-m --logs-path=./experiments/iter_mask/pancreas_hrnet64_iter/002_hrnet64_pancreas/evaluation_logs --iou-analysis --vis-preds
python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/scripts/evaluate_model.py NoBRS --checkpoint=./experiments/iter_mask/pancreas_hrnet64_iter/002_hrnet64_pancreas/checkpoints/epoch-159-val-loss-0.59.pth --datasets=Panc --structure=pancreas --n-clicks=50 --save-ious --print-ious --model-name=hrnet-64-iter-m --logs-path=./experiments/iter_mask/pancreas_hrnet64_iter/002_hrnet64_pancreas/evaluation_logs --iou-analysis --vis-preds
python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/scripts/evaluate_model.py NoBRS --checkpoint=./experiments/iter_mask/pancreatic_duct_hrnet64_iter/000_hrnet64_pancreatic_duct/checkpoints/epoch-29-val-loss-0.30.pth --datasets=Panc --structure=pancreatic_duct --n-clicks=50 --save-ious --print-ious --model-name=hrnet-64-iter-m --logs-path=./experiments/iter_mask/pancreatic_duct_hrnet64_iter/000_hrnet64_pancreatic_duct/evaluation_logs --iou-analysis --vis-preds
python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/scripts/evaluate_model.py NoBRS --checkpoint=./experiments/iter_mask/tumour_hrnet64_iter/000_hrnet64_tumour/checkpoints/epoch-29-val-loss-0.32.pth --datasets=Panc --structure=tumour --n-clicks=50 --save-ious --print-ious --model-name=hrnet-64-iter-m --logs-path=./experiments/iter_mask/tumour_hrnet64_iter/000_hrnet64_tumour/evaluation_logs --iou-analysis --vis-preds


