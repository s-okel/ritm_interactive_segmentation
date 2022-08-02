#!/bin/bash
#
#SBATCH -o /home/014118_emtic_oncology/Pancreas/interactivity/slurm/logs/%j.%x.%N.out
#SBATCH -e /home/014118_emtic_oncology/Pancreas/interactivity/slurm/logs/%j.%x.%N.err
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=sanne.okel@philips.com

python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/train.py models/iter_mask/hrnet64_common_bile_duct_itermask_3p.py --exp-name=hrnet64_common_bile_duct --batch-size=32
python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/train.py models/iter_mask/hrnet64_common_bile_duct_itermask_3p.py --exp-name=hrnet64_common_bile_duct --batch-size=32
python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/train.py models/iter_mask/hrnet64_common_bile_duct_itermask_3p.py --exp-name=hrnet64_common_bile_duct --batch-size=32
python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/train.py models/iter_mask/hrnet64_common_bile_duct_itermask_3p.py --exp-name=hrnet64_common_bile_duct --batch-size=32
python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/train.py models/iter_mask/hrnet64_common_bile_duct_itermask_3p.py --exp-name=hrnet64_common_bile_duct --batch-size=32
python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/train.py models/iter_mask/hrnet64_common_bile_duct_itermask_3p.py --exp-name=hrnet64_common_bile_duct --batch-size=32
python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/train.py models/iter_mask/hrnet64_common_bile_duct_itermask_3p.py --exp-name=hrnet64_common_bile_duct --batch-size=32
