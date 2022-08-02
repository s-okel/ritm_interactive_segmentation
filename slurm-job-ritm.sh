#!/bin/bash
#
#SBATCH -o /home/014118_emtic_oncology/Pancreas/interactivity/slurm/logs/%j.%x.%N.out
#SBATCH -e /home/014118_emtic_oncology/Pancreas/interactivity/slurm/logs/%j.%x.%N.err
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=sanne.okel@philips.com

python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/train.py models/iter_mask/hrnet64_pancreas_itermask_3p.py --exp-name=hrnet64_pancreas --batch-size=32
python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/train.py models/iter_mask/hrnet64_aorta_itermask_3p.py --exp-name=hrnet64_aorta --batch-size=32
python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/train.py models/iter_mask/hrnet64_arteria_mesenterica_superior_itermask_3p.py --exp-name=hrnet64_arteria_mesenterica_superior --batch-size=32
python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/train.py models/iter_mask/hrnet64_gastroduodenalis_itermask_3p.py --exp-name=hrnet64_gastroduodenalis --batch-size=32
python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/train.py models/iter_mask/hrnet64_common_bile_duct_itermask_3p.py --exp-name=hrnet64_common_bile_duct --batch-size=32
python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/train.py models/iter_mask/hrnet64_pancreatic_duct_itermask_3p.py --exp-name=hrnet64_pancreatic_duct --batch-size=32
python /home/014118_emtic_oncology/Pancreas/interactivity/repos/ritm_interactive_segmentation/train.py models/iter_mask/hrnet64_tumour_itermask_3p.py --exp-name=hrnet64_tumour --batch-size=32
