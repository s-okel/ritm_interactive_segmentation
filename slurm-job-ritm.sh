#!/bin/bash
#
#SBATCH -o /home/014118_emtic_oncology/Pancreas/interactivity/slurm/logs/%j.%x.%N.out
#SBATCH -e /home/014118_emtic_oncology/Pancreas/interactivity/slurm/logs/%j.%x.%N.err
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=sanne.okel@philips.com

python ./train.py ./models/iter_mask/hrnet64_aorta_itermask_3p.py --exp-name=hrnet64_aorta_radius_1 --batch-size=32
python ./train.py ./models/iter_mask/hrnet64_arteria_mesenterica_superior_itermask_3p.py --exp-name=hrnet64_arteria_mesenterica_superior_radius_1 --batch-size=32
python ./train.py ./models/iter_mask/hrnet64_common_bile_duct_itermask_3p.py --exp-name=hrnet64_common_bile_duct_radius_1 --batch-size=32
python ./train.py ./models/iter_mask/hrnet64_gastroduodenalis_itermask_3p.py --exp-name=hrnet64_gastroduodenalis_radius_1 --batch-size=32
python ./train.py ./models/iter_mask/hrnet64_pancreas_itermask_3p.py --exp-name=hrnet64_pancreas_radius_1 --batch-size=32
python ./train.py ./models/iter_mask/hrnet64_pancreatic_duct_itermask_3p.py --exp-name=hrnet64_pancreatic_duct_radius_1 --batch-size=32
python ./train.py ./models/iter_mask/hrnet64_tumour_itermask_3p.py --exp-name=hrnet64_tumour_radius_1 --batch-size=32
