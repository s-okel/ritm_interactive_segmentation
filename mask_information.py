import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import pickle
import torch
import torchvision
from tqdm import tqdm

if __name__ == "__main__":
    data_path = r'/home/014118_emtic_oncology/Pancreas/fullPixelAnnotRedo2/'

    label_dict = {}
    tbar = tqdm(os.listdir(data_path))

    for patient in tbar:
        tbar.set_description(f"patient {patient}")
        patient_path = data_path + patient
        for phase in os.listdir(patient_path):
            mask_path = patient_path + "/" + phase + "/labelsTr"
            for mask in os.listdir(mask_path):
                # get the label of the mask
                start = mask.find("_", 7)
                label = mask[start+1:-7]

                mask_array = np.array(nib.load(mask_path + "/" + mask).get_fdata())
                mask_size_per_slice = np.sum(mask_array, axis=(1, 2))
                total_mask_size = np.sum(mask_size_per_slice)

                if label not in label_dict:
                    label_dict[label] = {}

                if 'occurrence' not in label_dict[label]:
                    label_dict[label]['occurrence'] = 1
                else:
                    label_dict[label]['occurrence'] += 1

                if 'non_zero_slices' not in label_dict[label]:
                    label_dict[label]['non_zero_slices'] = np.sum(mask_size_per_slice > 0)
                else:
                    label_dict[label]['non_zero_slices'] += np.sum(mask_size_per_slice > 0)

                if 'masks_sum' not in label_dict[label]:
                    label_dict[label]['masks_sum'] = total_mask_size
                else:
                    label_dict[label]['masks_sum'] += total_mask_size

                if 'min_sum' not in label_dict[label]:
                    label_dict[label]['min_sum'] = np.min(mask_size_per_slice[np.nonzero(mask_size_per_slice)])
                else:
                    if total_mask_size < label_dict[label]['min_sum']:
                        label_dict[label]['min_sum'] = np.min(mask_size_per_slice[np.nonzero(mask_size_per_slice)])
                        
                if 'mean_sum' not in label_dict[label]:
                    label_dict[label]['mean_sum'] = np.mean(mask_size_per_slice[np.nonzero(mask_size_per_slice)])
                else:
                    if total_mask_size < label_dict[label]['mean_sum']:
                        label_dict[label]['mean_sum'] = np.mean(mask_size_per_slice[np.nonzero(mask_size_per_slice)])
                
                if 'std_sum' not in label_dict[label]:
                    label_dict[label]['std_sum'] = np.std(mask_size_per_slice[np.nonzero(mask_size_per_slice)])
                else:
                    if total_mask_size < label_dict[label]['std_sum']:
                        label_dict[label]['std_sum'] = np.std(mask_size_per_slice[np.nonzero(mask_size_per_slice)])

                if 'max_sum' not in label_dict[label]:
                    label_dict[label]['max_sum'] = total_mask_size
                else:
                    if total_mask_size > label_dict[label]['max_sum']:
                        label_dict[label]['max_sum'] = np.max(mask_size_per_slice[np.nonzero(mask_size_per_slice)])
                    label_dict[label]['max_sum'] = np.max(mask_size_per_slice[np.nonzero(mask_size_per_slice)])

                """ 
                if 'mask_size_per_slice' not in label_dict[label]:
                    label_dict[label]['mask_size_per_slice'] = list(mask_size_per_slice)
                else:
                    label_dict[label]['mask_size_per_slice'].append(mask_size_per_slice)
                """

        for label in label_dict:
            label_dict[label]['avg_mask_size'] = int(
                label_dict[label]['masks_sum'] / label_dict[label]['non_zero_slices'])
            label_dict[label]['masks_sum'] = int(label_dict[label]['masks_sum'])
            print(f"label: {label}")
            for thing in label_dict[label]:
                print(f"{thing}: {label_dict[label][thing]}")
            
            
        print(label_dict)
