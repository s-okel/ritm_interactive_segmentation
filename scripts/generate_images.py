import os.path

from torchvision.utils import save_image
import torch

labels = ['aorta', 'arteria_mesenterica_superior', 'common_bile_duct', 'gastroduodenalis', 'pancreas',
          'pancreatic_duct', 'tumour']

for label in labels:
    print(label)
    data = torch.load(f'../datasets/Panc/{label}_test_slices.pt')
    print(data.shape)

    images = data[0]  # 3016 x 128 x 128
    masks = data[1]

    idx = torch.randint(0, data.shape[1], (20,))
    print(idx)

    save_dir = f'images_with_masks/{label}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in idx:
        save_image(images[i] / 255, os.path.join(save_dir, f'{i}_image.png'))
        save_image(masks[i], os.path.join(save_dir, f'{i}_mask.png'))
