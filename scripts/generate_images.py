from torchvision.utils import save_image
import torch

label = 'tumour'
data = torch.load(f'../datasets/Panc/{label}_test_slices.pt')
print(data.shape)

images = data[0]  # 3016 x 128 x 128
masks = data[1]

idx = torch.randint(0, data.shape[1], (10,))
print(idx)

for i in idx:
    save_image(images[i] / 255, f'{i}_{label}_image.png')
    save_image(masks[i], f'{i}_{label}_mask.png')
