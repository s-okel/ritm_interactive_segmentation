import cv2
from isegm.data.base import ISDataset
from isegm.data.sample import DSample
import numpy as np
import torch
from torch.utils.data import DataLoader


class PancDataset(ISDataset):
    def __init__(self, split, one_input_channel=False, **kwargs):
        super(PancDataset, self).__init__(**kwargs)
        assert split in ['train', 'val']
        self.name = "Panc"
        self.one_input_channel = one_input_channel

        self.data = torch.load(f"./datasets/Panc/{split}_slices.pt")
        self.dataset_samples = range(len(self.data[0]))  # 4790

    def get_sample(self, index) -> DSample:
        img = np.array(self.data[0, index]).astype("uint8")
        if not self.one_input_channel:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        mask = np.array(self.data[1, index]).astype("int32")

        return DSample(img, mask, objects_ids=[1], sample_id=index)


if __name__ == "__main__":
    dataset = PancDataset('val', one_input_channel=False)
    dataloader = DataLoader(dataset)
    x = next(iter(dataloader))
    print(x)
