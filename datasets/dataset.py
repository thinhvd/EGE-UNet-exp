from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image


class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets, self)
        split = 'train' if train else 'val'
        images_dir = os.path.join(path_Data, split, 'images')
        masks_dir = os.path.join(path_Data, split, 'masks')
        images_list = sorted(os.listdir(images_dir))
        masks_list = sorted(os.listdir(masks_dir))
        self.data = []
        for i in range(len(images_list)):
            img_path = os.path.join(images_dir, images_list[i])
            mask_path = os.path.join(masks_dir, masks_list[i])
            self.data.append([img_path, mask_path])
        self.transformer = config.train_transformer if train else config.test_transformer
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)
        
    
