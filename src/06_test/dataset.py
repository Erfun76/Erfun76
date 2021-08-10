import numpy as np
import cv2
from torch.utils.data import Dataset
from os.path import join as opj
import rasterio
from rasterio.windows import Window

from transforms import get_transforms_test, get_transforms_mask
from get_config import get_config
import sys

sys.path.insert(0, '../')
from utils import rle2mask

config = get_config()


class HuBMAPDataset(Dataset):
    def __init__(self, idx, df, info_df, mode='train'):
        super().__init__()
        filename = info_df.loc[idx, 'image_file']

        path = opj(config['INPUT_PATH'], mode, filename)
        print("path: {}".format(path))
        self.data = rasterio.open(path)
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(rasterio.open(subdataset))
        self.h, self.w = self.data.height, self.data.width
        self.input_sz = config['input_resolution']
        self.sz = config['resolution']
        self.pad_sz = config['pad_size']  # add to each input tile
        self.pred_sz = self.sz - 2 * self.pad_sz
        self.pad_h = self.pred_sz - self.h % self.pred_sz  # add to whole slide
        self.pad_w = self.pred_sz - self.w % self.pred_sz  # add to whole slide
        self.num_h = (self.h + self.pad_h) // self.pred_sz
        self.num_w = (self.w + self.pad_w) // self.pred_sz

        self.rle = df.loc[df['id'] == filename[:-5], 'encoding'].values[0]
        self.mask = rle2mask(self.rle, shape=(self.h, self.w))
        self.transforms = get_transforms_test()
        self.transforms_mask = get_transforms_mask()

    def __len__(self):
        return self.num_h * self.num_w

    def __getitem__(self, idx):  # idx = i_h * self.num_w + i_w
        # prepare coordinates for rasterio
        i_h = idx // self.num_w
        i_w = idx % self.num_w
        y = i_h * self.pred_sz
        x = i_w * self.pred_sz
        ##TODO: changed second argument for batch size bigger than 1
        py0, py1 = max(0, y), y + self.pred_sz #min(y + self.pred_sz, self.h)
        px0, px1 = max(0, x), x + self.pred_sz #min(x + self.pred_sz, self.w)

        # padding coordinate for rasterio
        qy0, qy1 = max(0, y - self.pad_sz), min(y + self.pred_sz + self.pad_sz, self.h)
        qx0, qx1 = max(0, x - self.pad_sz), min(x + self.pred_sz + self.pad_sz, self.w)

        # placeholder for input tile (before resize)
        img = np.zeros((self.sz, self.sz, 3), np.uint8)
        mask = np.zeros((self.sz, self.sz), np.uint8)
        # replace the value
        if self.data.count == 3:
            img[0:qy1 - qy0, 0:qx1 - qx0] = \
                np.moveaxis(self.data.read([1, 2, 3], window=Window.from_slices((qy0, qy1), (qx0, qx1))), 0, -1)
        else:
            for i, layer in enumerate(self.layers):
                img[0:qy1 - qy0, 0:qx1 - qx0, i] = \
                    layer.read(1, window=Window.from_slices((qy0, qy1), (qx0, qx1)))
        mask[0:qy1 - qy0, 0:qx1 - qx0] = self.mask[qy0:qy1, qx0:qx1]
        if self.sz != self.input_sz:
            img  = cv2.resize(img,  (self.input_sz, self.input_sz), interpolation=cv2.INTER_AREA)
            # mask = cv2.resize(mask, (self.input_sz, self.input_sz), interpolation=cv2.INTER_AREA)

        augmented = self.transforms(image=img, mask=mask) # to normalized tensor
        img  = augmented['image']
        mask = augmented['mask']

        return {'img': img, 'mask': mask, 'p': [py0, py1, px0, px1], 'q': [qy0, qy1, qx0, qx1]}