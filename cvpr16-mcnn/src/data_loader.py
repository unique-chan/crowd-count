import numpy as np
import cv2
import os
import random
# import glob
import time
import pandas as pd
from tqdm import tqdm


class ImageDataLoader:
    def __init__(self, data_path, gt_path, shuffle=False, gt_downsample=False, pre_load=False):
        self.data_path = data_path
        self.gt_path = gt_path
        self.gt_downsample = gt_downsample
        # self.data_files = sorted(glob.glob(os.path.join(self.data_path, '*.jpg')))
        self.data_files = [fname for fname in os.listdir(self.data_path)
                           if os.path.isfile(os.path.join(self.data_path, fname))]
        self.shuffle = shuffle
        self.num_samples = len(self.data_files)
        self.blob_dic = {}
        self.id_list = list(range(self.num_samples))
        self.pre_load = pre_load
        if self.pre_load:
            for idx in tqdm(self.id_list, desc='Pre-load the dataset.', mininterval=0.01):
                blob = self.read_image_and_density_map(idx)
                self.blob_dic[idx] = blob
            time.sleep(0.01)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.id_list if self.pre_load else self.data_files)
        for idx in self.id_list:
            if self.pre_load:
                blob = self.blob_dic[idx]
                blob['idx'] = idx
            else:
                blob = self.read_image_and_density_map(idx)
            yield blob

    def read_image_and_density_map(self, idx):
        fname = self.data_files[idx]
        img = cv2.imread(os.path.join(self.data_path, fname), flags=cv2.IMREAD_GRAYSCALE).astype(np.float32)
        h, w = img.shape
        img = img.reshape((1, 1, h, w))
        density = pd.read_csv(os.path.join(self.gt_path, f'{os.path.splitext(fname)[0]}.csv'),
                              sep=',', header=None).values.astype(np.float32)
        if self.gt_downsample:
            h_, w_ = h // 4, w // 4
            density = cv2.resize(density, (w_, h_))
            density = density * ((h * w) / (h_ * w_))
        else:
            density = cv2.resize(density, (w, h))
        density = density.reshape((1, 1, density.shape[0], density.shape[1]))
        blob = {'data': img, 'gt_density': density, 'fname': fname}
        return blob

    def __len__(self):
        return self.num_samples
