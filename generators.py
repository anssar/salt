import keras
import numpy as np
from transforms import *


class TrainGenerator(keras.utils.Sequence):
    def __init__(self, images, masks, batch_size=32, img_size_target=128, img_channels=3):
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.img_size_target = img_size_target
        self.img_channels = img_channels

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        idx = 0
        batch_x = []
        batch_y = []
        for i in range(self.batch_size):
            x = upsample(
                self.images[index * self.batch_size + i],
                img_size_target=self.img_size_target
            ).reshape(self.img_size_target, self.img_size_target, self.img_channels)
            y = upsample(
                self.masks[index * self.batch_size + i],
                img_size_target=self.img_size_target
            ).reshape(self.img_size_target, self.img_size_target, 1)
            #x, y = augmentation_random(x, mask=y, img_size_target=self.img_size_target)
            #x = x.reshape(self.img_size_target, self.img_size_target, self.img_channels)
            #y = y.reshape(self.img_size_target, self.img_size_target, 1)
            batch_x.append(x)
            batch_y.append(y)
            x, y = augmentation(x, mask=y, img_size_target=self.img_size_target)
            x = [i.reshape(self.img_size_target, self.img_size_target, self.img_channels) for i in x]
            y = [i.reshape(self.img_size_target, self.img_size_target, 1) for i in y]
            batch_x.extend(x)
            batch_y.extend(y)
            if index * self.batch_size + i + 1 >= len(self.images):
                break
        return np.array(batch_x), np.array(batch_y)
        
    def on_epoch_end(self):
        pass


class ValidGenerator(keras.utils.Sequence):
    def __init__(self, images, masks, batch_size=32, img_size_target=128, img_channels=3):
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.img_size_target = img_size_target
        self.img_channels = img_channels

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        idx = 0
        batch_x = []
        batch_y = []
        for i in range(self.batch_size):
            x = upsample(
                self.images[index * self.batch_size + i],
                img_size_target=self.img_size_target
            ).reshape(self.img_size_target, self.img_size_target, self.img_channels)
            y = upsample(
                self.masks[index * self.batch_size + i],
                img_size_target=self.img_size_target
            ).reshape(self.img_size_target, self.img_size_target, 1)
            batch_x.append(x)
            batch_y.append(y)
            if index * self.batch_size + i + 1 >= len(self.images):
                break
        return np.array(batch_x), np.array(batch_y)
        
    def on_epoch_end(self):
        pass


class TestGenerator(keras.utils.Sequence):
    def __init__(self, images, batch_size=32, img_size_target=128, img_channels=3, tta_idx=None):
        self.images = images
        self.batch_size = batch_size
        self.img_size_target = img_size_target
        self.img_channels = img_channels
        self.tta_idx = tta_idx

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        idx = 0
        batch_x = []
        for i in range(self.batch_size):
            x = upsample(
                self.images[index * self.batch_size + i],
                img_size_target=self.img_size_target
            ).reshape(self.img_size_target, self.img_size_target, self.img_channels)
            if self.tta_idx is not None:
                x = augmentation_tta(x, self.tta_idx)
                x = x.reshape(self.img_size_target, self.img_size_target, self.img_channels)
            batch_x.append(x)
            if index * self.batch_size + i + 1 >= len(self.images):
                break
        return np.array(batch_x)
        
    def on_epoch_end(self):
        pass
