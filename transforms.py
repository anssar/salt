from albumentations import PadIfNeeded, CenterCrop, RandomBrightness, RandomScale, HorizontalFlip, Rotate, ShiftScaleRotate, Compose, OneOf, RandomSizedCrop, ElasticTransform, RandomGamma, GridDistortion
import numpy as np


AUGS = [
    HorizontalFlip(p=1),
    #OneOf([
    #    RandomGamma(gamma_limit=(90, 110), p=1),
    #    RandomBrightness(limit=0.08, p=1),
    #], p=1),
    #Compose([
    #    HorizontalFlip(p=1),
    #    OneOf([
    #        RandomGamma(gamma_limit=(90, 110), p=1),
    #        RandomBrightness(limit=0.08, p=1),
    #    ], p=1),
    #], p=1)
]
AUGS_TTA = [
    HorizontalFlip(p=1),
]
AUGS_TTA_INVERSE = [
    HorizontalFlip(p=1),
]


def upsample(img, img_size_ori=101, img_size_target=128):
    if img_size_ori == img_size_target:
        return img
    return PadIfNeeded(min_height=img_size_target,min_width=img_size_target)(image=img)['image']


def downsample(img, img_size_ori=101, img_size_target=128):
    if img_size_ori == img_size_target:
        return img
    return CenterCrop(img_size_ori, img_size_ori)(image=img)['image']


def augmentation_random(img, mask=None, img_size_target=128):
    aug = Compose([
        RandomBrightness(limit=0.2, p=0.4),
        HorizontalFlip(p=0.5),
        Rotate(limit=10, p=0.4),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.5),
        RandomSizedCrop((img_size_target // 2, img_size_target), img_size_target, img_size_target, p=0.4)
    ], p=0.95)
    if mask is not None:
        res = aug(image=img, mask=mask)
        return res['image'], res['mask']
    else:
        return aug(image=img)['image']


def augmentation(img, mask=None, img_size_target=128):
    reti = []
    if mask is not None:
        retm = []
        for aug in AUGS:
            res = aug(image=img, mask=mask)
            reti.append(res['image'])
            retm.append(res['mask'])
        return reti, retm
    else:
        for aug in augs:
            res = aug(image=img)
            reti.append(res['image'])
        return reti


def augmentation_tta(img, tta_idx):
    return AUGS_TTA[tta_idx](image=img)['image']


def augmentation_tta_inverse(img, tta_idx):
    return AUGS_TTA_INVERSE[tta_idx](image=img)['image']
