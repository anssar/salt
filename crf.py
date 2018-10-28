import numpy as np
from skimage.io import imread, imsave
from skimage.color import gray2rgb
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
import pandas as pd
from tqdm import tqdm
from utils import *


INPUT_DIR = './input'


def crf(original_image, mask_img):
    if(len(mask_img.shape)<3):
        mask_img = gray2rgb(mask_img)
    annotated_label = mask_img[:,:,0] + (mask_img[:,:,1]<<8) + (mask_img[:,:,2]<<16)
    colors, labels = np.unique(annotated_label, return_inverse=True)
    n_labels = 2
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                      normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(10)
    MAP = np.argmax(Q, axis=0)
    return MAP.reshape((original_image.shape[0],original_image.shape[1]))


def apply_crf(filename):
    df = pd.read_csv(filename)
    for i in tqdm(range(df.shape[0])):
        if str(df.loc[i,'rle_mask'])!=str(np.nan):        
            decoded_mask = rl_dec(df.loc[i,'rle_mask'])        
            orig_img = imread(INPUT_DIR + "/test/images/{}.png".format(df.loc[i,'id']))        
            crf_output = crf(orig_img,decoded_mask)
            df.loc[i,'rle_mask'] = rl_enc(crf_output)
    df.to_csv(filename.split('.')[0] + '_crf.csv', index=False)
