import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from keras.preprocessing.image import load_img


INPUT_DIR = './input'
N_FOLDS = 5
IMG_SIZE_ORI = 101


def cov_to_class(val):
    for i in range(0, 6):
        if val * 5 <= i :
            return i


def get_mask_type(mask, img_size_ori=101):
    border = 10
    outer = np.zeros((img_size_ori - 2 * border, img_size_ori - 2 * border), np.float32)
    outer = cv2.copyMakeBorder(
        outer, border, border, border, border, borderType = cv2.BORDER_CONSTANT, value = 1
    )
    cover = (mask > 0.5).sum()
    if cover < 8:
        return 0 # empty
    if cover == ((mask*outer) > 0.5).sum():
        return 1 #border
    if np.all(mask==mask[0]):
        return 2 #vertical
    percentage = cover / (img_size_ori * img_size_ori)
    if percentage < 0.15:
        return 3
    elif percentage < 0.25:
        return 4
    elif percentage < 0.50:
        return 5
    elif percentage < 0.75:
        return 6
    else:
        return 7


if __name__ == '__main__':
    train_df = pd.read_csv(INPUT_DIR + "/train.csv", index_col="id", usecols=[0])
    depths_df = pd.read_csv(INPUT_DIR + "/depths.csv", index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]
    train_df["images"] = [np.array(
        load_img(INPUT_DIR + "/train/images/{}.png".format(idx))
    ) / 255 for idx in tqdm(train_df.index)]
    train_df["masks"] = [np.array(
        load_img(INPUT_DIR + "/train/masks/{}.png".format(idx), grayscale=True)
    ) / 255 for idx in tqdm(train_df.index)]
    train_df["depth_p"] = train_df.z / 1000
    train_df["depth_class"] = train_df.depth_p.map(cov_to_class)
    train_df["coverage"] = train_df.masks.map(np.sum) / pow(IMG_SIZE_ORI, 2)
    train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
    train_df["strat"] = 5 * train_df["coverage_class"] + train_df["depth_class"]
    train_df["strat2"] = train_df.masks.map(get_mask_type)
    train_all = []
    evaluate_all = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)
    for train_index, evaluate_index in skf.split(train_df.index.values, train_df.strat2):
        train_all.append(train_index)
        evaluate_all.append(evaluate_index)
        print(train_index.shape, evaluate_index.shape)
    np.save(open(INPUT_DIR + '/folds.npy', 'wb'), np.array([train_all, evaluate_all]))
