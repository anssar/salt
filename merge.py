import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import *

N_FOLDS = 5
INPUT_DIR = './input'
OUTPUT_DIR = './output'
BASE_NAME = 'inceptionresnetv2'

train_df = pd.read_csv(INPUT_DIR + "/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv(INPUT_DIR + "/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]
x_test_ids = test_df.index.values

threshold = 0
for i in range(N_FOLDS):
    f = open(OUTPUT_DIR + '/{}/predictions/threshold_{}_fold_{}.txt'.format(
        BASE_NAME, BASE_NAME, i))
    threshold += float(f.readline())
    f.close()
threshold /= 5

t_file = open(
    OUTPUT_DIR + '/{}/predictions/threshold_{}_merged.txt'.format(
    BASE_NAME, BASE_NAME), 'w'
)
t_file.write(str(threshold))
t_file.close()

prediction = np.zeros((18000, 101, 101, 1))
for i in range(N_FOLDS):
    pred = np.load(OUTPUT_DIR + '/{}/predictions/prediction_{}_fold_{}.npy'.format(
        BASE_NAME, BASE_NAME, i))
    prediction += pred
prediction /= 5

np.save(
    open(
        OUTPUT_DIR + '/{}/predictions/prediction_{}_merged.npy'.format(
        BASE_NAME, BASE_NAME), 'wb'
    ),
    prediction
)

pred_dict = {
    idx: rl_enc(np.round(prediction[i] > threshold))
    for i, idx in enumerate(tqdm(x_test_ids))
}
sub = pd.DataFrame.from_dict(pred_dict, orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv(OUTPUT_DIR + '/{}/result/submission_{}_merged.csv'.format(BASE_NAME, BASE_NAME))