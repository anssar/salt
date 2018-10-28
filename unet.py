import numpy as np
import pandas as pd
import os
import keras
from keras.preprocessing.image import load_img
from keras import Model
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.models import load_model
from keras.optimizers import Adam
from tqdm import tqdm
from segmentation_models import Unet, utils as segmentation_utils
from utils import *
from metrics import *
from losses import *
from callbacks import *
from transforms import *
from generators import *


INPUT_DIR = './input'
OUTPUT_DIR = './output'
BASE_NAME = 'inceptionresnetv2'
BACKBONE = 'inceptionresnetv2'
WARM_EPOCHS = 3
EPOCHS = 300
BATCH_SIZE = 32
CUR_FOLD_INDEX = 0
IMG_SIZE_ORI = 101
IMG_SIZE_TARGET = 128
IMG_CHANNELS = 3


def prepare_output_dir():
    dirs = [
        'logs',
        'models',
        'result',
        'predictions'
    ]
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    if not os.path.exists(OUTPUT_DIR + '/' + BASE_NAME):
        os.mkdir(OUTPUT_DIR + '/' + BASE_NAME)
    for directory in dirs:
        if not os.path.exists(OUTPUT_DIR + '/' + BASE_NAME + '/' + directory):
            os.mkdir(OUTPUT_DIR + '/' + BASE_NAME + '/' + directory)
            

def prepare():
    prepare_output_dir()
    keras.losses.bce_dice_jaccard_loss = bce_dice_jaccard_loss
    keras.losses.lovasz_loss = lovasz_loss
    keras.metrics.my_iou_metric = my_iou_metric
    keras.metrics.my_iou_metric_2 = my_iou_metric_2
    train_df = pd.read_csv(INPUT_DIR + "/train.csv", index_col="id", usecols=[0])
    depths_df = pd.read_csv(INPUT_DIR + "/depths.csv", index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]
    train_df["images"] = [
        np.array(load_img(
            INPUT_DIR + "/train/images/{}.png".format(idx)
        )) / 255 for idx in tqdm(train_df.index)
    ]
    train_df["masks"] = [
        np.array(load_img(
            INPUT_DIR + "/train/masks/{}.png".format(idx), grayscale=True
        )) / 255 for idx in tqdm(train_df.index)
    ]
    folds = np.load(open(INPUT_DIR + '/folds.npy', 'rb'))
    x_test = np.array([
        np.array(load_img(
            INPUT_DIR + "/test/images/{}.png".format(idx)
        )) / 255 for idx in tqdm(test_df.index)
    ]).reshape(-1, IMG_SIZE_ORI, IMG_SIZE_ORI, IMG_CHANNELS)
    train_idx = folds[0][CUR_FOLD_INDEX]
    valid_idx = folds[1][CUR_FOLD_INDEX]
    x_train = np.array([x.reshape(IMG_SIZE_ORI, IMG_SIZE_ORI, IMG_CHANNELS) for x in
                        train_df.images.values[train_idx]])
    x_valid = np.array([x.reshape(IMG_SIZE_ORI, IMG_SIZE_ORI, IMG_CHANNELS) for x in
                        train_df.images.values[valid_idx]])
    y_train = np.array([x.reshape(IMG_SIZE_ORI, IMG_SIZE_ORI, 1) for x in
                        train_df.masks.values[train_idx]])
    y_valid = np.array([x.reshape(IMG_SIZE_ORI, IMG_SIZE_ORI, 1) for x in
                        train_df.masks.values[valid_idx]])
    return x_train, y_train, x_valid, y_valid, x_test, test_df.index.values


def train_stage_1(x_train, y_train, x_valid, y_valid):
    opt = optimizers.adam(lr=0.001)
    model = Unet(
        backbone_name=BACKBONE,
        encoder_weights='imagenet',
        freeze_encoder=True
    )
    model.compile(
        loss=bce_dice_jaccard_loss,
        optimizer=opt,
        metrics=[my_iou_metric]
    )
    model_checkpoint = ModelCheckpoint(
        OUTPUT_DIR + "/{}/models/{}_fold_{}_stage1.model".format(
            BASE_NAME, BASE_NAME, CUR_FOLD_INDEX),
        monitor='val_my_iou_metric',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_my_iou_metric',
        mode='max',
        factor=0.5,
        patience=6,
        min_lr=0.00001,
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_my_iou_metric',
        mode='max',
        patience=20,
        verbose=1
    )
    logger = CSVLogger(
        OUTPUT_DIR + '/{}/logs/{}_fold_{}_stage1.log'.format(
            BASE_NAME, BASE_NAME, CUR_FOLD_INDEX)
    )
    model.fit_generator(
        TrainGenerator(
            x_train, y_train,
            batch_size=int(np.ceil(BATCH_SIZE / (len(AUGS) + 1))),
            img_size_target=IMG_SIZE_TARGET
        ),
        steps_per_epoch=int(np.ceil(len(x_train) / int(np.ceil(BATCH_SIZE / (len(AUGS) + 1))))),
        epochs=WARM_EPOCHS,
        validation_data=ValidGenerator(
            x_valid, y_valid,
            batch_size=BATCH_SIZE,
            img_size_target=IMG_SIZE_TARGET
        ),
        callbacks=[model_checkpoint], 
        shuffle=True
    )
    segmentation_utils.set_trainable(model)
    model.fit_generator(
        TrainGenerator(
            x_train, y_train,
            batch_size=int(np.ceil(BATCH_SIZE / (len(AUGS) + 1))),
            img_size_target=IMG_SIZE_TARGET
        ),
        steps_per_epoch=int(np.ceil(len(x_train) / int(np.ceil(BATCH_SIZE / (len(AUGS) + 1))))), 
        epochs=EPOCHS,
        validation_data=ValidGenerator(
            x_valid, y_valid,
            batch_size=BATCH_SIZE,
            img_size_target=IMG_SIZE_TARGET
        ),
        callbacks=[early_stopping, model_checkpoint, reduce_lr, logger],
        shuffle=True
    )
    

def train_stage_2(x_train, y_train, x_valid, y_valid):
    model = load_model(
        OUTPUT_DIR + "/{}/models/{}_fold_{}_stage1.model".format(
            BASE_NAME, BASE_NAME, CUR_FOLD_INDEX),
        custom_objects={
            'my_iou_metric': my_iou_metric,
            'bce_dice_jaccard_loss': bce_dice_jaccard_loss
        }
    )
    opt = optimizers.adam(lr = 0.001)
    input_x = model.layers[0].input
    output_layer = model.layers[-1].input
    model = Model(input_x, output_layer)
    model.compile(
        loss=lovasz_loss,
        optimizer=opt,
        metrics=[my_iou_metric_2]
    )
    model_checkpoint = ModelCheckpoint(
        OUTPUT_DIR + "/{}/models/{}_fold_{}_stage2.model".format(
            BASE_NAME, BASE_NAME, CUR_FOLD_INDEX),
        monitor='val_my_iou_metric_2',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_my_iou_metric_2',
        mode='max',
        factor=0.5,
        patience=6,
        min_lr=0.00001,
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_my_iou_metric_2',
        mode='max',
        patience=20,
        verbose=1
    )
    logger = CSVLogger(
        OUTPUT_DIR + '/{}/logs/{}_fold_{}_stage2.log'.format(
            BASE_NAME, BASE_NAME, CUR_FOLD_INDEX)
    )
    model.fit_generator(
        TrainGenerator(
            x_train, y_train,
            batch_size=int(np.ceil(BATCH_SIZE / (len(AUGS) + 1))),
            img_size_target=IMG_SIZE_TARGET
        ),
        steps_per_epoch=int(np.ceil(len(x_train) / np.ceil(BATCH_SIZE / (len(AUGS) + 1)))), 
        epochs=EPOCHS,
        validation_data=ValidGenerator(
            x_valid, y_valid,
            batch_size=BATCH_SIZE,
            img_size_target=IMG_SIZE_TARGET
        ),
        callbacks=[early_stopping, model_checkpoint, reduce_lr, logger],
        shuffle=True
    )

    
def predict_tta(model, x_test):
    preds_test = model.predict_generator(
        TestGenerator(
            x_test,
            batch_size=100,
            img_size_target=IMG_SIZE_TARGET
        ),
        steps=np.ceil(len(x_test) / 100)
    ).reshape(-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1)
    for tta_idx in range(len(AUGS_TTA)):
        preds_test += np.array(
            [
                augmentation_tta_inverse(x, tta_idx).reshape(IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1)
                for x in tqdm(
                    model.predict_generator(
                        TestGenerator(
                            x_test,
                            batch_size=100,
                            tta_idx=tta_idx,
                            img_size_target=IMG_SIZE_TARGET
                        ),
                        steps=np.ceil(len(x_test) / 100)
                    )
                )
            ]
        ).reshape(-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1)
    preds_test /= (len(AUGS_TTA) + 1)
    return preds_test

    
def get_threshold(x_valid, y_valid):
    model = load_model(
        OUTPUT_DIR + "/{}/models/{}_fold_{}_stage2.model".format(
            BASE_NAME, BASE_NAME, CUR_FOLD_INDEX),
        custom_objects={
            'my_iou_metric_2': my_iou_metric_2,
            'lovasz_loss': lovasz_loss
        }
    )
    preds_valid = predict_tta(model, x_valid)
    preds_valid = np.array([downsample(x) for x in preds_valid])
    print(preds_valid.shape, y_valid.shape)
    thresholds = np.linspace(-1, 1, 250)
    ious = np.array(
        [
            iou_metric_batch(y_valid, preds_valid > threshold)
            for threshold in tqdm(thresholds)
        ]
    )
    threshold_best_index = np.argmax(ious)
    threshold_best = thresholds[threshold_best_index]
    print('BEST THRESHOLD:', threshold_best)
    return threshold_best


def evaluate(x_test, x_test_ids, threshold):
    t_file = open(
        OUTPUT_DIR + '/{}/predictions/threshold_{}_fold_{}.txt'.format(
        BASE_NAME, BASE_NAME, CUR_FOLD_INDEX), 'w'
    )
    t_file.write(str(threshold))
    t_file.close()
    model = load_model(
        OUTPUT_DIR + "/{}/models/{}_fold_{}_stage2.model".format(
            BASE_NAME, BASE_NAME, CUR_FOLD_INDEX),
        custom_objects={
            'my_iou_metric_2': my_iou_metric_2,
            'lovasz_loss': lovasz_loss
        }
    )
    preds_test = predict_tta(model, x_test)
    preds_test = np.array([downsample(x) for x in preds_test])
    np.save(
        open(
            OUTPUT_DIR + '/{}/predictions/prediction_{}_fold_{}.npy'.format(
            BASE_NAME, BASE_NAME, CUR_FOLD_INDEX), 'wb'
        ),
        preds_test
    )
    pred_dict = {
        idx: rl_enc(np.round(preds_test[i] > threshold))
        for i, idx in enumerate(tqdm(x_test_ids))
    }
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(OUTPUT_DIR + '/{}/result/submission_{}_fold_{}.csv'.format(
        BASE_NAME, BASE_NAME, CUR_FOLD_INDEX))
    
    
if __name__ == '__main__':
    for i in range(3, 4):
        CUR_FOLD_INDEX = i
        x_train, y_train, x_valid, y_valid, x_test, x_test_ids = prepare()
        train_stage_1(x_train, y_train, x_valid, y_valid)
        train_stage_2(x_train, y_train, x_valid, y_valid)
        threshold = get_threshold(x_valid, y_valid)
        evaluate(x_test, x_test_ids, threshold)
