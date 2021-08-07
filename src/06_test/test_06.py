import sys
sys.path.insert(0, '../')
import warnings
warnings.simplefilter('ignore')
from get_config import get_config
from utils import fix_seed, rle2mask, mask2rle
from models import build_model
from utils_inference import get_pred_mask, get_rle
from get_fold_idxs_list import get_fold_idxs_list

import numpy as np
import pandas as pd
import os
from os.path import join as opj
import gc
import cv2
import rasterio
from rasterio.windows import Window

import torch


if __name__=='__main__':
    # config
    fix_seed(2021)
    config = get_config()
    INPUT_PATH = config['INPUT_PATH']
    OUTPUT_PATH = config['OUTPUT_PATH']
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    device = config['device']
    print(device)
    
    # import data
    train_df = pd.read_csv(opj(INPUT_PATH, 'train.csv'))
    info_df  = pd.read_csv(opj(INPUT_PATH,'HuBMAP-20-dataset_information.csv'))
    sub_df = pd.read_csv(opj(INPUT_PATH, 'sample_submission.csv'))
    print('train_df.shape = ', train_df.shape)
    print('info_df.shape  = ', info_df.shape)
    print('sub_df.shape = ', sub_df.shape)

    # inference
    LOAD_LOCAL_WEIGHT_PATH_LIST = {}
    for seed in config['split_seed_list']:
        LOAD_LOCAL_WEIGHT_PATH_LIST[seed] = []
        for fold in config['FOLD_LIST']:
            LOAD_LOCAL_WEIGHT_PATH_LIST[seed].append(opj(config['model_path'],f'model_seed{seed}_fold{fold}_bestscore.pth'))
    model_list = {}
    for seed in config['split_seed_list']:
        model_list[seed] = []
        for path in LOAD_LOCAL_WEIGHT_PATH_LIST:
            print("Loading weights from %s" % path)
            model = build_model(model_name=config['model_name'],
                                resolution=(None,None),
                                deepsupervision=config['deepsupervision'], 
                                clfhead=config['clfhead'],
                                clf_threshold=config['clf_threshold'],
                                load_weights=False).to(device)
            model.load_state_dict(torch.load(path))
            model.eval()
            model_list[seed].append(model) 
    
    # pseudo-label for test data
    val_patient_numbers_list = [
        [68250],  # fold0
        [65631],  # fold1
        [67177],  # fold2
    ]

    test_patient_numbers_list = [
        [63921],  # fold0
        [63921],  # fold1
        [63921],  # fold2
    ]
    _, _, tst_idxs_list = get_fold_idxs_list(info_df, val_patient_numbers_list, test_patient_numbers_list)
    print("test index list: {}".format(tst_idxs_list))

    train_df['predicted'] = None
    for idx in tst_idxs_list:
        print('idx = ', idx)
        pred_mask,h,w = get_pred_mask(idx, train_df, info_df, model_list, mode='train')
        rle = get_rle(pred_mask,h,w)
        train_df.loc[idx,'predicted'] = rle
    train_df.to_csv(opj(OUTPUT_PATH, 'test_mask_predicted.csv'), index=False)
    
    # pseudo-label for test data
    # for idx in range(len(sub_df)):
    #     print('idx = ', idx)
    #     pred_mask,h,w = get_pred_mask(idx, sub_df, model_list, mode='test')
    #     rle = get_rle(pred_mask,h,w)
    #     sub_df.loc[idx,'predicted'] = rle
    # sub_df.to_csv(opj(OUTPUT_PATH, 'pseudo_test.csv'), index=False)