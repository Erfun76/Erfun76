from dataset import HuBMAPDataset
import numpy as np
import pandas as pd
import os
from os.path import join as opj
from get_config import get_config
import sys
sys.path.insert(0, '../')
from get_fold_idxs_list import get_fold_idxs_list

config = get_config()
INPUT_PATH = config['INPUT_PATH']
OUTPUT_PATH = config['OUTPUT_PATH']
os.makedirs(OUTPUT_PATH, exist_ok=True)
device = config['device']
print(device)

# import data
train_df = pd.read_csv(opj(INPUT_PATH, 'train.csv'))
info_df = pd.read_csv(opj(INPUT_PATH, 'HuBMAP-20-dataset_information.csv'))
sub_df = pd.read_csv(opj(INPUT_PATH, 'sample_submission.csv'))
print('train_df.shape = ', train_df.shape)
print('info_df.shape  = ', info_df.shape)
print('sub_df.shape = ', sub_df.shape)

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

print(train_df['id'])
ds = HuBMAPDataset(tst_idxs_list[0], train_df, info_df, "train")
for list in ds:
    print(list['img'].shape, list['mask'].shape)
