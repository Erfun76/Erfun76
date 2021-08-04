import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from get_config import get_config
config = get_config()

def get_fold_idxs_list(data_df, val_patient_numbers_list, test_patient_numbers_list):
    trn_idxs_list = []
    val_idxs_list = []
    tst_idxs_list = []
    for fold in range(len(val_patient_numbers_list)):
        trn_idxs = data_df[~data_df['patient_number'].isin(val_patient_numbers_list[fold] + test_patient_numbers_list[fold])].index.tolist()
        trn_idxs_list.append(np.array(trn_idxs))
        val_idxs = data_df[data_df['patient_number'].isin(val_patient_numbers_list[fold])].index.tolist()
        val_idxs_list.append(np.array(val_idxs))

    tst_idxs = data_df[data_df['patient_number'].isin(test_patient_numbers_list[0])].index.tolist()
    tst_idxs_list.append(np.array(tst_idxs))

    return trn_idxs_list, val_idxs_list, tst_idxs_list