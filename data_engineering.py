import pandas as pd
import numpy as np
from math import isnan
import multiprocessing as mp
import sys
from tqdm import tqdm


def calcu_trading_entropy(
        data_2: pd.DataFrame
) -> float:
    """calculate trading entropy of given data
    Args:
        data (pd.DataFrame): 2 cols, Amount and Type
    Returns:
        float: entropy
    """
    # if empty
    if len(data_2) == 0:
        return 0

    amounts = np.array([data_2[data_2['Type'] == type]['Amount'].sum()
                        for type in data_2['Type'].unique()])
    proportions = amounts / amounts.sum() if amounts.sum() else np.ones_like(amounts)
    ent = -np.array([proportion * np.log(1e-5 + proportion)
                     for proportion in proportions]).sum()
    return ent

def span_data_3d(
        data: pd.DataFrame,
        time_windows=None,
        spatio_windows=None,
) -> np.ndarray:
    if time_windows is None:
        time_windows = [1, 3, 5, 10, 20, 50, 100, 500]
    if spatio_windows is None:
        spatio_windows = [1, 2, 3, 4,5]
    data = data[data['Labels'] != 2].copy()
    data['Amount'] = np.log1p(data['Amount']) # Lọc giá trị ngoại lai và chuẩn hóa Logarit
    data['Location'] = data['Location'].apply(lambda x: int(x.split('L')[1]))
    data['Location'] = data['Location'].apply(lambda x: 1 if x == 100 else x)
    data['Location'] = data['Location'].apply(lambda x: 2 if 102 >= x > 100 else x)
    data['Location'] = data['Location'].apply(lambda x: 3 if 110 >= x > 102 else x)
    data['Location'] = data['Location'].apply(lambda x: 4 if 140 >= x > 110 else x)
    data['Location'] = data['Location'].apply(lambda x: 5 if x > 140 else x)

    nume_feature_ret, label_ret = [], []
    for row_idx in tqdm(range(len(data))):
        record = data.iloc[row_idx]
        acct_no = record['Source']
        location = int(record['Location'])
        feature_of_one_record = []

        # Quét qua cửa sổ thời gian
        for time_span in time_windows:
            feature_of_one_timestamp = []
            prev_records = data.iloc[(row_idx - time_span):row_idx, :]
            prev_and_now_records = data.iloc[(
                                                     row_idx - time_span):row_idx + 1, :]
            prev_records = prev_records[prev_records['Source'] == acct_no]

            # Quét qua cửa sổ không gian
            for spatio_span in spatio_windows:
                feature_of_one_spatio_stamp = []
                one_spatio_records = prev_records[prev_records['Location'] > location - spatio_span]
                one_spatio_records = one_spatio_records[one_spatio_records['Location'] < location + spatio_span]

                # 5 đặc trưng được tính: AvgAmountT, TotalAmountTs, BiasAmountT, NumberT, TradingEntropyT
                # AvgAmountT
                feature_of_one_spatio_stamp.append(
                    one_spatio_records['Amount'].sum() / time_span)
                # TotalAmountTs
                feature_of_one_spatio_stamp.append(one_spatio_records['Amount'].sum())
                # BiasAmountT
                feature_of_one_spatio_stamp.append(
                    record['Amount'] - feature_of_one_spatio_stamp[0])
                # NumberT
                feature_of_one_spatio_stamp.append(len(one_spatio_records))

                # TradingEntropyT ->  TradingEntropyT = EntT − NewEntT
                old_ent = calcu_trading_entropy(prev_records[['Amount', 'Type']])
                new_ent = calcu_trading_entropy(
                    prev_and_now_records[['Amount', 'Type']])
                feature_of_one_spatio_stamp.append(old_ent - new_ent)

                feature_of_one_timestamp.append(feature_of_one_spatio_stamp)
            feature_of_one_record.append(feature_of_one_timestamp)
        nume_feature_ret.append(feature_of_one_record)
        label_ret.append(record['Labels'])

    nume_feature_ret = np.array(nume_feature_ret)
    print(nume_feature_ret.shape)
    # sanity check
    assert nume_feature_ret.shape == (
        len(data), len(time_windows), len(spatio_windows), 5), "output shape invalid."

    return nume_feature_ret.astype(np.float32), np.array(label_ret).astype(np.int64)
