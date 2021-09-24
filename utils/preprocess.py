import torch
import numpy as np
import pandas as pd


def one_hot(cat_data: pd.Series, prefix='', postfix='') -> pd.DataFrame:
    categories = sorted(set(cat_data))
    n_category = len(categories)
    
    df = []
    for value in cat_data:
        vec = [0] * n_category
        find = np.where(np.array(categories) == value)[0][0]
        vec[find] = 1.0
        df.append(vec)
    
    column_names = []
    for category in categories:
        column_names.append(f"{prefix}{category}{postfix}")

    df = pd.DataFrame(df, columns=column_names)
    return df

def one_hot_encoding(data):
    mapping_set = {}
    for i, d in enumerate(data.unique()):
        mapping_set[d] = i
    data = data.map(mapping_set)
    data = data.astype("float")
    return data

def normalize(data):
    """Normalize input in [-1,1] range, saving statics for denormalization"""
    ### 1. DATA Log 씌우기 

    ### 2. Column 별 unique 값 찾기
    ### 2.1   2% 지점의 값  98% 지점의 값
    # 2 * (x - x.min) / (x.max - x.min) - 1
    max = data.iloc[:, 1:].max(0)
    min = data.iloc[:, 1:].min()

    data.iloc[:, 1:] = data.iloc[:, 1:] - min
    data.iloc[:, 1:] = data.iloc[:, 1:] / (max - min)
    data.iloc[:, 1:] = 2 * data.iloc[:, 1:] - 1


    return data, max, min


def denormalize(data, max, min):
    """Revert [-1,1] normalization"""
    DELTA = (max - min)
    
    data.iloc[:, 1:] = 0.5 * (data.iloc[:, 1:] + 1)
    data.iloc[:, 1:] = data.iloc[:, 1:] * DELTA
    data.iloc[:, 1:] = data.iloc[:, 1:] + min


    ### 1. DATA Log 벗기기

    return data

if __name__ == "__main__":
    data_path = "data/agricultural_product.csv"
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])

    # Weekday encoding
    data['weekday'] = one_hot_encoding(data['weekday'])
    data = data.set_index('date')

    print("\n\n\n*** INPUT ***\n\n\n")
    print(data.iloc[:10, :10])

    data, min, max = normalize(data)
    
    print("\n\n\n*** NORMALIZED ***\n\n\n")
    print(data.iloc[:10, :10])
    data = denormalize(data, min, max)
    
    print("\n\n\n*** DENORMALIZED ***\n\n\n")
    print(data.iloc[:10, :10])
    