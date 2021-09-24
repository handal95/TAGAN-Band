# import torch
import numpy as np
import pandas as pd

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
    data.iloc[:, 1:] = np.log10(data.iloc[:, 1:]+ 1) 

    ### 2. Column 별 unique 값 찾기
    for col in data.columns[1:]:
        unique_values = sorted(data[col].unique())
        two_percent = int(len(unique_values) * 0.02) 
        front_two = unique_values[:two_percent]
        back_two = unique_values[-two_percent:]
        for i, value in enumerate(data[col]):
            if value in front_two:
                data[col][i] = unique_values[two_percent - 1]
            elif value in back_two:
                data[col][i] = unique_values[-two_percent]
        print(f'{col} {len(unique_values) - len(front_two) - len(back_two) - len(data[col].unique())}')
    
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
    data.iloc[:, 1:] = np.power(10, data.iloc[:, 1:]) - 1 
    return data

if __name__ == "__main__":
    data_path = "./data/public_data/train.csv"
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])

    # Weekday encoding
    data['요일'] = one_hot_encoding(data['요일'])
    data = data.set_index('date')

    print("\n\n\n*** INPUT ***\n\n\n")
    print(data.iloc[:10, :10])

    data, min, max = normalize(data)
    
    print("\n\n\n*** NORMALIZED ***\n\n\n")
    print(data.iloc[:10, :10])
    data = denormalize(data, min, max)
    
    print("\n\n\n*** DENORMALIZED ***\n\n\n")
    print(data.iloc[:10, :10])
    