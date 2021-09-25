import torch
import numpy as np
import pandas as pd


def one_hot(cat_data: pd.Series, prefix="", postfix="") -> pd.DataFrame:
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
