import numpy as np

def one_hot_encoding(data):
    mapping_set = {}
    for i, d in enumerate(data.unique()):
        mapping_set[d] = i
    data = data.map(mapping_set)
    data = data.astype('float')
    return data
