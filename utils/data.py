import pandas as pd

def read_data(meta_config, ref_key):
    data = pd.read_csv(meta_config[ref_key])
    return  data

def parse_dataset(meta_config):
    train_data = read_data(meta_config, 'train')
    test_data = read_data(meta_config, 'test')
    return train_data, test_data


