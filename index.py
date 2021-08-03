import pandas as pd

def get_data(file_name):
    return pd.read_csv(file_name)

baseDados = get_data('data/AG.csv')

print(baseDados)