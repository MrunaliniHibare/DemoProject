import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    dataset = pd.read_csv(filepath)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    return X, y

def split_data(X, y, test_size=1/3, random_state=0):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
