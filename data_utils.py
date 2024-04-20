import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_and_preprocess_data(data_path):
    df = pd.read_csv(data_path)
    X = df.drop(['A_id', 'Quality'], axis=1)
    y = df.filter(['Quality'])

    quality_mapping = {'good': 1, 'bad': 0}
    y['Quality'] = y['Quality'].map(quality_mapping)

    X = X.apply(pd.to_numeric, errors='coerce')

    missing_mask = X.isnull().any(axis=1)
    X = X[~missing_mask]
    y = y[~missing_mask]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=42)

    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    return train_loader, test_loader, X_train.shape[1]
