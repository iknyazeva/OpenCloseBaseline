import pytest
from scripts.augmentation_models import LSTMBaseline
from scripts.data_utils import *
from torch.utils.data import TensorDataset, DataLoader
import torch


def test_LSTMBaseline():
    path_to_open  = '/data/Projects/OpenCloseIHB/opened_ihb.npy'
    path_to_close = '/data/Projects/OpenCloseIHB/closed_ihb.npy'
    X, y, groups = load_data(path_to_open, path_to_close)
    X_train, X_test, y_train, y_test, _, _ = get_random_split(X, y, groups, test_size=0.15)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    train_dataset = TensorDataset(X_train, X_train)
    test_dataset = TensorDataset(X_test, X_test)

    batch_size = 64
    lstm_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    lstm_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    lstm = LSTMBaseline(input_dim=420)
    train_loss, val_loss = lstm.model_training(lstm_train_loader, lstm_test_loader,
                                               epochs=20, lr=0.001)
    
    gen = lstm.generate(X_test)
    
    assert len(train_loss) == 20
    assert len(gen) == len(X_test)
