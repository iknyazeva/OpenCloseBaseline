import pytest
from scripts.classification_models import *
from scripts.data_utils import *


def test_LogRegPCA():
    # data
    path_to_open  = '/data/Projects/OpenCloseIHB/opened_ihb.npy'
    path_to_close = '/data/Projects/OpenCloseIHB/closed_ihb.npy'
    X, y, groups = load_data(path_to_open, path_to_close)
    X_train, X_test, y_train, y_test, train_groups, test_groups = get_random_split(X, y, groups, test_size=0.15)

    logreg = LogRegPCA()
    logreg.model.set_params(**{'C': 0.002})
    logreg.pca.set_params(**{'n_components': 0.95})

    train_acc = logreg.model_training(X_train, y_train)
    conf_mat, acc = logreg.model_testing(X_test, y_test)

    # --------------
    assert logreg.model.get_params()['C'] == 0.002
    assert logreg.pca.get_params()['n_components'] == 0.95
    assert train_acc > 0.8

