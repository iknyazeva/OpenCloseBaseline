import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import pickle
from typing import Tuple, List, Optional, Callable
from nilearn.connectome import ConnectivityMeasure
import numpy as np
import h5py


#TODO function for ICA aggregation
def networks_aggregation():
    pass


def get_connectome(timeseries: np.ndarray,
                   conn_type: str = 'corr') -> np.ndarray:
    
    if conn_type == 'corr':
        conn = ConnectivityMeasure(kind='correlation', standardize=False).fit_transform(timeseries)
        conn[conn == 1] = 0.999999

        for i in conn:
            np.fill_diagonal(i, 0)

        conn = np.arctanh(conn)
    
    else:
        raise NotImplementedError
    
    return conn


def load_hdf5(path):
    opened, closed = np.zeros((84, 120, 420)), np.zeros((84, 120, 420))
    with h5py.File(path, "r") as data:
        for i in range(84):
            opened[i] = data[f'sub-{i+1:03d}']['opened'][:]
            closed[i] = data[f'sub-{i+1:03d}']['closed'][:]

    return opened, closed


def load_data(path_to_dataset: str,
              path_to_idx: Optional[str]=None):
    
    """ Function for loading the data from pickle files

    The function loads two datasets from given paths, normalizes them using z-score normalization,
    concatenates them along the second axis, and generates the corresponding labels and groups.

    :param path_to_opened: Path to a hdf5 file containing the 'opened' dataset.
    :param path_to_closed: Path to a hdf5 file containing the 'closed' dataset.
    :param path_to_idx: Path to a hdf5 file containing subject indexes.
    :return: A tuple containing:
             - X: Concatenated and normalized dataset.
             - y: Labels for the data (0 for 'closed', 1 for 'opened').
             - groups: Group identifiers for the samples.

    """

    opened, closed = load_hdf5(path_to_dataset)

    closed = zscore(closed, nan_policy='omit')
    opened = zscore(opened, nan_policy='omit')
    
    np.nan_to_num(closed, copy=False)
    np.nan_to_num(opened, copy=False)

    X = np.concatenate([closed, opened], axis=0)

    # fix у китайцев разное количество открытых и закрытых
    n_closed, n_opened = closed.shape[0], opened.shape[0]
    #num_people = n_closed + n_opened
    num_states = 2
    y = np.array([0] * n_closed + [1] * n_opened)
    groups = np.tile(np.arange(n_closed), num_states)

    return X, y, groups


def get_random_split(X: np.ndarray,
                     y: np.ndarray,
                     groups: np.ndarray,
                     is_real_data: Optional[np.ndarray]=None,
                     test_size: float = 0.2,
                     random_state: int = 42,
                     shuffle_train: bool = True) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    unique_groups = np.unique(groups)
    train_groups, test_groups = train_test_split(unique_groups,
                                                 test_size=test_size,
                                                 random_state=random_state)
    
    train_index = np.isin(groups, train_groups)
    test_index = np.isin(groups, test_groups)

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    groups_train, groups_test = groups[train_index], groups[test_index]

    if is_real_data is not None:
        is_real_test = is_real_data[test_index]
        X_test = X_test[is_real_test]
        y_test = y_test[is_real_test]
        groups_test = groups_test[is_real_test]

    if shuffle_train:
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
        groups_train = groups_train[indices]

    return X_train, X_test, y_train, y_test, groups_train, groups_test


def get_multiple_splits(X: np.ndarray,
                        y: np.ndarray,
                        groups: np.ndarray,
                        is_real_data: Optional[np.ndarray]=None,
                        test_size: float = 0.2,
                        random_seeds: Optional[List[int]] = None) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
        Generates multiple random splits of the data based on different random seeds.

        Parameters:
        - X: np.ndarray - Features array.
        - y: np.ndarray - Target array.
        - groups: np.ndarray - Groups array.
        - is_real_data: np.ndarray - Boolean array indicating whether the data is real or not.
        - test_size: float - Proportion of the dataset to include in the test split.
        - random_seeds: Optional[List[int]] - List of random seeds for reproducibility. If None, generates 10 random seeds.

        Returns:
        - List of tuples, each containing (X_train, X_test, y_train, y_test, train_groups, test_groups)
        """

    if random_seeds is None:
        np.random.seed(42)  # Ensure reproducibility of the generated seeds
        random_seeds = np.random.randint(0, 10000, size=10).tolist()

    splits = []
    for seed in random_seeds:
        splits.append(get_random_split(X, y, groups, is_real_data, test_size, seed))

    return splits


def augment_data(augmentation_func: Callable,
                 X: np.ndarray,
                 y: np.ndarray,
                 groups: np.ndarray,
                 n_aug = 1) -> Tuple[np.ndarray, np.ndarray,np. ndarray, np.ndarray]:
    """
    Applies an augmentation function to the data and returns the augmented data along with indicators.
    Data consists of X, labels and groups.

    Args:
        augmentation_func (Callable[[Any], Any]): Function to augment the data, one examlple.
        - X: np.ndarray - Features array.
        - y: np.ndarray - Target array.
        - groups: np.ndarray - Groups array.
        - n_aug: int - Number of augmentations.

    Returns:
        Tuple[np.ndarray, np.ndarray,np. ndarray, np.ndarray]: A tuple containing the array of augmented data
         corresponding labels and groups and array of indicators,
         where True indicates real data and False indicates augmented data.
    """

    new_shape = (X.shape[0] * (n_aug+1), *X.shape[1:])
    augmented_data = np.empty(new_shape)
    is_real_data = np.empty((X.shape[0]*(n_aug+1),), dtype=bool)
    y_augmented = np.empty((X.shape[0]*(n_aug+1),), dtype=int)
    groups_augmented = np.empty((X.shape[0]*(n_aug+1),), dtype=int)

    augmented_data[:X.shape[0],:] = X
    is_real_data[:X.shape[0]] = True
    y_augmented[:X.shape[0]] = y
    groups_augmented[:X.shape[0]] = groups

    for k in range(1,n_aug+1):
        augmented_array = np.array([augmentation_func(slice) for slice in X])
        augmented_data[X.shape[0]*k:X.shape[0]*(k+1), :] = augmented_array
        is_real_data[X.shape[0]*k:X.shape[0]*(k+1)] = False
        y_augmented[X.shape[0]*k:X.shape[0]*(k+1)] = y
        groups_augmented[X.shape[0]*k:X.shape[0]*(k+1)] = groups

    return augmented_data, y_augmented, groups_augmented, is_real_data




