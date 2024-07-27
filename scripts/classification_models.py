from nilearn.connectome import ConnectivityMeasure
import numpy as np


def get_connectome(timeseries: np.ndarray,
                   conn_type: str = 'corr') -> np.ndarray:
    if conn_type == 'corr':
        conn = ConnectivityMeasure(kind='correlation').fit_transform(timeseries)
        conn[conn == 1] = 0.999999
        conn = np.arctanh(conn)
    else:
        raise NotImplementedError
    return conn

#TODO function for ICA aggregation


#TODO baseline for LinearModel with PCA on vectors


#TODO graph model baseline
