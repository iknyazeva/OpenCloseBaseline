import numpy as np


def augmentaion_noise_jittering(input_data: np.ndarray, jitter_ratio: float = 0.2) -> np.ndarray:
    std_emp = np.std(input_data.flatten())
    noise = np.random.normal(0, jitter_ratio * std_emp, input_data.shape)
    jittered_data = input_data + noise
    return jittered_data

#TODO LSTM baseline


#TODO Transformer baseline
