from unittest import TestCase
import unittest

import numpy as np
from augmentation_models import augmentaion_noise_jittering

from scripts.data_utils import (load_data,
                                get_random_split,
                                get_multiple_splits,
                                augment_data)


class Test(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path_to_dataset = '/data/Projects/OpenCloseIHB/ihb_dataset.hdf5'

    def test_load_data(self):
        X, y, groups = load_data(self.path_to_dataset)
        y_vals = np.unique(y)
        self.assertTrue(X.shape == (168, 120, 423))
        self.assertTrue(list(y_vals) == [0, 1])
        self.assertEqual(len(groups), len(y))
        self.assertEqual(len(groups), X.shape[0])

    def test_get_random_split(self):
        X, y, groups = load_data(self.path_to_open, self.path_to_close)
        X_train, X_test, y_train, y_test, train_groups, test_groups = get_random_split(X,
                                                                                       y,
                                                                                       groups,
                                                                                       test_size=0.15)

        self.assertTrue(len(set(train_groups).intersection(set(test_groups))) == 0)

    def test_get_multiply_split(self):
        X, y, groups = load_data(self.path_to_open, self.path_to_close)
        random_seeds = [42, 123, 456]
        splits_with_seeds = get_multiple_splits(X, y, groups, test_size=0.2, random_seeds=random_seeds)

        self.assertEqual(len(splits_with_seeds), len(random_seeds))

    def test_augmentaion_noise_jittering(self):

        X, y, groups = load_data(self.path_to_open, self.path_to_close)
        X_augmented = augmentaion_noise_jittering(X[0])
        self.assertTrue(X_augmented.shape == X[0].shape)

    def test_augment_data(self):
        X, y, groups = load_data(self.path_to_open, self.path_to_close)
        n_aug=1
        augmented_data, y_augmented, groups_augmented, is_real_data = augment_data(augmentaion_noise_jittering,
                                                                                   X,
                                                                                   y,
                                                                                   groups,
                                                                                   n_aug=n_aug)

        self.assertEqual(len(augmented_data), len(X)*(n_aug+1))

    def test_get_random_split_with_aug(self):
        X, y, groups = load_data(self.path_to_open, self.path_to_close)
        X_aug, y_aug, groups_aug, is_real_data = augment_data(augmentaion_noise_jittering,
                                                                                   X,
                                                                                   y,
                                                                                   groups,
                                                                                   n_aug=1)

        X_train, X_test, y_train, y_test, train_groups, test_groups = get_random_split(X_aug,
                                                                                       y_aug,
                                                                                       groups_aug,
                                                                                       is_real_data=is_real_data,
                                                                                       test_size=0.15)
        train_unique = set(np.unique(train_groups))
        test_unique = set(np.unique(test_groups))

        self.assertEqual(len(train_unique.intersection(test_unique)),0)

if __name__ == '__main__':
    unittest.main()