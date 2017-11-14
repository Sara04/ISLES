"""Class for BRATS 2017 patch extraction."""

import os
import numpy as np
import cv2
import time
from scipy.ndimage import interpolation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys


class PatchExtractorISLES(object):
    """Class for ISLES 2017 patch extraction."""

    """
        Attributes:
            w, h, d: volume's widht, height and depth (number of slices)
            augment_train: flag indicating whether to augment training data

        Methods:
            extract_train_or_valid_data: extract data for
                training or validation (with equal distribution of classes)
            extract_test_patches: extract data for testing
    """
    def __init__(self, augment_train=True):
        """Initialization of PatchExtractorISLES attributes."""
        self.augment_train = augment_train in [True, 'True', 'true', 'yes', 'Yes']

    def extract_train_or_valid_data(self, db, meta, pp, seg, exp_out, mode='train'):
        """Extraction of training and validation data."""
        """
            Arguments:
                db: DatabaseISLES object
                pp: PreprocessorISLES object
                seg: SegmentatorISLES object
                exp_put: path to the experiment output
                mode: train, valid or train_valid
                    valid and train_valid modes are without augmentation
            Returns:
                data and corresponding labels
        """
        raise NotImplementedError()

    def extract_test_patches(self, scan, db, pp, volumes, ind_part):
        """Extraction of test patches."""
        """
            Arguments:
                scan: selected scan
                db: DatabaseISLES object
                pp: PreprocessorISLES object
                volumes: scan volumes
                ind_part: list of voxel indices at which patches will be
                    extracted
            Returns:
                extracted test patches
        """
        raise NotImplementedError()
