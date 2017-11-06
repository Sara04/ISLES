"""Class for ISLES 2017 database management."""
import os
import natsort as ns
import numpy as np

from scan import ScanISLES


class DatabaseISLES(object):
    """Class for ISLES 2017 database management."""

    """
        Attributes:
            db_path: path where training and testing subsets are stored
            n_classes: number of brain region labels
            classes: class labels
                0 - non-lesion
                1 - stroke lesion
            n_modalities: number of MRI modalities
            modalities: list of MRI modalities, namely
                ADC - apparent diffusion coefitients
                MTT - mean transit time
                rCBF - regional cerebral blood flow
                rCBV - regional cerebral blood volume
                Tmax - time to maximum
                TTP - time to peak
            sizes: list of possible volume's slices' size (width and height)
            valid_p: percentage of training data that will be used for
                algorithm's training validation

            train_dict: dictionary for storing training scans
            test_dict: dictionary for storing test scans

            train_scans: list for storing scans used for algorithm training
            valid_scans: list for storing scans used for algorithm validation

            Directories:
            brain_masks_dir: path to the brain's masks
            normalized_volumes_dir: path to the normalized volumes
            lesion_dist_dir: path to the lesion distance maps
            seg_results_dir: path to the segmentation results

        Methods:
            load_training_dict: creating a dictionary of training scans
            load_testing_dict: creating a dictionary of testing scans
            train_valid_split: split training database into train and valid
                subsets (validation dataset is used for evaluation)
            name: returns database name with train valid split parameter
    """

    def __init__(self, db_path, n_classes=2, classes=[0, 1],
                 n_modalities=6, modalities=['ADC', 'MTT', 'rCBF',
                                             'rCBV', 'Tmax', 'TTP'],
                 sizes=[128, 192, 256], valid_p=0.2):
        """Initialization of DatabaseISLES attributes."""
        self.db_path = db_path
        self.n_classes = n_classes
        self.classes = classes
        self.n_modalities = n_modalities
        self.modalities = modalities
        self.sizes = sizes

        self.valid_p = valid_p

        self.train_dict = {}
        self.test_dict = {}

        self.train_scans = []
        self.valid_scans = []

        self.brain_masks_dir = None
        self.norm_volumes_dir = None
        self.augmented_volumes_dir = None
        self.lesion_dist_dir = None
        self.seg_results_dir = None

    def load_training_dict(self, folder_name='ISLES2017_Training'):
        """Loading training dictionary."""
        """
            Arguments:
                folder_name: folder where the training data is stored
        """
        folders = os.listdir(os.path.join(self.db_path, folder_name))
        for fo in folders:
            if fo.startswith('_'):
                continue
            if fo not in self.train_dict:
                self.train_dict[fo] = {}

                modalities_dict = {}

                modalities = os.listdir(os.path.join(self.db_path,
                                                     folder_name, fo))

                for m in modalities:
                    m_split = str.split(str(m), '.')
                    mb = str.split(m_split[-2], '_')[-1]
                    if mb not in modalities_dict:
                        modalities_dict[mb] = m

                scan_name = fo
                s_relative_path = os.path.join(folder_name, fo)
                self.train_dict[fo] = ScanISLES(self, scan_name,
                                                s_relative_path,
                                                modalities_dict,
                                                'train')

    def load_testing_dict(self, folder_name='ISLES2017_Testing'):
        """Loading training dictionary."""
        """
            Arguments:
                folder_name: folder where the training data is stored
        """
        folders = os.listdir(os.path.join(self.db_path, folder_name))
        for fo in folders:
            if fo.startswith('_'):
                continue
            if fo not in self.test_dict:
                self.test_dict[fo] = {}

                modalities_dict = {}

                modalities = os.listdir(os.path.join(self.db_path,
                                                     folder_name, fo))

                for m in modalities:
                    m_split = str.split(str(m), '.')
                    mb = str.split(m_split[-2], '_')[-1]
                    if mb not in modalities_dict:
                        modalities_dict[mb] = m

                scan_name = fo
                s_relative_path = os.path.join(folder_name, fo)
                self.test_dict[fo] = ScanISLES(self, scan_name,
                                               s_relative_path,
                                               modalities_dict,
                                               'test')

    def train_valid_split(self, folder_name='ISLES2017_Training'):
        """Splitting training data into train and valid subsets."""
        """
            Arguments:
                folder_name: name of the folder where training data is stored
        """

        np.random.seed(123456)
        scan_list = ns.natsort(self.train_dict.keys())

        n_scans = len(self.train_dict)
        select_valid = np.random.choice(n_scans,
                                        int(np.round(n_scans *
                                                     self.valid_p)),
                                        replace=False)

        for s_idx, s in enumerate(scan_list):
            if s_idx in select_valid:
                self.valid_scans.append(s)
            else:
                self.train_scans.append(s)

    def name(self):
        """Return database name."""
        return "%s(valid_p=%s)" % (type(self).__name__, self.valid_p)
