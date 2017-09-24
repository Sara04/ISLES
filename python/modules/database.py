"""Class for ISLES 2017 database management."""
import os
import natsort as ns
import numpy as np

from scan import ScanISLES


class DatabaseISLES(object):
    """Class for ISLES 2017 database management."""

    def __init__(self, db_path,
                 modalities=['ADC', 'MTT', 'rCBF',
                             'rCBV', 'Tmax', 'TTP', 'OT'],
                 sizes=[128, 192, 256],
                 classes=[0, 1], n_modalities=6, n_classes=2):
        """Initialization of DatabaseISLES attributes."""
        self.db_path = db_path
        self.train_dict = {}
        self.test_dict = {}
        self.valid_scans = []
        self.train_scans = []
        self.modalities = modalities
        self.sizes = sizes
        self.classes = classes
        self.n_ms = n_modalities
        self.n_classes = n_classes

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
                    m_split = str.split(m, '.')
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
                    m_split = str.split(m, '.')
                    mb = str.split(m_split[-2], '_')[-1]
                    if mb not in modalities_dict:
                        modalities_dict[mb] = m

                scan_name = fo
                s_relative_path = os.path.join(folder_name, fo)
                self.test_dict[fo] = ScanISLES(self, scan_name,
                                               s_relative_path,
                                               modalities_dict,
                                               'test')

    def train_valid_split(self, split_no, folder_name='ISLES2017_Training'):
        """Splitting training data into train and valid subsets."""
        """
            Arguments:
                folder_name: name of the folder where training data is stored

            Note:
                the splitting of the training data is done since at the moment
                of the algorithm creation, testing dataset is not available,
                so validation dataset would be used for testing
        """

        scan_list = ns.natsort(self.train_dict.keys())

        self.valid_scans = scan_list[split_no * 5:(split_no + 1) * 5]
        self.train_scans = []
        for s in scan_list:
            if s not in self.valid_scans:
                self.train_scans.append(s)
