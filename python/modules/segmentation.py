import numpy as np
import cv2
import time

#from cnn import CnnBRATS
from cnn3 import CnnISLES3
import os


class SegmentatorISLES(object):

    def __init__(self, train_iters=6001,
                 scans_per_batch=5,
                 scans_per_batch_valid=5,
                 patches_per_scan=100,
                 patches_per_scan_valid=200,
                 test_patches_per_scan=5000,
                 scans_per_test=2):

        self.train_valid_iters = train_iters
        self.scans_per_batch = scans_per_batch
        self.scans_per_batch_valid = scans_per_batch_valid
        self.patches_per_scan = patches_per_scan
        self.patches_per_scan_valid = patches_per_scan_valid
        self.test_patches_per_scan = test_patches_per_scan
        self.scans_per_test = scans_per_test

        self.cnn = CnnISLES3()

    def train_and_valid(self, train_r1, train_r2, valid_r1, valid_r2):

        self.cnn.train_valid_accuracy(train_r1, train_r2, valid_r1, valid_r2)

    def train(self, train_r1, train_r2):

        n1 = train_r1['labels'].shape[0]
        random_s = np.random.choice(n1, int(0.2 * n1), replace=False)
        for idx, r in enumerate(random_s):
            train_r1['labels'][r, :] = np.zeros(2)
            train_r1['labels'][r, np.random.randint(2)] = 1

        n2 = train_r2['labels'].shape[0]
        random_s = np.random.choice(n2, int(0.2 * n2), replace=False)
        for idx, r in enumerate(random_s):
            train_r2['labels'][r, :] = np.zeros(2)
            train_r2['labels'][r, np.random.randint(2)] = 1

        self.cnn.train(train_r1, train_r2)

    def dice_score(self, seg_gt, seg_test, label):

        if label == -1:
            a = seg_gt != 0
            b = seg_test != 0
        else:
            a = seg_gt == label
            b = seg_test == label

        return float(2 * np.sum(a * b)) / (np.sum(a) + np.sum(b))

    def evaluate_dice(self, db, pp, patch_ex, seg_results_path):

        for s_idx, s in enumerate(db.valid_scans):
            scan_path = os.path.join(seg_results_path, s + '.bin')
            scan_sc_path = os.path.join(seg_results_path, s + '_scores.bin')
            scan = db.train_dict[s]
            for sz in db.sizes:
                for orr in [0, 3]:
                    print "s, s idx, size, orient:", s_idx, s, sz, orr

                    scan_path = os.path.join(seg_results_path, s + '_' + str(sz) + '_' + str(orr) + '.bin')
                    scan_sc_path = os.path.join(seg_results_path, s + '_' + str(sz) + '_' + str(orr) + '_scores.bin')
                    volumes = scan.load_volumes_norm_aligned(db, sz, orr)
                    patch_ex._get_coordinates(volumes[0].shape)
                    test_ind = np.where(volumes[7])
                    test_out = np.zeros((volumes[0].shape))
                    test_out_sc = np.zeros((volumes[0].shape))
                    n_indices = len(test_ind[0])
                    i = 0
                    time_start = time.time()
                    while i < n_indices:
                        indx = [test_ind[0][i:i + self.test_patches_per_scan],
                                test_ind[1][i:i + self.test_patches_per_scan],
                                test_ind[2][i:i + self.test_patches_per_scan]]
                        i += self.test_patches_per_scan
                        test_patches =\
                            patch_ex.extract_test_patches(scan, db, pp, volumes, indx)
                        test_labels = self.cnn.test_accuracy(test_patches)
                        for j in range(test_labels.shape[0]):
                            test_out[indx[0][j], indx[1][j], indx[2][j]] =\
                                db.classes[np.argmax(test_labels[j, :])]
                            test_out_sc[indx[0][j], indx[1][j], indx[2][j]] =\
                                np.max(test_labels[j, :])
                    if not os.path.exists(os.path.dirname(scan_path)):
                        os.makedirs(os.path.dirname(scan_path))
                    test_out.tofile(scan_path)
                    test_out_sc.tofile(scan_sc_path)
                    time_end = time.time()
                    print("time elapsed:", time_end - time_start)
                    dice1 = self.dice_score(volumes[6], test_out, 1)
                    print("dice 1:", dice1)

    def evaluate_dice_test(self, db, pp, patch_ex, seg_results_path):

        for s_idx, s in enumerate(db.test_dict.keys()):
            scan = db.test_dict[s]
            for sz in db.sizes:
                for orr in [0, 3]:
                    print "s, s_idx, size, orient:", s_idx, s, sz, orr
                    scan_path = os.path.join(seg_results_path, s + '_' + str(sz) + '_' + str(orr) + '.bin')
                    scan_sc_path = os.path.join(seg_results_path, s + '_' + str(sz) + '_' + str(orr) + '_scores.bin')
                    volumes = scan.load_volumes_norm_aligned(db, sz, orr, test=True)
                    patch_ex._get_coordinates(volumes[0].shape)
                    test_ind = np.where(volumes[6])
                    test_out = np.zeros((volumes[0].shape))
                    test_out_sc = np.zeros((volumes[0].shape))
                    n_indices = len(test_ind[0])
                    i = 0
                    time_start = time.time()
                    while i < n_indices:
                        indx = [test_ind[0][i:i + self.test_patches_per_scan],
                                test_ind[1][i:i + self.test_patches_per_scan],
                                test_ind[2][i:i + self.test_patches_per_scan]]
                        i += self.test_patches_per_scan
                        test_patches =\
                            patch_ex.extract_test_patches(scan, db, pp, volumes, indx)
                        test_labels = self.cnn.test_accuracy(test_patches)
                        for j in range(test_labels.shape[0]):
                            test_out[indx[0][j], indx[1][j], indx[2][j]] =\
                                db.classes[np.argmax(test_labels[j, :])]
                            test_out_sc[indx[0][j], indx[1][j], indx[2][j]] =\
                                np.max(test_labels[j, :])
                    if not os.path.exists(os.path.dirname(scan_path)):
                        os.makedirs(os.path.dirname(scan_path))
                    test_out.tofile(scan_path)
                    test_out_sc.tofile(scan_sc_path)
                    time_end = time.time()
                    print("time elapsed:", time_end - time_start)

    def save_model(self, output_path, it):

        self.cnn.save_model(output_path, it)

    def restore_model(self, output_path, it):

        self.cnn.restore_model(output_path, it)

