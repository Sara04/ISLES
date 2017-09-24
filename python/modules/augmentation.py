"""Class for ISLES 2017 data augmentation."""

import numpy as np
import json
import os
import cv2
from scipy import ndimage

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.ndimage import interpolation
from scipy.misc import imresize


class AugmentatorISLES(object):
    """Class for ISLES 2017 data preprocessing."""
    def __init__(self, x=0):

        self.x = x

    def augment(self, db, pp):

        for scan_name in db.train_dict.keys():
            scan = db.train_dict[scan_name]
            volumes = scan.load_volumes(db)
            print("Number of volumes:", len(volumes))

            volumes_r = []
            for m_idx, m in enumerate(db.modalities[:-1]):
                volumes[m_idx] = pp.normalize(scan, m,
                                              volumes[m_idx], volumes[-1])
                volumes_r.append(pp.rotate(db, scan, m, volumes[m_idx]))

            volumes_r.append(pp.rotate(db, scan, m, volumes[-2]))
            volumes_r.append(pp.rotate(db, scan, m, volumes[-1]))

            for s in db.sizes:
                volumes_a = []
                if s == scan.h:
                    for v_idx, v in enumerate(volumes_r):
                        if v_idx >= 6:
                            volumes_a.append((v > 0.5).astype('uint8'))
                        else:
                            volumes_a.append(v.astype('float32'))
                else:
                    for i in range(6):
                        vr = np.zeros((s, s, scan.d), dtype='float32')
                        for j in range(scan.d):
                            vr[:, :, j] = imresize(volumes_r[i][:, :, j],
                                                   (s, s), mode='F')
                        volumes_a.append(vr.astype('float32'))
                    for i in range(6, 8):
                        vr = np.zeros((s, s, scan.d))
                        for j in range(scan.d):
                            vr[:, :, j] = imresize(volumes_r[i][:, :, j],
                                                   (s, s), mode='F') > 0.3
                        volumes_a.append(vr.astype('uint8'))

                for i in range(4):
                    for m_idx, m in enumerate(db.modalities):
                        output_path = os.path.join(db.aug_data_dir,
                                                   scan_name + '_' + str(s) +
                                                   '_' + m + '_' + str(i) +
                                                   '.bin')
                        if not os.path.exists(os.path.dirname(output_path)):
                            os.makedirs(os.path.dirname(output_path))
                        if i == 0:
                            volumes_a[m_idx].tofile(output_path)
                        elif i == 1:
                            volumes_a[m_idx][::-1, :, :].tofile(output_path)
                        elif i == 2:
                            volumes_a[m_idx][:, ::-1, :].tofile(output_path)
                        elif i == 3:
                            volumes_a[m_idx][::-1, ::-1, :].tofile(output_path)

                    if not os.path.exists(os.path.dirname(output_path)):
                        os.makedirs(os.path.dirname(output_path))
                    output_path = os.path.join(db.aug_data_dir,
                                               scan_name + '_' + str(s) +
                                               '_brain_mask_' + str(i) +
                                               '.bin')
                    if i == 0:
                        volumes_a[-1].tofile(output_path)
                    elif i == 1:
                        volumes_a[-1][::-1, :, :].tofile(output_path)
                    elif i == 2:
                        volumes_a[-1][:, ::-1, :].tofile(output_path)
                    elif i == 3:
                        volumes_a[-1][::-1, ::-1, :].tofile(output_path)

    def augment_test(self, db, pp):

        for scan_name in db.test_dict.keys():
            scan = db.test_dict[scan_name]
            volumes = scan.load_volumes(db, test=True)
            print("Number of volumes:", len(volumes))

            volumes_r = []
            for m_idx, m in enumerate(db.modalities[:-1]):
                volumes[m_idx] = pp.normalize(scan, m,
                                              volumes[m_idx], volumes[-1])
                volumes_r.append(pp.rotate(db, scan, m, volumes[m_idx]))

            volumes_r.append(pp.rotate(db, scan, m, volumes[-2]))
            volumes_r.append(pp.rotate(db, scan, m, volumes[-1]))

            for s in db.sizes:
                volumes_a = []
                if s == scan.h:
                    for v_idx, v in enumerate(volumes_r):
                        if v_idx >= 6:
                            volumes_a.append((v > 0.5).astype('uint8'))
                        else:
                            volumes_a.append(v.astype('float32'))
                else:
                    for i in range(6):
                        vr = np.zeros((s, s, scan.d), dtype='float32')
                        for j in range(scan.d):
                            vr[:, :, j] = imresize(volumes_r[i][:, :, j],
                                                   (s, s), mode='F')
                        volumes_a.append(vr.astype('float32'))
                    for i in range(6, 8):
                        vr = np.zeros((s, s, scan.d))
                        for j in range(scan.d):
                            vr[:, :, j] = imresize(volumes_r[i][:, :, j],
                                                   (s, s), mode='F') > 0.3
                        volumes_a.append(vr.astype('uint8'))

                for i in range(4):
                    for m_idx, m in enumerate(db.modalities[:-1]):
                        output_path = os.path.join(db.aug_data_dir,
                                                   scan_name + '_' + str(s) +
                                                   '_' + m + '_' + str(i) +
                                                   '.bin')
                        if not os.path.exists(os.path.dirname(output_path)):
                            os.makedirs(os.path.dirname(output_path))
                        if i == 0:
                            volumes_a[m_idx].tofile(output_path)
                        elif i == 1:
                            volumes_a[m_idx][::-1, :, :].tofile(output_path)
                        elif i == 2:
                            volumes_a[m_idx][:, ::-1, :].tofile(output_path)
                        elif i == 3:
                            volumes_a[m_idx][::-1, ::-1, :].tofile(output_path)

                    if not os.path.exists(os.path.dirname(output_path)):
                        os.makedirs(os.path.dirname(output_path))
                    output_path = os.path.join(db.aug_data_dir,
                                               scan_name + '_' + str(s) +
                                               '_brain_mask_' + str(i) +
                                               '.bin')
                    if i == 0:
                        volumes_a[-1].tofile(output_path)
                    elif i == 1:
                        volumes_a[-1][::-1, :, :].tofile(output_path)
                    elif i == 2:
                        volumes_a[-1][:, ::-1, :].tofile(output_path)
                    elif i == 3:
                        volumes_a[-1][::-1, ::-1, :].tofile(output_path)