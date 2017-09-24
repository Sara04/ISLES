"""Class for BRATS 2017 data postprocessing."""

import numpy as np
import json
import os
import cv2
import nibabel as nib
from scipy.ndimage import measurements
from scipy.ndimage import morphology

from scipy.misc import imresize
from scipy import ndimage


class PostprocessorISLES(object):
    """Class for BRATS 2017 data postprocessing."""

    def __init__(self, size_th_1=3000, size_th_2=1, score_th_1=0.915, score_th_2=1.0):
        """Initialization of PostprocessorBRATS attributes."""
        self.size_th_1 = size_th_1
        self.size_th_2 = size_th_2

        self.score_th_1 = score_th_1
        self.score_th_2 = score_th_2

    def _load_segmentation(self, db, scan, sz, orr, score=False):

        if not score:
            test_path = os.path.join(db.seg_results_dir,
                                     scan.name + '_' + str(sz) + '_' + str(orr) + '.bin')
        else:
            test_path = os.path.join(db.seg_results_dir,
                                     scan.name + '_' + str(sz) + '_' + str(orr) + '_scores.bin')

        segmentation_test = np.reshape(np.fromfile(test_path), [sz, sz, -1])
        return segmentation_test

    def _compute_dice(self, a, b):

        sum_a = np.sum(a)
        sum_b = np.sum(b)
        sum_o = np.sum(a * b)

        return 2 * float(sum_o) / (sum_a + sum_b + 0.0001)

    def derotate(self, db, prep, scan, volume, mode='train'):
        """Rotation of the input scan."""

        if mode == 'train':
            r_params = prep.train_rotate_params[scan.name]
        else:
            r_params = prep.test_rotate_params[scan.name]

        rot_matrix = np.asarray(r_params['r_matrix']).transpose()
        r_center = np.copy(r_params['r_center'])

        r_center[0] *= float(volume.shape[0]) / scan.h
        r_center[1] *= float(volume.shape[1]) / scan.w
        volume_c = np.zeros(volume.shape)

        i_s, j_s = [volume.shape[0] / 2 - r_center[0],
                    volume.shape[1] / 2 - r_center[1]]

        rot_matri_1 = np.eye(2)

        for k in range(volume.shape[2]):
            volume_c[:, :, k] =\
                ndimage.interpolation.affine_transform(volume[:, :, k],
                                                       rot_matri_1,
                                                       [i_s, j_s])
            volume_c[:, :, k] =\
                ndimage.interpolation.affine_transform(volume_c[:, :, k],
                                                       rot_matrix,
                                                       [0, 0])
        return volume_c

    def determine_parameters(self, db, prep):

        print "Entered!!!"

        dice_avg = 0
        count = 0
        for s_idx, scan_name in enumerate(db.valid_scans):
            print("s idx:", s_idx, scan_name)

            scan = db.train_dict[scan_name]
            gt = scan.load_volume(db, 'OT')

            scores_avg = np.zeros(gt.shape)
            for sz in db.sizes[0:1]:
                for orr in [0, 3]:
                    segment_test = self._load_segmentation(db, scan, sz, orr)
                    segment_test_sc = self._load_segmentation(db, scan, sz, orr, score=True)

                    if sz == scan.h:
                        if not orr:
                            add = np.copy(segment_test_sc)
                            add[segment_test == 0] = 0.0
                            scores_avg += add
                            #scores_avg = np.maximum(scores_avg, add)
                        else:
                            add = np.copy(segment_test_sc)
                            add[segment_test == 0] = 0.0
                            scores_avg += add[::-1, ::-1, :]
                            #scores_avg = np.maximum(scores_avg, add[::-1, ::-1, :])
                    else:

                        sc_tmp = np.zeros(gt.shape, dtype='float32')
                        cls_tmp = np.zeros(gt.shape, dtype='float32')
                        for j in range(scan.d):
                            sc_tmp[:, :, j] = imresize(segment_test_sc[:, :, j], (scan.h, scan.w), mode='F')
                            cls_tmp[:, :, j] = imresize(segment_test[:, :, j], (scan.h, scan.w), mode='F') > 0.3

                        if not orr:
                            add = np.copy(sc_tmp)
                            add[cls_tmp == 0] = 0.0
                            scores_avg += add
                            #scores_avg = np.maximum(scores_avg, add)
                        else:
                            add = np.copy(sc_tmp)
                            add[cls_tmp == 0] = 0.0
                            scores_avg += add[::-1, ::-1, :]
                            #scores_avg = np.maximum(scores_avg, add[::-1, ::-1, :])

            scores_avg /= 2

            score_final = self.derotate(db, prep, scan, scores_avg)

            mask_whole = score_final > 0.70
            M, label = measurements.label(mask_whole)

            for i in range(1, label + 1):
                p = (M == i)
                p_sum = np.sum(p)
                p_sc = (M == i) * score_final
                p_sc_sum = np.sum(p_sc)
                p_avg = p_sc_sum / p_sum

                if p_sum < 100:  # or p_avg < 0.74:
                    mask_whole[p] = 0

            dice = self._compute_dice(mask_whole, gt)
            print "Dice:", dice
            print "seg shape:", gt.shape
            for i in range(scan.d):
                img = np.hstack((score_final[:, :, i], gt[:, :, i]))
                cv2.imshow('img', img)
                cv2.waitKey(0)

    def postprocess(self, db):

        for s_idx, scan_name in enumerate(db.valid_dict.keys()):
            print("s idx:", s_idx)

            segment_test = self._load_segmentation(db, db.valid_dict[scan_name])
            segment_sc_test = self._load_segmentation(db, db.valid_dict[scan_name], True)
            #segment_sc_test_0 = self._load_segmentation(db, db.valid_dict[scan_name], True, 0)
            segment_sc_test_1 = self._load_segmentation(db, db.valid_dict[scan_name], True, 1)
            segment_sc_test_2 = self._load_segmentation(db, db.valid_dict[scan_name], True, 2)
            segment_sc_test_4 = self._load_segmentation(db, db.valid_dict[scan_name], True, 4)
            segment_sc_test_t = segment_sc_test_1 + segment_sc_test_2 + segment_sc_test_4
            segment_sc_test_m = np.maximum(np.maximum(segment_sc_test_1, segment_sc_test_2), segment_sc_test_4)

            mask_whole = (segment_test != 0) * (segment_sc_test_t > self.score_th_1) * (segment_sc_test_m > 0.4)
            M, label = measurements.label(mask_whole)

            for i in range(1, label + 1):
                p = (M == i)
                p_sum = np.sum(p)
                p_sc = (M == i) * segment_sc_test
                p_sc_sum = np.sum(p_sc)
                p_avg = p_sc_sum / p_sum
                #print("p_sum, p_avg:", p_sum, p_avg)
                if p_sum < self.size_th_1:
                    mask_whole[p] = 0

            se = np.ones((3, 3, 3))
            mask_whole = morphology.binary_closing(mask_whole, se)
            segment_test *= mask_whole

            segment_test.astype('int16')

            segment_nib = nib.Nifti1Image(segment_test, np.eye(4))

            output_path = os.path.join(db.seg_results_final_dir, scan_name + '.nii.gz')
            segment_nib.to_filename(output_path)

    def postprocess_test(self, db, prep):

        print "Entered!!!"

        dice_avg = 0
        count = 0
        for s_idx, scan_name in enumerate(db.test_dict):
            print("s idx:", s_idx, scan_name)

            scan = db.test_dict[scan_name]

            scores_avg = np.zeros((scan.h, scan.w, scan.d))
            for sz in db.sizes[0:1]:
                for orr in [0, 3]:
                    segment_test = self._load_segmentation(db, scan, sz, orr)
                    segment_test_sc = self._load_segmentation(db, scan, sz, orr, score=True)

                    if sz == scan.h:
                        if not orr:
                            add = np.copy(segment_test_sc)
                            add[segment_test == 0] = 0.0
                            scores_avg += add
                            #scores_avg = np.maximum(scores_avg, add)
                        else:
                            add = np.copy(segment_test_sc)
                            add[segment_test == 0] = 0.0
                            scores_avg += add[::-1, ::-1, :]
                            #scores_avg = np.maximum(scores_avg, add[::-1, ::-1, :])
                    else:

                        sc_tmp = np.zeros((scan.h, scan.w, scan.d), dtype='float32')
                        cls_tmp = np.zeros((scan.h, scan.w, scan.d), dtype='float32')
                        for j in range(scan.d):
                            sc_tmp[:, :, j] = imresize(segment_test_sc[:, :, j], (scan.h, scan.w), mode='F')
                            cls_tmp[:, :, j] = imresize(segment_test[:, :, j], (scan.h, scan.w), mode='F') > 0.3

                        if not orr:
                            add = np.copy(sc_tmp)
                            add[cls_tmp == 0] = 0.0
                            scores_avg += add
                            #scores_avg = np.maximum(scores_avg, add)
                        else:
                            add = np.copy(sc_tmp)
                            add[cls_tmp == 0] = 0.0
                            scores_avg += add[::-1, ::-1, :]
                            #scores_avg = np.maximum(scores_avg, add[::-1, ::-1, :])

            scores_avg /= 2
            print "point 1"
            score_final = self.derotate(db, prep, scan, scores_avg)

            mask_whole = score_final > 0.70
            M, label = measurements.label(mask_whole)

            for i in range(1, label + 1):
                p = (M == i)
                p_sum = np.sum(p)
                p_sc = (M == i) * score_final
                p_sc_sum = np.sum(p_sc)
                p_avg = p_sc_sum / p_sum

                if p_sum < 100:  # or p_avg < 0.74:
                    mask_whole[p] = 0

            mask_whole = mask_whole.astype('int16')

            segment_nib = nib.Nifti1Image(mask_whole, np.eye(4))

            print "Scan relative path:", scan.relative_path
            modalities = os.listdir(os.path.join(db.db_path, scan.relative_path))
            for m in modalities:
                m_split = str.split(m, '.')
                if 'MR_MTT' in m_split:
                    id_ = m_split[-1]
                    break
            print "id_:", id_
            output_path = os.path.join(db.seg_final_results_dir, 'SMIR.' + scan_name + '.' + id_ + '.nii')
            segment_nib.to_filename(output_path)

    def name(self):
        """Class name reproduction."""
        return "%s()" % (type(self).__name__)
