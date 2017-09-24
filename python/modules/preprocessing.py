"""Class for ISLES 2017 data preprocessing."""

import numpy as np
import json
import os
import cv2
from scipy import ndimage

from scipy.ndimage import morphology

import matplotlib.pyplot as plt
import matplotlib.cm as cm


class PreprocessorISLES(object):
    """Class for ISLES 2017 data preprocessing."""

    def __init__(self, norm_type='mean_std'):
        """Initialization of PreprocessorBRATS attributes."""
        self.norm_type = norm_type
        self.train_norm_params = {}
        self.train_rotate_params = {}
        self.test_norm_params = {}
        self.test_rotate_params = {}

    def _compute_mean_std_params(self, volume, brain_mask):

        mean_ = np.mean(volume[brain_mask == 1])
        std_ = np.std(volume[brain_mask == 1])

        return {'mean': float(mean_), 'std': float(std_)}

    def _compute_min_max_params(self, volume, brain_mask):

        max_ = np.max(volume[brain_mask == 1])
        min_ = np.min(volume[brain_mask == 1])

        return {'min': float(min_), 'max': float(max_)}

    def compute_norm_params(self, volume, brain_mask):
        """Normalization parameters computation."""
        """
            Arguments:
                volume: input volume

            Returns:
                dictionary of computed parameters
        """
        if self.norm_type == 'mean_std':
            return self._compute_mean_std_params(volume, brain_mask)
        if self.norm_type == 'min_max':
            return self._compute_min_max_params(volume, brain_mask)

    def _compute_preprocess_parameters(self, db, data_dict):
        n_params = {}
        for s in data_dict:
            brain_mask_path = os.path.join(db.brain_masks_dir,
                                           s + '_brain_mask.bin')
            brain_mask = np.fromfile(brain_mask_path, dtype='uint8')
            brain_mask = np.reshape(brain_mask,
                                    [data_dict[s].h,
                                     data_dict[s].w,
                                     data_dict[s].d])
            for m in db.modalities:
                if m in['OT']:
                    continue

                volume = data_dict[s].load_volume(db, m)
                volume_norm_params =\
                    self.compute_norm_params(volume, brain_mask)

                if s not in n_params:
                    n_params[s] = {}
                if m not in n_params[s]:
                    n_params[s][m] = {}

                for p in volume_norm_params:
                    n_params[s][m][p] = volume_norm_params[p]
        return n_params

    def _compute_rotation_parameters(self, db, data_dict):
        r_params = {}
        for s in data_dict:

            if s not in r_params:
                r_params[s] = {}

            brain_mask_path = os.path.join(db.brain_masks_dir,
                                           s + '_brain_mask.bin')
            brain_mask = np.fromfile(brain_mask_path, dtype='uint8')
            brain_mask = np.reshape(brain_mask,
                                    [data_dict[s].h,
                                     data_dict[s].w,
                                     data_dict[s].d])
            mask = (np.sum(brain_mask != 0, axis=2) != 0).astype('float32')

            x, y = np.where(mask)
            p = np.polyfit(y, x, 1)

            S = np.sum(mask)

            center_x =\
                np.sum(np.arange(mask.shape[0]) * np.sum(mask, axis=1)) / S
            center_y =\
                np.sum(np.arange(mask.shape[1]) * np.sum(mask, axis=0)) / S
            angle = np.arctan(p[0])
            rot_matrix = [[0, 0], [0, 0]]
            rot_matrix[0] = [np.cos(-angle), -np.sin(-angle)]
            rot_matrix[1] = [np.sin(-angle), np.cos(-angle)]

            r_params[s]['r_matrix'] = rot_matrix
            r_params[s]['r_center'] = [center_x, center_y]

        return r_params

    def _load_preprocess_parameters(self, params_output_path):

        with open(params_output_path, 'r') as f:
            return json.load(f)

    def _load_rotation_parameters(self, params_output_path):

        with open(params_output_path, 'r') as f:
            return json.load(f)

    def _save_preprocess_parameters(self, params_output_dir, data_dict):

        pp_done_path = os.path.join(params_output_dir, 'done')
        pp_params_output_path = os.path.join(params_output_dir, 'params.json')

        if not os.path.exists(params_output_dir):
            os.makedirs(params_output_dir)
        with open(pp_params_output_path, 'w') as f:
            json.dump(data_dict, f)
        with open(pp_done_path, 'w') as f:
            f.close()

    def _save_rotation_parameters(self, params_output_dir, data_dict):

        pp_done_path = os.path.join(params_output_dir, 'done_r')
        pp_params_output_path = os.path.join(params_output_dir, 'rotations.json')

        if not os.path.exists(params_output_dir):
            os.makedirs(params_output_dir)
        with open(pp_params_output_path, 'w') as f:
            json.dump(data_dict, f)
        with open(pp_done_path, 'w') as f:
            f.close()

    def get_preprocessing_parameters(self, db, exp_out, mode):
        """Getting a dictionary of the preprocessing parameters."""
        """
            Arguments:
                db: DatabaseBRATS
                exp_out: path to the experiment meta output
                mode: training or validation subsets
        """
        if mode == 'train':
            data_dict = db.train_dict
        elif mode == 'test':
            data_dict = db.test_dict

        out_path = os.path.join(exp_out, 'preprocessing', self.name(), mode)
        done_path = os.path.join(out_path, 'done')

        if os.path.exists(done_path):
            params_out_path = os.path.join(out_path, 'params.json')
            n_params = self._load_preprocess_parameters(params_out_path)
        else:
            n_params = self._compute_preprocess_parameters(db, data_dict)
            self._save_preprocess_parameters(out_path, n_params)

        self.train_norm_params = n_params

    def get_rotation_parameters(self, db, exp_out, mode):
        """Getting a dictionary of the preprocessing parameters."""
        """
            Arguments:
                db: DatabaseBRATS
                exp_out: path to the experiment meta output
                mode: training or validation subsets
        """
        if mode == 'train':
            data_dict = db.train_dict
        elif mode == 'test':
            data_dict = db.test_dict

        out_path = os.path.join(exp_out, 'preprocessing', self.name(), mode)
        done_path = os.path.join(out_path, 'done_r')

        if os.path.exists(done_path):
            params_out_path = os.path.join(out_path, 'rotations.json')
            r_params = self._load_rotation_parameters(params_out_path)
        else:
            r_params = self._compute_rotation_parameters(db, data_dict)
            self._save_rotation_parameters(out_path, r_params)

        self.train_rotate_params = r_params

    def normalize(self, scan, m, x, mask):
        """Normalization of the input array x."""
        n_params = self.train_norm_params[scan.name][m]

        if self.norm_type == 'mean_std':
            return mask * (x - n_params['mean']) / n_params['std']
        if self.norm_type == 'min_max':
            return mask * (x - n_params['min']) / (n_params['max'] - n_params['max'])

    def rotate(self, db, scan, m, volume):
        """Rotation of the input scan."""
        r_params = self.train_rotate_params[scan.name]
        rot_matrix = np.asarray(r_params['r_matrix'])
        r_center = np.copy(r_params['r_center'])

        r_center[0] *= float(volume.shape[0]) / scan.h
        r_center[1] *= float(volume.shape[1]) / scan.w
        volume_c = np.zeros(volume.shape)

        i_s, j_s = [r_center[0] - volume.shape[0] / 2, r_center[1] - volume.shape[1] / 2]

        for k in range(volume.shape[2]):
            volume_c[:, :, k] =\
                ndimage.interpolation.affine_transform(volume[:, :, k],
                                                       rot_matrix,
                                                       [i_s, j_s])
        return volume_c

    def derotate(self, db, scan, m, volume):
        """Rotation of the input scan."""
        r_params = self.train_rotate_params[scan.name]
        rot_matrix = np.asarray(r_params['r_matrix']).transpose()
        r_center = np.copy(r_params['r_center'])

        r_center[0] *= float(volume.shape[0]) / scan.h
        r_center[1] *= float(volume.shape[1]) / scan.w
        volume_c = np.zeros(volume.shape)

        i_s, j_s = [volume.shape[0] / 2 - r_center[0],
                    volume.shape[1] / 2 - r_center[1]]

        for k in range(volume.shape[2]):
            volume_c[:, :, k] =\
                ndimage.interpolation.affine_transform(volume[:, :, k],
                                                       rot_matrix,
                                                       [i_s, j_s])
        return volume_c

    def compute_tumor_distance_maps(self, db):

        se = np.ones((3, 3))
        for scan_name in db.train_dict:
            for s in db.sizes:
                for i in range(4):
                    gt_path =\
                        os.path.join(db.aug_data_dir,
                                     scan_name + '_' + str(s) +
                                     '_OT_' + str(i) + '.bin')
                    gt_v = np.reshape(np.fromfile(gt_path, dtype='uint8'),
                                      [s, s, -1])
                    brain_mask_path =\
                        os.path.join(db.aug_data_dir,
                                     scan_name + '_' + str(s) + '_brain_mask_' +
                                     str(i) + '.bin')
                    brain_mask =\
                        np.reshape(np.fromfile(brain_mask_path,
                                   dtype='uint8'), [s, s, -1])
                    dist_map = np.zeros(gt_v.shape)
                    for j in range(gt_v.shape[2]):
                        if not np.sum(gt_v[:, :, j]):
                            dist_map[:, :, j] = 255 * brain_mask[:, :, j]
                        else:
                            v = 1.0
                            mask = gt_v[:, :, j]
                            while(np.sum(mask) != s * s):
                                mask_d = morphology.binary_dilation(mask, se)
                                dist_map[:, :, j] += (mask_d - mask) * v
                                mask = np.copy(mask_d)
                                v += 1
                            dist_map[:, :, j] *= brain_mask[:, :, j]

                    output_path = os.path.join(db.tumor_dist_dir,
                                               scan_name + '_' + str(s) +
                                               '_dist_map_' + str(i) + '.bin')
                    if not os.path.exists(os.path.dirname(output_path)):
                        os.makedirs(os.path.dirname(output_path))
                    dist_map = np.clip(dist_map, 0, 255)
                    dist_map.astype('uint8').tofile(output_path)

    def name(self):
        """Class name reproduction."""
        return "%s(norm_type=%s)" % (type(self).__name__, self.norm_type)
