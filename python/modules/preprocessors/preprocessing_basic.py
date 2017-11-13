"""Class for ISLES 2017 data preprocessing."""

import sys
import os
import json
import numpy as np
from scipy import ndimage
from .. import PreprocessorISLES


class PreprocessorISLESBasic(PreprocessorISLES):
    """Class for ISLES 2017 data preprocessing."""

    """
        Attributes:
            norm_type: normalization type (mean_std or min_max)

            train_norm_params: dictionary for train normalization parameters
            test_norm_params: dictionary for test normalization parameters

            train_align_params: dictionary for train alignment parameters
            test_align_params: dictionary for test alignment parameters

            clip: flag indicating whether to clip values after normalization
            clip_u: upper clip value
            clip_l: lower clip value

        Methods:
            get_normalization_parameters: get normalization parameters
                for the selected database (train or test)
            get_alignment_parameters: get alignment parameters
                for the selected database (train or test)
            normalize: normalization of the input volume
            align: alignment of the input volume
            dealign: dealignment of the input volume
            name: reproduce PreprocessorISLESBasic object's name
    """
    def __init__(self, norm_type,
                 clip=True, clip_l=-2.0, clip_u=2.0):
        """Initialization of PreprocessorISLESBasic attributes."""
        self.norm_type = norm_type

        self.train_norm_params = {}
        self.test_norm_params = {}
        self.train_align_params = {}
        self.test_align_params = {}

        self.clip, self.clip_l, self.clip_u = [clip, clip_l, clip_u]

    def _compute_mean_std_params(self, volume):
        mean_ = np.mean(volume[volume != 0.0])
        std_ = np.std(volume[volume != 0.0])
        return {'mean': float(mean_), 'std': float(std_)}

    def _compute_min_max_params(self, volume):
        max_ = np.max(volume[volume != 0])
        min_ = np.min(volume[volume != 0])
        return {'min': float(min_), 'max': float(max_)}

    def _compute_norm_params(self, volume):
        """Normalization parameters computation."""
        """
            Arguments:
                volume: input volume

            Returns:
                dictionary of computed parameters
        """
        if self.norm_type == 'mean_std':
            return self._compute_mean_std_params(volume)
        if self.norm_type == 'min_max':
            return self._compute_min_max_params(volume)

    def _compute_normalization_parameters(self, db, data_dict):
        n_params = {}
        n_subjects = len(data_dict)
        for s_idx, s in enumerate(data_dict):
            for m in db.modalities:

                volume = data_dict[s].load_volume(db, m)
                volume_norm_params = self._compute_norm_params(volume)

                if s not in n_params:
                    n_params[s] = {}
                if m not in n_params[s]:
                    n_params[s][m] = {}

                for p in volume_norm_params:
                    n_params[s][m][p] = volume_norm_params[p]
            sys.stdout.write("\rNormalization parameters computation: "
                             "%.3f %% / 100 %%" %
                             (100 * float(s_idx + 1) / n_subjects))
            sys.stdout.flush()
        sys.stdout.write("\n")
        return n_params

    def _compute_alignment_parameters(self, db, meta, data_dict):
        r_params = {}
        n_subjects = len(data_dict)
        for s_idx, s in enumerate(data_dict):

            if s not in r_params:
                r_params[s] = {}

            brain_mask = meta.load_brain_mask(db, data_dict[s])
            mask = (np.sum(brain_mask != 0, axis=2) != 0).astype('float32')
            x, y = np.where(mask)
            p = np.polyfit(y, x, 1)
            sum_ = np.sum(mask)

            center_x =\
                np.sum(np.arange(mask.shape[0]) * np.sum(mask, axis=1)) / sum_
            center_y =\
                np.sum(np.arange(mask.shape[1]) * np.sum(mask, axis=0)) / sum_

            angle = np.arctan(p[0])
            rot_matrix = [[0, 0], [0, 0]]
            rot_matrix[0] = [np.cos(angle), np.sin(angle)]
            rot_matrix[1] = [-np.sin(angle), np.cos(angle)]

            r_params[s]['r_matrix'] = rot_matrix
            r_params[s]['r_center'] = [center_x, center_y]

            sys.stdout.write("\rAlignment parameters computation: "
                             "%.3f %% / 100 %%" %
                             (100 * float(s_idx + 1) / n_subjects))
            sys.stdout.flush()
        sys.stdout.write("\n")
        return r_params

    def _load_normalization_parameters(self, params_output_path):
        with open(params_output_path, 'r') as f:
            return json.load(f)

    def _load_alignment_parameters(self, params_output_path):
        with open(params_output_path, 'r') as f:
            return json.load(f)

    def _save_normalization_parameters(self, norm_output_dir, data_dict):

        norm_done_path = os.path.join(norm_output_dir, 'done')
        norm_output_path = os.path.join(norm_output_dir, 'normalizations.json')

        if not os.path.exists(norm_output_dir):
            os.makedirs(norm_output_dir)
        with open(norm_output_path, 'w') as f:
            json.dump(data_dict, f)
        with open(norm_done_path, 'w') as f:
            f.close()

    def _save_alignment_parameters(self, align_output_dir, data_dict):

        align_done_path = os.path.join(align_output_dir, 'done')
        align_output_path = os.path.join(align_output_dir, 'alignments.json')

        if not os.path.exists(align_output_dir):
            os.makedirs(align_output_dir)
        with open(align_output_path, 'w') as f:
            json.dump(data_dict, f)
        with open(align_done_path, 'w') as f:
            f.close()

    def get_normalization_parameters(self, db, exp_out, mode):
        """Getting a dictionary of the normalization parameters."""
        """
            Arguments:
                db: DatabaseISLES
                exp_out: path to the experiment meta output
                mode: training or validation subsets
        """
        if mode == 'train':
            data_dict = db.train_dict
        elif mode == 'test':
            data_dict = db.test_dict

        out_path = os.path.join(exp_out, 'normalization', self.name(), mode)
        done_path = os.path.join(out_path, 'done')

        if os.path.exists(done_path):
            params_out_path = os.path.join(out_path, 'normalizations.json')
            n_params = self._load_normalization_parameters(params_out_path)
        else:
            n_params = self._compute_normalization_parameters(db, data_dict)
            self._save_normalization_parameters(out_path, n_params)

        self.train_norm_params = n_params

    def get_alignment_parameters(self, db, meta, exp_out, mode):
        """Getting a dictionary of the alignment parameters."""
        """
            Arguments:
                db: DatabaseISLES
                exp_out: path to the experiment meta output
                mode: training or validation subsets
        """
        if mode == 'train':
            data_dict = db.train_dict
        elif mode == 'test':
            data_dict = db.test_dict

        out_path = os.path.join(exp_out, 'alignment', self.name(), mode)
        done_path = os.path.join(out_path, 'done')

        if os.path.exists(done_path):
            params_out_path = os.path.join(out_path, 'alignments.json')
            r_params = self._load_alignment_parameters(params_out_path)
        else:
            r_params = self._compute_alignment_parameters(db, meta, data_dict)
            self._save_alignment_parameters(out_path, r_params)

        self.train_rotate_params = r_params

    def normalize(self, scan, m, x, mask):
        """Normalization of the input array x."""
        """
            Arguments:
                scan: ScanISLES
                m: modality
                x: piece of volume to be normalized
                mask: mask indicating voxels to be normalized
        """
        n_params = self.train_norm_params[scan.name][m]

        if self.norm_type == 'mean_std':
            return mask * (x - n_params['mean']) / n_params['std']
        if self.norm_type == 'min_max':
            return mask * (x - n_params['min']) / (n_params['max'] -
                                                   n_params['max'])

    def align(self, db, scan, m, volume):
        """Alignment of the input scan."""
        """
            Arguments:
                db: DatabaseISLES object
                scan: ScanISLES object
                m: modality
                volume: input volume
            Returns:
                aligned volume
        """
        r_params = self.train_rotate_params[scan.name]
        rot_matrix = np.asarray(r_params['r_matrix'])
        r_center = np.copy(r_params['r_center'])

        r_center[0] *= float(volume.shape[0]) / scan.h
        r_center[1] *= float(volume.shape[1]) / scan.w
        volume_c = np.zeros(volume.shape)

        i_s, j_s = [r_center[0] - volume.shape[0] / 2,
                    r_center[1] - volume.shape[1] / 2]

        for k in range(volume.shape[2]):
            volume_c[:, :, k] =\
                ndimage.interpolation.affine_transform(volume[:, :, k],
                                                       rot_matrix,
                                                       [i_s, j_s])
        return volume_c

    def dealign(self, db, scan, m, volume):
        """De-alignment of the input scan."""
        """
            Arguments:
                db: DatabaseISLES object
                scan: ScalISLES object
                m: modality
                volume: input volume
            Returns:
                de-aligned volume
        """
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

    def name(self):
        """Class name reproduction."""
        """
            Returns:
                PreprocessingISLES object's name
        """
        if not self.clip:
            return "%s(norm_type=%s)" % (type(self).__name__, self.norm_type)
        else:
            return ("%s(norm_type=%s, clip_l=%s, clip_u=%s)"
                    % (type(self).__name__, self.norm_type,
                       self.clip_l, self.clip_u))
