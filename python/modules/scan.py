"""Class for ISLES 2017 scan/sample info loading and generating."""
import os
import nibabel as nib
import numpy as np
from scipy.ndimage import morphology
import cv2
from scipy.ndimage import interpolation
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class ScanISLES(object):
    """Class for ISLES 2017 scan info loading and generating."""

    def __init__(self, db, name, relative_path, modalities_dict, mode):
        """Initialization of ScanISLES attributes."""
        self.name = name
        self.relative_path = relative_path
        self.modalities_dict = modalities_dict
        self.mode = mode
        self.h = None
        self.w = None
        self.d = None

        _ = self.load_volume(db, 'ADC')

    def load_volume(self, db, m):
        """Loading volume as numpy array."""
        """
            Arguments:
                db: DatabaseISLES
                m: image modality
            Returns:
                volume as numpy array
        """
        volume_path = os.path.join(db.db_path, self.relative_path,
                                   self.modalities_dict[m],
                                   self.modalities_dict[m] + '.nii')

        volume = nib.load(volume_path).get_data().astype('float32')
        self.h, self.w, self.d = volume.shape
        return volume

    def load_volumes(self, db, test=False):
        """Loading all volumes as a list numpy arrays."""
        """
            Arguments:
                db: DatabaseBRATS
            Returns:
                list of volumes
        """
        if test:
            volumes = [self.load_volume(db, m) for m in db.modalities[:-1]]
        else:
            volumes = [self.load_volume(db, m) for m in db.modalities]
        volumes.append(self.load_brain_mask(db.brain_masks_dir))
        return volumes

    def load_volumes_norm_aligned(self, db, size, orient=0, test=False):

        volumes = []
        for m in db.modalities[:-1]:
            volume_path = os.path.join(db.aug_data_dir,
                                       self.name + '_' + str(size) +
                                       '_' + m + '_' + str(orient) + '.bin')
            v = np.reshape(np.fromfile(volume_path, dtype='float32'),
                           [size, size, -1])
            volumes.append(v)

        if not test:
            volume_path = os.path.join(db.aug_data_dir,
                                       self.name + '_' + str(size) +
                                       '_OT_' + str(orient) + '.bin')
            v = np.reshape(np.fromfile(volume_path, dtype='uint8'),
                           [size, size, -1])
            volumes.append(v)

        volume_path = os.path.join(db.aug_data_dir,
                                   self.name + '_' + str(size) +
                                   '_brain_mask_' + str(orient) + '.bin')
        v = np.reshape(np.fromfile(volume_path, dtype='uint8'),
                       [size, size, -1])
        volumes.append(v)

        if not test:
            volume_path = os.path.join(db.tumor_dist_dir,
                                       self.name + '_' + str(size) +
                                       '_dist_map_' + str(orient) + '.bin')
            v = np.reshape(np.fromfile(volume_path, dtype='uint8'),
                           [size, size, -1])
            volumes.append(v)

        return volumes

    def load_brain_mask(self, exp_out):
        """Loading brain mask as numpy array."""
        """
            Arguments:
                exp_out: experiment output for meta data
            Returns:
                brain mask as a numpy array
        """
        brain_mask_path = os.path.join(exp_out,
                                       self.name + '_brain_mask.bin')

        np_array = np.fromfile(brain_mask_path, dtype='uint8')

        return np.reshape(np_array, (self.h, self.w, self.d))

    def load_tumor_dist_maps(self, exp_out, s):
        """Loading brain mask as numpy array."""
        """
            Arguments:
                exp_out: experiment output for meta data
            Returns:
                brain mask as a numpy array
        """
        tdm_path = os.path.join(exp_out, self.name + '_tumor_dist.bin_' + str(s))
        if os.path.exists(tdm_path):
            np_array = np.fromfile(tdm_path, dtype='uint8')
            return np.reshape(np_array, (s, s, self.d))
        else:
            return np.zeros((s, s, self.d))

    def _compute_and_save_brain_mask(self, db, brain_mask_path):

        for m in db.modalities[:-1]:
            volume = self.load_volume(db, m)
            if 'bm' in locals():
                bm = np.logical_or(bm, volume != 0)
            else:
                bm = volume != 0
        bm.tofile(brain_mask_path)

    def compute_brain_mask(self, db, exp_out):
        """Compute brain mask."""
        """
            Arguments:
                db: DatabaseBRATS
                exp_out: experiment output for meta data
        """
        bm_path = os.path.join(exp_out, self.name + '_brain_mask.bin')
        self._compute_and_save_brain_mask(db, bm_path)

    def _compute_and_save_tumor_distance_map(self, db, brain_mask,
                                             tumor_distance_map_path):

        v_seg = self.load_volume(db, 'OT')
        struct_elem = np.ones((3, 3, 3))
        for s in db.sizes:
            if v_seg.shape[0] == s:
                mask = v_seg
                brain = brain_mask
            else:
                mask =\
                    interpolation.zoom(v_seg,
                                       [float(s) / self.h,
                                        float(s) / self.w,
                                        1]) >= 0.5
                brain =\
                    interpolation.zoom(brain_mask,
                                       [float(s) / self.h,
                                        float(s) / self.w,
                                        1]) >= 0.5
            v = 1.0
            tumor_dist_map = np.zeros((s, s, self.d))
            while(np.sum(mask) != s * s * self.d):
                mask_d = morphology.binary_dilation(mask, struct_elem)
                tumor_dist_map += (mask_d - mask) * v
                mask = np.copy(mask_d)
                v += 1

            tumor_dist_map *= brain
            tumor_dist_map = np.clip(tumor_dist_map, a_min=0.0, a_max=255.0)
            tumor_dist_map.astype('uint8').\
                tofile(tumor_distance_map_path + '_' + str(s))

    def compute_tumor_distance_map(self, db, bm_exp_out, tdm_exp_out):
        """Tumor distance map computation."""
        """
            Arguments:
                db: DatabaseBRATS
                exp_out: experiment output for meta data
        """
        tdm_path = os.path.join(tdm_exp_out, self.name + '_tumor_dist.bin')

        brain_mask = self.load_brain_mask(bm_exp_out)
        self._compute_and_save_tumor_distance_map(db, brain_mask, tdm_path)
