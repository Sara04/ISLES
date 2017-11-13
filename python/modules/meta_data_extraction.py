"""Class for ISLES 2017 meta data extraction."""

import os
import sys
import numpy as np
from scipy.ndimage import morphology


class MetaDataExtractorISLES(object):
    """Class for ISLES 2017 meta data extraction."""

    """
        Methods:
            compute_brain_masks: compute and save brain masks for
                the selected dataset (train or test)
            load_brain_masks: loading brain mask for a given scan
            compute_lesion_distance_maps: compute and save
                lesion distance maps for the selected dataset
                (train or test)
            load_lesion_dist_map: loading lesion distance map for
                a given scan
            compute_normalized_volumes: compute and save
                normalized volumes
            load_volumes_norm_aligned: loading normalized and
                aligned volumes of all modalities
    """

    def _compute_and_save_brain_mask(self, scan, db):

        bm_path = os.path.join(db.brain_masks_dir,
                               scan.name + '_brain_mask.bin')
        bm = np.ones(scan.h, scan.w, scan.d)
        for m in db.modalities:
            v = scan.load_volume(db, m)
            bm *= (v != 0)
        bm.tofile(bm_path)

    def compute_brain_masks(self, db, exp_out, mode):
        """Compute and save brain masks."""
        """
            Arguments:
                db: DatabaseISLES object
                exp_out: path to the experiment meta output
                mode: train, valid or test database
        """
        if mode == 'train':
            data_dict = db.train_dict
        elif mode == 'test':
            data_dict = db.test_dict

        db.brain_masks_dir = os.path.join(exp_out, 'brain_masks', mode)
        if not os.path.exists(os.path.join(db.brain_masks_dir, 'done')):
            n_subjects = len(data_dict)
            if not os.path.exists(db.brain_masks_dir):
                os.makedirs(db.brain_masks_dir)
            for s_idx, s in enumerate(data_dict):
                self._compute_and_save_brain_mask(data_dict[s], db)
                sys.stdout.write("\rComputing and saving brain masks: "
                                 "%.3f %% / 100 %%" %
                                 (100 * float(s_idx + 1) / n_subjects))
                sys.stdout.flush()
            sys.stdout.write("\n")
            with open(os.path.join(db.brain_masks_dir, 'done'), 'w') as f:
                f.close()
        else:
            print "Brain masks already computed"

    def load_brain_mask(self, db, scan):
        """Load brain mask."""
        """
            Arguments:
                db: DatabaseISLES object
                scan: ScanISLES object
        """
        brain_mask_path = os.path.join(db.brain_masks_dir,
                                       scan.name + '_brain_mask.bin')
        brain_mask = np.fromfile(brain_mask_path, dtype='uint8')

        return np.reshape(brain_mask, [scan.h, scan.w, scan.d])

    def _compute_and_save_lesion_distance_maps(self, scan, db):
        for s in db.sizes:
            for i in range(4):
                gt_path = os.path.join(db.augmented_volumes_dir,
                                       scan.name + '_' + str(s) +
                                       '_OT_' + str(i) + '.bin')
                bm_path = os.path.join(db.augmented_volumes_dir,
                                       scan.name + '_' + str(s) +
                                       '_brain_mask_' + str(i) + '.bin')

                ldm_path = os.path.join(db.lesion_dist_dir,
                                        scan.name + '_' + str(s) +
                                        '_lesion_dist_map_' + str(i) + '.bin')

                brain_mask = np.reshape(np.fromfile(bm_path, dtype='uint8'),
                                        [s, s, -1])
                v_seg = np.reshape(np.fromfile(gt_path, dtype='uint8'),
                                   [s, s, -1])

                seg_mask = v_seg != 0
                struct_elem = np.ones((3, 3, 3))
                v = 1.0

                lesion_dist_map = np.zeros((s, s, scan.d))
                while(np.sum(seg_mask) != s * s * scan.d):
                    seg_mask_d =\
                        morphology.binary_dilation(seg_mask, struct_elem)
                    lesion_dist_map += (seg_mask_d - seg_mask) * v
                    seg_mask = np.copy(seg_mask_d)
                    v += 1

                lesion_dist_map *= brain_mask
                lesion_dist_map =\
                    np.clip(lesion_dist_map, a_min=0.0, a_max=255.0)
                lesion_dist_map.astype('uint8').tofile(ldm_path)

    def compute_lesion_distance_maps(self, db, exp_out):
        """Compute and save tumor distance maps."""
        """
            Arguments:
                db: DatabaseISLES object
                exp_out: path to the experiment meta output
        """
        db.lesion_dist_dir = os.path.join(exp_out, 'lesion_dist_maps', 'train')
        if not os.path.exists(os.path.join(db.lesion_dist_dir, 'done')):
            n_subjects = len(db.train_dict)
            if not os.path.exists(db.lesion_dist_dir):
                os.makedirs(db.lesion_dist_dir)
            for s_idx, s in enumerate(db.train_dict):
                self._compute_and_save_lesion_distance_maps(db.train_dict[s], db)
                sys.stdout.write("\rComputing and saving lesion distance maps: "
                                 "%.3f %% / 100 %%" %
                                 (100 * float(s_idx + 1) / n_subjects))
                sys.stdout.flush()
            sys.stdout.write("\n")
            with open(os.path.join(db.lesion_dist_dir, 'done'), 'w') as f:
                f.close()
        else:
            print "Lesion distance maps already computed"

    def load_lesion_dist_map(self, exp_out, scan, s):
        """Loading brain mask as numpy array."""
        """
            Arguments:
                exp_out: experiment output for meta data
            Returns:
                brain mask as a numpy array
        """
        tdm_path = os.path.join(exp_out,
                                scan.name + '_lesion_dist.bin_' + str(s))
        if os.path.exists(tdm_path):
            np_array = np.fromfile(tdm_path, dtype='uint8')
            return np.reshape(np_array, (s, s, scan.d))
        else:
            return np.zeros((s, s, scan.d))

    def _normalize_volumes(self, scan, db, prep):
        n_volumes = prep.normalize_volumes(db, scan)
        for m in db.modalities:
            n_volume_path = os.path.join(db.norm_volumes_dir,
                                         scan.name,
                                         scan.name + '_' + m + '.bin')
            if not os.path.exists(os.path.dirname(n_volume_path)):
                os.makedirs(os.path.dirname(n_volume_path))
            n_volumes[m].tofile(n_volume_path)

    def compute_normalized_volumes(self, db, prep, exp_out, mode):
        """Compute and save normalized volumes."""
        """
            Arguments:
                db: DatabaseISLES object
                prep: PreprocessorBRATS object
                exp_out: path to the experiment meta data output
                mode: train, valid or test database
        """
        if mode == 'train':
            data_dict = db.train_dict
        elif mode == 'test':
            data_dict = db.test_dict

        db.norm_volumes_dir = os.path.join(exp_out,
                                           'normalized_volumes', mode)
        if not os.path.exists(os.path.join(db.norm_volumes_dir, 'done')):
            n_subjects = len(data_dict)
            if not os.path.exists(db.norm_volumes_dir):
                os.makedirs(db.norm_volumes_dir)
            for s_idx, s in enumerate(data_dict):
                self._normalize_volumes(data_dict[s], db, prep)
                sys.stdout.write("\rComputing and saving normalized volumes: "
                                 "%.3f %% / 100 %%" %
                                 (100 * float(s_idx + 1) / n_subjects))
                sys.stdout.flush()
            sys.stdout.write("\n")

            with open(os.path.join(db.norm_volumes_dir, 'done'), 'w') as f:
                f.close()
        else:
            print "Volumes already normalized"

    def load_volumes_norm_aligned(self, db, scan, size, orient=0, test=False):
        """Loading normalized and aligned volumes."""
        """
            Arguments:
                db: DatabaseISLES object
                scan: ScanISLES object
                size: scan's slices' width and height (128, 192 or 256)
                orient: selected orientation of volume
                test: in case of test ground truth and lesion distance map
                    are not loaded
            Returns:
                list of volumes
        """
        volumes = []
        for m in db.modalities:
            volume_path = os.path.join(db.augmented_volumes_dir,
                                       scan.name + '_' + str(size) +
                                       '_' + m + '_' + str(orient) + '.bin')
            v = np.reshape(np.fromfile(volume_path, dtype='float32'),
                           [size, size, -1])
            volumes.append(v)

        if not test:
            volume_path = os.path.join(db.augmented_volumes_dir,
                                       scan.name + '_' + str(size) +
                                       '_OT_' + str(orient) + '.bin')
            v = np.reshape(np.fromfile(volume_path, dtype='uint8'),
                           [size, size, -1])
            volumes.append(v)

        volume_path = os.path.join(db.augmented_volumes_dir,
                                   scan.name + '_' + str(size) +
                                   '_brain_mask_' + str(orient) + '.bin')
        v = np.reshape(np.fromfile(volume_path, dtype='uint8'),
                       [size, size, -1])
        volumes.append(v)

        if not test:
            volume_path = os.path.join(db.lesion_dist_dir,
                                       scan.name + '_' + str(size) +
                                       '_lesion_dist_map_' +
                                       str(orient) + '.bin')
            v = np.reshape(np.fromfile(volume_path, dtype='uint8'),
                           [size, size, -1])
            volumes.append(v)

        return volumes
