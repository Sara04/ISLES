"""Class for ISLES 2017 data augmentation."""

import sys

import numpy as np
import os
from scipy.misc import imresize


class AugmentatorISLES(object):
    """Class for ISLES 2017 data preprocessing."""

    def _mirror_flip_and_save(self, volume, output_path):

        volume.tofile(output_path.format(0))
        volume[::-1, :, :].tofile(output_path.format(1))
        volume[:, ::-1, :].tofile(output_path.format(2))
        volume[::-1, ::-1, :].tofile(output_path.format(3))

    def augment_data(self, db, meta, exp_out, prep, mode):
        """Method for data augmentation."""
        """
            Arguments:
                db: DatabaseISLES object
                meta: MetaDataExtractorISLES object
                exp_out: experiment output
                prep: PreprocessorISLES object
                mode: selected mode train/test
        """
        if mode == 'train':
            data_dict = db.train_dict
        elif mode == 'test':
            data_dict = db.test_dict

        db.augmented_volumes_dir =\
            os.path.join(exp_out, 'augmented_data', mode)

        if not os.path.exists(os.path.join(db.augmented_volumes_dir, 'done')):
            n_subjects = len(data_dict)
            for s_idx, scan_name in enumerate(data_dict):
                scan = data_dict[scan_name]
                if mode == 'test':
                    volumes = scan.load_volumes(db, meta, True)
                else:
                    volumes = scan.load_volumes(db, meta)

                volumes_r = []
                for m_idx, m in enumerate(db.modalities):
                    volumes[m_idx] = prep.normalize(scan, m,
                                                    volumes[m_idx],
                                                    volumes[-1])
                    volumes_r.append(prep.align(db, scan, m, volumes[m_idx]))

                if mode == 'train':
                    volumes_r.append(prep.align(db, scan, m, volumes[-2]))
                volumes_r.append(prep.align(db, scan, m, volumes[-1]))

                for s in db.sizes:
                    volumes_a = []
                    if s == scan.h:
                        for v_idx, v in enumerate(volumes_r):
                            if v_idx >= len(db.modalities):
                                volumes_a.append((v > 0.5).astype('uint8'))
                            else:
                                volumes_a.append(v.astype('float32'))
                    else:
                        for i in range(len(volumes_r)):
                            vr = np.zeros((s, s, scan.d), dtype='float32')
                            for j in range(scan.d):
                                vr[:, :, j] = imresize(volumes_r[i][:, :, j],
                                                       (s, s), mode='F')
                            if i >= len(db.modalities):
                                volumes_a.append((vr > 0.3).astype('uint8'))
                            else:
                                volumes_a.append(vr.astype('float32'))

                    for m_idx, m in enumerate(db.modalities):
                        output_path = os.path.join(db.augmented_volumes_dir,
                                                   scan_name + '_' + str(s) +
                                                   '_' + m + '_' + '{0}' + '.bin')
                        if not os.path.exists(os.path.dirname(output_path)):
                            os.makedirs(os.path.dirname(output_path))
                        self._mirror_flip_and_save(volumes_a[m_idx], output_path)

                    if mode == 'train':
                        output_path = os.path.join(db.augmented_volumes_dir,
                                                   scan_name + '_' + str(s) +
                                                   '_OT_' + '{0}' + '.bin')
                        if not os.path.exists(os.path.dirname(output_path)):
                            os.makedirs(os.path.dirname(output_path))
                        self._mirror_flip_and_save(volumes_a[-2], output_path)

                    output_path = os.path.join(db.augmented_volumes_dir,
                                               scan_name + '_' + str(s) +
                                               '_brain_mask_' + '{0}' + '.bin')
                    self._mirror_flip_and_save(volumes_a[-1], output_path)

                sys.stdout.write("\rData augmentation: %.3f %% / 100 %%" %
                                 (100 * float(s_idx + 1) / n_subjects))
                sys.stdout.flush()
            sys.stdout.write("\n")
            with open(os.path.join(db.augmented_volumes_dir, 'done'), 'w') as f:
                f.close()
        else:
            print "Data already augmented"
