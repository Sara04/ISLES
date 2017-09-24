"""Toolchain fro segmentation training and validation."""
import argparse
import sys
import os

from modules.database import DatabaseISLES
from modules.preprocessing import PreprocessorISLES
from modules.augmentation import AugmentatorISLES
from modules.patch_extraction import PatchExtractorISLES
from modules.segmentation import SegmentatorISLES
from modules.postprocessing import PostprocessorISLES
import cv2
import numpy as np
import time

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def main():
    """Function that runs segmentation training and validation."""
    # _______________________________________________________________________ #
    parser = argparse.ArgumentParser(description=''
                                     'Ischemic Stroke Lesion Segmentation')

    parser.add_argument('-i', dest='db_path', required=True,
                        help='Path to the directory where the training and '
                             'validation are placed.')
    parser.add_argument('-o', dest='exp_out', required=True,
                        help='Path where the intermediate and the final '
                             'results would be placed.')
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        print "\nInput database path does not exist!\n"
        sys.exit(1)

    if not os.path.exists(args.exp_out):
        os.mkdir(args.exp_out)

    # 1. Loading toolchain modules
    # _______________________________________________________________________ #
    db = DatabaseISLES(args.db_path)
    prep = PreprocessorISLES('mean_std')
    aug = AugmentatorISLES()
    patch_ex = PatchExtractorISLES()
    seg = SegmentatorISLES()
    post = PostprocessorISLES()
    # _______________________________________________________________________ #

    # 2. Loading training and validation lists
    # _______________________________________________________________________ #
    db.load_testing_dict()
    # _______________________________________________________________________ #

    # 3. Computing brain masks
    # _______________________________________________________________________ #
    print "Computing brain masks..."
    bm_output_path = os.path.join(args.exp_out, 'brain_masks', 'test')
    db.brain_masks_dir = bm_output_path
    done_path = os.path.join(bm_output_path, 'done')
    if not os.path.exists(done_path):
        if not os.path.exists(bm_output_path):
            os.makedirs(bm_output_path)
        for s in db.test_dict:
            db.test_dict[s].compute_brain_mask(db, bm_output_path)
        with open(done_path, 'w') as f:
            f.close()
    else:
        print "Brain masks already computed"

    # _______________________________________________________________________ #

    # 4. Computing pre-processing parameters
    # _______________________________________________________________________ #
    prep.get_preprocessing_parameters(db, args.exp_out, 'test')
    prep.get_rotation_parameters(db, args.exp_out, 'test')
    # _______________________________________________________________________ #
    # 5. Preprocessing and augmentation
    # _______________________________________________________________________ #
    augment_output_path = os.path.join(args.exp_out, 'augmented', 'test')
    db.aug_data_dir = augment_output_path
    done_path = os.path.join(augment_output_path, 'done')
    if not os.path.exists(done_path):
        aug.augment_test(db, prep)
        with open(done_path, 'w') as f:
            f.close()
    else:
        print "Data already augmented"

    # _______________________________________________________________________ #

    # 6. Segmentator training and validation
    # _______________________________________________________________________ #

    print "CNN training and validation..."
    for split in range(0, 1):
        db.train_valid_split(split)

        seg_path = os.path.join(args.exp_out, 'segmentators', 'split_' + str(split))
        done_path = os.path.join(seg_path, 'done')
        restore = True
        restore_it = 3100
        if restore:
            seg.restore_model(seg_path, restore_it)

        # _______________________________________________________________________ #

        # 6. Segmentation of validation scans
        # _______________________________________________________________________ #
        print "CNN validation..."
        seg_results_path = os.path.join(args.exp_out, 'segmentations', 'test')
        done_path = os.path.join(seg_results_path, 'done')
        db.seg_results_dir = seg_results_path
        if not os.path.exists(done_path):
            seg.evaluate_dice_test(db, prep, patch_ex, seg_results_path)
            with open(done_path, 'w') as f:
                f.close()

    # _______________________________________________________________________ #

    # 7. Preprocessing parameters
    # _______________________________________________________________________ #

    db.seg_final_results_dir = os.path.join(args.exp_out, 'segmentations_final', 'test')
    post.postprocess_test(db, prep)

if __name__ == '__main__':
    main()
