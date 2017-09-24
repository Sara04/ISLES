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
    db.load_training_dict()
    # _______________________________________________________________________ #

    # 3. Computing brain masks
    # _______________________________________________________________________ #
    print "Computing brain masks..."
    bm_output_path = os.path.join(args.exp_out, 'brain_masks', 'train')
    db.brain_masks_dir = bm_output_path
    done_path = os.path.join(bm_output_path, 'done')
    if not os.path.exists(done_path):
        if not os.path.exists(bm_output_path):
            os.makedirs(bm_output_path)
        for s in db.train_dict:
            db.train_dict[s].compute_brain_mask(db, bm_output_path)
        with open(done_path, 'w') as f:
            f.close()
    else:
        print "Brain masks already computed"

    # _______________________________________________________________________ #

    # 4. Computing pre-processing parameters
    # _______________________________________________________________________ #
    prep.get_preprocessing_parameters(db, args.exp_out, 'train')
    prep.get_rotation_parameters(db, args.exp_out, 'train')
    # _______________________________________________________________________ #
    # 5. Preprocessing and augmentation
    # _______________________________________________________________________ #
    augment_output_path = os.path.join(args.exp_out, 'augmented', 'train')
    db.aug_data_dir = augment_output_path
    done_path = os.path.join(augment_output_path, 'done')
    if not os.path.exists(done_path):
        aug.augment(db, prep)
        with open(done_path, 'w') as f:
            f.close()
    else:
        print "Data already augmented"

    # _______________________________________________________________________ #
    # 6. Computing tumor distance maps
    # _______________________________________________________________________ #
    print "Computing tumor distance maps..."
    tdm_output_path = os.path.join(args.exp_out, 'tumor_dist_maps', 'train')
    db.tumor_dist_dir = tdm_output_path
    done_path = os.path.join(tdm_output_path, 'done')
    if not os.path.exists(done_path):
        prep.compute_tumor_distance_maps(db)
        with open(done_path, 'w') as f:
            f.close()
    else:
        print "Tumor distance maps already computed"

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
        if not os.path.exists(done_path):
            for it in range(restore_it + 1, seg.train_valid_iters):

                train_data_r1, train_data_r2 =\
                    patch_ex.extract_train_data(db, prep, seg, args.exp_out)

                if train_data_r1:
                    seg.train(train_data_r1, train_data_r2)
                if it % 10 == 0:
                    train_data_r1, train_data_r2 =\
                        patch_ex.extract_train_data(db, prep, seg, args.exp_out, train=False)
                    valid_data_r1, valid_data_r2 =\
                        patch_ex.extract_valid_data(db, prep, seg, args.exp_out)
                    if train_data_r1 and valid_data_r1:
                        seg.train_and_valid(train_data_r1, train_data_r2,
                                            valid_data_r1, valid_data_r2)
                    seg.save_model(seg_path, it)

        # _______________________________________________________________________ #

        # 6. Segmentation of validation scans
        # _______________________________________________________________________ #
        print "CNN validation..."
        seg_results_path = os.path.join(args.exp_out, 'segmentations', 'train', 'split_' + str(split))
        done_path = os.path.join(seg_results_path, 'done')
        db.seg_results_dir = seg_results_path
        if not os.path.exists(done_path):
            seg.evaluate_dice(db, prep, patch_ex, seg_results_path)
            with open(done_path, 'w') as f:
                f.close()

    # _______________________________________________________________________ #

    # 7. Preprocessing parameters
    # _______________________________________________________________________ #

    post.determine_parameters(db, prep)

if __name__ == '__main__':
    main()
