"""Segmentation class."""
import tensorflow as tf
import os
import cv2
import numpy as np

from .. import SegmentatorISLES


def _weight_variable(shape):

    initial = tf.get_variable("W", shape,
                              initializer=tf.contrib.layers.xavier_initializer())
    return initial


def _bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def _conv2d(x, weights, strides=[1, 1, 1, 1], padding_mode='VALID'):
    return tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1],
                        padding=padding_mode)


def _max_pool_2x2(x, ksize=[1, 2, 2, 1]):
    return tf.nn.max_pool(x, ksize=ksize, strides=ksize, padding='VALID')


class CnnISLES2(SegmentatorISLES):
    """Segmentation class."""

    # restore_it=206
    def __init__(self, lr, lw, kp=0.5, restore=False, restore_it=0,
                 train_iters=500,
                 lp_w=25, lp_h=25, lp_d=14,
                 mp_w=15, mp_h=15, mp_d=12,
                 sp_w=7, sp_h=7, sp_d=6):
        """Class initialization."""
        self.restore, self.restore_it = [restore, restore_it]
        self.train_iters = train_iters
        self.lr = lr
        self.lw = lw
        self.kp = kp
        self.lp_w, self.lp_h, self.lp_d = [lp_w, lp_h, lp_d]
        self.mp_w, self.mp_h, self.mp_d = [mp_w, mp_h, mp_d]
        self.sp_w, self.sp_h, self.sp_d = [sp_w, sp_h, sp_d]

        self.lp_x_r1 =\
            tf.placeholder(tf.float32,
                           shape=[None, self.lp_h * self.lp_w * self.lp_d])
        self.mp_x_r1 =\
            tf.placeholder(tf.float32,
                           shape=[None, self.mp_h * self.mp_w * self.mp_d])
        self.sp_x_r1 =\
            tf.placeholder(tf.float32,
                           shape=[None, self.sp_h * self.sp_w * self.sp_d])

        self.gt_r1 = tf.placeholder(tf.float32, shape=[None, 2])

        self.lp_x_r2 =\
            tf.placeholder(tf.float32,
                           shape=[None, self.lp_h * self.lp_w * self.lp_d])
        self.mp_x_r2 =\
            tf.placeholder(tf.float32,
                           shape=[None, self.mp_h * self.mp_w * self.mp_d])
        self.sp_x_r2 =\
            tf.placeholder(tf.float32,
                           shape=[None, self.sp_h * self.sp_w * self.sp_d])
        self.gt_r2 = tf.placeholder(tf.float32, shape=[None, 2])

        self.keep_prob = tf.placeholder(tf.float32)

        self.sess = tf.Session()

        lp_imgs_r1 = tf.reshape(self.lp_x_r1,
                                [-1, self.lp_h, self.lp_w, self.lp_d])
        mp_imgs_r1 = tf.reshape(self.mp_x_r1,
                                [-1, self.mp_h, self.mp_w, self.mp_d])
        sp_imgs_r1 = tf.reshape(self.sp_x_r1,
                                [-1, self.sp_h, self.sp_w, self.sp_d])

        lp_imgs_r2 = tf.reshape(self.lp_x_r2,
                                [-1, self.lp_h, self.lp_w, self.lp_d])
        mp_imgs_r2 = tf.reshape(self.mp_x_r2,
                                [-1, self.mp_h, self.mp_w, self.mp_d])
        sp_imgs_r2 = tf.reshape(self.sp_x_r2,
                                [-1, self.sp_h, self.sp_w, self.sp_d])

        with tf.variable_scope('l_patches'):
            with tf.variable_scope('layer_1'):
                lp_w_conv1 = _weight_variable([3, 3, 14, 16])
                lp_b_conv1 = _bias_variable([16])
                lp_h_conv1_r1 = tf.nn.relu(_conv2d(lp_imgs_r1, lp_w_conv1) +
                                           lp_b_conv1)
                lp_h_conv1_r2 = tf.nn.relu(_conv2d(lp_imgs_r2, lp_w_conv1) +
                                           lp_b_conv1)
            with tf.variable_scope('layer_2'):
                lp_w_conv2 = _weight_variable([3, 3, 16, 32])
                lp_b_conv2 = _bias_variable([32])
                lp_h_conv2_r1 = tf.nn.relu(_conv2d(lp_h_conv1_r1, lp_w_conv2) +
                                           lp_b_conv2)
                lp_h_conv2_r2 = tf.nn.relu(_conv2d(lp_h_conv1_r2, lp_w_conv2) +
                                           lp_b_conv2)
                lp_h_pool2_r1 = _max_pool_2x2(lp_h_conv2_r1)
                lp_h_pool2_r2 = _max_pool_2x2(lp_h_conv2_r2)
            with tf.variable_scope('layer_3'):
                lp_w_conv3 = _weight_variable([3, 3, 32, 64])
                lp_b_conv3 = _bias_variable([64])
                lp_h_conv3_r1 = tf.nn.relu(_conv2d(lp_h_pool2_r1, lp_w_conv3) +
                                           lp_b_conv3)
                lp_h_conv3_r2 = tf.nn.relu(_conv2d(lp_h_pool2_r2, lp_w_conv3) +
                                           lp_b_conv3)
                lp_h_pool3_r1 = _max_pool_2x2(lp_h_conv3_r1)
                lp_h_pool3_r2 = _max_pool_2x2(lp_h_conv3_r2)
            with tf.variable_scope('layer_4'):
                lp_w_conv4 = _weight_variable([3, 3, 64, 128])
                lp_b_conv4 = _bias_variable([128])
                lp_h_conv4_r1 = tf.nn.relu(_conv2d(lp_h_pool3_r1, lp_w_conv4) +
                                           lp_b_conv4)
                lp_h_conv4_r2 = tf.nn.relu(_conv2d(lp_h_pool3_r2, lp_w_conv4) +
                                           lp_b_conv4)
                lp_h_conv4_flat_r1 =\
                    tf.reshape(lp_h_conv4_r1, [-1, 2 * 2 * 128])
                lp_h_conv4_flat_r2 =\
                    tf.reshape(lp_h_conv4_r2, [-1, 2 * 2 * 128])
            with tf.variable_scope('layer_5'):
                lp_w_fcn1 = _weight_variable([2 * 2 * 128, 64])
                lp_b_fcn1 = _bias_variable([64])
                lp_h_fcn1_r1 =\
                    tf.nn.dropout(tf.nn.relu(tf.matmul(lp_h_conv4_flat_r1,
                                                       lp_w_fcn1) +
                                  lp_b_fcn1), self.keep_prob)
                lp_h_fcn1_r2 =\
                    tf.nn.dropout(tf.nn.relu(tf.matmul(lp_h_conv4_flat_r2,
                                                       lp_w_fcn1) +
                                  lp_b_fcn1), self.keep_prob)

            with tf.variable_scope('layer_6'):
                lp_w_fcn2 = _weight_variable([64, 32])
                lp_b_fcn2 = _bias_variable([32])
                lp_h_fcn2_r1 = tf.nn.relu(tf.matmul(lp_h_fcn1_r1, lp_w_fcn2) +
                                          lp_b_fcn2)
                lp_h_fcn2_r2 = tf.nn.relu(tf.matmul(lp_h_fcn1_r2, lp_w_fcn2) +
                                          lp_b_fcn2)
        with tf.variable_scope('m_patches'):
            with tf.variable_scope('layer_1'):
                mp_w_conv1 = _weight_variable([3, 3, 12, 16])
                mp_b_conv1 = _bias_variable([16])
                mp_h_conv1_r1 = tf.nn.relu(_conv2d(mp_imgs_r1, mp_w_conv1) +
                                           mp_b_conv1)
                mp_h_conv1_r2 = tf.nn.relu(_conv2d(mp_imgs_r2, mp_w_conv1) +
                                           mp_b_conv1)
            with tf.variable_scope('layer_2'):
                mp_w_conv2 = _weight_variable([3, 3, 16, 32])
                mp_b_conv2 = _bias_variable([32])
                mp_h_conv2_r1 = tf.nn.relu(_conv2d(mp_h_conv1_r1, mp_w_conv2) +
                                           mp_b_conv2)
                mp_h_conv2_r2 = tf.nn.relu(_conv2d(mp_h_conv1_r2, mp_w_conv2) +
                                           mp_b_conv2)
                mp_h_pool2_r1 = _max_pool_2x2(mp_h_conv2_r1)
                mp_h_pool2_r2 = _max_pool_2x2(mp_h_conv2_r2)
            with tf.variable_scope('layer_3'):
                mp_w_conv2 = _weight_variable([3, 3, 32, 64])
                mp_b_conv2 = _bias_variable([64])
                mp_h_conv2_r1 = tf.nn.relu(_conv2d(mp_h_pool2_r1, mp_w_conv2) +
                                           mp_b_conv2)
                mp_h_conv2_r2 = tf.nn.relu(_conv2d(mp_h_pool2_r2, mp_w_conv2) +
                                           mp_b_conv2)
                mp_h_conv2_flat_r1 =\
                    tf.reshape(mp_h_conv2_r1, [-1, 3 * 3 * 64])
                mp_h_conv2_flat_r2 =\
                    tf.reshape(mp_h_conv2_r2, [-1, 3 * 3 * 64])
            with tf.variable_scope('layer_4'):
                mp_w_fcn1 = _weight_variable([3 * 3 * 64, 64])
                mp_b_fcn1 = _bias_variable([64])
                mp_h_fcn1_r1 =\
                    tf.nn.dropout(tf.nn.relu(tf.matmul(mp_h_conv2_flat_r1,
                                                       mp_w_fcn1) +
                                  mp_b_fcn1), self.keep_prob)
                mp_h_fcn1_r2 =\
                    tf.nn.dropout(tf.nn.relu(tf.matmul(mp_h_conv2_flat_r2,
                                                       mp_w_fcn1) +
                                  mp_b_fcn1), self.keep_prob)
            with tf.variable_scope('layer_5'):
                mp_w_fcn2 = _weight_variable([64, 32])
                mp_b_fcn2 = _bias_variable([32])
                mp_h_fcn2_r1 = tf.nn.relu(tf.matmul(mp_h_fcn1_r1, mp_w_fcn2) +
                                          mp_b_fcn2)
                mp_h_fcn2_r2 = tf.nn.relu(tf.matmul(mp_h_fcn1_r2, mp_w_fcn2) +
                                          mp_b_fcn2)
        with tf.variable_scope('s_patches'):
            with tf.variable_scope('layer_1'):
                sp_w_conv1 = _weight_variable([3, 3, 6, 16])
                sp_b_conv1 = _bias_variable([16])
                sp_h_conv1_r1 = tf.nn.relu(_conv2d(sp_imgs_r1, sp_w_conv1) +
                                           sp_b_conv1)
                sp_h_conv1_r2 = tf.nn.relu(_conv2d(sp_imgs_r2, sp_w_conv1) +
                                           sp_b_conv1)
            with tf.variable_scope('layer_2'):
                sp_w_conv2 = _weight_variable([3, 3, 16, 32])
                sp_b_conv2 = _bias_variable([32])
                sp_h_conv2_r1 = tf.nn.relu(_conv2d(sp_h_conv1_r1, sp_w_conv2) +
                                           sp_b_conv2)
                sp_h_conv2_r2 = tf.nn.relu(_conv2d(sp_h_conv1_r2, sp_w_conv2) +
                                           sp_b_conv2)
                sp_h_conv2_flat_r1 =\
                    tf.reshape(sp_h_conv2_r1, [-1, 3 * 3 * 32])
                sp_h_conv2_flat_r2 =\
                    tf.reshape(sp_h_conv2_r2, [-1, 3 * 3 * 32])
            with tf.variable_scope('layer_3'):
                sp_w_fcn1 = _weight_variable([3 * 3 * 32, 32])
                sp_b_fcn1 = _bias_variable([32])
                sp_h_fcn1_r1 =\
                    tf.nn.dropout(tf.nn.relu(tf.matmul(sp_h_conv2_flat_r1,
                                                       sp_w_fcn1) +
                                  sp_b_fcn1), self.keep_prob)
                sp_h_fcn1_r2 =\
                    tf.nn.dropout(tf.nn.relu(tf.matmul(sp_h_conv2_flat_r2,
                                                       sp_w_fcn1) +
                                  sp_b_fcn1), self.keep_prob)
            with tf.variable_scope('layer_4'):
                sp_w_fcn2 = _weight_variable([32, 32])
                sp_b_fcn2 = _bias_variable([32])
                sp_h_fcn2_r1 = tf.nn.relu(tf.matmul(sp_h_fcn1_r1, sp_w_fcn2) +
                                          sp_b_fcn2)
                sp_h_fcn2_r2 = tf.nn.relu(tf.matmul(sp_h_fcn1_r2, sp_w_fcn2) +
                                          sp_b_fcn2)

        with tf.variable_scope('patch_merge'):
            with tf.variable_scope('layer_1'):
                feat_r1 = tf.concat([lp_h_fcn2_r1, mp_h_fcn2_r1, sp_h_fcn2_r1], 1)
                feat_r2 = tf.concat([lp_h_fcn2_r2, mp_h_fcn2_r2, sp_h_fcn2_r2], 1)
                mp_w_fcn1 = _weight_variable([96, 32])
                mp_b_fcn1 = _bias_variable([32])
                mp_h_fcn1_r1 = tf.nn.relu(tf.matmul(feat_r1, mp_w_fcn1) +
                                          mp_b_fcn1)
                mp_h_fcn1_r2 = tf.nn.relu(tf.matmul(feat_r2, mp_w_fcn1) +
                                          mp_b_fcn1)
            with tf.variable_scope('layer_2'):
                mp_w_fcn2 = _weight_variable([32, 2])
                mp_b_fcn2 = _bias_variable([2])
                mp_h_fcn1_r1 =\
                    tf.nn.softmax(tf.matmul(mp_h_fcn1_r1, mp_w_fcn2) +
                                  mp_b_fcn2)
                mp_h_fcn1_r2 =\
                    tf.nn.softmax(tf.matmul(mp_h_fcn1_r2, mp_w_fcn2) +
                                  mp_b_fcn2)

        # ___________________________________________________________________ #
        cross_entropy =\
            self.lw[0] * tf.reduce_mean(-tf.reduce_sum(self.gt_r1 *
                                                       tf.log(mp_h_fcn1_r1),
                                                       reduction_indices=[1])) +\
            self.lw[1] * tf.reduce_mean(-tf.reduce_sum(self.gt_r2 *
                                                       tf.log(mp_h_fcn1_r2),
                                                       reduction_indices=[1]))

        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(cross_entropy)
        correct_prediction_r1 = tf.equal(tf.argmax(mp_h_fcn1_r1, 1),
                                         tf.argmax(self.gt_r1, 1))
        self.accuracy_r1 =\
            tf.reduce_mean(tf.cast(correct_prediction_r1, tf.float32))
        self.probabilities_1 = mp_h_fcn1_r1

        correct_prediction_r2 = tf.equal(tf.argmax(mp_h_fcn1_r2, 1),
                                         tf.argmax(self.gt_r2, 1))
        self.accuracy_r2 =\
            tf.reduce_mean(tf.cast(correct_prediction_r2, tf.float32))
        self.probabilities_2 = mp_h_fcn1_r2

        vars_to_save = tf.trainable_variables()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(vars_to_save, max_to_keep=100000)

    def training_and_validation(self, db, meta, prep, patch_ex, exp_out):
        """Run slgorithm's training and validation."""
        """
            Arguments:
                db: DatabaseISLES object
                prep: PreprocessorISLES object
                patch_ex: PatchExtractorISLES object
                exp_out: path to the experiment output
        """
        seg_path = os.path.join(exp_out, 'segmentators', db.name(),
                                prep.name(), patch_ex.name(), self.name())
        seg_done = os.path.join(seg_path, 'done')

        if self.restore:
            self.restore_model(seg_path, self.restore_it)
            print "RESTORED!!!"

        if not os.path.exists(seg_done):
            if not os.path.exists(seg_path):
                os.makedirs(seg_path)

            for it in range(self.restore_it, self.train_iters):

                self._train(db, meta, prep, patch_ex, exp_out)

                if it % 2 == 0:

                    train_acc_l_region, train_acc_s_region =\
                        self._validate(db, meta, prep, patch_ex, exp_out,
                                       'train')
                    valid_acc_l_region, valid_acc_s_region =\
                        self._validate(db, meta, prep, patch_ex, exp_out,
                                       'valid')

                    print("train, valid accuracy:" + " " +
                          str(train_acc_l_region) + " " +
                          str(train_acc_s_region) + " " +
                          str(valid_acc_l_region) + " " +
                          str(valid_acc_s_region))
                if it % 2 == 0:
                    self.save_model(seg_path, it)
            with open(seg_done, 'w') as f:
                f.close()
        else:
            print "Segmentator is already trained!"

    def _train(self, db, meta, prep, patch_ex, exp_out):

        data = patch_ex.extract_train_or_valid_data(db, meta, prep, exp_out, 'train')

        if data['region_1']['labels'].shape[0]:
            random_s =\
                np.random.choice(data['region_1']['labels'].shape[0],
                                 int(0.2 * data['region_1']['labels'].shape[0]),
                                 replace=False)
            for idx, r in enumerate(random_s):
                data['region_1']['labels'][r, :] = np.zeros(2)
                data['region_1']['labels'][r, np.random.randint(2)] = 1

            random_s =\
                np.random.choice(data['region_2']['labels'].shape[0],
                                 int(0.2 * data['region_2']['labels'].shape[0]),
                                 replace=False)
            for idx, r in enumerate(random_s):
                data['region_2']['labels'][r, :] = np.zeros(2)
                data['region_2']['labels'][r, np.random.randint(2)] = 1

            self.sess.run(self.train_step,
                          feed_dict={self.lp_x_r1: data['region_1']['l_patch'],
                                     self.mp_x_r1: data['region_1']['m_patch'],
                                     self.sp_x_r1: data['region_1']['s_patch'],
                                     self.gt_r1: data['region_1']['labels'],
                                     self.lp_x_r2: data['region_2']['l_patch'],
                                     self.mp_x_r2: data['region_2']['m_patch'],
                                     self.sp_x_r2: data['region_2']['s_patch'],
                                     self.gt_r2: data['region_2']['labels'],
                                     self.keep_prob: self.kp})

    def _validate(self, db, meta, prep, patch_ex, exp_out, subset):
        if subset == 'train':
            data = patch_ex.extract_train_or_valid_data(db, meta, prep,
                                                        exp_out,
                                                        'train_valid')
        elif subset == 'valid':
            data = patch_ex.extract_train_or_valid_data(db, meta, prep,
                                                        exp_out,
                                                        'valid')

        accuracy_l_region =\
            self.sess.run(self.accuracy_r1,
                          feed_dict={self.lp_x_r1: data['region_1']['l_patch'],
                                     self.mp_x_r1: data['region_1']['m_patch'],
                                     self.sp_x_r1: data['region_1']['s_patch'],
                                     self.gt_r1: data['region_1']['labels'],
                                     self.keep_prob: 1.0})
        accuracy_s_region =\
            self.sess.run(self.accuracy_r2,
                          feed_dict={self.lp_x_r2: data['region_2']['l_patch'],
                                     self.mp_x_r2: data['region_2']['m_patch'],
                                     self.sp_x_r2: data['region_2']['s_patch'],
                                     self.gt_r2: data['region_2']['labels'],
                                     self.keep_prob: 1.0})

        return accuracy_l_region, accuracy_s_region

    def _compute_clf_scores_per_scan(self, db, prep, patch_ex, clf_out, scan):

        for s in db.sizes:
            for i_fm in [0]:

                scan_prob_path = os.path.join(clf_out,
                                              scan.name + '_' + str(s) +
                                              '_scores_' + '{0}' + '.bin')

                volumes = scan.load_volumes_norm_aligned(db, s, i_fm, True)

                patch_ex._get_coordinates(volumes[0].shape)

                indices = np.where(volumes[-1])
                class_number = np.zeros((s, s, scan.d))
                p_1 = np.zeros((s, s, scan.d))

                n_indices = len(indices[0])
                i = 0
                while i < n_indices:
                    s_idx = [indices[0][i:i + patch_ex.test_patches_per_scan],
                             indices[1][i:i + patch_ex.test_patches_per_scan],
                             indices[2][i:i + patch_ex.test_patches_per_scan]]
                    i += patch_ex.test_patches_per_scan
                    patches = patch_ex.extract_test_patches(scan, db, prep,
                                                            volumes, s_idx)
                    labels =\
                        self.sess.run(self.probabilities_1,
                                      feed_dict={self.lp_x_r1: patches['l_patch'],
                                                 self.mp_x_r1: patches['m_patch'],
                                                 self.sp_x_r1: patches['s_patch'],
                                                 self.keep_prob: 1.0})
                    for j in range(labels.shape[0]):
                        #class_number[s_idx[0][j], s_idx[1][j], s_idx[2][j]] =\
                        #    db.classes[np.argmax(labels[j, :])]
                        p_1[s_idx[0][j], s_idx[1][j], s_idx[2][j]] = labels[j, 1]

                #class_number.tofile(scan_class_path.format(i_fm))
                p_1.tofile(scan_prob_path.format(i_fm) + '_1')

    def name(self):
        """Class name reproduction."""
        return "%s()" % (type(self).__name__)

    def restore_model(self, output_path, it):
        """Restore model."""
        model_path = os.path.join(output_path, self.name(),
                                  'model_' + str(it))
        self.saver.restore(self.sess, model_path)

    def save_model(self, output_path, it):
        """Saving model."""
        model_path = os.path.join(output_path, self.name(), 'model_' + str(it))

        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))

        self.saver.save(self.sess, model_path)
