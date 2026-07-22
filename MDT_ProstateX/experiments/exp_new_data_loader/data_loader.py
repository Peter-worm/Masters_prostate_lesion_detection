#!/usr/bin/env python
# Copyright 2021 Oscar J. Pellicer-Valero
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# CHANGELOG:
#
# This file has been modified to be able to load ProstateX imaging data
# The main changes have been aplied to get_train_generators
# It includes a dictionary (ss_v2) with the different data subsets: train_fix (used always 
# for training), train_val (used for training and cross-validation), and test (used for test)

'''
Data Loader for prostate lesion segmentation. 
This dataloader expects preprocessed data per patient in .npy or .npz files and
a pandas dataframe in the same directory containing the meta-info e.g. labels, foregound slice-ids.
'''


import numpy as np
import os
from collections import OrderedDict
import pandas as pd
import pickle
import time
import subprocess

ss_v2= {
'train_fix': ['3307', '711', '1057', '665', '302', '1242', '160', '618', '621', '116', '312', '3009', '3333', '3311', '563', '3047', '3354', '066', '247', '1205', '464', '3052', '161', '433', '075', '765', '644', '1193', '3348', '932', '3023', '140', '846', '156', '031', '3103', '055', '033', '634', '175', '3029', '304', '957', '1019', '3062', '3030', '3020', '930', '1025', '3309', '095', '760', '358', '429', '1031', '094', '798', '3053', '1239', '950', '1041', '326', '784', '496', '372', '3089', '3344', '409', '3086', '562', '3357', '212', '1085', '3100', '1056', '3090', '3031', '359', '652', '135', '830', '1042', '3006', '3314', '3027', '022', '597', '3345', '1127', '386', '1189', '1021', '027', '1228', '472', '229', '1238', '106', '909', '1036', '639', '452', '330', '3066', '715', '197', '3353', '828', '061', '549', '3358', '124', '202', '653', '431', '082', '103', '608', '3059', '167', '3087', '3084', '853', '631', '1165', '893', '3048', '086', '993', '353', '1141', '692', '725', '099', '268', '3063', '3088', '345', '340', '3078', '568', '3085', '3005', '878', '003', '191', '757', '1044', '1023', '731', '523', '3080', '1178', '3083', '525', '3072', '905', '511', '3081', '1183', '026', '007', '017', '417', '3061', '460', '3012', '3018', '566', '3332', '356', '3082', '314', '1150', '1177', '897', '3064', '1051', '1144', '176', '3049', '1038', '534', '339', '815', '187', '691', '3359', '1225', '3057', '014', '004', '683', '3001', '449', '3101', '3075', '221', '001', '090', '203', '3320', '667', '016', '768', '3337', '3334', '1161', '1224', '519', '091', '3038', '3074', '1192', '3007', '3037', '1221', '3003', '1013', '3321', '104', '601', '1090', '967', '742', '466', '706', '805', '438', '927', '3317', '3042', '3004', '3056', '311', '470', '3099', '3022', '3021', '112', '088', '077', '3060', '3335', '165', '3040', '134', '3091', '008', '906', '1232', '3343', '159', '3032', '1066', '737', '3014', '506', '1138', '317', '655', '766', '205', '921', '096', '650', '696', '524', '367', '371', '3360', '484', '708', '473', '3322', '3036', '583', '432', '1187', '720', '3319', '3058', '3076', '3341', '322', '507', '684', '847', '912', '3340', '119', '3019', '079', '424', '590', '3015', '097', '3026', '3093', '754', '3098', '195', '856', '1244'],
'train_val': ['724', '3323', '172', '389', '3339', '543', '378', '1014', '131', '323', '183', '3033', '889', '3011', '076', '005', '584', '3071', '670', '495', '1093', '098', '227', '756', '1184', '3041', '1151', '738', '3017', '3351', '444', '210', '3025', '355', '3328', '829', '321', '3326'],
'test': ['679', '054', '1149', '285', '498', '337', '3044', '3355', '380', '1188', '966', '3092', '3077', '233', '1063', '3073', '977', '1159', '3055', '393', '269', '482', '049', '841', '989', '3039', '3069', '174', '986', '1086', '585', '023', '611', '011', '3094', '1218', '399', '794'],       
}



# batch generator tools from https://github.com/MIC-DKFZ/batchgenerators
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.transforms.spatial_transforms import MirrorTransform as Mirror
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading import SingleThreadedAugmenter
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform
from batchgenerators.transforms.utility_transforms import ConvertSegToBoundingBoxCoordinates
from custom_transform import RandomChannelDeleteTransform

import utils.dataloader_utils as dutils
import utils.exp_utils as utils

def get_train_generators(cf, logger):
    """
    wrapper function for creating the training batch generator pipeline. returns the train/val generators.
    selects patients according to cv folds (generated by first run/fold of experiment):
    splits the data into n-folds, where 1 split is used for val, 1 split for testing and the rest for training. 
    (inner loop test set)
    """
        
    #Generate info_df.pickle when training for the first time
    files = [os.path.join(cf.pp_dir, f) for f in os.listdir(cf.pp_dir) if 'meta_info' in f]
    df = pd.DataFrame(columns=['pid', 'class_target', 'spacing', 'fg_slices'], dtype=object)
    for f in files:
        with open(f, 'rb') as handle:
            df.loc[len(df)] = pickle.load(handle)
    df.to_pickle(os.path.join(cf.pp_dir, 'info_df.pickle'))
    print ("Aggregated meta info to df with length", len(df))
    
    #load data
    all_data = load_dataset(cf, logger)
    splits_file = os.path.join(cf.exp_dir, 'fold_ids.pickle')
    ss= ss_v2
        
    #Get subset idx
    pids= list(pd.read_pickle(os.path.join(cf.pp_dir, 'info_df.pickle')).pid.values)

    #Regenerate fold_ids.pickle?
    if not os.path.isfile(splits_file) and not cf.created_fold_id_pickle:           
        #Get subsets
        prostatex_train_fix, prostatex_train_val, prostatex_test= ss['train_fix'] if cf.use_prostatex_test else [], \
                                                                  ss['train_val'], ss['test']
        #Build samples for all folds     
        from sklearn.model_selection import KFold

        prostatex_kf= KFold(n_splits=cf.n_cv_splits, random_state=5, shuffle=True)
        prostatex_kf.get_n_splits(prostatex_train_val)

        fg= []
        for i, (prostatex_train_index, prostatex_val_index) in \
            enumerate(prostatex_kf.split(prostatex_train_val)) :
            fg.append([
                list(np.array(prostatex_train_val)[prostatex_train_index]) + prostatex_train_fix,
                list(np.array(prostatex_train_val)[prostatex_val_index]),
                prostatex_test,
                i
            ])

        #Save it
        pickle.dump(fg, open(os.path.join(cf.exp_dir, 'fold_ids.pickle'), 'wb'))
        cf.created_fold_id_pickle = True
    else:
        with open(splits_file, 'rb') as handle:
            fg = pickle.load(handle)

    train_pids, val_pids, test_pids, _ = fg[cf.fold]
    
    if cf.verbose == True:
        print('Train IDs:', train_pids)
        print('\nValidation IDs:', val_pids)
        print('\nTest IDs:', test_pids)

    train_data = {k: v for (k, v) in all_data.items() if any(p == v['pid'] for p in train_pids)}
    val_data = {k: v for (k, v) in all_data.items() if any(p == v['pid'] for p in val_pids)}

    logger.info("data set loaded with: {} train / {} val / {} test patients".format(len(train_pids), len(val_pids), len(test_pids)))
    batch_gen = {}
    batch_gen['train'] = create_data_gen_pipeline(train_data, cf=cf, is_training=True)
    batch_gen['val_sampling'] = create_data_gen_pipeline(val_data, cf=cf, is_training=False)
    if cf.val_mode == 'val_patient':
        batch_gen['val_patient'] = PatientBatchIterator(val_data, cf=cf)
        batch_gen['n_val'] = len(val_pids) if cf.max_val_patients is None else min(len(val_pids), cf.max_val_patients)
    else:
        batch_gen['n_val'] = cf.num_val_batches

    return batch_gen

def get_test_generator(cf, logger):
    """
    wrapper function for creating the test batch generator pipeline.
    selects patients according to cv folds (generated by first run/fold of experiment)
    """
    with open(os.path.join(cf.exp_dir, 'fold_ids.pickle'), 'rb') as handle:
        fold_list = pickle.load(handle)
    train_pids, val_pids, test_pids, _ = fold_list[cf.fold]
    pp_name = None
    if cf.test_subset == 'train':
        pids= train_pids 
        logger.warn('INFO: using train set for testing')
    elif cf.test_subset == 'val':
        pids= val_pids 
        logger.warn('INFO: using cross-validation set for testing')
    elif cf.test_subset == 'test':
        pids= ss_v2['test']
        logger.warn('INFO: using test set for testing')
    else:
        raise ValueError('cf.test_subset must be either train, val or test')
    print('Pids:', pids)    
    
    test_data = load_dataset(cf, logger, pids, pp_data_path=cf.pp_test_data_path, pp_name=pp_name)
    logger.info("data set loaded with: {} test patients".format(len(pids)))
    batch_gen = {}
    batch_gen['test'] = PatientBatchIterator(test_data, cf=cf)
    batch_gen['n_test'] = len(pids) if cf.max_test_patients=="all" else \
        min(cf.max_test_patients, len(pids))
    return batch_gen

def load_dataset(cf, logger, subset_pids=None, pp_data_path=None, pp_name=None):
    """
    loads the dataset. if deployed in cloud also copies and unpacks the data to the working directory.
    :param subset_pids: subset pids to be loaded from the dataset. used e.g. for testing to only load the test folds.
    :return: data: dictionary with one entry per patient (in this case per patient-breast, since they are treated as
    individual images for training) each entry is a dictionary containing respective meta-info as well as paths to the preprocessed
    numpy arrays to be loaded during batch-generation
    """
    if pp_data_path is None:
        pp_data_path = cf.pp_data_path
    if pp_name is None:
        pp_name = cf.pp_name
    if cf.server_env:
        copy_data = True
        target_dir = os.path.join(cf.data_dest, pp_name)
        if not os.path.exists(target_dir):
            cf.data_source_dir = pp_data_path
            os.makedirs(target_dir)
            subprocess.call('rsync -av {} {}'.format(
                os.path.join(cf.data_source_dir, cf.input_df_name), os.path.join(target_dir, cf.input_df_name)), shell=True)
            logger.info('created target dir and info df at {}'.format(os.path.join(target_dir, cf.input_df_name)))

        elif subset_pids is None:
            copy_data = False

        pp_data_path = target_dir


    p_df = pd.read_pickle(os.path.join(pp_data_path, cf.input_df_name))

    if cf.select_prototype_subset is not None:
        prototype_pids = p_df.pid.tolist()[:cf.select_prototype_subset]
        p_df = p_df[p_df.pid.isin(prototype_pids)]
        logger.warning('WARNING: using prototyping data subset!!!')

    if subset_pids is not None:
        p_df = p_df[p_df.pid.isin(subset_pids)]
        logger.info('subset: selected {} instances from df'.format(len(p_df)))

    if cf.server_env:
        if copy_data:
            copy_and_unpack_data(logger, p_df.pid.tolist(), cf.fold_dir, cf.data_source_dir, target_dir)

    class_targets = p_df['class_target'].tolist()
    pids = p_df.pid.tolist()
    imgs = [os.path.join(pp_data_path, '{}_img.npy'.format(pid)) for pid in pids]
    segs = [os.path.join(pp_data_path,'{}_rois.npy'.format(pid)) for pid in pids]

    data = OrderedDict()
    # for the experiment conducted here, malignancy scores are binarized following self.cf.modify_class_target_fn
    for ix, pid in enumerate(pids):
        targets = cf.modify_class_target_fn(class_targets[ix])
        data[pid] = {'data': imgs[ix], 'seg': segs[ix], 'pid': pid, 'class_target': targets}
        data[pid]['fg_slices'] = p_df['fg_slices'].tolist()[ix]
        #data[pid]['class_unknown']= 

    return data

def create_data_gen_pipeline(patient_data, cf, is_training=True):
    """
    create mutli-threaded train/val/test batch generation and augmentation pipeline.
    :param patient_data: dictionary containing one dictionary per patient in the train/test subset.
    :param is_training: (optional) whether to perform data augmentation (training) or not (validation/testing)
    :return: multithreaded_generator
    """

    # create instance of batch generator as first element in pipeline.
    data_gen = BatchGenerator(patient_data, batch_size=cf.batch_size, cf=cf)

    # add transformations to pipeline.
    my_transforms = []
    if is_training:
        mirror_transform = Mirror(axes=cf.mirror_axes) #np.arange(cf.dim)
        my_transforms.append(mirror_transform)
        spatial_transform = SpatialTransform(patch_size=cf.patch_size[:cf.dim],
                                             patch_center_dist_from_border=cf.da_kwargs['rand_crop_dist'],
                                             do_elastic_deform=cf.da_kwargs['do_elastic_deform'],
                                             alpha=cf.da_kwargs['alpha'], sigma=cf.da_kwargs['sigma'],
                                             do_rotation=cf.da_kwargs['do_rotation'], angle_x=cf.da_kwargs['angle_x'],
                                             angle_y=cf.da_kwargs['angle_y'], angle_z=cf.da_kwargs['angle_z'],
                                             do_scale=cf.da_kwargs['do_scale'], scale=cf.da_kwargs['scale'],
                                             random_crop=cf.da_kwargs['random_crop'])

        my_transforms.append(spatial_transform)
        my_transforms.append(RandomChannelDeleteTransform(cf.droppable_channels, cf.channel_drop_p))
    else:
        my_transforms.append(CenterCropTransform(crop_size=cf.patch_size[:cf.dim]))

    my_transforms.append(ConvertSegToBoundingBoxCoordinates(cf.dim, get_rois_from_seg_flag=False, class_specific_seg_flag=cf.class_specific_seg_flag))
    all_transforms = Compose(my_transforms)
    if cf.debugging:
        multithreaded_generator = SingleThreadedAugmenter(data_gen, all_transforms)
    else:
        multithreaded_generator = MultiThreadedAugmenter(data_gen, all_transforms, num_processes=cf.n_workers, seeds=range(cf.n_workers))
    return multithreaded_generator


class BatchGenerator(SlimDataLoaderBase):
    """
    creates the training/validation batch generator. Samples n_batch_size patients (draws a slice from each patient if 2D)
    from the data set while maintaining foreground-class balance. Returned patches are cropped/padded to pre_crop_size.
    Actual patch_size is obtained after data augmentation.
    :param data: data dictionary as provided by 'load_dataset'.
    :param batch_size: number of patients to sample for the batch
    :return dictionary containing the batch data (b, c, y, x(, z)) / seg (b, 1, y, x(, z)) / pids / class_target
    """
    def __init__(self, data, batch_size, cf):
        super(BatchGenerator, self).__init__(data, batch_size)
        
        self.cf = cf
        self.crop_margin = np.array(self.cf.patch_size)/8. #min distance of ROI center to edge of cropped_patch.
        self.p_fg = 0.5

    def generate_train_batch(self):
        print("Number of patients:", len(self._data))
        print("Patient IDs:", list(self._data.keys())[:10])
        batch_data, batch_segs, batch_pids, batch_targets, batch_patient_labels = [], [], [], [], []
        class_targets_list =  [v['class_target'] for (k, v) in self._data.items()]

        #I am turning this off, because it is problematic with my class 20
        if False: #self.cf.head_classes > 2:
            # samples patients towards equilibrium of foreground classes on a roi-level (after randomly sampling the ratio "batch_sample_slack).
            batch_ixs = dutils.get_class_balanced_patients(
                class_targets_list, self.batch_size, self.cf.head_classes - 1, slack_factor=self.cf.batch_sample_slack)
        else:
            print("class_targets_list:", class_targets_list)
            print("len(class_targets_list):", len(class_targets_list))
            print("batch_size:", self.batch_size)
            batch_ixs = np.random.choice(len(class_targets_list), self.batch_size)

        patients = list(self._data.items())

        for b in batch_ixs:
            patient = patients[b][1]

            # data shape: from (c, z, y, x) to (c, y, x, z).
            data = np.transpose(np.load(patient['data'], mmap_mode='r'), axes=(3, 1, 2, 0))
            data = data[:4] # XDDDDDD
            seg = np.transpose(np.load(patient['seg'], mmap_mode='r'), axes=(3, 1, 2, 0))

            print("inside loader hihi")
            print(data.shape)
            print(seg.shape)
            batch_pids.append(patient['pid'])
            batch_targets.append(patient['class_target'])

            batch_pids.append(patient['pid'])
            batch_targets.append(patient['class_target'])

            if self.cf.dim == 2:
                # draw random slice from patient while oversampling slices containing foreground objects with p_fg.
                if len(patient['fg_slices']) > 0:
                    fg_prob = self.p_fg / len(patient['fg_slices'])
                    bg_prob = (1 - self.p_fg) / (data.shape[3] - len(patient['fg_slices']))
                    slices_prob = [fg_prob if ix in patient['fg_slices'] else bg_prob for ix in range(data.shape[3])]
                    slice_id = np.random.choice(data.shape[3], p=slices_prob)
                else:
                    slice_id = np.random.choice(data.shape[3])

                # if set to not None, add neighbouring slices to each selected slice in channel dimension.
                if self.cf.n_3D_context is not None:
                    padded_data = dutils.pad_nd_image(data[0], [(data.shape[-1] + (self.cf.n_3D_context*2))], mode='constant')
                    padded_slice_id = slice_id + self.cf.n_3D_context
                    data = (np.concatenate([padded_data[..., ii][np.newaxis] for ii in range(
                        padded_slice_id - self.cf.n_3D_context, padded_slice_id + self.cf.n_3D_context + 1)], axis=0))
                else:
                    data = data[..., slice_id]
                seg = seg[..., slice_id]

            # pad data if smaller than pre_crop_size.
            if np.any([data.shape[dim + 1] < ps for dim, ps in enumerate(self.cf.pre_crop_size)]):
                new_shape = [np.max([data.shape[dim + 1], ps]) for dim, ps in enumerate(self.cf.pre_crop_size)]
                data = dutils.pad_nd_image(data, new_shape, mode='constant')
                seg = dutils.pad_nd_image(seg, new_shape, mode='constant')



            # crop patches of size pre_crop_size, while sampling patches containing foreground with p_fg.
            crop_dims = [dim for dim, ps in enumerate(self.cf.pre_crop_size) if data.shape[dim + 1] > ps]
            print("Before cropping")
            print("data.shape:", data.shape)
            print("seg.shape :", seg.shape)
            print("crop_dims:", crop_dims)
            if len(crop_dims) > 0:
                fg_prob_sample = np.random.rand(1)
                # with p_fg: sample random pixel from random ROI and shift center by random value.
                if fg_prob_sample < self.p_fg and np.sum(seg) > 0:
                    seg_ixs = np.argwhere(seg == np.random.choice(np.unique(seg)[1:], 1))
                    roi_anchor_pixel = seg_ixs[np.random.choice(seg_ixs.shape[0], 1)][0]
                    assert seg[tuple(roi_anchor_pixel)] > 0
                    # sample the patch center coords. constrained by edges of images - pre_crop_size /2. And by
                    # distance to the desired ROI < patch_size /2.
                    # (here final patch size to account for center_crop after data augmentation).
                    sample_seg_center = {}
                    for ii in crop_dims:
                        low = np.max((self.cf.pre_crop_size[ii]//2, roi_anchor_pixel[ii] - (self.cf.patch_size[ii]//2 - self.crop_margin[ii])))
                        high = np.min((data.shape[ii + 1] - self.cf.pre_crop_size[ii]//2,
                                       roi_anchor_pixel[ii] + (self.cf.patch_size[ii]//2 - self.crop_margin[ii])))
                        # happens if lesion on the edge of the image. dont care about roi anymore,
                        # just make sure pre-crop is inside image.
                        if low >= high:
                            low = data.shape[ii + 1] // 2 - (data.shape[ii + 1] // 2 - self.cf.pre_crop_size[ii] // 2)
                            high = data.shape[ii + 1] // 2 + (data.shape[ii + 1] // 2 - self.cf.pre_crop_size[ii] // 2)
                        sample_seg_center[ii] = np.random.randint(low=low, high=high)

                else:
                    # not guaranteed to be empty. probability of emptiness depends on the data.
                    sample_seg_center = {ii: np.random.randint(low=self.cf.pre_crop_size[ii]//2,
                                                           high=data.shape[ii + 1] - self.cf.pre_crop_size[ii]//2) for ii in crop_dims}

                for ii in crop_dims:
                    
                    print(f"\nCropping dimension {ii}")
                    print("Before:", data.shape, seg.shape)
                    min_crop = int(sample_seg_center[ii] - self.cf.pre_crop_size[ii] // 2)
                    max_crop = int(sample_seg_center[ii] + self.cf.pre_crop_size[ii] // 2)
                    data = np.take(data, indices=range(min_crop, max_crop), axis=ii + 1)
                    seg = np.take(seg, indices=range(min_crop, max_crop), axis=ii + 1)
                    print("After :", data.shape, seg.shape)

            batch_data.append(data)
            batch_segs.append(seg)

        data = np.array(batch_data)
        seg = np.array(batch_segs).astype(np.uint8)
        print("inside loader hihi 2")
        print(data.shape)
        print(seg.shape)
        class_target = np.array(batch_targets, dtype=object)
        return {'data': data, 'seg': seg, 'pid': batch_pids, 'class_target': class_target}


class PatientBatchIterator(SlimDataLoaderBase):
    """
    creates a test generator that iterates over entire given dataset returning 1 patient per batch.
    Can be used for monitoring if cf.val_mode = 'patient_val' for a monitoring closer to actualy evaluation (done in 3D),
    if willing to accept speed-loss during training.
    :return: out_batch: dictionary containing one patient with batch_size = n_3D_patches in 3D or
    batch_size = n_2D_patches in 2D .
    """
    def __init__(self, data, cf): #threads in augmenter
        super(PatientBatchIterator, self).__init__(data, 0)
        self.cf = cf
        self.patient_ix = 0
        self.dataset_pids = [v['pid'] for (k, v) in data.items()]
        self.patch_size = cf.patch_size
        if len(self.patch_size) == 2:
            self.patch_size = self.patch_size + [1]


    def generate_train_batch(self):
        pid = self.dataset_pids[self.patient_ix]
        patient = self._data[pid]
        # data shape: from (c, z, y, x) to (c, y, x, z).
        data = np.transpose(np.load(patient['data'], mmap_mode='r'), axes=(3, 1, 2, 0)).copy()
        seg = np.transpose(np.load(patient['seg'], mmap_mode='r'), axes=(3, 1, 2, 0))[0].copy()
        batch_class_targets = np.array([patient['class_target']])

        # pad data if smaller than patch_size seen during training.
        if np.any([data.shape[dim + 1] < ps for dim, ps in enumerate(self.patch_size)]):
            new_shape = [data.shape[0]] + [np.max([data.shape[dim + 1], self.patch_size[dim]]) for dim, ps in enumerate(self.patch_size)]
            data = dutils.pad_nd_image(data, new_shape) # use 'return_slicer' to crop image back to original shape.
            seg = dutils.pad_nd_image(seg, new_shape)

        # get 3D targets for evaluation, even if network operates in 2D. 2D predictions will be merged to 3D in predictor.
        if self.cf.dim == 3 or self.cf.merge_2D_to_3D_preds:
            out_data = data[np.newaxis]
            out_seg = seg[np.newaxis, np.newaxis]
            out_targets = batch_class_targets

            batch_3D = {'data': out_data, 'seg': out_seg, 'class_target': out_targets, 'pid': pid}
            converter = ConvertSegToBoundingBoxCoordinates(dim=3, get_rois_from_seg_flag=False, class_specific_seg_flag=self.cf.class_specific_seg_flag)
            batch_3D = converter(**batch_3D)
            batch_3D.update({'patient_bb_target': batch_3D['bb_target'],
                                  'patient_roi_labels': batch_3D['class_target'],
                                  'original_img_shape': out_data.shape})

        if self.cf.dim == 2:
            out_data = np.transpose(data, axes=(3, 0, 1, 2))  # (z, c, y, x )
            out_seg = np.transpose(seg, axes=(2, 0, 1))[:, np.newaxis]
            out_targets = np.array(np.repeat(batch_class_targets, out_data.shape[0], axis=0))

            # if set to not None, add neighbouring slices to each selected slice in channel dimension.
            if self.cf.n_3D_context is not None:
                slice_range = range(self.cf.n_3D_context, out_data.shape[0] + self.cf.n_3D_context)
                out_data = np.pad(out_data, ((self.cf.n_3D_context, self.cf.n_3D_context), (0, 0), (0, 0), (0, 0)), 'constant', constant_values=0)
                out_data = np.array(
                    [np.concatenate([out_data[ii] for ii in range(
                        slice_id - self.cf.n_3D_context, slice_id + self.cf.n_3D_context + 1)], axis=0) for slice_id in
                     slice_range])

            batch_2D = {'data': out_data, 'seg': out_seg, 'class_target': out_targets, 'pid': pid}
            converter = ConvertSegToBoundingBoxCoordinates(dim=2, get_rois_from_seg_flag=False, class_specific_seg_flag=self.cf.class_specific_seg_flag)
            batch_2D = converter(**batch_2D)

            if self.cf.merge_2D_to_3D_preds:
                batch_2D.update({'patient_bb_target': batch_3D['patient_bb_target'],
                                      'patient_roi_labels': batch_3D['patient_roi_labels'],
                                      'original_img_shape': out_data.shape})
            else:
                batch_2D.update({'patient_bb_target': batch_2D['bb_target'],
                                 'patient_roi_labels': batch_2D['class_target'],
                                 'original_img_shape': out_data.shape})

        out_batch = batch_3D if self.cf.dim == 3 else batch_2D
        patient_batch = out_batch

        # crop patient-volume to patches of patch_size used during training. stack patches up in batch dimension.
        # in this case, 2D is treated as a special case of 3D with patch_size[z] = 1.
        if np.any([data.shape[dim + 1] > self.patch_size[dim] for dim in range(3)]):
            patch_crop_coords_list = dutils.get_patch_crop_coords(data[0], self.patch_size)
            new_img_batch, new_seg_batch, new_class_targets_batch = [], [], []

            for cix, c in enumerate(patch_crop_coords_list):

                seg_patch = seg[c[0]:c[1], c[2]: c[3], c[4]:c[5]]
                new_seg_batch.append(seg_patch)

                # if set to not None, add neighbouring slices to each selected slice in channel dimension.
                # correct patch_crop coordinates by added slices of 3D context.
                if self.cf.dim == 2 and self.cf.n_3D_context is not None:
                    tmp_c_5 = c[5] + (self.cf.n_3D_context * 2)
                    if cix == 0:
                        data = np.pad(data, ((0, 0), (0, 0), (0, 0), (self.cf.n_3D_context, self.cf.n_3D_context)), 'constant', constant_values=0)
                else:
                    tmp_c_5 = c[5]

                new_img_batch.append(data[:, c[0]:c[1], c[2]:c[3], c[4]:tmp_c_5])

            data = np.array(new_img_batch) # (n_patches, c, x, y, z)
            seg = np.array(new_seg_batch)[:, np.newaxis]  # (n_patches, 1, x, y, z)
            batch_class_targets = np.repeat(batch_class_targets, len(patch_crop_coords_list), axis=0)

            if self.cf.dim == 2:
                if self.cf.n_3D_context is not None:
                    data = np.transpose(data[:, 0], axes=(0, 3, 1, 2))
                else:
                    # all patches have z dimension 1 (slices). discard dimension
                    data = data[..., 0]
                seg = seg[..., 0]

            patch_batch = {'data': data, 'seg': seg, 'class_target': batch_class_targets, 'pid': pid}
            patch_batch['patch_crop_coords'] = np.array(patch_crop_coords_list)
            patch_batch['patient_bb_target'] = patient_batch['patient_bb_target']
            patch_batch['patient_roi_labels'] = patient_batch['patient_roi_labels']
            patch_batch['original_img_shape'] = patient_batch['original_img_shape']

            converter = ConvertSegToBoundingBoxCoordinates(self.cf.dim, get_rois_from_seg_flag=False, class_specific_seg_flag=self.cf.class_specific_seg_flag)
            patch_batch = converter(**patch_batch)
            out_batch = patch_batch

        self.patient_ix += 1
        if self.patient_ix == len(self.dataset_pids):
            self.patient_ix = 0

        out_batch['data'][:,self.cf.drop_channels_test,]= 0.
        return out_batch

def copy_and_unpack_data(logger, pids, fold_dir, source_dir, target_dir):

    start_time = time.time()
    with open(os.path.join(fold_dir, 'file_list.txt'), 'w') as handle:
        for pid in pids:
            handle.write('{}_img.npz\n'.format(pid))
            handle.write('{}_rois.npz\n'.format(pid))

    subprocess.call('rsync -av --files-from {} {} {}'.format(os.path.join(fold_dir, 'file_list.txt'),
        source_dir, target_dir), shell=True)
    n_threads = 8
    dutils.unpack_dataset(target_dir, threads=n_threads)
    copied_files = os.listdir(target_dir)
    t = utils.get_formatted_duration(time.time() - start_time)
    logger.info("\ncopying and unpacking data set finished using {} threads.\n{} files in target dir: {}. Took {}\n"
        .format(n_threads, len(copied_files), target_dir, t))


if __name__=="__main__":

    total_stime = time.time()

    cf_file = utils.import_module("cf", "configs.py")
    cf = cf_file.configs()

    cf.created_fold_id_pickle = False
    cf.exp_dir = "dev/"
    cf.plot_dir = cf.exp_dir + "plots"
    os.makedirs(cf.exp_dir, exist_ok=True)
    cf.fold = 0
    logger = utils.get_logger(cf.exp_dir)

    #batch_gen = get_train_generators(cf, logger)
    #train_batch = next(batch_gen["train"])

    test_gen = get_test_generator(cf, logger)
    test_batch = next(test_gen["test"])

    mins, secs = divmod((time.time() - total_stime), 60)
    h, mins = divmod(mins, 60)
    t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs))
    print("{} total runtime: {}".format(os.path.split(__file__)[1], t))