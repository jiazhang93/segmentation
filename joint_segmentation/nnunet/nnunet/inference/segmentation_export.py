#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import sys
from copy import deepcopy
from typing import Union, Tuple

import numpy as np
import SimpleITK as sitk
from batchgenerators.augmentations.utils import resize_segmentation
from nnunet.preprocessing.preprocessing import get_lowres_axis, get_do_separate_z, resample_data_or_seg
from batchgenerators.utilities.file_and_folder_operations import *
import cc3d


def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 18
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    return conn_comp


def postprocess_delete_small_edge_leison(mask, min_lesion_size=4, max_edge_slice=3):
    filter_size_func = lambda x:len(np.where(x==True)[0])<min_lesion_size # sitk read
    filer_edge_func = lambda x: (np.min(np.where(x==True)[0])==0) and (np.max(np.where(x==True)[0])<=max_edge_slice)

    pred_conn_comp = con_comp(mask)
    num_lesions = pred_conn_comp.max()

    new_mask = np.zeros_like(mask)

    for i in range(1, num_lesions+1):
        comp_mask = np.isin(pred_conn_comp, i)
        if filter_size_func(comp_mask):
            continue
        if filer_edge_func(comp_mask):
            # print('delete', i)
            continue
        else:
            # print('reserve', i)
            new_mask += comp_mask

    return new_mask

def save_segmentation_nifti_from_predict_2dplus3d(segmentation_softmax_2d: Union[str, np.ndarray], 
                                        segmentation_softmax_3d: Union[str, np.ndarray], out_fname: str,
                                         properties_dict_2d: dict, properties_dict_3d: dict, order: int = 1,
                                         region_class_order: Tuple[Tuple[int]] = None,
                                         seg_postprogess_fn: callable = None, seg_postprocess_args: tuple = None,
                                         resampled_npz_fname: str = None,
                                         non_postprocessed_fname: str = None, force_separate_z: bool = None,
                                         interpolation_order_z: int = 0, verbose: bool = True):
    
    
    seg_old_size_postprocessed_2d = get_predict_original(segmentation_softmax_2d, properties_dict_2d, 
                        order, region_class_order, seg_postprogess_fn, seg_postprocess_args,
                        resampled_npz_fname, non_postprocessed_fname, force_separate_z,
                        interpolation_order_z, verbose)

    seg_old_size_postprocessed_3d = get_predict_original(segmentation_softmax_3d, properties_dict_3d, 
                        order, region_class_order, seg_postprogess_fn, seg_postprocess_args,
                        resampled_npz_fname, non_postprocessed_fname, force_separate_z,
                        interpolation_order_z, verbose)

    
    pred2d_seg = seg_old_size_postprocessed_2d.astype(np.uint8)
    pred3d_seg = seg_old_size_postprocessed_3d.astype(np.uint8)


    ###### combine 2d and 3d mask ##########

    seg_and_fusion = np.bitwise_and(pred3d_seg.astype(np.bool8), pred2d_seg.astype(np.bool8)).astype(np.uint8)
    
    pred_seg3d_conn_comp = con_comp(pred3d_seg)
    pred_seg2d_conn_comp = con_comp(pred2d_seg)


    lesionId_reserved_in3d = set(np.unique(seg_and_fusion*pred_seg3d_conn_comp))
    lesionId_deleted_in3d =  set(np.unique(pred_seg3d_conn_comp)) - lesionId_reserved_in3d
    lesionId_reserved_in3d = lesionId_reserved_in3d - set([0])

    lesionId_reserved_in2d = set(np.unique(seg_and_fusion*pred_seg2d_conn_comp))
    lesionId_deleted_in2d =  set(np.unique(pred_seg2d_conn_comp)) - lesionId_reserved_in2d
    lesionId_reserved_in2d = lesionId_reserved_in2d - set([0])
    

    if len(lesionId_reserved_in3d)>len(lesionId_deleted_in3d):
        #delete lesion
        new_mask_3d = pred3d_seg.copy()
        for lesion_id in list(lesionId_deleted_in3d):
            new_mask_3d[pred_seg3d_conn_comp==lesion_id] = 0
        new_mask_3d[new_mask_3d>0]=1
    else:
        new_mask_3d = np.zeros_like(pred3d_seg)
        for lesion_id in list(lesionId_reserved_in3d):
            new_mask_3d[pred_seg3d_conn_comp==lesion_id] = 1

    if len(lesionId_reserved_in2d)>len(lesionId_deleted_in2d):
        #delete lesion
        new_mask_2d = pred2d_seg.copy()
        for lesion_id in list(lesionId_deleted_in2d):
            new_mask_2d[pred_seg2d_conn_comp==lesion_id] = 0
        new_mask_2d[new_mask_2d>0]=1
    else:
        new_mask_2d = np.zeros_like(pred2d_seg)
        for lesion_id in list(lesionId_reserved_in2d):
            new_mask_2d[pred_seg2d_conn_comp==lesion_id] = 1
            

    pred_seg = np.bitwise_or(new_mask_3d.astype(np.bool8), new_mask_2d.astype(np.bool8)).astype(np.uint8)


    ## delete small and border lesions

    seg_old_size_postprocessed_out = postprocess_delete_small_edge_leison(pred_seg, 4, 3)

    seg_resized_itk = sitk.GetImageFromArray(seg_old_size_postprocessed_out.astype(np.uint8))
    seg_resized_itk.SetSpacing(properties_dict_2d['itk_spacing'])
    seg_resized_itk.SetOrigin(properties_dict_2d['itk_origin'])
    seg_resized_itk.SetDirection(properties_dict_2d['itk_direction'])
    sitk.WriteImage(seg_resized_itk, out_fname)



def get_predict_original(segmentation_softmax: Union[str, np.ndarray],
                                         properties_dict: dict, order: int = 1,
                                         region_class_order: Tuple[Tuple[int]] = None,
                                         seg_postprogess_fn: callable = None, seg_postprocess_args: tuple = None,
                                         resampled_npz_fname: str = None,
                                         non_postprocessed_fname: str = None, force_separate_z: bool = None,
                                         interpolation_order_z: int = 0, verbose: bool = True):
    if verbose: print("force_separate_z:", force_separate_z, "interpolation order:", order)

    if isinstance(segmentation_softmax, str):
        assert isfile(segmentation_softmax), "If isinstance(segmentation_softmax, str) then " \
                                             "isfile(segmentation_softmax) must be True"
        del_file = deepcopy(segmentation_softmax)
        if segmentation_softmax.endswith('.npy'):
            segmentation_softmax = np.load(segmentation_softmax)
        elif segmentation_softmax.endswith('.npz'):
            segmentation_softmax = np.load(segmentation_softmax)['softmax']
        os.remove(del_file)

    # first resample, then put result into bbox of cropping, then save
    current_shape = segmentation_softmax.shape
    shape_original_after_cropping = properties_dict.get('size_after_cropping')
    shape_original_before_cropping = properties_dict.get('original_size_of_raw_data')
    # current_spacing = dct.get('spacing_after_resampling')
    # original_spacing = dct.get('original_spacing')

    if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(shape_original_after_cropping))]):
        if force_separate_z is None:
            if get_do_separate_z(properties_dict.get('original_spacing')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            elif get_do_separate_z(properties_dict.get('spacing_after_resampling')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('spacing_after_resampling'))
            else:
                do_separate_z = False
                lowres_axis = None
        else:
            do_separate_z = force_separate_z
            if do_separate_z:
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            else:
                lowres_axis = None

        if lowres_axis is not None and len(lowres_axis) != 1:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False

        if verbose: print("separate z:", do_separate_z, "lowres axis", lowres_axis)
        seg_old_spacing = resample_data_or_seg(segmentation_softmax, shape_original_after_cropping, is_seg=False,
                                               axis=lowres_axis, order=order, do_separate_z=do_separate_z,
                                               order_z=interpolation_order_z)
        # seg_old_spacing = resize_softmax_output(segmentation_softmax, shape_original_after_cropping, order=order)
    else:
        if verbose: print("no resampling necessary")
        seg_old_spacing = segmentation_softmax

    if resampled_npz_fname is not None:
        np.savez_compressed(resampled_npz_fname, softmax=seg_old_spacing.astype(np.float16))
        # this is needed for ensembling if the nonlinearity is sigmoid
        if region_class_order is not None:
            properties_dict['regions_class_order'] = region_class_order
        save_pickle(properties_dict, resampled_npz_fname[:-4] + ".pkl")

                      
    if region_class_order is None:
        seg_old_spacing = seg_old_spacing.argmax(0)
    else:
        seg_old_spacing_final = np.zeros(seg_old_spacing.shape[1:])
        for i, c in enumerate(region_class_order):
            seg_old_spacing_final[seg_old_spacing[i] > 0.5] = c
        seg_old_spacing = seg_old_spacing_final

    print('seg_old_spacing argmax shape, ', seg_old_spacing.shape)

    bbox = properties_dict.get('crop_bbox')

    if bbox is not None:
        seg_old_size = np.zeros(shape_original_before_cropping, dtype=np.uint8)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + seg_old_spacing.shape[c], shape_original_before_cropping[c]))
        seg_old_size[bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1]] = seg_old_spacing
    else:
        seg_old_size = seg_old_spacing

    if seg_postprogess_fn is not None:
        seg_old_size_postprocessed = seg_postprogess_fn(np.copy(seg_old_size), *seg_postprocess_args)
    else:
        seg_old_size_postprocessed = seg_old_size


    seg_old_size_postprocessed_out = np.zeros(seg_old_size_postprocessed.shape, dtype=seg_old_size_postprocessed.dtype)
    seg_old_size_postprocessed_out[seg_old_size_postprocessed==3]=1

    return seg_old_size_postprocessed_out
    

def save_segmentation_nifti_from_softmax_2dplus3d(segmentation_softmax_2d: Union[str, np.ndarray], 
                                        segmentation_softmax_3d: Union[str, np.ndarray], out_fname: str,
                                         properties_dict_2d: dict, properties_dict_3d: dict, order: int = 1,
                                         region_class_order: Tuple[Tuple[int]] = None,
                                         seg_postprogess_fn: callable = None, seg_postprocess_args: tuple = None,
                                         resampled_npz_fname: str = None,
                                         non_postprocessed_fname: str = None, force_separate_z: bool = None,
                                         interpolation_order_z: int = 0, verbose: bool = True):
    
    # softmax output
    seg_old_spacing_2d = get_softmax_original(segmentation_softmax_2d, properties_dict_2d, 
                        order, region_class_order, seg_postprogess_fn, seg_postprocess_args,
                        resampled_npz_fname, non_postprocessed_fname, force_separate_z,
                        interpolation_order_z, verbose)

    seg_old_spacing_3d = get_softmax_original(segmentation_softmax_3d, properties_dict_3d, 
                        order, region_class_order, seg_postprogess_fn, seg_postprocess_args,
                        resampled_npz_fname, non_postprocessed_fname, force_separate_z,
                        interpolation_order_z, verbose)

    # softmax_lesion_2d =  np.around(seg_old_spacing_2d[3,], 6)
    # softmax_lesion_3d =  np.around(seg_old_spacing_3d[3,], 6)

    # softmax_lesion_2d =  seg_old_spacing_2d[3,]
    # softmax_lesion_3d =  seg_old_spacing_3d[3,]

    # seg_prob_mask = np.around(seg_old_spacing_2d[3,], 6).astype(np.float16)
    # seg_prob_itk = sitk.GetImageFromArray(seg_prob_mask.astype(np.single))
    # seg_prob_itk.SetSpacing(properties_dict_2d['itk_spacing'])
    # seg_prob_itk.SetOrigin(properties_dict_2d['itk_origin'])
    # seg_prob_itk.SetDirection(properties_dict_2d['itk_direction'])
    # sitk.WriteImage(seg_prob_itk, 
    # out_fname.replace(out_fname.split('/')[-1], 
    #                   out_fname.split('/')[-1].replace('.nii.gz', '_2dsoftmax.nii.gz')))


    # seg_prob_mask = np.around(seg_old_spacing_3d[3,], 6).astype(np.float16)
    # seg_prob_itk = sitk.GetImageFromArray(seg_prob_mask.astype(np.single))
    # seg_prob_itk.SetSpacing(properties_dict_3d['itk_spacing'])
    # seg_prob_itk.SetOrigin(properties_dict_3d['itk_origin'])
    # seg_prob_itk.SetDirection(properties_dict_3d['itk_direction'])
    # sitk.WriteImage(seg_prob_itk, 
    # out_fname.replace(out_fname.split('/')[-1], 
    #                   out_fname.split('/')[-1].replace('.nii.gz', '_3dsoftmax.nii.gz')))


    softmax_lesion_2d =  np.around(seg_old_spacing_2d[1,], 6)
    softmax_lesion_3d =  np.around(seg_old_spacing_3d[1,], 6)


    soft_alpha = 0.55
    # print('#softmax'*50)
    # print('seg_old_spacing_2d[3,].dtype', seg_old_spacing_2d[3,].dtype)

    seg_old_spacing = soft_alpha*softmax_lesion_3d + (1-soft_alpha)*softmax_lesion_2d

    seg_old_spacing[seg_old_spacing>0.5] = 1
    seg_old_spacing[seg_old_spacing!=1] = 0


    # print('seg_old_spacing argmax shape, ', seg_old_spacing.shape)

    if seg_postprogess_fn is not None:
        seg_old_size_postprocessed = seg_postprogess_fn(np.copy(seg_old_spacing), *seg_postprocess_args)
    else:
        seg_old_size_postprocessed = seg_old_spacing


    ## delete small and border lesions

    seg_old_size_postprocessed_out = postprocess_delete_small_edge_leison(seg_old_size_postprocessed, 4, 3)

    seg_resized_itk = sitk.GetImageFromArray(seg_old_size_postprocessed_out.astype(np.uint8))
    seg_resized_itk.SetSpacing(properties_dict_2d['itk_spacing'])
    seg_resized_itk.SetOrigin(properties_dict_2d['itk_origin'])
    seg_resized_itk.SetDirection(properties_dict_2d['itk_direction'])
    sitk.WriteImage(seg_resized_itk, out_fname)


def get_softmax_original(segmentation_softmax: Union[str, np.ndarray],
                                         properties_dict: dict, order: int = 1,
                                         region_class_order: Tuple[Tuple[int]] = None,
                                         seg_postprogess_fn: callable = None, seg_postprocess_args: tuple = None,
                                         resampled_npz_fname: str = None,
                                         non_postprocessed_fname: str = None, force_separate_z: bool = None,
                                         interpolation_order_z: int = 0, verbose: bool = True):
    if verbose: print("force_separate_z:", force_separate_z, "interpolation order:", order)

    if isinstance(segmentation_softmax, str):
        assert isfile(segmentation_softmax), "If isinstance(segmentation_softmax, str) then " \
                                             "isfile(segmentation_softmax) must be True"
        del_file = deepcopy(segmentation_softmax)
        if segmentation_softmax.endswith('.npy'):
            segmentation_softmax = np.load(segmentation_softmax)
        elif segmentation_softmax.endswith('.npz'):
            segmentation_softmax = np.load(segmentation_softmax)['softmax']
        os.remove(del_file)

    # first resample, then put result into bbox of cropping, then save
    current_shape = segmentation_softmax.shape
    shape_original_after_cropping = properties_dict.get('size_after_cropping')
    shape_original_before_cropping = properties_dict.get('original_size_of_raw_data')
    # current_spacing = dct.get('spacing_after_resampling')
    # original_spacing = dct.get('original_spacing')

    if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(shape_original_after_cropping))]):
        if force_separate_z is None:
            if get_do_separate_z(properties_dict.get('original_spacing')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            elif get_do_separate_z(properties_dict.get('spacing_after_resampling')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('spacing_after_resampling'))
            else:
                do_separate_z = False
                lowres_axis = None
        else:
            do_separate_z = force_separate_z
            if do_separate_z:
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            else:
                lowres_axis = None

        if lowres_axis is not None and len(lowres_axis) != 1:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False

        if verbose: print("separate z:", do_separate_z, "lowres axis", lowres_axis)
        seg_old_spacing = resample_data_or_seg(segmentation_softmax, shape_original_after_cropping, is_seg=False,
                                               axis=lowres_axis, order=order, do_separate_z=do_separate_z,
                                               order_z=interpolation_order_z)
        # seg_old_spacing = resize_softmax_output(segmentation_softmax, shape_original_after_cropping, order=order)
    else:
        if verbose: print("no resampling necessary")
        seg_old_spacing = segmentation_softmax

    
    return seg_old_spacing




def save_segmentation_nifti_from_softmax(segmentation_softmax: Union[str, np.ndarray], out_fname: str,
                                         properties_dict: dict, order: int = 1,
                                         region_class_order: Tuple[Tuple[int]] = None,
                                         seg_postprogess_fn: callable = None, seg_postprocess_args: tuple = None,
                                         resampled_npz_fname: str = None,
                                         non_postprocessed_fname: str = None, force_separate_z: bool = None,
                                         interpolation_order_z: int = 0, verbose: bool = True):
    """
    This is a utility for writing segmentations to nifty and npz. It requires the data to have been preprocessed by
    GenericPreprocessor because it depends on the property dictionary output (dct) to know the geometry of the original
    data. segmentation_softmax does not have to have the same size in pixels as the original data, it will be
    resampled to match that. This is generally useful because the spacings our networks operate on are most of the time
    not the native spacings of the image data.
    If seg_postprogess_fn is not None then seg_postprogess_fnseg_postprogess_fn(segmentation, *seg_postprocess_args)
    will be called before nifty export
    There is a problem with python process communication that prevents us from communicating objects
    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
    patching system python code.) We circumvent that problem here by saving softmax_pred to a npy file that will
    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
    filename or np.ndarray for segmentation_softmax and will handle this automatically
    :param segmentation_softmax:
    :param out_fname:
    :param properties_dict:
    :param order:
    :param region_class_order:
    :param seg_postprogess_fn:
    :param seg_postprocess_args:
    :param resampled_npz_fname:
    :param non_postprocessed_fname:
    :param force_separate_z: if None then we dynamically decide how to resample along z, if True/False then always
    /never resample along z separately. Do not touch unless you know what you are doing
    :param interpolation_order_z: if separate z resampling is done then this is the order for resampling in z
    :param verbose:
    :return:
    """
    if verbose: print("force_separate_z:", force_separate_z, "interpolation order:", order)

    if isinstance(segmentation_softmax, str):
        assert isfile(segmentation_softmax), "If isinstance(segmentation_softmax, str) then " \
                                             "isfile(segmentation_softmax) must be True"
        del_file = deepcopy(segmentation_softmax)
        if segmentation_softmax.endswith('.npy'):
            segmentation_softmax = np.load(segmentation_softmax)
        elif segmentation_softmax.endswith('.npz'):
            segmentation_softmax = np.load(segmentation_softmax)['softmax']
        os.remove(del_file)

    # first resample, then put result into bbox of cropping, then save
    current_shape = segmentation_softmax.shape
    shape_original_after_cropping = properties_dict.get('size_after_cropping')
    shape_original_before_cropping = properties_dict.get('original_size_of_raw_data')
    # current_spacing = dct.get('spacing_after_resampling')
    # original_spacing = dct.get('original_spacing')

    if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(shape_original_after_cropping))]):
        if force_separate_z is None:
            if get_do_separate_z(properties_dict.get('original_spacing')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            elif get_do_separate_z(properties_dict.get('spacing_after_resampling')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('spacing_after_resampling'))
            else:
                do_separate_z = False
                lowres_axis = None
        else:
            do_separate_z = force_separate_z
            if do_separate_z:
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            else:
                lowres_axis = None

        if lowres_axis is not None and len(lowres_axis) != 1:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False

        if verbose: print("separate z:", do_separate_z, "lowres axis", lowres_axis)
        seg_old_spacing = resample_data_or_seg(segmentation_softmax, shape_original_after_cropping, is_seg=False,
                                               axis=lowres_axis, order=order, do_separate_z=do_separate_z,
                                               order_z=interpolation_order_z)
        # seg_old_spacing = resize_softmax_output(segmentation_softmax, shape_original_after_cropping, order=order)
    else:
        if verbose: print("no resampling necessary")
        seg_old_spacing = segmentation_softmax

    if resampled_npz_fname is not None:
        np.savez_compressed(resampled_npz_fname, softmax=seg_old_spacing.astype(np.float16))
        # this is needed for ensembling if the nonlinearity is sigmoid
        if region_class_order is not None:
            properties_dict['regions_class_order'] = region_class_order
        save_pickle(properties_dict, resampled_npz_fname[:-4] + ".pkl")

    
    # prob mask save
    seg_prob_mask = np.around(seg_old_spacing[1,], 6).astype(np.float16)
    seg_prob_itk = sitk.GetImageFromArray(seg_prob_mask.astype(np.single))
    seg_prob_itk.SetSpacing(properties_dict['itk_spacing'])
    seg_prob_itk.SetOrigin(properties_dict['itk_origin'])
    seg_prob_itk.SetDirection(properties_dict['itk_direction'])
    sitk.WriteImage(seg_prob_itk, 
    out_fname.replace(out_fname.split('/')[-1], 
                      out_fname.split('/')[-1].replace('.nii.gz', '_softmax.nii.gz')))
                      

    if region_class_order is None:
        seg_old_spacing = seg_old_spacing.argmax(0)
    else:
        seg_old_spacing_final = np.zeros(seg_old_spacing.shape[1:])
        for i, c in enumerate(region_class_order):
            seg_old_spacing_final[seg_old_spacing[i] > 0.5] = c
        seg_old_spacing = seg_old_spacing_final

    bbox = properties_dict.get('crop_bbox')

    if bbox is not None:
        seg_old_size = np.zeros(shape_original_before_cropping, dtype=np.uint8)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + seg_old_spacing.shape[c], shape_original_before_cropping[c]))
        seg_old_size[bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1]] = seg_old_spacing
    else:
        seg_old_size = seg_old_spacing

    if seg_postprogess_fn is not None:
        seg_old_size_postprocessed = seg_postprogess_fn(np.copy(seg_old_size), *seg_postprocess_args)
    else:
        seg_old_size_postprocessed = seg_old_size
    
    

    seg_resized_itk = sitk.GetImageFromArray(seg_old_size_postprocessed.astype(np.uint8))
    seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
    seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
    seg_resized_itk.SetDirection(properties_dict['itk_direction'])
    sitk.WriteImage(seg_resized_itk, out_fname)


    if (non_postprocessed_fname is not None) and (seg_postprogess_fn is not None):
        seg_resized_itk = sitk.GetImageFromArray(seg_old_size.astype(np.uint8))
        seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
        seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
        seg_resized_itk.SetDirection(properties_dict['itk_direction'])
        sitk.WriteImage(seg_resized_itk, non_postprocessed_fname)


def save_segmentation_nifti(segmentation, out_fname, dct, order=1, force_separate_z=None, order_z=0, verbose: bool = False):
    """
    faster and uses less ram than save_segmentation_nifti_from_softmax, but maybe less precise and also does not support
    softmax export (which is needed for ensembling). So it's a niche function that may be useful in some cases.
    :param segmentation:
    :param out_fname:
    :param dct:
    :param order:
    :param force_separate_z:
    :return:
    """
    # suppress output
    print("force_separate_z:", force_separate_z, "interpolation order:", order)
    if not verbose:
        sys.stdout = open(os.devnull, 'w')

    if isinstance(segmentation, str):
        assert isfile(segmentation), "If isinstance(segmentation_softmax, str) then " \
                                     "isfile(segmentation_softmax) must be True"
        del_file = deepcopy(segmentation)
        segmentation = np.load(segmentation)
        os.remove(del_file)

    # first resample, then put result into bbox of cropping, then save
    current_shape = segmentation.shape
    shape_original_after_cropping = dct.get('size_after_cropping')
    shape_original_before_cropping = dct.get('original_size_of_raw_data')
    # current_spacing = dct.get('spacing_after_resampling')
    # original_spacing = dct.get('original_spacing')

    if np.any(np.array(current_shape) != np.array(shape_original_after_cropping)):
        if order == 0:
            seg_old_spacing = resize_segmentation(segmentation, shape_original_after_cropping, 0)
        else:
            if force_separate_z is None:
                if get_do_separate_z(dct.get('original_spacing')):
                    do_separate_z = True
                    lowres_axis = get_lowres_axis(dct.get('original_spacing'))
                elif get_do_separate_z(dct.get('spacing_after_resampling')):
                    do_separate_z = True
                    lowres_axis = get_lowres_axis(dct.get('spacing_after_resampling'))
                else:
                    do_separate_z = False
                    lowres_axis = None
            else:
                do_separate_z = force_separate_z
                if do_separate_z:
                    lowres_axis = get_lowres_axis(dct.get('original_spacing'))
                else:
                    lowres_axis = None

            print("separate z:", do_separate_z, "lowres axis", lowres_axis)
            seg_old_spacing = resample_data_or_seg(segmentation[None], shape_original_after_cropping, is_seg=True,
                                                   axis=lowres_axis, order=order, do_separate_z=do_separate_z,
                                                   order_z=order_z)[0]
    else:
        seg_old_spacing = segmentation

    bbox = dct.get('crop_bbox')

    if bbox is not None:
        seg_old_size = np.zeros(shape_original_before_cropping)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + seg_old_spacing.shape[c], shape_original_before_cropping[c]))
        seg_old_size[bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1]] = seg_old_spacing
    else:
        seg_old_size = seg_old_spacing

    seg_resized_itk = sitk.GetImageFromArray(seg_old_size.astype(np.uint8))
    seg_resized_itk.SetSpacing(dct['itk_spacing'])
    seg_resized_itk.SetOrigin(dct['itk_origin'])
    seg_resized_itk.SetDirection(dct['itk_direction'])
    sitk.WriteImage(seg_resized_itk, out_fname)

    if not verbose:
        sys.stdout = sys.__stdout__
