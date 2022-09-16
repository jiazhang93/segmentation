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
from typing import Tuple

import numpy as np
import torch
from nnunet.network_architecture.generic_modular_residual_UNet import FabiansUNet, get_default_network_config
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper


from nnunet.network_architecture.neural_network import SegmentationNetwork
from torch import nn
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch.cuda.amp import autocast
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss #DC_and_CE_and_ClassificationCE_loss
# from nnunet.training.data_augmentation.data_augmentation_moreDA_classification import get_moreDA_augmentation_classification
from nnunet.training.dataloading.dataset_loading import unpack_dataset

class nnUNetTrainerV2_ResencUNet_ori(nnUNetTrainerV2):
    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss

            self.seg_loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
            # deep supervision
            self.ds_seg_loss =  MultipleOutputLoss2(self.seg_loss, self.ds_loss_weights)
            # self.classification_loss = ClassificationCE_loss()
            # self.loss = self.ds_seg_loss + self.classification_loss


            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            # if training:
            #     self.dl_tr, self.dl_val = self.get_basic_generators()
            #     if self.unpack_data:
            #         print("unpacking dataset")
            #         unpack_dataset(self.folder_with_preprocessed_data)
            #         # unpack_dataset(self.folder_with_preprocessed_data+'_organmask')
            #         print("done")
            #     else:
            #         print(
            #             "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
            #             "will wait all winter for your model to finish!")

            #     self.tr_gen, self.val_gen = get_moreDA_augmentation_classification(
            #         self.dl_tr, self.dl_val,
            #         self.data_aug_params[
            #             'patch_size_for_spatialtransform'],
            #         self.data_aug_params,
            #         deep_supervision_scales=self.deep_supervision_scales,
            #         pin_memory=self.pin_memory,
            #         use_nondetMultiThreadedAugmenter=False
            #     )

            #     self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
            #                            also_print_to_console=False)
            #     self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
            #                            also_print_to_console=False)
            # else:
            #     pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        if self.threeD:
            cfg = get_default_network_config(3, None, norm_type="in") 
            """
            {'conv_op': torch.nn.modules.conv.Conv3d,
             'dropout_op': None,
             'norm_op': torch.nn.modules.instancenorm.InstanceNorm3d,
             'norm_op_kwargs': {'eps': 1e-05, 'affine': True},
             'dropout_op_kwargs': {'p': 0, 'inplace': True},
             'conv_op_kwargs': {'stride': 1, 'dilation': 1, 'bias': True},
             'nonlin': torch.nn.modules.activation.LeakyReLU,
             'nonlin_kwargs': {'negative_slope': 0.01, 'inplace': True}}
            """
        else:
            cfg = get_default_network_config(1, None, norm_type="in")

        stage_plans = self.plans['plans_per_stage'][self.stage]
        conv_kernel_sizes = stage_plans['conv_kernel_sizes']
#         blocks_per_stage_encoder = stage_plans['num_blocks_encoder']
#         blocks_per_stage_decoder = stage_plans['num_blocks_decoder']

        # optimizer and lr reset
        # self.initial_lr = 1e-3
        # self.optimizer.param_groups[0]["momentum"] = 0.95
        # blocks_per_stage_encoder = FabiansUNet.default_blocks_per_stage_encoder
        # blocks_per_stage_decoder = FabiansUNet.default_blocks_per_stage_decoder
        blocks_per_stage_encoder = (1,2,2,2,2,2)
        blocks_per_stage_decoder = (1,1,1,1,1,1)
     
        # pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']
        
        pool_op_kernel_sizes = [[1, 1, 1],
                                [2, 2, 2],
                                [2, 2, 2],
                                [2, 2, 2],
                                [2, 2, 2],
                                [2, 2, 2]
                                ]

        conv_op_kernel_sizes = [[3, 3, 3],
                                [3, 3, 3],
                                [3, 3, 3],
                                [3, 3, 3],
                                [3, 3, 3],
                                [3, 3, 3]
                                ]
        
        # self.num_classes = 2
        self.network = FabiansUNet(self.num_input_channels, self.base_num_features, 
                                   blocks_per_stage_encoder[:len(conv_op_kernel_sizes)], 2,
                                   pool_op_kernel_sizes, conv_op_kernel_sizes, cfg, self.num_classes,
                                   blocks_per_stage_decoder[:len(conv_op_kernel_sizes)-1], True, False, 
                                   320, InitWeights_He(1e-2))

        self.num_classes = self.plans['num_classes'] + 1  # background is no longer in num_classes

        # self.network = ResidualClassificationNet(self.num_input_channels, self.base_num_features, 
        #                            blocks_per_stage_encoder[:len(conv_op_kernel_sizes)], 2,
        #                            pool_op_kernel_sizes, conv_op_kernel_sizes, cfg, self.num_classes,
        #                            blocks_per_stage_decoder[:len(conv_op_kernel_sizes)-1], True, False, 
        #                            320, InitWeights_He(1e-2))


        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        # self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                                  momentum=0.99, nesterov=True)

        # self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                                  momentum=0.95, nesterov=True)

        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.95, nesterov=True)

        self.lr_scheduler = None
        
    def setup_DA_params(self):
        """
        net_num_pool_op_kernel_sizes is different in resunet
        """
        self.net_num_pool_op_kernel_sizes= [[1, 1, 1],
                                            [2, 2, 2],
                                            [2, 2, 2],
                                            [2, 2, 2],
                                            [2, 2, 2],
                                            [2, 2, 2]
                                            ]

        super().setup_DA_params()
        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes[1:]), axis=0))[:-1]

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 force_separate_z: bool = None, interpolation_order: int = 3, interpolation_order_z=0,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = False
        ret = nnUNetTrainer.validate(self, do_mirroring=do_mirroring, use_sliding_window=use_sliding_window,
                                     step_size=step_size, save_softmax=save_softmax, use_gaussian=use_gaussian,
                                     overwrite=overwrite, validation_folder_name=validation_folder_name,
                                     debug=debug, all_in_gpu=all_in_gpu,
                                     segmentation_export_kwargs=segmentation_export_kwargs,
                                     run_postprocessing_on_folds=run_postprocessing_on_folds)
        self.network.decoder.deep_supervision = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = False
        ret = nnUNetTrainer.predict_preprocessed_data_return_seg_and_softmax(self, data, do_mirroring=do_mirroring,
                                                                             mirror_axes=mirror_axes,
                                                                             use_sliding_window=use_sliding_window,
                                                                             step_size=step_size,
                                                                             use_gaussian=use_gaussian,
                                                                             pad_border_mode=pad_border_mode,
                                                                             pad_kwargs=pad_kwargs,
                                                                             all_in_gpu=all_in_gpu,
                                                                             verbose=verbose,
                                                                             mixed_precision=mixed_precision)
        self.network.decoder.deep_supervision = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax_cls(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision: bool = True, classification_flag=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param data:
        :param do_mirroring:
        :param mirror_axes:
        :param use_sliding_window:
        :param step_size:
        :param use_gaussian:
        :param pad_border_mode:
        :param pad_kwargs:
        :param all_in_gpu:
        :param verbose:
        :return:
        """

        ### here classification_flag set to True, to calculate classification results

        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = False
        # self.network.do_ds = False
        # self.network.no_grad_state = True


        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']

        if do_mirroring:
            assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"

        valid = list((SegmentationNetwork, nn.DataParallel))
        assert isinstance(self.network, tuple(valid))

        current_mode = self.network.training
        self.network.eval()


        # jz add classification output
        ret, softmax, dummy_cls = self.network.predict_3D(data, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                      use_sliding_window=use_sliding_window, step_size=step_size,
                                      patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                                      use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                      pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                      mixed_precision=mixed_precision,classification_flag=classification_flag)


        self.network.train(current_mode)

        self.network.decoder.deep_supervision = ds
        # self.network.do_ds = ds
        # self.network.no_grad_state = no_grad_state
        print('IN nnUNetTrainerV2_PercentPriorNet, predict softmax done!!!')
        return ret, softmax, dummy_cls


    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
            
        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.ds_seg_loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.ds_seg_loss(output, target) 

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()


    def run_training(self):
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = True
        ret = nnUNetTrainer.run_training(self)
        self.network.decoder.deep_supervision = ds
        return ret