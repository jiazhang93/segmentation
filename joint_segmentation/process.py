import SimpleITK
import time
import os

import subprocess
import shutil


#from nnunet.inference.predict import predict_from_folder
from predict import predict_from_folder
from nnunet.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
from batchgenerators.utilities.file_and_folder_operations import join, isdir
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
import torch


class noorgan2plus3():  # SegmentationAlgorithm is not inherited in this class anymore

    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        # set some paths and parameters
        # self.input_path = '/input/'  # according to the specified grand-challenge interfaces
        # self.output_path = '/output/images/automated-petct-lesion-segmentation/'  # according to the specified grand-challenge interfaces
        # self.nii_path = '/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs'
        # self.result_path = '/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result'
        # self.nii_seg_file = 'TCIA_001.nii.gz'

        self.input_path = '/input/'  # according to the specified grand-challenge interfaces
        # self.input_path = '/input/images/automated-petct-lesion-segmentation/'
        self.output_path = '/output/images/automated-petct-lesion-segmentation/'
        self.nii_path = '/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task501_pet/imagesTs'
        self.result_path = '/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task501_pet/result'
        # self.nii_seg_file = 'pet_001.nii.gz'
        self.nii_seg_file = 'autopet_501.nii.gz'
        # self.chk = 'model_best'
        self.chk = 'model_final_checkpoint'



        # self.ckpt_path = '/opt/algorithm/checkpoints/nnUNet/res18organbestmodel'
        
        self.ckpt_path_2d = '/opt/algorithm/checkpoints/nnUNet/weights/weights_2d'
        self.ckpt_path_3d = '/opt/algorithm/checkpoints/nnUNet/weights/weights_3d'


        self.model_2d = '2d'
        self.model_3d = '3d_fullres'
        self.folds_2d = 'None'
        self.folds_3d = [1,2]

        print(os.listdir('/opt/algorithm/checkpoints/nnUNet'))
        print(os.listdir('/opt/algorithm/'))
        print('WETIGHTS PATH, 2d, 3d ', os.listdir(self.ckpt_path_2d),os.listdir(self.ckpt_path_3d) )
        print(self.input_path, self.output_path, self.nii_seg_file)

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):  #nnUNet specific
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):  #nnUNet specific
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print('Checking GPU availability')
        is_available = torch.cuda.is_available()
        print('Available: ' + str(is_available))
        print(f'Device count: {torch.cuda.device_count()}')
        if is_available:
            print(f'Current device: {torch.cuda.current_device()}')
            print('Device name: ' + torch.cuda.get_device_name(0))
            print('Device memory: ' + str(torch.cuda.get_device_properties(0).total_memory))

    def load_inputs(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        print(os.listdir(self.input_path))
        # print(os.listdir())
        ct_mha = os.listdir(os.path.join(self.input_path, 'images/ct/'))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, 'images/pet/'))[0]
        uuid = os.path.splitext(ct_mha)[0]

        # self.convert_mha_to_nii(os.path.join(self.input_path, 'images/pet/', pet_mha),
        #                         os.path.join(self.nii_path, 'TCIA_001_0000.nii.gz'))
        # self.convert_mha_to_nii(os.path.join(self.input_path, 'images/ct/', ct_mha),
        #                         os.path.join(self.nii_path, 'TCIA_001_0001.nii.gz'))

        self.convert_mha_to_nii(os.path.join(self.input_path, 'images/pet/', pet_mha),
                                os.path.join(self.nii_path,  'autopet_501_0000.nii.gz'))
        self.convert_mha_to_nii(os.path.join(self.input_path, 'images/ct/', ct_mha),
                                os.path.join(self.nii_path, 'autopet_501_0001.nii.gz'))
        print('uuid, ', uuid)
        return uuid

    def write_outputs(self, uuid):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        pred_img = SimpleITK.ReadImage(os.path.join(self.result_path, self.nii_seg_file))
        tmp_pred = SimpleITK.GetArrayFromImage(pred_img)
        import numpy as np

        print('pred shape, max,min, ', tmp_pred.shape, np.max(np.ravel(tmp_pred)), np.min(np.ravel(tmp_pred)))
        self.convert_nii_to_mha(os.path.join(self.result_path, self.nii_seg_file), os.path.join(self.output_path, uuid + ".mha"))
        print('Output written to: ' + os.path.join(self.output_path, uuid + ".mha"))
        print('outpath isdir, ', os.listdir(self.output_path))

    def predict(self):
        """
        Your algorithm goes here
        """
        #cproc = subprocess.run(f'nnUNet_predict -i {self.nii_path} -o {self.result_path} -t 001 -m 3d_fullres', shell=True, check=True)
        #os.system(f'nnUNet_predict -i {self.nii_path} -o {self.result_path} -t 001 -m 3d_fullres')
        print("nnUNet segmentation starting!")
        input_folder = self.nii_path
        output_folder = self.result_path
        part_id = 0  #args.part_id
        num_parts = 1  #args.num_parts
        # folds = 'None'  # args.folds
        # folds = [4] # use folder_4
        folds_2d = self.folds_2d
        folds_3d = self.folds_3d

        save_npz = False  #args.save_npz
        lowres_segmentations = 'None'  #args.lowres_segmentations
        num_threads_preprocessing = 1  #args.num_threads_preprocessing
        num_threads_nifti_save = 1  # args.num_threads_nifti_save
        disable_tta = False  #args.disable_tta
        step_size = 0.5  #args.step_size
        # interp_order = args.interp_order
        # interp_order_z = args.interp_order_z
        # force_separate_z = args.force_separate_z
        overwrite_existing = True  #args.overwrite_existing
        mode = 'normal'  #args.mode
        all_in_gpu = 'None'  #args.all_in_gpu
        # model = '3d_fullres'  # args.model
        # model = '3d_cascade_fullres'
        # model = self.model
        model_2d = self.model_2d
        model_3d = self.model_3d
        trainer_class_name = default_trainer  #args.trainer_class_name
        cascade_trainer_class_name = default_cascade_trainer  # args.cascade_trainer_class_name
        disable_mixed_precision = False  #args.disable_mixed_precision
        plans_identifier = default_plans_identifier
        # chk = 'model_final_checkpoint'
        chk = self.chk

        # task_name = '001'
        task_name = '501'

        if not task_name.startswith("Task"):
            task_id = int(task_name)
            task_name = convert_id_to_task_name(task_id)

        assert model_2d in ["2d", "3d_lowres", "3d_fullres",
                         "3d_cascade_fullres"], "-m must be 2d, 3d_lowres, 3d_fullres or " \
                                                "3d_cascade_fullres"

        assert model_3d in ["2d", "3d_lowres", "3d_fullres",
                         "3d_cascade_fullres"], "-m must be 2d, 3d_lowres, 3d_fullres or " \
                                                "3d_cascade_fullres"

        if lowres_segmentations == "None":
            lowres_segmentations = None

        if isinstance(folds_2d, list):
            if folds_2d[0] == 'all' and len(folds_2d) == 1:
                pass
            else:
                folds_2d = [int(i) for i in folds_2d]
        elif folds_2d == "None":
            folds_2d = None
        else:
            raise ValueError("Unexpected value for argument folds")


        if isinstance(folds_3d, list):
            if folds_3d[0] == 'all' and len(folds_3d) == 1:
                pass
            else:
                folds_3d = [int(i) for i in folds_3d]
        elif folds_3d == "None":
            folds_3d = None
        else:
            raise ValueError("Unexpected value for argument folds")

        assert all_in_gpu in ['None', 'False', 'True']
        if all_in_gpu == "None":
            all_in_gpu = None
        elif all_in_gpu == "True":
            all_in_gpu = True
        elif all_in_gpu == "False":
            all_in_gpu = False

        model_folder_name_2d = self.ckpt_path_2d
        model_folder_name_3d = self.ckpt_path_3d
        print("using model stored in ", model_folder_name_2d)
        assert isdir(model_folder_name_2d), "model output folder not found. Expected: %s" % model_folder_name_2d


        predict_from_folder(model_folder_name_2d, model_folder_name_3d, input_folder, output_folder, folds_2d, folds_3d, 
                           save_npz, num_threads_preprocessing,
                           num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                           overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                           mixed_precision=not disable_mixed_precision,
                           step_size=step_size, checkpoint_name=chk, disable_postprocessing=True)

        print("nnUNet segmentation done!")
        if not os.path.exists(os.path.join(self.result_path, self.nii_seg_file)):
            print('waiting for nnUNet segmentation to be created')
        while not os.path.exists(os.path.join(self.result_path, self.nii_seg_file)):
            print('.', end='')
            time.sleep(5)
        #print(cproc)  # since nnUNet_predict call is split into prediction and postprocess, a pre-mature exit code is received but segmentation file not yet written. This hack ensures that all spawned subprocesses are finished before being printed.
        print('Prediction finished')

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        # process function will be called once for each test sample
        self.check_gpu()
        print('Start processing')
        uuid = self.load_inputs()
        print('Start prediction')
        self.predict()
        print('Start output writing')
        self.write_outputs(uuid)


if __name__ == "__main__":
    print("START")
    noorgan2plus3().process()