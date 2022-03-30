import os.path
import subprocess
import SimpleITK as sitk
import shutil
import numpy as np
from SpatialTransformer import SpatialTransformer
import torch
import matplotlib.pyplot as plt


def transform_affine():
    transformix = "/exports/lkeb-hpc/tlandman/elastix-5.0.0-Linux/bin/transformix"
    plan_scan_idx = 1
    data_dir = "/exports/lkeb-hpc/tlandman/Data/Patient_MHA/"
    patient_list = sorted(os.listdir(data_dir))
    for patient_idx in range(0, len(patient_list)):
        patient_dir = os.path.join(data_dir, patient_list[patient_idx])
        scan_list = sorted([f for f in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, f))])
        planning_dose = os.path.join(patient_dir, scan_list[plan_scan_idx], 'Dose.mha')
        for scan_idx in range(0, len(scan_list)):
            if scan_idx == plan_scan_idx:
                continue
            scan_dir = os.path.join(patient_dir, scan_list[scan_idx])
            output_dir = os.path.join(scan_dir, 'planning')
            result_file = os.path.join(output_dir, 'result.mha')
            transform_file = os.path.join(output_dir, 'TransformParameters.0.txt')

            if not os.path.isfile(result_file):
                cmdStr = "%s -in %s -out %s -tp %s" % (transformix, planning_dose, output_dir, transform_file)
                with open(os.devnull, "w") as fnull:
                    affine_dose = os.path.join(output_dir, 'Planning_Dose.mha')
                    subprocess.call(cmdStr, stdout=fnull, stderr=subprocess.STDOUT, shell=True)
                    os.rename(result_file, affine_dose)
            print(patient_list[patient_idx], scan_list[scan_idx])


def check_dose():
    ## in loop scans ##
    CT_path = os.path.join(scan_dir, 'CTImage.mha')
    reader = sitk.ImageFileReader()
    reader.SetImageIO("MetaImageIO")
    reader.SetFileName(CT_path)
    CT_sitk = reader.Execute()
    dose_path = os.path.join(scan_dir, 'Dose.mha')
    reader.SetFileName(dose_path)
    dose_sitk = reader.Execute()
    if CT_sitk.GetOrigin() != dose_sitk.GetOrigin() or CT_sitk.GetSpacing() != dose_sitk.GetSpacing():
        print(patient_idx, scan_idx)
        print(CT_sitk.GetOrigin(), dose_sitk.GetOrigin())
        print(CT_sitk.GetSpacing(), dose_sitk.GetSpacing())
        dose_sitk.SetOrigin(CT_sitk.GetOrigin())
        dose_sitk.SetSpacing(CT_sitk.GetSpacing())
        writer = sitk.ImageFileWriter()
        writer.SetFileName(dose_path)
        writer.Execute(dose_sitk)

def copy_trans_files():
    ## in loop scan ##
    planning_dir = os.path.join(scan_dir, 'planning')
    if not os.path.exists(planning_dir):
        os.mkdir(planning_dir)
    transform_dir = "/exports/lkeb-hpc/mseelmahdy/ProstateReg/MICCAI_Experiments/Haukeland_NCC_Elastix2/results_NCC/NCC/FASGD100"
    transform_old = os.path.join(transform_dir, patient_list[patient_idx], scan_list[scan_idx], 'OriginalImage/TransformParameters.0.txt')
    transform_new = os.path.join(planning_dir, 'TransformParameters.0.txt')
    a_file = open(transform_old, "r")
    list_of_lines = a_file.readlines()
    list_of_lines[28] = "(DefaultPixelValue 0.000000)\n"
    list_of_lines[30] = '(ResultImagePixelType "float")\n'
    a_file = open(transform_new, "w")
    a_file.writelines(list_of_lines)
    a_file.close()
    print(patient_list[patient_idx], scan_list[scan_idx])

def loop_scans():
    data_dir = "/exports/lkeb-hpc/tlandman/Data/Patient_MHA/"
    patient_list = sorted(os.listdir(data_dir))
    for patient_idx in range(0, len(patient_list)):
        patient_dir = os.path.join(data_dir, patient_list[patient_idx])
        scan_list = sorted([f for f in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, f))])
        print(patient_list[patient_idx])
        for scan_idx in range(0, len(scan_list)):
            if scan_idx == 1:
                continue
            scan_dir = os.path.join(patient_dir, scan_list[scan_idx])

            reader = sitk.ImageFileReader()
            reader.SetImageIO("MetaImageIO")
            file = os.path.join(scan_dir, 'planning/Planning_CTImage.mha')
            reader.SetFileName(file)
            image = reader.Execute()
            test = np.array(sitk.GetArrayFromImage(image), dtype=type)
            if np.shape(test)[0]==280:
                print(patient_list[patient_idx])
                print(scan_list[scan_idx])

def loop_output_scans():
    data_dir = "/exports/lkeb-hpc/tlandman/Thesis/MultiTask/experiments/Single-Task/Reg_input_If_Im_Sm/output/HMC/"
    out_dir = '/exports/lkeb-hpc/tlandman/Thesis/MultiTask/experiments/Single-Task/Dose_input_Dm_DVF/output/HMC/'

    patient_list = sorted(f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)))
    for patient_idx in range(0, len(patient_list)):
        patient_dir = os.path.join(data_dir, patient_list[patient_idx])
        out_pat_dir = os.path.join(out_dir, patient_list[patient_idx])
        if not os.path.exists(out_pat_dir):
            os.mkdir(out_pat_dir)
        scan_list = sorted([f for f in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, f))])
        for scan_idx in range(1, 2): #len(scan_list)):
            # if scan_idx == 1:
            #     continue
            dvf_path = os.path.join(patient_dir, scan_list[scan_idx], 'DVF.mha')
            planning_dose_path = os.path.join('/exports/lkeb-hpc/tlandman/Data/Patient_MHA/', patient_list[patient_idx], scan_list[scan_idx], 'planning/Planning_Dose.mha')

            reader = sitk.ImageFileReader()
            reader.SetImageIO("MetaImageIO")
            reader.SetFileName(dvf_path)
            dvf_itk = reader.Execute()
            dvf = torch.Tensor(np.transpose(np.expand_dims(sitk.GetArrayFromImage(dvf_itk), 0), (0, 4, 1, 2, 3)))
            reader.SetFileName(planning_dose_path)
            dose_itk = reader.Execute()
            dose = torch.Tensor(np.expand_dims(np.expand_dims(sitk.GetArrayFromImage(dose_itk), 0), 0))
            spatial_transform = SpatialTransformer(dim=1)
            dose_transform = spatial_transform(dose, dvf)

            dose_transform = np.squeeze(dose_transform.numpy())
            dose_transform_itk = sitk.GetImageFromArray(dose_transform)
            dose_transform_itk.SetOrigin(dose_itk.GetOrigin())
            dose_transform_itk.SetSpacing(dose_itk.GetSpacing())

            writer = sitk.ImageFileWriter()
            trans_dir = os.path.join('/exports/lkeb-hpc/tlandman/Thesis/MultiTask/experiments/Single-Task/Dose_input_Dm_DVF/output/HMC/', patient_list[patient_idx], scan_list[scan_idx])
            if not os.path.exists(trans_dir):
                os.mkdir(trans_dir)
            trans_file = os.path.join(trans_dir, 'Dose.mha')
            writer.SetFileName(trans_file)
            writer.Execute(dose_transform_itk)

            print(patient_list[patient_idx], scan_list[scan_idx])
def make_2D_masks():
    ## in create masks ##
    mask = np.zeros(np.shape(cont))
    coords = np.where(np.any(cont == 4, axis=2))
    for i in range(np.shape(coords)[1]):
        mask[coords[0][i], coords[1][i], :] = 1
    coords = np.where(np.any(cont==3, axis=2))
    for i in range(np.shape(coords)[1]):
        mask[coords[0][i], coords[1][i], :] = 2
    file = os.path.join(scan_dir, 'masks/Torso.mha')
    reader.SetFileName(file)
    image = reader.Execute()
    torso = sitk.GetArrayFromImage(image)
    mask = np.where((torso == 1) & ((mask == 1) | (mask == 2)), mask, 0)
    mask = np.array(mask, dtype=np.uint8)
    mask_itk = sitk.GetImageFromArray(mask)
    writer = sitk.ImageFileWriter()
    mask_file = os.path.join(scan_dir, 'target_mask.mha')
    writer.SetFileName(mask_file)
    writer.Execute(mask_itk)


def create_masks():
    data_dir = "/exports/lkeb-hpc/tlandman/Data/Patient_MHA/"
    data_dir_2 = "/exports/lkeb-hpc/tlandman/Thesis/MultiTask/experiments/Single-Task/Seg_input_If_Im_Sm/output/HMC/"
    patient_list = sorted([f for f in os.listdir(data_dir_2) if os.path.isdir(os.path.join(data_dir_2, f))])
    for patient_idx in range(0, len(patient_list)):
        patient_dir = os.path.join(data_dir, patient_list[patient_idx])
        patient_dir_2 = os.path.join(data_dir_2, patient_list[patient_idx])
        scan_list = sorted([f for f in os.listdir(patient_dir_2) if os.path.isdir(os.path.join(patient_dir_2, f))])
        print(patient_list[patient_idx])
        for scan_idx in range(0, len(scan_list)):
            # if scan_idx == 1:
            #     continue
            scan_dir = os.path.join(patient_dir, scan_list[scan_idx])
            scan_dir_2 = os.path.join(patient_dir_2, scan_list[scan_idx])

            reader = sitk.ImageFileReader()
            reader.SetImageIO("MetaImageIO")
            file = os.path.join(scan_dir_2, 'Segmentation.mha')
            reader.SetFileName(file)
            # image = reader.Execute()
            # dose = np.array(sitk.GetArrayFromImage(image))

            # reader.SetFileName(file)
            cont_itk = reader.Execute()
            cont = np.array(sitk.GetArrayFromImage(cont_itk))
            mask = np.zeros(np.shape(cont))
            coords = np.where(np.any(cont == 4, axis=2))
            for i in range(np.shape(coords)[1]):
                mask[coords[0][i], coords[1][i], :] = 1
            coords = np.where(np.any(cont == 3, axis=2))
            for i in range(np.shape(coords)[1]):
                mask[coords[0][i], coords[1][i], :] = 2

            file = os.path.join(scan_dir, 'masks/Torso.mha')
            reader.SetFileName(file)
            image = reader.Execute()
            torso = sitk.GetArrayFromImage(image)
            mask = np.where((torso == 1) & ((mask == 1) | (mask == 2)), mask, 0)
            mask = np.array(mask, dtype=np.uint8)
            mask_itk = sitk.GetImageFromArray(mask)
            writer = sitk.ImageFileWriter()
            output_seg_mask = os.path.join(scan_dir, 'output_targets_mask.mha')
            writer.SetFileName(output_seg_mask)
            # writer.Execute(mask_itk)
            writer = sitk.ImageFileWriter()
            output_cont= os.path.join(scan_dir, 'output_segmentation.mha')
            writer.SetFileName(output_cont)
            # writer.Execute(cont_itk)
            plt.imshow(mask[50,:,:])
            plt.show()


create_masks()
# loop_output_scans()
# loop_scans()
# transform_affine()

