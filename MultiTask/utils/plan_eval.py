# from dose_char import *
import csv
import os
import matplotlib.pyplot as plt
import pydicom
import numpy as np
from rt_utils import RTStructBuilder
import SimpleITK as sitk

def get_dose_paths():
    dose_paths = []
    with open("/exports/lkeb-hpc/tlandman/Data/dose_paths.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            dose_paths.append(str(row[0]))
    return dose_paths

def get_struc_paths():
    struc_paths = []
    with open("/exports/lkeb-hpc/tlandman/Data/GTV_paths.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            struc_paths.append(str(row[0]))
    return struc_paths

def load_dose_dcm(data_path):
    dose_path = os.path.join(data_path, "Dose", "RTDose_4_physicalDose.dcm")
    ds = pydicom.dcmread(dose_path)
    dose_dcm = np.float32(ds.pixel_array*ds.DoseGridScaling)
    return dose_dcm

def load_struc_dcm(data_path):
    # in principe niet meer nodig
    struct_path = os.path.join(data_path, "Struct", "RTstruct.dcm")
    CT_path = os.path.join(data_path , "CT")
    # Load existing RT Struct. Requires the series path and existing RT Struct path

    rtstruct = RTStructBuilder.create_from(CT_path, struct_path)

    # Obtain all of the ROI names from within the image
    contours = rtstruct.get_roi_names()

    # Get the dimensions of the structures and create emty array
    refDs = np.shape(rtstruct.get_roi_mask_by_name("Bladder"))
    structs_array = np.zeros((1, len(contours), refDs[0], refDs[1], refDs[2]), dtype=np.float32)

    # Loading the 3D Masks from within the RT Struct
    for i in range(len(contours)):
        structs_array[:,i,:,:,:] = rtstruct.get_roi_mask_by_name(contours[i])

    return structs_array

def load_dose_mha(dose_path):
    """Calculates mean dose within a structure

    :params dose: array with dose information
    :params struct_arr: structure information
    :params structure: name of structure
    :return dmean: Mean dose
    """
    reader = sitk.ImageFileReader()
    reader.SetImageIO("MetaImageIO")
    reader.SetFileName(dose_path)
    image = reader.Execute()
    dose = np.moveaxis(sitk.GetArrayFromImage(image),1,-1)
    return dose

def load_struc_mha(struc_path):
    reader = sitk.ImageFileReader()
    reader.SetImageIO("MetaImageIO")
    reader.SetFileName(struc_path)
    image = reader.Execute()
    struc = np.array(sitk.GetArrayFromImage(image), dtype=bool)
    return struc

def load_struc_nrrd(struc_path):
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NrrdImageIO")
    reader.SetFileName(struc_path)
    image = reader.Execute()
    struc = np.array(sitk.GetArrayFromImage(image), dtype=bool)
    return struc

def plan_hist():
    dose_paths = get_dose_paths()
    struc_paths = get_struc_paths()
    mean_doses_mha = []
    mean_doses_dcm = []
    with open("/exports/lkeb-hpc/tlandman/Thesis/temp/plan_quality.csv", 'w') as f:
        for i in range(len(dose_paths)):
            dose_path = dose_paths[i]
            struc_path = struc_paths[i]
            struc_mha = load_struc_mha(struc_path)
            dose_mha = load_dose_mha(dose_path)
            mean_gtv =

            mean_doses_mha.append(np.mean(dose_mha[struc_mha]))
            mean_doses_dcm.append(np.mean(dose_dcm[struc_mha]))

            print(mean_doses_mha[-1])

    with open("/exports/lkeb-hpc/tlandman/Thesis/temp/mean_rectum.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(mean_doses)


    plt.figure()
    plt.hist(mean_doses)
    plt.xlabel('V95 GTV [-]')
    plt.show()

def check_doses():
    dose_paths = get_dose_paths()
    struc_paths = get_struc_paths()

    mean_doses = []
    for i in range(22,40): #len(dose_paths)):
        dose_path = dose_paths[i]
        struc_path = struc_paths[i]
        dcm_path = "/".join(dose_path.split('/')[:-2])
        dose, struc, gtv_dose = mean_dose(dose_path, struc_path)
        struc_dcm = np.array(np.moveaxis(np.moveaxis(np.squeeze(load_struct(dcm_path)[:,0,:,:,:]),0,-1),0,1), dtype=bool)
        dose_dcm = np.moveaxis(np.moveaxis(np.squeeze(load_dose(dcm_path)), 0, -1), 0, 1)

        if struc.all() == struc_dcm.all():
            print("fout")
            print(np.sum(struc))
            print(np.sum(struc_dcm))
        else:
            print("goed")
        # print(np.shape(dose))
        # print(np.shape(dose_dcm))
        # print(np.shape(struc))
        # print(np.shape(struc_dcm))
        # print(np.shape(dose[np.nonzero(struc)]))
        # print(np.shape(dose_dcm[np.nonzero(struc_dcm)]))
        # mean_doses.append(np.mean(dose[struc_dcm]))
        # print(mean_doses[-1])
        # if struc.all() == struc_dcm.all():
        #     print("Het gaat goed!")
        # else:
        #     print("Het gaat fout")
        #     print(np.max(struc-struc_dcm))
        #     print(dose_path)
    plt.figure()
    plt.hist(mean_doses)
    plt.show()

def check_strucs():
    dose_paths = get_dose_paths()
    struc_paths = get_struc_paths()
    for i in range(97,98): # len(dose_paths)):
        struc_mha = load_struc_mha(struc_paths[i])
        dose_mha = load_dose_mha(dose_paths[i])
        # dcm_path = "/".join(dose_paths[i].split('/')[:-2])
        dcm_path = "/exports/lkeb-hpc/tlandman/temp/plan/"
        dose_dcm = np.moveaxis(np.squeeze(load_dose_dcm(dcm_path)),-1,0)
        struc_dcm = np.array(np.moveaxis(np.squeeze(load_struc_dcm(dcm_path)[:, 0, :, :, :]), -1, 0) ,dtype=bool)
        # ms_path = "/exports/lkeb-hpc/mseelmahdy/HaukelandAffine/"+"/".join(dose_paths[i].split('/')[6:-2])+"/Contours_Cleaned/GTV.mha"
        # struc_ms = load_struc_mha(ms_path)
        # nrrd_path = "/exports/lkeb-hpc/tlandman/Data/Patient_NRRD/" + "/".join(dose_paths[i].split('/')[6:-2]) + "/masks/Torso.nrrd"
        # struc_nrrd = load_struc_nrrd(nrrd_path)
        diff = struc_dcm.astype(np.float32) - struc_mha.astype(np.float32)
        print(np.sum(struc_dcm.astype(np.float32)))
        print(np.sum(struc_mha.astype(np.float32)))
        print(np.nonzero(diff))
        print(np.shape(struc_mha))
        print(type(struc_mha[0,0,0]))
        print(np.shape(dose_dcm))
        print(type(dose_dcm[0, 0, 0]))
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(struc_mha[42,:,:])
        plt.subplot(1,3,2)
        plt.imshow(dose_mha[42, :, :])
        plt.subplot(1,3,3)
        plt.imshow(dose_dcm[42, :, :])
        plt.show()
        # if (struc_dcm == struc_ms).all():
        #     print("Goed!!!")
        # else:
        #     print("Fout!!!")
        #     print(struc_dcm.sum())
        #     print(struc_ms.sum())

plan_hist()
# plan_hist()


        # GTV_vox = dose[struc]
        # NVox = GTV_vox.size
        # ReqVox = GTV_vox >= 63.65
        # V95 = sum(ReqVox) / NVox
        # ReqVox = GTV_vox >= 71.69
        # V107 = sum(ReqVox) / NVox
        #
        # print(str(V95)+" "+str(V107))
        # mean_doses.append(V95)
        # mean_dose.append(V107)

# patient = "Patient_16/visit_20070903"
# dose_path_mha = "/exports/lkeb-hpc/tlandman/Data/Patient_DCM/"+patient+"/Dose/RTDose_4_physicalDose.mha"
# dcm_path = "/".join(dose_path_mha.split('/')[:-2])
# print(dcm_path)
# struc_path_mha = "/exports/lkeb-hpc/tlandman/Data/Patient_MHA/"+patient+"/masks/GTV.mha"
# dose_path_dcm = "/exports/lkeb-hpc/tlandman/Data/Patient_DCM/"+patient
# struc_path_dcm = "/exports/lkeb-hpc/tlandman/Data/Patient_DCM/"+patient
# dose, struc = mean_dose(dose_path_mha, struc_path_mha)
# # print(mean_dose_mha)
# dose_dcm = np.squeeze(load_dose(dose_path_dcm))
# struc_dcm = load_struct(struc_path_dcm)
# print(np.shape(dose_dcm))
# struc_dcm = np.array(np.squeeze(struc_dcm[:,0,:,:,:]), dtype=bool)
# print(np.shape(struc_dcm))
# print(np.mean(dose_dcm[struc_dcm]))
# print(np.max(dose_dcm))
# dose = np.moveaxis(dose, 0,-1)
# dose = np.moveaxis(dose, 0, 1)
#
# print(np.mean(dose[struc_dcm]))