# from dose_char import *
import csv
import os
import matplotlib.pyplot as plt
import pydicom
import numpy as np
from rt_utils import RTStructBuilder
import SimpleITK as sitk
import pandas as pd

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

def get_paths_csv(csv_path):
    data_paths = []
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            data_paths.append(str(row[1]))
    return data_paths

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

def mean_doses():
    dose_paths = get_paths_csv("/exports/lkeb-hpc/tlandman/Thesis/MultiTask/data/thomas/fixed_Dose.csv")
    gtv_paths = get_paths_csv("/exports/lkeb-hpc/tlandman/Thesis/MultiTask/data/thomas/fixed_GTV.csv")
    sv_paths = get_paths_csv("/exports/lkeb-hpc/tlandman/Thesis/MultiTask/data/thomas/fixed_SeminalVesicle.csv")
    bladder_paths = get_paths_csv("/exports/lkeb-hpc/tlandman/Thesis/MultiTask/data/thomas/fixed_Bladder.csv")
    rectum_paths = get_paths_csv("/exports/lkeb-hpc/tlandman/Thesis/MultiTask/data/thomas/fixed_Rectum.csv")

    with open("/exports/lkeb-hpc/tlandman/Thesis/temp/mean_doses.csv***SAFE***", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["mean dose gtv","mean dose sv","mean dose rectum","mean dose bladder"])
        for i in range(len(dose_paths)):
            dose_path = dose_paths[i]
            gtv_path = gtv_paths[i]
            sv_path = sv_paths[i]
            bladder_path = bladder_paths[i]
            rectum_path = rectum_paths[i]
            gtv_mha = load_struc_mha(gtv_path)
            sv_mha = load_struc_mha(sv_path)
            rectum_mha = load_struc_mha(rectum_path)
            bladder_mha = load_struc_mha(bladder_path)
            dose_mha = load_dose_mha(dose_path)
            mean_gtv = np.mean(dose_mha[gtv_mha])
            mean_sv = np.mean(dose_mha[sv_mha])
            mean_bladder = np.mean(dose_mha[bladder_mha])
            mean_rectum = np.mean(dose_mha[rectum_mha])
            writer.writerow([mean_gtv, mean_sv, mean_rectum, mean_bladder])

def V95():
    dose_paths = get_paths_csv("/exports/lkeb-hpc/tlandman/Thesis/MultiTask/data/thomas/fixed_Dose.csv")
    gtv_paths = get_paths_csv("/exports/lkeb-hpc/tlandman/Thesis/MultiTask/data/thomas/fixed_GTV.csv")
    sv_paths = get_paths_csv("/exports/lkeb-hpc/tlandman/Thesis/MultiTask/data/thomas/fixed_SeminalVesicle.csv")

    with open("/exports/lkeb-hpc/tlandman/Thesis/temp/V95_doses.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["V95 gtv","V107 gtv","v95 sv","v107 sv"])
        for i in range(len(dose_paths)):
            dose_path = dose_paths[i]
            gtv_path = gtv_paths[i]
            sv_path = sv_paths[i]
            dose = load_dose_mha(dose_path)
            gtv_dose = dose[load_struc_mha(gtv_path)]
            sv_dose = dose[load_struc_mha(sv_path)]

            v95_gtv = np.sum(gtv_dose >= 68 * 0.95) / np.size(gtv_dose)
            v107_gtv = np.sum(gtv_dose >= 68 * 1.07) / np.size(gtv_dose)
            v95_sv = np.sum(sv_dose >= 50 * 0.95) / np.size(sv_dose)
            v107_sv = np.sum(sv_dose >= 50 * 1.07) / np.size(sv_dose)

            writer.writerow([v95_gtv, v107_gtv, v95_sv, v107_sv])

def dose_hist():
    csv_path = "/exports/lkeb-hpc/tlandman/Thesis/temp/mean_doses.csv"
    with open(csv_path, newline='') as f:
        df = pd.read_csv(csv_path)
        mean_doses = df.to_numpy(float)

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Histogram of mean doses')
    axs[0, 0].hist(mean_doses[:,0], bins=np.linspace(66,74,33), color='b', label='GTV', rwidth=0.8)
    axs[0, 1].hist(mean_doses[:,1], bins=np.linspace(48,56,33), color='r', label='Seminal Vesicle', rwidth=0.8)
    axs[1, 0].hist(mean_doses[:,2], bins=np.linspace(1,9,33), color='g', label='Rectum', rwidth=0.8)
    axs[1, 1].hist(mean_doses[:,3], bins=np.linspace(1,9,33), color='m', label='Bladder', rwidth=0.8)

    for _, ax in enumerate(axs.flat):
        ax.legend()
        ax.set_ylim([0, 50])
        ax.set_xlabel('Mean Dose [Gy]')
    plt.show()

def v95_hist():
    csv_path = "/exports/lkeb-hpc/tlandman/Thesis/temp/V95_doses.csv"
    with open(csv_path, newline='') as f:
        df = pd.read_csv(csv_path)
        mean_doses = df.to_numpy(float)
    print(np.sum((mean_doses[:, 0] >= 0.95)) / np.size(mean_doses[:, 0]))
    print(np.sum((mean_doses[:, 1] <= 0.05)) / np.size(mean_doses[:, 1]))
    print(np.sum((mean_doses[:, 2] >= 0.95)) / np.size(mean_doses[:, 2]))
    print(np.sum((mean_doses[:, 3] <= 0.05)) / np.size(mean_doses[:, 3]))

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].hist(mean_doses[:,0], bins=np.linspace(0.92,1,33), color='b', label='V95 GTV', rwidth=0.8)
    axs[0, 1].hist(mean_doses[:,1], bins=np.linspace(0,0.08,33), color='r', label='V107 GTV', rwidth=0.8)
    axs[1, 0].hist(mean_doses[:,2], bins=np.linspace(0.92,1,33), color='g', label='V95 Seminal Vesicle', rwidth=0.8)
    axs[1, 1].hist(mean_doses[:,3], bins=np.linspace(0,0.08,33), color='m', label='V107 Seminal Vesicle', rwidth=0.8)

    for _, ax in enumerate(axs.flat):
        ax.legend()
        ax.set_ylim([0, 25])

    plt.show()



### Check what is needed from functions beneath and order functions above correctly ###

def check_doses():

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






