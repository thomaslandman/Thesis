# import sys
# from numpy.core.defchararray import rpartition
# sys.path.append('C:\\Users\Thomas\AppData\Local\Programs\Python\Python39\Lib\site-packages')
import pydicom
import numpy as np
import os
from matplotlib import pyplot as plt
from rt_utils import RTStructBuilder
# import torch

def load_data(data_path, dim=[128,256,64]):
    """Load the data, slice the desired dimension and return 5D struct array and 5D Dose Array"""

    struct_array = load_struct(data_path)
    # struct_array = struct_array[:, :, 256-int(dim[0]/2):256+int(dim[0]/2), 256-int(dim[1]/2):256+int(dim[1]/2), int(struct_array.shape[3]/2)-int(dim[2]/2):int(struct_array.shape[3]/2)+int(dim[2]/2)]
    struct_array = struct_array[:, :, 184:312, 256-int(dim[1]/2) : 256+int(dim[1]/2), 17:81]

    dose_array = load_dose(data_path)
    # dose_array = dose_array[:, :, 256-int(dim[0]/2):256+int(dim[0]/2), 256-int(dim[1]/2):256+int(dim[1]/2), int(dose_array.shape[2]/2)-int(dim[2]/2):int(dose_array.shape[2]/2)+int(dim[2]/2)]
    dose_array = dose_array[:, :, 184:312, 256-int(dim[1]/2) : 256+int(dim[1]/2), 17:81]

    return struct_array, dose_array

def load_struct(data_path): 
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

def load_dose(data_path): 
    dose_path = os.path.join(data_path, "Dose", "RTDose_4_physicalDose.dcm")
    ds = pydicom.dcmread(dose_path)
    dose_array = np.float32(ds.pixel_array*ds.DoseGridScaling)
    dose_array = np.moveaxis(np.float32(ds.pixel_array*ds.DoseGridScaling), 0, -1)
    dose_array = np.expand_dims(np.expand_dims(dose_array, axis=0), axis=0)
    return dose_array

def load_CT(data_path): 
    CT_path = os.path.join(data_path, "CT")
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(CT_path):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName,filename))
                
    # Get ref file
    RefDs = pydicom.read_file(lstFilesDCM[0])

    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

    # Load spacing values (in mm)
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

    # The array is sized based on 'ConstPixelDims'
    CT_Array = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    # loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = pydicom.read_file(filenameDCM)
        # store the raw image data
        CT_Array[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

    return CT_Array




# Paths
def input_data(N):
    """ Extracts a dicom file of certain patient,
    converts to numpy arrays and makes structure masks of correct size.

    :param N: number of the patient in the patient list
    :return structures_mask: 4 dimensional array of structure mask booleans
    :return dose_arr: 3 dimensional array of dose
    :return start_mod: Amount of voxels added at begin of each dimension
    :return end_mod: Amount of voxels added at end of each dimension
    """
    ### MODIFY START ###
    # MODIFY THIS PART FOR DIFFERENT FILE LOCATIONS #
    with open(r'C:\Users\t.meerbothe\Desktop\project_Thierry_64_dev\Scripts\Cleaned_scripts\Lists\patient_IDs.txt') as f:                             #Location of list with patient IDs
        patIDs = [line.rstrip()[1:-1] for line in f]
    patID = patIDs[N]
    struc_list = r'C:\Users\t.meerbothe\Desktop\project_Thierry_64_dev\Scripts\Cleaned_scripts\Lists\Structures.txt'    #Location of list with structures
    PTV_struct = r'C:\Users\t.meerbothe\Desktop\project_Thierry_64_dev\Scripts\Cleaned_scripts\Lists\PTV.txt'           #Location of list with PTV names
    data_dir = r'D:\dicomserver1417d\data'                                                                              #Location of DICOM data
    # MODIFY STANDARDIZED ARRAY SIZE #
    input_size = [144, 64, 96]                                                                                          #Standardized size
    ### MODIFY STOP ###

    dicom_paths = get_dicom_paths(data_dir, patID)

    # load plan, structures, scan and dose
    dose = load_RTDOSE(dicom_paths['RTDOSE'])
    structures = load_structures(dicom_paths['RTSTRUCT'])

    # Make structures
    struct_arrays = struct_arr(struc_list, structures, dose)
    with open(struc_list) as s:
        structs = [line.rstrip()[1:-1] for line in s]

    ptv_array = struct_arr(PTV_struct, structures, dose)

    structures_mask = np.zeros([(len(structs)+1), input_size[0], input_size[1], input_size[2]])
    for i in range(len(structs)):
        name = structs[i]
        structures_mask[i, :, :, :], start_mod, end_mod = standard_size(input_size, struct_arrays[name])
    structures_mask[-1, :, :, :], start_mod, end_mod = standard_size(input_size, ptv_array['PTVpros+vs'])
    structures_mask = structures_mask > 0

    dose_arr = TAVSFldToNumpyNdArray(dose.Data)
    dose_arr, start_mod, end_mod = standard_size(input_size, dose_arr)
    return structures_mask, dose_arr, start_mod, end_mod

def data_import_new():
    scan_ID = "Patient_04"
    # CT_Array = load_CT(scan_ID)
    Dose_Array = load_Dose(scan_ID)
    input_size = [144, 64, 96]  
    Dose_Array, start_mod, end_mod = standard_size(input_size, Dose_Array)
    Struct_Array = load_Struct(scan_ID)
    print(np.shape(Struct_Array))

    # plt.figure()
    # plt.subplot(251)
    # plt.imshow(CT_Array[:,:,50])
    # plt.subplot(252)
    # plt.imshow(Dose_Array[:,:,50])
    # for i in range(5):
    #     plt.subplot(2, 5, 6+i)
    #     plt.imshow(Struct_Array[i, :, :, 50])
    # plt.show()

    print(np.shape(Dose_Array))
    return Struct_Array, Dose_Array, start_mod, end_mod



# scan_ID = "Patient_04"
# CT_Array = load_CT(scan_ID)
# Dose_Array = load_Dose(scan_ID)
# Struct_Array = load_Struct(scan_ID)
# print(np.shape(Struct_Array))

# plt.figure()
# plt.subplot(251)
# plt.imshow(CT_Array[:,:,50])
# plt.subplot(252)
# plt.imshow(Dose_Array[:,:,50])
# for i in range(5):
#     plt.subplot(2, 5, 6+i)
#     plt.imshow(Struct_Array[i, :, :, 50])
# plt.show()

# filenameDCM = "C:\\Users\Thomas\Desktop\Applied Physics\Master Thesis\\treatment planning\Thyrza plans\Patient_AarhusProstate04\Studies\\2382\Dose\\1.2.826.0.1.3680043.2.968.3.70231572.14382.1468485904.720\\1.2.826.0.1.3680043.2.968.3.70231572.14382.1468485904.719.dcm"
# ds = pydicom.dcmread(filenameDCM, force=True)
# # Dose_Array = np.moveaxis(ds.pixel_array, 0, -1)
# Dose_Array = ds.pixel_array
# print(type(ds))
# print(np.shape(ds))
# plt.figure()
# # plt.subplot(251)
# # plt.imshow(CT_Array[:,:,50])
# plt.subplot(252)
# plt.imshow(Dose_Array[:,:,50])
# for i in range(5):
#     plt.subplot(2, 5, 6+i)
#     plt.imshow(Struct_Array[i, :, :, 50])
# plt.show()



### EXTRA CODE ###
    # Dose_Array = image(filenameDCM)
    # # loop through all the DICOM files
    # for filenameDCM in lstFilesDCM:
    #     # read the file
    #     ds = pydicom.read_file(filenameDCM)
    #     # store the raw image data
    #     CT_Array[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

    # print(np.shape(Dose_Array))
    # print(type(Dose_Array))
    # reference_filepath = pymedphys.data_path("C:\\Users\Thomas\Desktop\Applied Physics\Master Thesis\\treatment planning\Thyrza plans\Patient_AarhusProstate04\Studies\\1992\Dose\\1.2.826.0.1.3680043.2.968.3.05114902.10029.1468486999.645\\1.2.826.0.1.3680043.2.968.3.05114902.10029.1468486999.644.dcm")


    # myfirstimage = image("C:\\Users\Thomas\Desktop\Applied Physics\Master Thesis\\treatment planning\Patient_04\RTDose_4_physicalDose.dcm")
    # print(type(myfirstimage))
    # x,y,z= myfirstimage.get_slices_at_index() #defaults to central voxel
    # print(type(x))
    # print(np.shape(x))
    # import scipy.misc
    # scipy.misc.imsave("C:\\Users\Thomas\Desktop\Applied Physics\Master Thesis\\treatment planning\Patient_04\slicex.png",x)

    # mask_3d = rtstruct.get_roi_mask_by_name("Bladder")
    # print(mask_3d[:,:,50])
    # # Display one slice of the region
    # first_mask_slice = mask_3d[:, :, 50]
    # plt.imshow(first_mask_slice)
    # plt.show()
    #ctrs = ds.ROIContourSequence
    #mask_3d = rtstruct.get_roi_mask_by_name("ROI NAME")
    #print(ctrs[0])
    # contourA = np.array([])
    # for sequence in range(len(ctrs[0].ContourSequence)):
    #     #np.concatenate(contourA,ctrs[0].ContourSequence[sequence].ContourData)
    #     #print(type(ctrs[0].ContourSequence[sequence].ContourData[0]))
    #     Struct_Array = 2
    # import pymedphys
    # from medimage import image
    # import cv2