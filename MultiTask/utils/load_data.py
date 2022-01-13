import csv
import SimpleITK as sitk
import numpy as np
import os
import pydicom

def get_paths_csv(csv_path):
    """Gets a list of all data paths from a csv

    :params csv_path: path to the csv file
    :return data_paths: list of data paths
    """
    data_paths = []
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            data_paths.append(str(row[1]))
    return data_paths

def read_mha(data_path, type=np.float32):
    """Reads a mha image to an array

    :params data_path: path to the image file
    :params type: type of the array
    :return image_array: np.array of dtype
    """
    reader = sitk.ImageFileReader()
    reader.SetImageIO("MetaImageIO")
    reader.SetFileName(data_path)
    image = reader.Execute()
    image_array = np.array(sitk.GetArrayFromImage(image), dtype=type)
    return image_array

def read_nrrd(data_path, type=np.float32):
    """Reads a nrrd image to an array

    :params data_path: path to the image file
    :params type: type of the array
    :return image_array: np.array of dtype
    """
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NrrdImageIO")
    reader.SetFileName(data_path)
    image = reader.Execute()
    image_array = np.array(sitk.GetArrayFromImage(image), dtype=type)
    return image_array

def read_dose_dcm(dose_path, type=np.float32):
    """Reads a Dicom dose to an array

    :params data_path: path to the image file
    :params type: type of the array
    :return dose_array: np.array of dtype
    """
    ds = pydicom.dcmread(dose_path)
    dose_array = np.array(ds.pixel_array*ds.DoseGridScaling, dtype=type)
    return dose_array

def read_struc_dcm(data_path):
    # moet nog opgeschoond worden
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