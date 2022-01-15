import os
import csv
import shutil

def list_dirs(bef_patient, aft_visit):
    data_paths = []
    dir_path = bef_patient
    patients = os.listdir(dir_path)
    patients.sort()
    for patient in patients:
        visits = os.listdir(os.path.join(dir_path, patient))
        visits.sort()
        for visit in visits:
            data_paths.append(os.path.join(dir_path, patient, visit, aft_visit))
    return data_paths

def copy_files(data_paths_from, data_paths_to):
    for i in range(len(data_paths_from)):
        original = data_paths_from[i]
        target = data_paths_to[i]
        shutil.copyfile(original, target)


def create_csv(data_paths, csv_loc):
    with open(csv_loc, 'w') as f:
        writer = csv.writer(f)
        for i in range(len(data_paths)):
            subject = "subject_"+f"{i+1:03d}"
            writer.writerow([subject, data_paths[i]])

data_paths_from = list_dirs('/exports/lkeb-hpc/mseelmahdy/HaukelandAffine/', 'Contours_Cleaned/Segmentation.mha')
data_paths_to = list_dirs('/exports/lkeb-hpc/tlandman/Data/Patient_MHA/', 'Segmentation.mha')
copy_files(data_paths_from, data_paths_to)