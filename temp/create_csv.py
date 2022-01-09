import os
import csv

def list_dirs():
    data_paths = []
    dir_path = "/exports/lkeb-hpc/tlandman/Data/Patient_MHA/"
    patients = os.listdir(dir_path)
    patients.sort()
    for patient in patients:
        visits = os.listdir(os.path.join(dir_path, patient))
        visits.sort()
        for visit in visits:
            data_paths.append(os.path.join(dir_path, patient, visit, "masks/GTV.mha"))
    return data_paths

def create_csv(data_paths):
    with open("/exports/lkeb-hpc/tlandman/Thesis/MultiTask/data/thomas/fixed_GTV.csv", 'w') as f:
        writer = csv.writer(f)
        for i in range(len(data_paths)):
            subject = "subject_"+f"{i+1:03d}"
            writer.writerow([subject, data_paths[i]])

data_paths = list_dirs()
create_csv(data_paths)