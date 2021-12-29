import pydicom
import os
import csv
import numpy as np

patient_list = []
visit_date_list = []
with open("/exports/lkeb-hpc/tlandman/Patient_Data/visits.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        patient_list.append(str(row[0]))
        visit_date_list.append(str(row[1]))

dir_path = "/exports/lkeb-hpc/tlandman/Patient_Dose/"
scanID = 25
data_path = os.path.join(dir_path, patient_list[scanID], visit_date_list[scanID])
dose_path = os.path.join(data_path, "Dose", "RTDose_4_physicalDose.dcm")
ds = pydicom.dcmread(dose_path)
dose_array = np.moveaxis(ds.pixel_array, 0, -1)
dose_array = np.expand_dims(np.expand_dims(dose_array, axis=0), axis=0)
print(type(dose_array[0,0,0,4,5]))
scaling = ds.DoseGridScaling
print(type(scaling))
dose_array = np.float32(dose_array*scaling)
print(type(dose_array[0,0,0,4,5]))
#comdose_array = np.moveaxis(np.float32(ds.pixel_array), 0, -1)
#dose_array = np.expand_dims(np.expand_dims(dose_array, axis=0), axis=0)