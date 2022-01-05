import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os
# import sys
# sys.path.append('C:\\Users\Thomas\AppData\Local\Programs\Python\Python39\Lib\site-packages')
import medpy.io.save as mha_save
import csv

patient_list = []
visit_date_list = []
with open("/exports/lkeb-hpc/tlandman/Patient_Data/visits.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        patient_list.append(str(row[0]))
        visit_date_list.append(str(row[1]))
dir_path = "/exports/lkeb-hpc/tlandman/Patient_Dose/"
for scanID in range(len(visit_date_list)):
    data_path = os.path.join(dir_path, patient_list[scanID], visit_date_list[scanID])
    old_dose_path = os.path.join(data_path, 'Dose/RTDose_4_physicalDose.dcm')
    ds = pydicom.dcmread(old_dose_path)
    dose_array = np.moveaxis(np.float32(ds.pixel_array * ds.DoseGridScaling), 0, -1)
    dose_path_new = os.path.join(data_path, 'Dose/RTDose_4_physicalDose.mha')
    mha_save(dose_array, dose_path_new)
    print(str(scanID)+'klaar')


# dose_path =
# dose_path = "/exports/lkeb-hpc/tlandman/Patient_Dose/Patient_01/visit_20070510/Dose/RTDose_4_physicalDose.dcm"
# ds = pydicom.dcmread(dose_path)
# dose_array = np.moveaxis(ds.pixel_array, 0, -1)
# ds = pydicom.dcmread(dose_path)
# dose_array = np.moveaxis(np.float32(ds.pixel_array*ds.DoseGridScaling), 0, -1)

# plt.imshow(dose_array[:,:,30])
# plt.colorbar()
# plt.show()
# dose_path_new = "/exports/lkeb-hpc/tlandman/Patient_Dose/Patient_01/visit_20070510/Dose/RTDose_4_physicalDose.mha"
mha_save(dose_array, dose_path_new)