from dose_char import *
import csv
import os
import matplotlib.pyplot as plt

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



dose_paths = get_dose_paths()
struc_paths = get_struc_paths()
mean_doses = []
for i in range(30): # (len(dose_dir)):
    dose_path = dose_paths[i]
    struc_path = struc_paths[i]
    mean_doses.append(mean_dose(dose_path, struc_path))
    print(str(i)+ "done")
plt.figure()
plt.hist(mean_doses)
plt.show()
