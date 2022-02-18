import os.path
import subprocess
import SimpleITK as sitk
import shutil


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
        for scan_idx in range(0, len(scan_list)):
            if scan_idx == 1:
                continue
            scan_dir = os.path.join(patient_dir, scan_list[scan_idx])

            source = os.path.join(scan_dir, 'planning/CTImage.mha')
            destination = os.path.join(scan_dir, 'planning/Planning_CTImage.mha')
            # shutil.copyfile(source, destination)
            os.rename(source, destination)

# loop_scans()
# transform_affine()

