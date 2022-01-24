import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from load_data import read_mha, get_paths_csv
import SimpleITK as sitk
import os

# pred_dose_path = '/exports/lkeb-hpc/tlandman/Thesis/MultiTask/experiments/Single-Task/Dose_input_Sf/output/HMC/Patient_22/visit_20071029/Dose.mha'
# dose_path= '/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_22/visit_20071029/Dose.mha'
# dose = read_mha(dose_path)
# dose_pred = read_mha(pred_dose_path)
#
# diff = dose - dose_pred
#
# fig, axs = plt.subplots(2, 2)
# slice = 55
#
# plot = axs[0, 0].imshow(dose[slice, :, :], cmap='jet')
# axs[0, 0].set_title('Plan Dose [Gy]')
# # for i in range(np.shape(struct)[0]):
# #     axs[0, 0].contour(struct[i, :, :, slice], levels=1, colors='white', linestyles='--')
# fig.colorbar(plot, ax=axs[0,0])
#
# plot = axs[0, 1].imshow(dose_pred[slice, :, :], cmap='jet')
# axs[0, 1].set_title('Predicted Dose [Gy]')
# # for i in range(np.shape(struct)[0]):
# #     axs[0, 1].contour(struct[i, :, :, slice], levels=1, colors='white', linestyles='--')
# fig.colorbar(plot, ax=axs[0,1])
#
# plot = axs[1, 0].imshow(diff[slice, :, :], cmap='bwr', vmax=20, vmin=-20)
# axs[1, 0].set_title('Dose Difference [Gy]')
# # for i in range(np.shape(struct)[0]):
# #     axs[1, 0].contour(struct[i, :, :, slice], levels=1, colors='green', linestyles='--')
# fig.colorbar(plot, ax=axs[1,0])
#
# # plot = axs[1, 1].imshow(gamma_map[:, :, slice], cmap='Greys')
# # axs[1, 1].set_title('3%/3mm Gamma Map')
# # for i in range(np.shape(struct)[0]):
# #     axs[1, 1].contour(struct[i, :, :, slice], levels=1, colors='green', linestyles='--')
# # fig.colorbar(plot, ax=axs[1, 1])
# # plt.savefig('/exports/lkeb-hpc/tlandman/')
# plt.show()
dose_paths = get_paths_csv("/exports/lkeb-hpc/tlandman/Thesis/MultiTask/data/thomas/fixed_Dose.csv")
for i in range(len(dose_paths)):
    dose_path = dose_paths[i]
    dose = read_mha(dose_path, type=np.float32)
    sampler = np.array(np.where(dose>0, 1, 0), dtype=np.uint8)
    print(np.shape(sampler))
    print(sampler[5,5,5])
    sampler_itk = sitk.GetImageFromArray(sampler)
    writer = sitk.ImageFileWriter()
    sampler_path = os.path.join('/'.join(dose_path.split('/')[:-1]),'Sampler_Dose.mha')
    print(sampler_path)
    writer.SetFileName(sampler_path)
    writer.Execute(sampler_itk)

def dose_eval(csv_path):
    dose_paths = get_paths_csv("/exports/lkeb-hpc/tlandman/Thesis/MultiTask/data/thomas/fixed_Dose.csv")
    gtv_paths = get_paths_csv("/exports/lkeb-hpc/tlandman/Thesis/MultiTask/data/thomas/fixed_GTV.csv")
    sv_paths = get_paths_csv("/exports/lkeb-hpc/tlandman/Thesis/MultiTask/data/thomas/fixed_SeminalVesicle.csv")
    bladder_paths = get_paths_csv("/exports/lkeb-hpc/tlandman/Thesis/MultiTask/data/thomas/fixed_Bladder.csv")
    rectum_paths = get_paths_csv("/exports/lkeb-hpc/tlandman/Thesis/MultiTask/data/thomas/fixed_Rectum.csv")

    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Dmean gtv", "V95 gtv", "V110 gtv", "Dmean sv", "V95 sv", "V110 sv", "Dmean rectum", "D2 rectum", "Dmean bladder", "D2 bladder"])
        for i in range(len(dose_paths)):
            dose_path = dose_paths[i]
            gtv_path = gtv_paths[i]
            sv_path = sv_paths[i]
            bladder_path = bladder_paths[i]
            rectum_path = rectum_paths[i]

            dose = read_mha(dose_path, type=np.float32)
            gtv_dose = dose[read_mha(gtv_path, type=np.bool)]
            sv_dose = dose[read_mha(sv_path, type=np.bool)]
            rectum_dose = dose[read_mha(rectum_path,  type=np.bool)]
            bladder_dose = dose[read_mha(bladder_path,  type=np.bool)]

            Dmean_gtv = np.mean(gtv_dose)
            V95_gtv = np.sum(gtv_dose >= 74 * 0.95) / np.size(gtv_dose)
            V110_gtv = np.sum(gtv_dose >= 74 * 1.10) / np.size(gtv_dose)
            Dmean_sv = np.mean(sv_dose)
            V95_sv = np.sum(sv_dose >= 55 * 0.95) / np.size(sv_dose)
            V110_sv = np.sum(sv_dose >= 55 * 1.10) / np.size(sv_dose)
            Dmean_rectum = np.mean(rectum_dose)
            D2_rectum = np.sort(rectum_dose.flatten())[int(-np.size(rectum_dose) * 0.02)]
            Dmean_bladder = np.mean(bladder_dose)
            D2_bladder = np.sort(bladder_dose.flatten())[int(-np.size(bladder_dose) * 0.02)]
            print([Dmean_gtv, V95_gtv, V110_gtv, Dmean_sv, V95_sv, V110_sv, Dmean_rectum, D2_rectum, Dmean_bladder, D2_bladder])
            writer.writerow([Dmean_gtv, V95_gtv, V110_gtv, Dmean_sv, V95_sv, V110_sv, Dmean_rectum, D2_rectum, Dmean_bladder, D2_bladder])


def dose_hist(data_csv_path):

    with open(data_csv_path, newline='') as f:
        df = pd.read_csv(f)
        dose_data = df[['Dmean gtv', 'Dmean sv', 'Dmean rectum', 'Dmean bladder']].to_numpy(float)

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Histogram of mean doses')
    axs[0, 0].hist(dose_data[:,0], bins=np.linspace(72,80,33), color='b', label='GTV', rwidth=0.8)
    axs[0, 1].hist(dose_data[:,1], bins=np.linspace(52,60,33), color='r', label='Seminal Vesicle', rwidth=0.8)
    axs[1, 0].hist(dose_data[:,2], bins=np.linspace(1,9,33), color='g', label='Rectum', rwidth=0.8)
    axs[1, 1].hist(dose_data[:,3], bins=np.linspace(1,9,33), color='m', label='Bladder', rwidth=0.8)

    for _, ax in enumerate(axs.flat):
        ax.legend()
        ax.set_ylim([0, 50])
        ax.set_xlabel('Mean Dose [Gy]')
        ax.set_ylabel('Counts [-]')

    plt.show()


def target_dose_hist(data_csv_path):

    with open(data_csv_path, newline='') as f:
        df = pd.read_csv(f)
        target_Vx = df[['V95 gtv', 'V110 gtv', 'V95 sv', 'V110 sv']].to_numpy(float)
    print(np.sum((target_Vx[:, 0] >= 0.95)) / np.size(target_Vx[:, 0]))
    print(np.sum((target_Vx[:, 1] <= 0.05)) / np.size(target_Vx[:, 1]))
    print(np.sum((target_Vx[:, 2] >= 0.90)) / np.size(target_Vx[:, 2]))
    print(np.sum((target_Vx[:, 3] <= 0.10)) / np.size(target_Vx[:, 3]))

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].hist(target_Vx[:,0], bins=np.linspace(0.92,1,33), color='b', label='V95 GTV', rwidth=0.8)
    axs[0, 1].hist(target_Vx[:,1], bins=np.linspace(0,0.08,33), color='r', label='V110 GTV', rwidth=0.8)
    axs[1, 0].hist(target_Vx[:,2], bins=np.linspace(0.92,1,33), color='g', label='V95 Seminal Vesicle', rwidth=0.8)
    axs[1, 1].hist(target_Vx[:,3], bins=np.linspace(0,0.08,33), color='m', label='V110 Seminal Vesicle', rwidth=0.8)

    for _, ax in enumerate(axs.flat):
        ax.legend()
        ax.set_ylim([0, 35])
        ax.set_ylabel('Counts [-]')
        ax.set_xlabel('Volume Percentage [-]')

    plt.show()


def oar_dose_hist(data_csv_path):
    with open(data_csv_path, newline='') as f:
        df = pd.read_csv(f)
        oar_doses = df[['Dmean rectum', 'D2 rectum', 'Dmean bladder', 'D2 bladder']].to_numpy(float)


    fig, axs = plt.subplots(2, 2)

    axs[0, 0].hist(oar_doses[:,0], color='b', bins=np.linspace(0,15,31), rwidth=0.8)
    axs[0, 1].hist(oar_doses[:,1],  color='b', bins=np.linspace(16,76,31), label='Rectum', rwidth=0.8)
    axs[1, 0].hist(oar_doses[:,2], color='g', bins=np.linspace(0,15,31), rwidth=0.8)
    axs[1, 1].hist(oar_doses[:,3], color='g', bins=np.linspace(16,76,31), label='Bladder', rwidth=0.8) # bins=np.linspace(0,0.08,33),

    for _, ax in enumerate(axs.flat):
        ax.legend()
        ax.set_ylim([0, 30])
        ax.set_ylabel('Counts [-]')
        ax.set_xlabel('D mean [Gy]')
    axs[0, 1].set_xlabel('D2 [Gy]')
    axs[1, 1].set_xlabel('D2 [Gy]')

    plt.show()

# data_path = '/exports/lkeb-hpc/tlandman/Thesis/Treatment_Plans/Plan_Eval_13_01.csv'
# # target_dose_hist(data_path)
# oar_dose_hist(data_path)