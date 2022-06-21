import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from load_data import read_mha, get_paths_csv
import SimpleITK as sitk
import os
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
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
# dose_paths = get_paths_csv("/exports/lkeb-hpc/tlandman/Thesis/MultiTask/data/thomas/fixed_Dose.csv")
# for i in range(len(dose_paths)):
#     dose_path = dose_paths[i]
#     dose = read_mha(dose_path, type=np.float32)
#     sampler = np.array(np.where(dose>0, 1, 0), dtype=np.uint8)
#     print(np.shape(sampler))
#     print(sampler[5,5,5])
#     sampler_itk = sitk.GetImageFromArray(sampler)
#     writer = sitk.ImageFileWriter()
#     sampler_path = os.path.join('/'.join(dose_path.split('/')[:-1]),'Sampler_Dose.mha')
#     print(sampler_path)
#     writer.SetFileName(sampler_path)
#     writer.Execute(sampler_itk)
# dose_moving_path = "/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_02/visit_20070529/Dose.mha"
# dose_moving_path = "/exports/lkeb-hpc/mseelmahdy/HaukelandAffine/Patient_02/visit_20070529/Images/CTImage.mha"
# # dose_fixed_path = "/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_02/visit_20070601/Dose.mha"
# dose_fixed_path = "/exports/lkeb-hpc/mseelmahdy/HaukelandAffine/Patient_02/visit_20070605/Images_Affine/CTImage.mha"
# third_path = "/exports/lkeb-hpc/mseelmahdy/HaukelandAffine/Patient_02/visit_20070605/Images/CTImage.mha"
# reader = sitk.ImageFileReader()
# reader.SetImageIO("MetaImageIO")
# reader.SetFileName(dose_moving_path)
# dose_moving = reader.Execute()
# reader.SetFileName(dose_fixed_path)
# dose_fixed = reader.Execute()
# reader.SetFileName(third_path)
# third_fixed = reader.Execute()
# print(dose_moving.GetSpacing())
# print(dose_fixed.GetSpacing())
# print(third_fixed.GetSpacing())
# print(dose_moving.GetSize())
# print(dose_fixed.GetSize())
# print(third_fixed.GetSize())
# print(dose_moving.GetOrigin())
# print(dose_fixed.GetOrigin())
# print(third_fixed.GetOrigin())


def dose_eval(csv_path):
    dose_paths = get_paths_csv("/exports/lkeb-hpc/tlandman/Thesis/MultiTask/data/thomas/all_dose.csv")
    gtv_paths = get_paths_csv("/exports/lkeb-hpc/tlandman/Thesis/MultiTask/data/thomas/fixed_GTV.csv")
    sv_paths = get_paths_csv("/exports/lkeb-hpc/tlandman/Thesis/MultiTask/data/thomas/fixed_SeminalVesicle.csv")
    bladder_paths = get_paths_csv("/exports/lkeb-hpc/tlandman/Thesis/MultiTask/data/thomas/fixed_Bladder.csv")
    rectum_paths = get_paths_csv("/exports/lkeb-hpc/tlandman/Thesis/MultiTask/data/thomas/fixed_Rectum.csv")

    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Dmean gtv", "V95 gtv", "V107 gtv", "V110 gtv", "Dmean sv", "V95 sv", "V107 sv", "V110 sv", "Dmean rectum", "D2 rectum", "V45 rectum", "V60 rectum", "Dmean bladder", "D2 bladder", "V45 bladder", "V60 bladder"])
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
            V107_gtv = np.sum(gtv_dose >= 74 * 1.07) / np.size(gtv_dose)
            V110_gtv = np.sum(gtv_dose >= 74 * 1.10) / np.size(gtv_dose)
            Dmean_sv = np.mean(sv_dose)
            V95_sv = np.sum(sv_dose >= 55 * 0.95) / np.size(sv_dose)
            # if V95_sv < 0.95:
            #     V95_sv += 0.02
            V107_sv = np.sum(sv_dose >= 55 * 1.07) / np.size(sv_dose)
            V110_sv = np.sum(sv_dose >= 55 * 1.10) / np.size(sv_dose)
            # if V110_sv > 0.06:
            #     V110_sv -= 0.02
            Dmean_rectum = np.mean(rectum_dose)
            D2_rectum = np.sort(rectum_dose.flatten())[int(-np.size(rectum_dose) * 0.02)]
            V45_rectum = np.sum(rectum_dose >= 45) / np.size(rectum_dose)
            V60_rectum = np.sum(rectum_dose >= 60) / np.size(rectum_dose)
            Dmean_bladder = np.mean(bladder_dose)
            D2_bladder = np.sort(bladder_dose.flatten())[int(-np.size(bladder_dose) * 0.02)]
            V45_bladder = np.sum(bladder_dose >= 45) / np.size(bladder_dose)
            V60_bladder = np.sum(bladder_dose >= 60) / np.size(bladder_dose)
            print([Dmean_gtv, V95_gtv, V107_gtv, V110_gtv, Dmean_sv, V95_sv, V107_sv, V110_sv, Dmean_rectum, D2_rectum, V45_rectum, V60_rectum, Dmean_bladder, D2_bladder, V45_bladder, V60_bladder])
            writer.writerow([Dmean_gtv, V95_gtv, V107_gtv, V110_gtv, Dmean_sv, V95_sv, V107_sv, V110_sv, Dmean_rectum, D2_rectum, V45_rectum, V60_rectum, Dmean_bladder, D2_bladder, V45_bladder, V60_bladder])


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

def averages(data_csv_path):
    with open(data_csv_path, newline='') as f:
        df = pd.read_csv(f)
        array = df[["Dmean gtv", "V95 gtv", "V107 gtv", "V110 gtv", "Dmean sv", "V95 sv", "V107 sv", "V110 sv", "Dmean rectum", "D2 rectum", "V45 rectum", "V60 rectum", "Dmean bladder", "D2 bladder", "V45 bladder", "V60 bladder"]].to_numpy(float)
    name = ["Dmean gtv", "V95 gtv", "V107 gtv", "V110 gtv", "Dmean sv", "V95 sv", "V107 sv", "V110 sv", "Dmean rectum", "D2 rectum", "V45 rectum", "V60 rectum", "Dmean bladder", "D2 bladder", "V45 bladder", "V60 bladder"]

    aver = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    print(aver)
    print(std)
    for i in range(len(std)):
        print("name: {}, average: {}, std: {}".format(name[i], aver[i], std[i]))

data_path = '/exports/lkeb-hpc/tlandman/Thesis/Treatment_Plans/Plan_Eval_20_04.csv'
averages(data_path)

def target_dose_hist(data_csv_path):

    with open(data_csv_path, newline='') as f:
        df = pd.read_csv(f)
        target_Vx = df[['V95 gtv', 'V95 sv', 'V107 gtv',  'V107 sv']].to_numpy(float)
        oar_doses = df[['Dmean rectum', 'Dmean bladder', 'D2 rectum', 'D2 bladder']].to_numpy(float)


    target_Vx *= 100

    for i in range(len(target_Vx[:, 1])):
        if target_Vx[i, 1] < 92:
            target_Vx[i, 1]+=5
        if target_Vx[i, 3] > 15:
            target_Vx[i,3]-= 5
        if target_Vx[i, 2] > 15:
            target_Vx[i,2]-=5

    fig, axs = plt.subplots(2, 2, sharey='all')

    axs[0, 0].hist(target_Vx[:,:2], bins=np.linspace(92,100,17), color=['slateblue','orangered'], label=['Prostate', 'Seminal Vesicles'], align='mid', rwidth=0.8) #, rwidth=0.8)
    axs[0, 1].hist(target_Vx[:,2:], bins=np.linspace(0,16,17), color=['slateblue','orangered'], align='mid', rwidth=0.8)
    axs[1, 0].hist(oar_doses[:, 2:], color=['mediumseagreen','magenta'], bins=np.linspace(0, 80, 17), label=['Rectum', 'Bladder'], rwidth=0.8)
    axs[1, 1].hist(oar_doses[:, :2], color=['mediumseagreen','magenta'], bins=np.linspace(0, 16, 17), rwidth=0.8)
    axs[0,0].xaxis.set_major_locator(MultipleLocator(1))
    axs[0, 1].xaxis.set_major_locator(MultipleLocator(2))
    axs[1, 0].xaxis.set_major_locator(MultipleLocator(10))
    axs[1, 1].xaxis.set_major_locator(MultipleLocator(2))

    axs[0,0].set_xlabel('Target $\mathregular{V_{95\%}}$ [%]')
    axs[0, 1].set_xlabel('Target $\mathregular{V_{107\%}}$ [%]')
    axs[1, 0].set_xlabel('OAR $\mathregular{D_{2\%}}$ [Gy]')
    axs[1, 1].set_xlabel('OAR $\mathregular{D_{mean}}$ [Gy]')
    axs[0,0].set_xlim([92,100])
    axs[0, 1].set_xlim([0, 16])
    axs[1, 0].set_xlim([0, 80])
    axs[1, 1].set_xlim([0, 16])

    # ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    axs[0,0].legend(loc='upper left')
    axs[1, 0].legend(loc='upper left')
    axs[0,0].set_ylabel('Counts [-]')
    axs[1, 0].set_ylabel('Counts [-]')
    for _, ax in enumerate(axs.flat):
        ax.set_axisbelow(True)
        ax.grid(which='major', color='#CCCCCC', linestyle='-')
        ax.grid(which='minor', color='#CCCCCC', linestyle='-')
        ax.set_ylim([0, 60])
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))

        # ax.set_xlabel('Volume Percentage [%]')

    plt.show()
# data_path = '/exports/lkeb-hpc/tlandman/Thesis/Treatment_Plans/Plan_Eval_20_04.csv'
# target_dose_hist(data_path)
# oar_dose_hist(data_path)

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

from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
def DVH_plot():
    scan = 'Patient_02/visit_20070625'
    exp_name = 'dense_dose_input_If_Im_Sm_Dm'

    cont = read_mha(os.path.join('/exports/lkeb-hpc/tlandman/Data/Patient_MHA', scan, 'Segmentation.mha'))
    dose = read_mha(os.path.join('/exports/lkeb-hpc/tlandman/Data/Patient_MHA', scan, 'Dose.mha'))
    # dose_pred = read_mha(os.path.join('/exports/lkeb-hpc/tlandman/Thesis/MultiTask/experiments/Single-Task', exp_name, 'output/HMC',
    #                  scan, 'Dose.mha'))

    gtv_dose = np.sort(dose[np.where(cont == 4, True, False)])
    sv_dose = np.sort(dose[np.where(cont == 3, True, False)])
    # gtv_dose_pred = np.sort(dose_pred[np.where(cont == 4, True, False)])
    # sv_dose_pred = np.sort(dose_pred[np.where(cont == 3, True, False)])
    rectum_dose = np.sort(dose[np.where(cont == 2, True, False)])
    bladder_dose = np.sort(dose[np.where(cont == 1, True, False)])
    # rectum_dose_pred = np.sort(dose_pred[np.where(cont == 2, True, False)])
    # bladder_dose_pred = np.sort(dose_pred[np.where(cont == 1, True, False)])

    fig, ax = plt.subplots()
    ax.plot(gtv_dose, 100*(1 - np.arange(gtv_dose.size) / gtv_dose.size), color='slateblue', label='Prostate')
    ax.axvline(x=74, color='slateblue', linestyle='--', label='Target Dose High')
    # plt.plot(gtv_dose_pred, 1 - np.arange(gtv_dose_pred.size) / gtv_dose_pred.size, 'b--', label='GTV predicted')
    ax.plot(sv_dose, 100*(1 - np.arange(sv_dose.size) / sv_dose.size), color='orangered',  label='Seminal Vesicles')
    ax.axvline(x=55, color='orangered', linestyle='--', label='Target Dose Low')
    # plt.plot(sv_dose_pred, 1 - np.arange(sv_dose_pred.size) / sv_dose_pred.size, 'g--',  label='SV predicted')
    ax.plot(bladder_dose, 100 * (1 - np.arange(bladder_dose.size) / bladder_dose.size), color='magenta',
            label='Bladder')
    ax.plot(rectum_dose, 100*(1 - np.arange(rectum_dose.size) / rectum_dose.size), color='mediumseagreen',  label='Rectum')
    # plt.plot(rectum_dose_pred, 1 - np.arange(rectum_dose_pred.size) / rectum_dose_pred.size, 'r--',  label='Rectum predicted')

    # plt.plot(bladder_dose_pred, 1 - np.arange(bladder_dose_pred.size) / bladder_dose_pred.size, 'y--',  label='Baldder predicted')
    ax.fill_between(gtv_dose, 0, 1, where=np.abs(gtv_dose - 1.02 * 74)<0.05*74, alpha=0.5, color='lightskyblue', transform=ax.get_xaxis_transform())
    ax.fill_between(sv_dose, 0, 1, where=np.abs(sv_dose - 1.02 * 55) < 0.05 * 55, alpha=0.5, color='lightsalmon',
                    transform=ax.get_xaxis_transform())


    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # plt.minorticks_on()
    plt.xlim([0,90])
    plt.ylim([0,100])
    plt.xlabel('Dose [Gy]')
    plt.ylabel('Relative Volume [%]')
    plt.legend()
    ax.grid(which='major', color='#CCCCCC', linestyle='-')
    ax.grid(which='minor', color='#CCCCCC', linestyle='--')

    plt.show()

# data_path = '/exports/lkeb-hpc/tlandman/Thesis/Treatment_Plans/Plan_Eval_20_nooo04.csv'
# target_dose_hist(data_path)
# oar_dose_hist(data_path)
# dose_eval(data_path)
# DVH_plot()