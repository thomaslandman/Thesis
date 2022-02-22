import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from load_data import read_mha

def show_CT(fig, ax, CT=None, dose=None, cont=None, slice=50, title=None):
    ax.imshow(CT[slice, :, :], cmap='binary_r')

    if type(dose) != None:
        m = cm.ScalarMappable(cmap='jet')
        m.set_array([])
        m.set_clim(vmin=np.min(dose), vmax=np.max(dose))
        dose_overlay = np.where(dose > 1, 0.2, 0)
        dose_overlay = np.where(dose > 3, 0.3, dose_overlay)
        dose_overlay = np.where(dose > 5, 0.4, dose_overlay)
        dose_overlay = np.where(dose > 7, 0.6, dose_overlay)
        dose_overlay = np.where(dose > 10, 0.8, dose_overlay)

        dose = cm.jet((dose - np.min(dose)) / (np.max(dose) - np.min(dose)))
        dose[:, :, :, 3] = dose_overlay
        ax.imshow(dose[slice, :, :, :])
        fig.colorbar(m, ax=ax)


    if type(cont) != None:
        ax.contour(cont[slice, :, :], levels=[0, 1, 2, 3], colors=['white', 'white', 'white', 'white'], linewidths=0.8) # 'violet', 'lime', 'cyan', 'mediumblue'],


    if isinstance(title, str):
        ax.set_title(title)
    ax.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    return ax

def show_diff(fig, ax, CT, diff, cont=None, slice=50, lim=10, title=None):
    ax.imshow(CT[slice, :, :], cmap='binary_r')

    m = cm.ScalarMappable(cmap='bwr')
    m.set_array([])
    m.set_clim(vmin=-lim, vmax=lim)
    dose_overlay = np.where(abs(diff) > 0.001, 0.1, 0)
    dose_overlay = np.where(abs(diff) > 0.5, 0.2, dose_overlay)
    dose_overlay = np.where(abs(diff) > 1, 0.4, dose_overlay)
    dose_overlay = np.where(abs(diff) > 1.5, 0.6, dose_overlay)
    dose_overlay = np.where(abs(diff) > 2, 0.8, dose_overlay)
    diff = np.where(diff > lim, lim, diff)
    diff = np.where(diff < -lim, -lim, diff)
    diff = cm.bwr((diff - np.min(diff)) / (np.max(diff) - np.min(diff)))
    diff[:, :, :, 3] = dose_overlay

    ax.imshow(diff[slice, :, :, :])
    fig.colorbar(m, ax=ax)

    if type(cont) != None:
        ax.contour(cont[slice, :, :], levels=[0, 1, 2, 3], colors=['white', 'white', 'white', 'white'], linewidths=0.8) #'cyan', 'mediumblue', 'violet', 'lime'],


    if isinstance(title, str):
        ax.set_title(title)
    ax.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    return ax


def planning_daily_dose():
    daily_CT = read_mha('/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_22/visit_20071102/CTImage.mha')
    daily_cont = read_mha('/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_22/visit_20071102/Segmentation.mha')
    daily_dose = read_mha('/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_22/visit_20071102/Dose.mha')
    affine_dose = read_mha('/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_22/visit_20071102/planning/Planning_Dose.mha')
    affine_cont = read_mha('/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_22/visit_20071102/planning/Planning_Segmentation.mha')
    affine_CT = read_mha('/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_22/visit_20071102/planning/Planning_CTImage.mha')
    planning_dose = read_mha('/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_22/visit_20071022/Dose.mha')
    planning_cont = read_mha('/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_22/visit_20071022/Segmentation.mha')
    planning_CT = read_mha('/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_22/visit_20071022/CTImage.mha')
    predicted_dose = read_mha('/exports/lkeb-hpc/tlandman/Thesis/MultiTask/experiments/Single-Task/Dose_input_Sf_If/output/HMC/Patient_22/visit_20071102/Dose.mha')

    fig, axs = plt.subplots(2 ,2)
    # fig.suptitle('Comparing the daily scan with the registered planning scan')
    axs[0, 0] = show_CT(fig, axs[0, 0], daily_CT, dose=daily_dose, cont=daily_cont, slice=60,
                        title='Ground Truth Dose [Gy]')
    axs[0, 1] = show_CT(fig, axs[0, 1], daily_CT, dose=predicted_dose, cont=daily_cont, slice=60,
                        title='Predicted Dose [Gy]')
    diff_dose = daily_dose - predicted_dose
    axs[1, 0] = show_diff(fig, axs[1, 0], daily_CT, diff=diff_dose, cont=daily_cont, lim=15, slice=60,
                        title='Difference [Gy]')
    # axs[1, 1] = show_CT(fig, axs[1, 1], affine_CT, dose=affine_dose, cont=affine_cont, slice=60,
    #                     title='Registered Planning Scan')
    plt.show()

planning_daily_dose()