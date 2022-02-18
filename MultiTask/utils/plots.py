import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from load_data import read_mha

def show_CT(ax, CT_path, dose_path=None, cont_path=None, slice=50, title=None):
    CT = read_mha(CT_path)
    ax.imshow(CT[slice, :, :], cmap='binary_r')

    if dose_path != None:
        dose = read_mha(dose_path)
        dose_overlay = np.where(dose > 1, 0.2, 0)
        dose_overlay = np.where(dose > 3, 0.3, dose_overlay)
        dose_overlay = np.where(dose > 5, 0.4, dose_overlay)
        dose_overlay = np.where(dose > 7, 0.5, dose_overlay)
        dose_overlay = np.where(dose > 10, 0.6, dose_overlay)
        dose = cm.hot_r(dose / dose.max())
        dose[:, :, :, 3] = dose_overlay
        ax.imshow(dose[slice, :, :, :])

    if cont_path != None:
        cont = read_mha(cont_path)
        ax.contour(cont[slice, :, :], levels=[0, 1, 2, 3], colors=['cyan', 'mediumblue', 'violet', 'lime'],
                   linewidths=1.2)

    if isinstance(title, str):
        ax.set_title(title)
    ax.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    return ax

def planning_daily_dose():
    daily_CT_path = '/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_22/visit_20071102/CTImage.mha'
    daily_cont_path = '/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_22/visit_20071102/Segmentation.mha'
    daily_dose_path = '/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_22/visit_20071102/Dose.mha'
    affine_dose_path = '/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_22/visit_20071102/planning/Planning_Dose.mha'
    affine_cont_path = '/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_22/visit_20071102/planning/Planning_Segmentation.mha'
    affine_CT_path = '/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_22/visit_20071102/planning/Planning_CTImage.mha'
    planning_dose_path = '/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_22/visit_20071022/Dose.mha'
    planning_cont_path = '/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_22/visit_20071022/Segmentation.mha'
    planning_CT_path = '/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_22/visit_20071022/CTImage.mha'


    fig, axs = plt.subplots(2 ,2)
    fig.suptitle('Comparing the daily scan with the registered planning scan')
    axs[0,0] = show_CT(axs[0,0], daily_CT_path, dose_path=daily_dose_path, cont_path=daily_cont_path, slice=60,
                        title='Daily Scan')
    axs[0,1] = show_CT(axs[0,1], planning_CT_path, dose_path=planning_dose_path, cont_path=planning_cont_path, slice=60,
                        title='Planning Scan')
    axs[0, 0] = show_CT(axs[1, 0], daily_CT_path, dose_path=daily_dose_path, cont_path=daily_cont_path, slice=60,
                        title='Daily Scan')
    axs[0, 1] = show_CT(axs[1, 1], affine_CT_path, dose_path=affine_dose_path, cont_path=affine_cont_path, slice=60,
                        title='Registered Planning Scan')
    plt.show()

planning_daily_dose()