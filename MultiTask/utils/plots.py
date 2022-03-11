import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from load_data import read_mha
import SimpleITK as sitk
from pymedphys import gamma
import pandas as pd
import os


def get_loss(csv_path):
    df = pd.read_csv(csv_path)
    loss = df.to_numpy()
    prev_epoch = 74
    train_loss = []
    val_loss = []
    for i in range(np.shape(loss)[0]):
        if loss[i,1] == prev_epoch:
            val_loss.append([loss[i,1], loss[i,2]])
        else:
            train_loss.append([loss[i,1], loss[i,2]])
        prev_epoch = loss[i,1]
    return np.array(train_loss), np.array(val_loss)



def loss_plot():
    csv_path = '/home/tlandman/Downloads/Sf_If_Dm'
    train_loss, val_loss = get_loss(csv_path)

    plt.plot(train_loss[:,0], train_loss[:,1], label='Training Loss')
    plt.plot(val_loss[:,0], val_loss[:,1], label='Validation Loss')
    plt.ylim(0,50)
    plt.legend()
    plt.show()


def DVH_plot():
    scan = 'Patient_23/visit_20071105'
    exp_name = 'Dose_Masks_input_Sf_If_Dm'

    cont = read_mha(os.path.join('/exports/lkeb-hpc/tlandman/Data/Patient_MHA', scan, 'Segmentation.mha'))
    dose = read_mha(os.path.join('/exports/lkeb-hpc/tlandman/Data/Patient_MHA', scan, 'Dose.mha'))
    dose_pred = read_mha(os.path.join('/exports/lkeb-hpc/tlandman/Thesis/MultiTask/experiments/Single-Task', exp_name, 'output/HMC',
                     scan, 'Dose.mha'))

    gtv_dose = np.sort(dose[np.where(cont == 4, True, False)])
    sv_dose = np.sort(dose[np.where(cont == 3, True, False)])
    gtv_dose_pred = np.sort(dose_pred[np.where(cont == 4, True, False)])
    sv_dose_pred = np.sort(dose_pred[np.where(cont == 3, True, False)])
    rectum_dose = np.sort(dose[np.where(cont == 2, True, False)])
    bladder_dose = np.sort(dose[np.where(cont == 1, True, False)])
    rectum_dose_pred = np.sort(dose_pred[np.where(cont == 2, True, False)])
    bladder_dose_pred = np.sort(dose_pred[np.where(cont == 1, True, False)])

    plt.plot(gtv_dose, 1 - np.arange(gtv_dose.size) / gtv_dose.size, 'b-')
    plt.plot(gtv_dose_pred, 1 - np.arange(gtv_dose_pred.size) / gtv_dose_pred.size, 'b--')
    plt.plot(sv_dose, 1 - np.arange(sv_dose.size) / sv_dose.size, 'g-')
    plt.plot(sv_dose_pred, 1 - np.arange(sv_dose_pred.size) / sv_dose_pred.size, 'g--')
    plt.plot(rectum_dose, 1 - np.arange(rectum_dose.size) / rectum_dose.size, 'r-')
    plt.plot(rectum_dose_pred, 1 - np.arange(rectum_dose_pred.size) / rectum_dose_pred.size, 'r--')
    plt.plot(bladder_dose, 1 - np.arange(bladder_dose.size)/bladder_dose.size, 'y-')
    plt.plot(bladder_dose_pred, 1 - np.arange(bladder_dose_pred.size) / bladder_dose_pred.size, 'y--')

    plt.show()

def show_CT(fig, ax, CT=None, dose=None, cont=None, slice=50, title=None, max_dose=None):
    ax.imshow(CT[slice, :, :], cmap='binary_r')

    if type(dose) != None:
        m = cm.ScalarMappable(cmap='jet')
        m.set_array([])
        m.set_clim(vmin=0, vmax=np.max(max_dose))
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

def get_gamma_map(scan, dose, dose_pred, cont=None, print_values=False):
    reader = sitk.ImageFileReader()
    reader.SetImageIO("MetaImageIO")
    reader.SetFileName(os.path.join('/exports/lkeb-hpc/tlandman/Data/Patient_MHA', scan, 'Dose.mha'))
    image = reader.Execute()
    dims = image.GetSize()
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    x = np.linspace(origin[0], origin[0] + spacing[0] * (dims[0] - 1), dims[0])
    y = np.linspace(origin[1], origin[0] + spacing[1] * (dims[1] - 1), dims[1])
    z = np.linspace(origin[2], origin[2] + spacing[2] * (dims[2] - 1), dims[2])
    axes = (z, y, x)
    gamma_map = gamma(axes, dose, axes, dose_pred, 2, 2, lower_percent_dose_cutoff=0.1, interp_fraction=2,
                      local_gamma=False, max_gamma=2, skip_once_passed=True, quiet=False)

    if print_values == True:
        gamma_pass = np.where(gamma_map < 1, True, False)

        dose_mask = np.invert(np.isnan(gamma_map))
        gtv_mask = np.where(cont == 4, True, False)
        sv_mask = np.where(cont == 3, True, False)
        rectum_mask = np.where(cont == 2, True, False)
        bladder_mask = np.where(cont == 1, True, False)

        dose_gamma = np.count_nonzero(gamma_pass[dose_mask]) / np.size(gamma_pass[dose_mask])
        gtv_gamma = np.count_nonzero(gamma_pass[gtv_mask]) / np.size(gamma_pass[gtv_mask & dose_mask])
        sv_gamma = np.count_nonzero(gamma_pass[sv_mask]) / np.size(gamma_pass[sv_mask & dose_mask])
        rectum_gamma = np.count_nonzero(gamma_pass[rectum_mask]) / np.size(gamma_pass[rectum_mask & dose_mask])
        bladder_gamma = np.count_nonzero(gamma_pass[bladder_mask]) / np.size(gamma_pass[bladder_mask & dose_mask])

        print(dose_gamma)
        print(gtv_gamma)
        print(sv_gamma)
        print(rectum_gamma)
        print(bladder_gamma)

    return gamma_map

def show_gamma(fig, ax, CT, gamma, cont=None, slice=50, title=None):

    ax.imshow(CT[slice, :, :], cmap='binary_r')

    ax.imshow(gamma[slice, :, :], cmap='RdYlGn_r', vmin=0, vmax =2)
    m = cm.ScalarMappable(cmap='RdYlGn_r')
    m.set_array([])
    m.set_clim(vmin=0, vmax=2)
    fig.colorbar(m, ax=ax)

    if type(cont) != None:
        ax.contour(cont[slice, :, :], levels=[0, 1, 2, 3], colors=['white', 'white', 'white', 'white'], linewidths=0.8) #'cyan', 'mediumblue', 'violet', 'lime'],

    if isinstance(title, str):
        ax.set_title(title)
    ax.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)


def planning_daily_dose():
    scan = 'Patient_24/visit_20071207'
    exp_name = 'Dose_Masks_input_Sf_If_Dm_Ma'
    slice = 75
    daily_CT = read_mha(os.path.join('/exports/lkeb-hpc/tlandman/Data/Patient_MHA', scan, 'CTImage.mha'))
    daily_cont = read_mha(os.path.join('/exports/lkeb-hpc/tlandman/Data/Patient_MHA', scan, 'Segmentation.mha'))
    daily_dose = read_mha(os.path.join('/exports/lkeb-hpc/tlandman/Data/Patient_MHA', scan, 'Dose.mha'))
    predicted_dose = read_mha(os.path.join('/exports/lkeb-hpc/tlandman/Thesis/MultiTask/experiments/Single-Task', exp_name, 'output/HMC', scan, 'Dose.mha'))

    fig, axs = plt.subplots(2 ,2)
    # fig.suptitle('Comparing the daily scan with the registered planning scan')
    max_dose = np.max(np.append(daily_dose, predicted_dose))
    axs[0, 0] = show_CT(fig, axs[0, 0], daily_CT, dose=daily_dose, cont=daily_cont, slice=slice,
                        title='Ground Truth Dose [Gy]', max_dose=max_dose)
    axs[0, 1] = show_CT(fig, axs[0, 1], daily_CT, dose=predicted_dose, cont=daily_cont, slice=slice,
                        title='Predicted Dose [Gy]', max_dose=max_dose)

    diff_dose = daily_dose - predicted_dose
    axs[1, 0] = show_diff(fig, axs[1, 0], daily_CT, diff=diff_dose, cont=daily_cont, lim=10, slice=slice,
                        title='Difference [Gy]')

    gamma_map = get_gamma_map(scan, dose=daily_dose, dose_pred=predicted_dose, cont=daily_cont, print_values=True)
    axs[1, 1] = show_gamma(fig, axs[1,1], daily_CT, gamma_map, cont=daily_cont, slice=slice, title='Gamma Map')
    plt.show()

planning_daily_dose()
# DVH_plot()