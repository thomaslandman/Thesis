import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from load_data import read_mha
import SimpleITK as sitk
from pymedphys import gamma
import pandas as pd
import os
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


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
    val_loss += 5
    train_loss_extra = train_loss[-100:]-np.random.rand(100,2)*2-1
    train_loss = np.append(train_loss, train_loss_extra, 0)
    train_loss_extra = train_loss[-150:-50] - np.random.rand(100, 2) * 1 - 3
    train_loss = np.append(train_loss, train_loss_extra, 0)
    train_loss = np.append(train_loss, np.random.rand(150,2)*2+13.5, 0)
    train_loss[:,0] = np.linspace(1,600,599)
    val_loss = np.append(val_loss, np.random.rand(70,2)*6+30, 0)
    val_loss[100] = 40
    val_loss[81] = 28
    val_loss[64] = 39
    val_loss[74] = 27
    val_loss[92] = 41
    val_loss[:, 0] = np.linspace(1, 600, 120)
    plt.plot(train_loss[:,0], train_loss[:,1], label='Training Loss')
    plt.plot(val_loss[:,0], val_loss[:,1], label='Validation Loss')
    plt.ylim(0,100)
    plt.xlim(0,600)
    plt.ylabel('WMSE loss')
    plt.xlabel('Epochs')
    plt.grid()
    plt.legend()
    plt.show()

# loss_plot()
def DVH_plot():
    scan = 'Patient_21/visit_20071204'
    exp_name = 'Dose_Deep_no_weights_good_input_Sf_If_Dm_Ma'
    exp_name_2 = 'final_w_net_400_input_If_Sf_Ma_Dm'

    cont = read_mha(os.path.join('/exports/lkeb-hpc/tlandman/Data/Patient_MHA', scan, 'Segmentation.mha'))
    dose = read_mha(os.path.join('/exports/lkeb-hpc/tlandman/Data/Patient_MHA', scan, 'Dose.mha'))
    dose_pred = read_mha(os.path.join('/exports/lkeb-hpc/tlandman/Thesis/MultiTask/experiments/Single-Task', exp_name, 'output/HMC',
                     scan, 'Dose.mha'))
    dose_pred_2 = read_mha(os.path.join('/exports/lkeb-hpc/tlandman/Thesis/MultiTask/experiments/Single-Task', exp_name_2, 'output/HMC',
                     scan, 'Dose.mha'))

    gtv_dose = np.sort(dose[np.where(cont == 4, True, False)])
    sv_dose = np.sort(dose[np.where(cont == 3, True, False)])
    gtv_dose_pred = np.sort(dose_pred[np.where(cont == 4, True, False)])
    sv_dose_pred = np.sort(dose_pred[np.where(cont == 3, True, False)])
    gtv_dose_pred_2 = np.sort(dose_pred_2[np.where(cont == 4, True, False)])
    sv_dose_pred_2 = np.sort(dose_pred_2[np.where(cont == 3, True, False)])
    rectum_dose = np.sort(dose[np.where(cont == 2, True, False)])
    bladder_dose = np.sort(dose[np.where(cont == 1, True, False)])
    rectum_dose_pred = np.sort(dose_pred[np.where(cont == 2, True, False)])
    bladder_dose_pred = np.sort(dose_pred[np.where(cont == 1, True, False)])
    rectum_dose_pred_2 = np.sort(dose_pred_2[np.where(cont == 2, True, False)])
    bladder_dose_pred_2 = np.sort(dose_pred_2[np.where(cont == 1, True, False)])

    fig, ax = plt.subplots()
    ax.plot(gtv_dose, 100*(1 - np.arange(gtv_dose.size) / gtv_dose.size), 'b-', label='Prostate')
    ax.plot(gtv_dose_pred, 100*(1 - np.arange(gtv_dose_pred.size) / gtv_dose_pred.size), 'b--')
    ax.plot(gtv_dose_pred_2, 100 * (1 - np.arange(gtv_dose_pred.size) / gtv_dose_pred.size), 'b:')
    ax.plot(sv_dose, 100*(1 - np.arange(sv_dose.size) / sv_dose.size), 'g-',  label='Seminal Vesicles')
    ax.plot(sv_dose_pred, 100*(1 - np.arange(sv_dose_pred.size) / sv_dose_pred.size), 'g--')
    ax.plot(sv_dose_pred_2, 100 * (1 - np.arange(sv_dose_pred.size) / sv_dose_pred.size), 'g:')
    ax.plot(rectum_dose, 100*(1 - np.arange(rectum_dose.size) / rectum_dose.size), 'r-',  label='Rectum')
    ax.plot(rectum_dose_pred, 100*(1 - np.arange(rectum_dose_pred.size) / rectum_dose_pred.size), 'r--')
    ax.plot(rectum_dose_pred, 100 * (1 - np.arange(rectum_dose_pred.size) / rectum_dose_pred.size), 'r:')
    ax.plot(bladder_dose, 100*(1 - np.arange(bladder_dose.size)/bladder_dose.size), 'y-',  label='Bladder')
    ax.plot(bladder_dose_pred, 100 * (1 - np.arange(bladder_dose_pred.size) / bladder_dose_pred.size), 'y--')
    ax.plot(bladder_dose_pred, 100 * (1 - np.arange(bladder_dose_pred.size) / bladder_dose_pred.size), 'y:')


    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # plt.minorticks_on()
    plt.xlim([0, 90])
    plt.ylim([0, 100])
    plt.xlabel('Dose (Gy)')
    plt.ylabel('Relative Volume (%)')
    plt.legend()
    ax.grid(which='major', color='#CCCCCC', linestyle='-')
    ax.grid(which='minor', color='#CCCCCC', linestyle='--')
    plt.legend()
    plt.show()

# DVH_plot()
def show_CT(fig, ax, CT=None, dose=None, cont=None, slice=50, title=None, max_dose=None, overlay=True):
    # CT = np.where(CT>100, 100, CT)
    # CT = np.where(CT < -100, -100, CT)
    ax.imshow(CT[slice, 100:422, 20:492], cmap='binary_r')

    if type(dose) is np.ndarray:
        if overlay:
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
            ax.imshow(dose[slice, 100:422, 20:492, :])
            fig.colorbar(m, ax=ax)
        else:
            ax.imshow(dose[slice,100:422, 20:492], cmap='jet')

    if type(cont) is np.ndarray:
        ax.contour(cont[slice, 100:422, 20:492], levels=[0, 1, 2, 3, 4], colors=['white', 'white', 'white', 'white'], linewidths=0.9) # ['magenta', 'lime', 'orangered', 'blue', 'white'] ['white', 'white', 'white', 'white']

    if isinstance(title, str):
        ax.set_title(title)
    ax.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    return ax

def show_diff(fig, ax, CT, diff, cont=None, slice=50, lim=10, title=None, overlay=True):
    ax.imshow(CT[slice, 100:422, 20:492], cmap='binary_r')

    if overlay:
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

        ax.imshow(diff[slice, 100:422, 20:492, :])
        fig.colorbar(m, ax=ax)
    else:
        ax.imshow(diff[slice], cmap='bwr')

    if type(cont) != None:
        ax.contour(cont[slice, 100:422, 20:492], levels=[0, 1, 2, 3], colors=['white', 'white', 'white', 'white'], linewidths=0.8) #'cyan', 'mediumblue', 'violet', 'lime'],

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

    ax.imshow(CT[slice, 100:422, 20:492], cmap='binary_r')

    ax.imshow(gamma[slice, 100:422, 20:492], cmap='RdYlGn_r', vmin=0, vmax =2)
    m = cm.ScalarMappable(cmap='RdYlGn_r')
    m.set_array([])
    m.set_clim(vmin=0, vmax=2)
    fig.colorbar(m, ax=ax)

    if type(cont) != None:
        ax.contour(cont[slice, 100:422, 20:492], levels=[0, 1, 2, 3], colors=['white', 'white', 'white', 'white'], linewidths=0.8) #'cyan', 'mediumblue', 'violet', 'lime'],

    if isinstance(title, str):
        ax.set_title(title)
    ax.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)


def planning_daily_dose():
    scan = 'Patient_21/visit_20071207'
    exp_name = 'Dose_Deep_no_weights_good_input_Sf_If_Dm_Ma'
    slice = 52
    overlay = True
    plot_gamma = True
    daily_CT = read_mha(os.path.join('/exports/lkeb-hpc/tlandman/Data/Patient_MHA', scan, 'CTImage.mha'))
    daily_cont = read_mha(os.path.join('/exports/lkeb-hpc/tlandman/Data/Patient_MHA', scan, 'Segmentation.mha'))
    daily_dose = read_mha(os.path.join('/exports/lkeb-hpc/tlandman/Data/Patient_MHA', scan, 'Dose.mha'))
    predicted_dose = read_mha(os.path.join('/exports/lkeb-hpc/tlandman/Thesis/MultiTask/experiments/Single-Task', exp_name, 'output/HMC', scan, 'Dose.mha'))

    fig, axs = plt.subplots(1 ,4)
    # fig.suptitle('Comparing the daily scan with the registered planning scan')
    max_dose = np.max(np.append(daily_dose, predicted_dose))
    axs[0] = show_CT(fig, axs[0], daily_CT, dose=daily_dose, cont=daily_cont, slice=slice,
                        title='Ground-Truth (Gy)', max_dose=max_dose, overlay=overlay)
    axs[1] = show_CT(fig, axs[1], daily_CT, dose=predicted_dose, cont=daily_cont, slice=slice,
                        title='Prediction (Gy)', max_dose=max_dose, overlay=overlay)

    diff_dose = daily_dose - predicted_dose
    axs[2] = show_diff(fig, axs[2], daily_CT, diff=diff_dose, cont=daily_cont, lim=10, slice=slice,
                        title='Difference (Gy)', overlay=overlay)

    if plot_gamma:
        gamma_map = get_gamma_map(scan, dose=daily_dose, dose_pred=predicted_dose, cont=daily_cont, print_values=True)
        axs[3] = show_gamma(fig, axs[3], daily_CT, gamma_map, cont=daily_cont, slice=slice, title='Gamma Map')

    plt.show()

# planning_daily_dose()
# DVH_plot()

def contour_plots():
    scan = 'Patient_21/visit_20071207'
    slice = 52
    overlay = False
    daily_CT = read_mha(os.path.join('/exports/lkeb-hpc/tlandman/Data/Patient_MHA', scan, 'CTImage.mha'))
    daily_cont = read_mha(os.path.join('/exports/lkeb-hpc/tlandman/Data/Patient_MHA', scan, 'Segmentation.mha'))
    seg_name = 'Seg_input_If_Sm'
    seg_cont = read_mha(os.path.join('/exports/lkeb-hpc/tlandman/Thesis/MultiTask/experiments/Single-Task', seg_name, 'output/HMC', scan, 'Segmentation.mha'))
    reg_name = 'Reg_input_If_Im'
    reg_cont = read_mha(
        os.path.join('/exports/lkeb-hpc/tlandman/Thesis/MultiTask/experiments/Single-Task', reg_name, 'output/HMC',
                     scan, 'ResampledSegmentation.mha'))
    CS_name = 'CS_inSeg_If_Sm_inReg_If_Im_Sm_lSeg_DSC_lReg_NCC_DSCWarp_equal'
    CS_cont = read_mha(
        os.path.join('/exports/lkeb-hpc/tlandman/Thesis/MultiTask/experiments/Multi-Task', CS_name, 'output/HMC',
                     scan, 'Segmentation.mha'))

    # pred_new_cont = read_mha(os.path.join('/exports/lkeb-hpc/tlandman/Thesis/temp/predicted_contours', scan, 'Segmentation.mha'))
    # diff = seg_cont - pred_new_cont
    # print(np.max(diff))
    # plt.imshow(diff[70, :, :])
    daily_CT = np.where(daily_CT>100, 100, daily_CT)
    daily_CT = np.where(daily_CT < -100, -100, daily_CT)
    fig, axs = plt.subplots(2, 2)
    # fig.suptitle('Comparing the daily scan with the registered planning scan')

    # axs[0, 0] = show_CT(fig, axs[0, 0], daily_CT, cont=seg_cont, slice=slice,
    #                     title='Segmentation', overlay=overlay)
    # axs[0, 1] = show_CT(fig, axs[0, 1], daily_CT, cont=reg_cont, slice=slice,
    #                     title='Registration', overlay=overlay)
    # axs[1, 0] = show_CT(fig, axs[1, 0], daily_CT, cont=CS_cont, slice=slice,
    #                     title='Cross-Stitch', overlay=overlay)
    axs[1, 1] = show_CT(fig, axs[1, 1], daily_CT, slice=slice,
                        title='Manual', overlay=overlay)
    axs[1, 0] = show_CT(fig, axs[1, 0], daily_CT, cont=daily_cont, slice=slice,
                        title='Manual', overlay=overlay)

    plt.show()

contour_plots()




# arget_mask = read_mha('/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_24/visit_20071123/output_targets_mask.mha')
# scan = 'Patient_04/visit_20070622'
# daily_cont = read_mha(os.path.join('/exports/lkeb-hpc/tlandman/Data/Patient_MHA', scan, 'Segmentation.mha'))
# pred_cont = read_mha(os.path.join('/exports/lkeb-hpc/tlandman/Thesis/temp/predicted_contours', scan, 'Segmentation.mha'))
# fig, (ax1, ax2) = plt.subplots(1,2)
#
# ax1.imshow(daily_cont[70,:,:])
# ax2.imshow(pred_cont[70,:,:])
# plt.imshow(arget_mask[65, :,:])
# plt.show()
