import matplotlib.pyplot as plt
import numpy as np
import pylab

def loss_plot(training_loss, validation_loss, std_train, std_val):
    fig, ax = plt.subplots()
    epochs = len(training_loss)
    ax.plot(np.linspace(0, epochs, epochs), training_loss)
    ax.fill_between(np.linspace(0, epochs, epochs), training_loss - std_train, training_loss + std_train, alpha=0.3)
    ax.plot(np.linspace(0, epochs, epochs), validation_loss)
    ax.fill_between(np.linspace(0, epochs, epochs), validation_loss - std_val, validation_loss + std_val, alpha=0.3)
    plt.title('Training and validation loss')
    plt.legend(['Training_loss', 'Validation_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def dose_diff_plot(struct, dose, dose_pred, gamma_map):
    struct = np.squeeze(struct)
    dose = np.squeeze(dose)
    dose_pred = np.squeeze(dose_pred)
    diff = dose- dose_pred

    fig, axs = plt.subplots(2, 2)
    slice = 34

    plot = axs[0, 0].imshow(dose[:, :, slice], cmap='jet')
    axs[0, 0].set_title('Plan Dose [Gy]')
    for i in range(np.shape(struct)[0]):
        axs[0, 0].contour(struct[i, :, :, slice], levels=1, colors='white', linestyles='--')
    fig.colorbar(plot, ax=axs[0,0])

    plot = axs[0, 1].imshow(dose_pred[:, :, slice], cmap='jet')
    axs[0, 1].set_title('Predicted Dose [Gy]')
    for i in range(np.shape(struct)[0]):
        axs[0, 1].contour(struct[i, :, :, slice], levels=1, colors='white', linestyles='--')
    fig.colorbar(plot, ax=axs[0,1])

    plot = axs[1, 0].imshow(diff[:, :, slice], cmap='bwr', vmax=20, vmin=-20)
    axs[1, 0].set_title('Dose Difference [Gy]')
    for i in range(np.shape(struct)[0]):
        axs[1, 0].contour(struct[i, :, :, slice], levels=1, colors='green', linestyles='--')
    fig.colorbar(plot, ax=axs[1,0])

    plot = axs[1, 1].imshow(gamma_map[:, :, slice], cmap='Greys')
    axs[1, 1].set_title('3%/3mm Gamma Map')
    for i in range(np.shape(struct)[0]):
        axs[1, 1].contour(struct[i, :, :, slice], levels=1, colors='green', linestyles='--')
    fig.colorbar(plot, ax=axs[1, 1])
    # plt.savefig('/exports/lkeb-hpc/tlandman/')
    plt.show()


def DVH_plot(dose, structs, structure, color, style):
    struct_names = ['GTV', 'SeminalVesicle', 'Rectum', 'Bladder', 'Torso']

    if structure in struct_names:
        num = struct_names.index(structure)
    struct = np.squeeze(structs[:, num, :, :, :])
    dose = np.squeeze(dose)
    PTV_vox = dose[struct]
    Nvox = PTV_vox.size
    PTV_vox = np.reshape(PTV_vox, [Nvox])
    PTV_sort = np.sort(PTV_vox)
    vol = 1 - (np.linspace(0, Nvox-1, Nvox)/Nvox)
    plt.plot(PTV_sort, vol, color, linestyle=style)
    plt.xlabel('dose [Gy]')
    plt.ylabel('Volume')

def DVH_calc(structs, dose, structure, dose_limit):
    struct_names = ['GTV', 'SeminalVesicle', 'Rectum', 'Bladder', 'Torso']

    if structure in struct_names:
        num = struct_names.index(structure)
    struct = np.squeeze(structs[:, num, :, :, :])
    dose = np.squeeze(dose)
    dose_struct = dose[struct]
    vox_above = np.count_nonzero(dose_struct>dose_limit)
    vox_total = np.size(dose_struct)
    return vox_above/vox_total

def DICE_plot(DICE, Dice_step):
    ave_dice = np.average(DICE, axis=0)
    std_dice = np.std(DICE, axis=0)
    #fig, ax = plt.subplots()
    plt.plot(Dice_step, ave_dice)
    plt.fill_between(Dice_step, ave_dice - std_dice, ave_dice + std_dice, alpha=0.3)
    #ax.set_ylim(0, 1)
    plt.xlabel('isodose [%]')
    plt.ylabel('DICE []')
    #plt.title('DICE coefficient for isodose volumes')
    plt.show()


def Contour_plot():
    sw_struct = np.swapaxes(structure, 2, 3)
    test = np.zeros([144, 96, 64])
    test[sw_struct[3]] = 1
    test[sw_struct[-1]] = 2

    colmap = mcolors.ListedColormap(np.random.random((3, 3)))
    colmap.colors[0] = mcolors.ColorConverter.to_rgb(mcolors.CSS4_COLORS['darkblue'])
    colmap.colors[1] = mcolors.ColorConverter.to_rgb(mcolors.CSS4_COLORS['royalblue'])
    colmap.colors[2] = mcolors.ColorConverter.to_rgb(mcolors.CSS4_COLORS['yellow'])

    im = plt.imshow(test[:, 58, :].T, cmap=colmap)
    struct_lab = ['Exterior', 'Body', 'PTV']
    patches = [mpatches.Patch(color=colmap.colors[i], label=struct_lab[i]) for i in range(len(colmap.colors))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.axis('off')
    plt.show()

    sw_struct = np.swapaxes(structure, 2, 3)
    test = np.zeros([144, 96, 64])
    test[sw_struct[3]] = 1
    test[sw_struct[0]] = 2
    test[sw_struct[-1]] = 3

    colmap = mcolors.ListedColormap(np.random.random((4, 3)))
    colmap.colors[0] = mcolors.ColorConverter.to_rgb(mcolors.CSS4_COLORS['darkblue'])
    colmap.colors[1] = mcolors.ColorConverter.to_rgb(mcolors.CSS4_COLORS['royalblue'])
    colmap.colors[2] = mcolors.ColorConverter.to_rgb(mcolors.CSS4_COLORS['red'])
    colmap.colors[3] = mcolors.ColorConverter.to_rgb(mcolors.CSS4_COLORS['yellow'])

    im = plt.imshow(test[:, :, 32].T, cmap=colmap)
    struct_lab = ['Exterior', 'Body', 'Rectum', 'PTV']
    patches = [mpatches.Patch(color=colmap.colors[i], label=struct_lab[i]) for i in range(len(colmap.colors))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.axis('off')
    plt.show()