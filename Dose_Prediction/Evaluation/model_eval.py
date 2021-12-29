import torch.optim as optim
import torch
import numpy as np
import os

import sys
import matplotlib.pyplot as plt
# import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# sys.path.insert(1, r'C:\Users\t.meerbothe\Desktop\WS0102\Scripts\Final')
# sys.path.insert(2, r'C:\Users\t.meerbothe\Desktop\WS0102\Scripts\Model')
sys.path.insert(1, '/exports/lkeb-hpc/tlandman/Dose_Prediction/')
import load_data_new
#sys.path.insert(1, '/exports/lkeb-hpc/tlandman/Dose_Prediction/')
# import data_augmentation as aug
# import data_import
#from U_Net import UNet
#import dose_char
import plots
import gamma
import csv
import pydicom
# import Comparison
import SimpleITK as sitk

  

loss_path = "/exports/lkeb-hpc/tlandman/Dose_Prediction/log/training_loss5.csv"
patient_list = []
visit_date_list = []
with open("/exports/lkeb-hpc/tlandman/Patient_Data/visits.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        patient_list.append(str(row[0]))
        visit_date_list.append(str(row[1]))

dir_path = "/exports/lkeb-hpc/tlandman/Patient_Dose/"
print('Hello')
#model = UNet()
#model = model.cuda()
#optimizer = optim.Adam(model.parameters(), lr=1e-03)
#checkpoint = torch.load('/exports/lkeb-hpc/tlandman/Dose_Prediction/model/model5.pth.tar', map_location=torch.device('cpu'))
#model.load_state_dict(checkpoint['model_state_dict'])
#device = torch.device("cuda")

#data = []
#with open(loss_path, newline='') as csvfile:
#    losses = csv.reader(csvfile, delimiter=',')
#    for row in losses:
#        data.append(row)
#data = np.array(data[1:])
#training_loss = np.double(data[:,1])
#std_train = np.double(data[:,2])
#validation_loss = np.double(data[:,3])
#std_val = np.double(data[:,4])

gamma_list = []
#model.eval()
with torch.no_grad():
    scanID = 146
    data_path = os.path.join(dir_path, patient_list[scanID], visit_date_list[scanID])
    #struct, dose = load_data_new.load_data(data_path, dim=[128, 256, 64])
    #dose_tens = torch.from_numpy(dose).to(device)
    #struct_tens = torch.from_numpy(struct).to(device)
    #dose_pred = model(struct_tens)
    #dose_pred = np.array(dose_pred.cpu())
    #dose_path = os.path.join(data_path, "Dose", "RTDose_4_physicalDose.dcm")
    #ps = pydicom.dcmread(dose_path).PixelSpacing
    #ss = pydicom.dcmread(dose_path).SliceThickness
    #res = (ps[0], ps[1], ss)
    #gamma_map = gamma.gamma_evaluation(np.squeeze(dose_pred), np.squeeze(dose), 3., 3., res, signed=False)
    
    scanID = 146
    data_path = os.path.join(dir_path, patient_list[scanID], visit_date_list[scanID])
    struct, dose = load_data_new.load_data(data_path)
    print('still oke')
    reader = sitk.ImageFileReader()
    reader.SetImageIO("MetaImageIO")
    reader.SetFileName('/exports/lkeb-hpc/tlandman/New_Code/JRS-MTL/experiments/Single-Task/Dose_input_Sf/output/HMC/Patient_21/visit_20071204/Dose.mha')
    print('still ok 32')
    dose_pred = reader.Execute()

    dose_pred = sitk.GetArrayFromImage(dose_pred)
    dose_path = os.path.join(data_path, "Dose", "RTDose_4_physicalDose.dcm")
    dose_pred = dose_pred[0:128,:,:]
    ds = pydicom.dcmread(dose_path)
    dose = np.float16(ds.pixel_array * ds.DoseGridScaling)
    print(np.shape(dose_pred))
    print(np.shape(dose))
    gamma_map = gamma.gamma_evaluation(np.squeeze(dose_pred), np.squeeze(dose), 3., 3., (0.875,0.875,2), signed=False)
    print('almost')
    # for scanID in range(144, len(visit_date_list)):
    #     data_path = os.path.join(dir_path, patient_list[scanID], visit_date_list[scanID])
    #     struct, dose = load_data_new.load_data(data_path, dim=[128, 256, 64])
    #     dose_tens = torch.from_numpy(dose).to(device)
    #     struct_tens = torch.from_numpy(struct).to(device)
    #     dose_pred = model(struct_tens)
    #     dose = np.array(dose_tens.cpu())
    #     dose_pred = np.array(dose_pred.cpu())
        # dose_path = os.path.join(data_path, "Dose", "RTDose_4_physicalDose.dcm")
        # ps = pydicom.dcmread(dose_path).PixelSpacing
        # ss = pydicom.dcmread(dose_path).SliceThickness
        # res = (ps[0], ps[1], ss)
        # gamma_map = gamma.gamma_evaluation(np.squeeze(dose_pred), np.squeeze(dose), 3., 3., res, signed=False)
        # gamma_list.append(gamma.pass_rate(gamma_map))
        # print('Gamma for scan '+str(scanID)+' is '+str(gamma_list[-1]))
# print(gamma_list)
# print(np.median(gamma_list))
# gamma_list.sort()
# print(gamma_list)
# struct = np.array(struct_tens.cpu(), dtype=bool)
# dose = np.array(dose_tens.cpu())
# dose_pred = np.array(dose_pred.cpu())

# dose_path = os.path.join(data_path, "Dose", "RTDose_4_physicalDose.dcm")
# ps = pydicom.dcmread(dose_path).PixelSpacing
# ss = pydicom.dcmread(dose_path).SliceThickness
# res = (ps[0], ps[1], ss)
#
# # Perform gamma evaluation at 3mm, 3%, resolution x=1, y=1, z=3
# gamma_map = gamma.gamma_evaluation(np.squeeze(dose_pred), np.squeeze(dose), 3., 3., res, signed=False)
#
# gamma.gamma_plot(gamma_map,  only_pass=False, cmap='Greys')
# gamma.gamma_plot(gamma_map,  only_pass=True, cmap='Greys')
#
# print(gamma.pass_rate(gamma_map))



# plots.loss_plot(training_loss, validation_loss, std_train, std_val)

plots.dose_diff_plot(struct, dose, dose_pred, gamma_map)

# plt.figure()
# plots.DVH_plot(dose, struct, 'GTV', 'C0', '-')
# plots.DVH_plot(dose_pred, struct, 'GTV', 'C0', '--')
# plots.DVH_plot(dose, struct, 'SeminalVesicle', 'C1', '-')
# plots.DVH_plot(dose_pred, struct, 'SeminalVesicle', 'C1', '--')
# plots.DVH_plot(dose, struct, 'Rectum', 'C2', '-')
# plots.DVH_plot(dose_pred, struct, 'Rectum', 'C2', '--')
# plots.DVH_plot(dose, struct, 'Bladder', 'C3', '-')
# plots.DVH_plot(dose_pred, struct, 'Bladder', 'C3', '--')
# plt.legend(['GTV', 'GTV pred', 'Seminal Vesicle', 'Seminal Vesicle pred', 'Rectum', 'Rectum pred', 'Bladder', 'Bladder pred'])
# plt.title('DVH comparison scan %s' % scanID)
# plt.show()


# Plotting.dose_diff_plot(struct, dose, dose_pred)

# V95 = []
# for scanID in range(len(patient_list)):
#     data_path = os.path.join(dir_path, patient_list[scanID], visit_date_list[scanID])
#     struct, dose = load_data_new.load_data(data_path, dim=[96, 192, 48])
#     struct = np.bool_(struct)
#     V95.append(Plotting.DVH_calc(struct, dose, 'GTV', 0.95 * 74))
# V95.sort()
# plt.figure()
# plt.plot(V95)
# plt.show()

# plt.figure()
# Plotting.DVH_plot(dose, structure, 'Bladder', 'C1', '-')
# Plotting.DVH_plot(pr_dose, structure, 'Bladder', 'C2', '-')
# #plt.plot(resultOVH[1]/100, resultOVH[0]/100, 'C3')
# plt.legend(['Truth', 'DL pred'])
# plt.xlabel('Dose [Gy]')
# plt.ylabel('Volume')
# plt.title('Rectum predictions for scan %s' % scanID)
# plt.show()
#
# sw_dose = np.swapaxes(pr_dose,1,2)
# fig, ax = plt.subplots()
# plot = ax.imshow(sw_dose[72, :, :])
# plt.axis('off')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(plot,cax=cax)
#
# sw_struct = np.swapaxes(structure, 2, 3)
# test = np.zeros([144,96,64])
# test[sw_struct[3]] = 1
# test[sw_struct[0]] = 2
# test[sw_struct[1]] = 3
# test[sw_struct[-1]] = 4
#
# colmap = mcolors.ListedColormap(np.random.random((5,3)))
# colmap.colors[0] = mcolors.ColorConverter.to_rgb(mcolors.CSS4_COLORS['darkblue'])
# colmap.colors[1] = mcolors.ColorConverter.to_rgb(mcolors.CSS4_COLORS['royalblue'])
# colmap.colors[2] = mcolors.ColorConverter.to_rgb(mcolors.CSS4_COLORS['red'])
# colmap.colors[3] = mcolors.ColorConverter.to_rgb(mcolors.CSS4_COLORS['green'])
# colmap.colors[4] = mcolors.ColorConverter.to_rgb(mcolors.CSS4_COLORS['yellow'])
#
# im = plt.imshow(test[72,:,:], cmap=colmap)
# struct_lab = ['Exterior', 'Body', 'Rectum', 'Anal Sphincter', 'PTV']
# patches = [ mpatches.Patch(color=colmap.colors[i], label=struct_lab[i] ) for i in range(len(colmap.colors)) ]
# plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
#
# Plotting.dose_diff_plot(pr_dose, dose, structure[-1, :, :, :], axis=0)
# Plotting.dose_diff_plot(pr_dose, dose, structure[-1, :, :, :], axis=1)
# Plotting.dose_diff_plot(pr_dose, dose, structure[-1, :, :, :], axis=2)
#


# # Dose characteristics of all validation examples
# Ntest = 12
# Dice_step = np.linspace(0, 100, 101, dtype=int)/100
# PTV_Dmax = np.zeros([Ntest, 2])
# PTV_Dmean = np.zeros([Ntest, 2])
# PTV_D98 = np.zeros([Ntest, 2])
# PTV_D95 = np.zeros([Ntest, 2])
# RECT_Dmax = np.zeros([Ntest, 2])
# RECT_Dmean = np.zeros([Ntest, 2])
# RECT_V45 = np.zeros([Ntest, 2])
# CI = np.zeros([Ntest, 2])
# HI = np.zeros([Ntest, 2])
# DICE = np.zeros([Ntest, len(Dice_step)])
#
# for i in range(Ntest):
#     print(i)
#     with torch.no_grad():
#         structure, dose, emax, emin = data_import.input_data(pat_list[77+i])
#         PTV_Dmax[i, 0] = dose_char.Dx_calc(dose, structure, 0.02, 'PTVpros+vs')
#         PTV_Dmean[i, 0] = dose_char.mean_dose(dose, structure, 'PTVpros+vs')
#         PTV_D98[i, 0] = dose_char.Dx_calc(dose, structure, 0.98, 'PTVpros+vs')
#         PTV_D95[i, 0] = dose_char.Dx_calc(dose, structure, 0.95, 'PTVpros+vs')
#         RECT_Dmax[i, 0] = dose_char.Dx_calc(dose, structure, 0.02, 'RECTUM')
#         RECT_Dmean[i, 0] = dose_char.mean_dose(dose, structure, 'RECTUM')
#         RECT_V45[i, 0] = dose_char.Vx_calc(dose, structure, 45, 'RECTUM')
#         CI[i, 0] = dose_char.CI_calc(dose, structure, 60)
#         HI[i, 0] = dose_char.HI_calc(dose, structure, PTV_D98[i, 0])
#
#         #pred_dose = np.load(r'C:\Users\t.meerbothe\Desktop\WS0102\Data\hit_check\Test\pat' + str(i + 77) + '.npy')
#         #structure_6 = np.concatenate((structure, np.expand_dims(np.swapaxes(pred_dose, 1, 2), axis=0)), axis=0)
#         #pred_dose = np.load(
#         #    r'C:\Users\t.meerbothe\Desktop\project_Thierry_64_dev\Data\Segment\Orig_engine\hit_check\Test\pat' + str(
#         #        i + 77) + '.npy')
#         #pred_dose = np.swapaxes(pred_dose,1,2)
#         #structure_1 = np.expand_dims(pred_dose, 0)
#         str_gpu_tens = aug.structure_transform(structure.copy(), trans_list[0])
#         output = model(str_gpu_tens)
#         pr_dose = output.squeeze().detach().numpy()
#
#         PTV_Dmax[i, 1] = dose_char.Dx_calc(pr_dose, structure, 0.02, 'PTVpros+vs')
#         PTV_Dmean[i, 1] = dose_char.mean_dose(pr_dose, structure, 'PTVpros+vs')
#         PTV_D98[i, 1] = dose_char.Dx_calc(pr_dose, structure, 0.98, 'PTVpros+vs')
#         PTV_D95[i, 1] = dose_char.Dx_calc(pr_dose, structure, 0.95, 'PTVpros+vs')
#         RECT_Dmax[i, 1] = dose_char.Dx_calc(pr_dose, structure, 0.02, 'RECTUM')
#         RECT_Dmean[i, 1] = dose_char.mean_dose(pr_dose, structure, 'RECTUM')
#         RECT_V45[i, 1] = dose_char.Vx_calc(pr_dose, structure, 45, 'RECTUM')
#         CI[i, 1] = dose_char.CI_calc(pr_dose, structure, 60)
#         HI[i, 1] = dose_char.HI_calc(pr_dose, structure, PTV_D98[i, 1])
#         for j in range(len(Dice_step)):
#             DICE[i, j] = dose_char.isodose_dice(dose, pr_dose, Dice_step[j])
#
#
# def ave_perc_err(values):
#     perc_err = abs(100 * ((values[:, 0] - values[:, 1]) / 60))
#     meanperc_errstd = np.zeros(2)
#     meanperc_errstd[0] = np.average(perc_err)
#     meanperc_errstd[1] = np.std(perc_err)
#     return meanperc_errstd
#
#
# def rel_err(values):
#     perc_err = abs(100 * ((values[:, 0] - values[:, 1]) / values[:, 0]))
#     meanperc_errstd = np.zeros(2)
#     meanperc_errstd[0] = np.average(perc_err)
#     meanperc_errstd[1] = np.std(perc_err)
#     return meanperc_errstd
#
#
# def rel_err_ind(values):
#     perc_err = abs(100 * (values[:, 0] - values[:, 1]))
#     meanperc_errstd = np.zeros(2)
#     meanperc_errstd[0] = np.average(perc_err)
#     meanperc_errstd[1] = np.std(perc_err)
#     return meanperc_errstd
#
# meanperc_errstd_PTV_D95 = ave_perc_err(PTV_D95)
# meanperc_errstd_PTV_D98 = ave_perc_err(PTV_D98)
# meanperc_errstd_PTV_Dmax = ave_perc_err(PTV_Dmax)
# meanperc_errstd_PTV_Dmean = ave_perc_err(PTV_Dmean)
# meanperc_errstd_RECT_Dmax = rel_err(RECT_Dmax)
# meanperc_errstd_RECT_Dmean = rel_err(RECT_Dmean)
# meanperc_errstd_RECT_V45 = rel_err(RECT_V45)
# rel_err_CI = rel_err_ind(CI)
# rel_err_HI = rel_err_ind(HI)
#
# res_D95_mean = np.mean(PTV_D95, axis=0)
# res_D95_std = np.std(PTV_D95, axis=0)
# res_D98_mean = np.mean(PTV_D98, axis=0)
# res_D98_std = np.std(PTV_D98, axis=0)
# res_Dmax_mean = np.mean(PTV_Dmax, axis=0)
# res_Dmax_std = np.std(PTV_Dmax, axis=0)
# res_Dmean_mean = np.mean(PTV_Dmean, axis=0)
# res_Dmean_std = np.std(PTV_Dmean, axis=0)
# res_RDmax_mean = np.mean(RECT_Dmax, axis=0)
# res_RDmax_std = np.std(RECT_Dmax, axis=0)
# res_RDmean_mean = np.mean(RECT_Dmean, axis=0)
# res_RDmean_std = np.std(RECT_Dmean, axis=0)
# res_RV45_mean = np.mean(RECT_V45, axis=0)
# res_RV45_std = np.std(RECT_V45, axis=0)
# res_CI_mean = np.mean(CI, axis=0)
# res_CI_std = np.std(CI, axis=0)
# res_HI_mean = np.mean(HI, axis=0)
# res_HI_std = np.std(HI, axis=0)
#
# Plotting.DICE_plot(DICE, Dice_step)
#
# box = np.zeros([12,7])
# box[:,0] = 100 * ((PTV_D98[:, 0] - PTV_D98[:, 1]) / 60)
# box[:,1] = 100 * ((PTV_D95[:, 0] - PTV_D95[:, 1]) / 60)
# box[:,2] = 100 * ((PTV_Dmax[:, 0] - PTV_Dmax[:, 1]) / 60)
# box[:,3] = 100 * ((PTV_Dmean[:, 0] - PTV_Dmean[:, 1]) / 60)
# box[:,4] = 100 * ((RECT_Dmax[:, 0] - RECT_Dmax[:, 1]) / RECT_Dmax[:, 0])
# box[:,5] = 100 * ((RECT_Dmean[:, 0] - RECT_Dmean[:, 1]) / RECT_Dmean[:, 0])
# box[:,6] = 100 * ((RECT_V45[:, 0] - RECT_V45[:, 1]) / RECT_V45[:, 0])
# fig, ax = plt.subplots()
# ax.boxplot(box, labels=['PTV_D98', 'PTV_D95', 'PTV_Dmax', 'PTV_Dmean', 'Rect_Dmax', 'Rect_Dmean', 'Rect_V45'])
# plt.xticks(rotation=30, ha='right')
# plt.ylabel(r'$100 \cdot (\frac{D_{True} - D_{Pred}}{D_{True}}) $')
# ax.set_ylim(-20, 20)
#
# def DVH_calc_nodf(dose, structure, struct_arr):
#     struc_list = r'C:\Users\t.meerbothe\Desktop\project_Thierry_64_dev\Lists/Structures.txt'
#     with open(struc_list) as s:
#         structs = [line.rstrip()[1:-1] for line in s]
#     structs.append('PTVpros+vs')
#     if structure in structs:
#         num = structs.index(structure)
#     struct_arr_inb = np.squeeze(struct_arr[num, :, :, :])
#     PTV_vox = dose[struct_arr_inb]
#     Nvox = PTV_vox.size
#     PTV_vox = np.reshape(PTV_vox, [Nvox])
#     PTV_sort = np.sort(PTV_vox)
#
#     return PTV_sort
#
# def ave_abs_err(values):
#     diff = abs(values[:, 0] - values[:, 1])
#     ADD = np.zeros(2)
#     ADD[0] = np.average(diff)
#     ADD[1] = np.std(diff)
#     return ADD
#
#
# def DVH_to_list(DVH_vals):
#     DVH_points = np.linspace(1,60,60)
#     int_DVH = np.zeros([2,61])
#     int_DVH[0,:] = np.linspace(0,60,61)
#     int_DVH[1,0] = 1
#     n = 1
#     for i in DVH_points:
#         check = np.where(DVH_vals > i)
#         if len(check[0]) == len(DVH_vals):
#             int_DVH[1,n] = 1
#             n += 1
#         else:
#             ind = check[0][0]
#             lower = DVH_vals[ind-1]
#             upper = DVH_vals[ind]
#             intlow = i - lower
#             intup = upper - i
#             inter_val = intlow/(intlow + intup)
#             loc = ind - inter_val
#             volume = 1 - loc/len(DVH_vals)
#             int_DVH[1,n] = volume
#             n+=1
#     return int_DVH
#
# # Absolute DVH error
# struct = 'RECTUM'
# out_PTV_tr = DVH_calc_nodf(dose, struct, structure)
# out_PTV_pr = DVH_calc_nodf(pr_dose, struct, structure)
# vals = np.zeros([len(out_PTV_tr),2])
# vals[:, 0] = out_PTV_tr
# vals[:, 1] = out_PTV_pr
# test_out = ave_abs_err(vals)
#
# struct = 'RECTUM'
# patID = patIDs[pat]
# resultOVH = Comparison.OVHpred(patID)
# out_PTV_tr = DVH_calc_nodf(dose, struct, structure)
# out_PTV_pr = DVH_calc_nodf(pr_dose, struct, structure)
# out_DVH = DVH_to_list(out_PTV_tr)
# out_DVH_pr = DVH_to_list(out_PTV_pr)
# vals = np.zeros([61,2])
# vals[:,0] = out_DVH[1,:]
# vals[:,1] = resultOVH[0]/100
# test_out_OVH = ave_abs_err(vals)
# vals[:,1] = out_DVH_pr[1,:]
# test_out_pr = ave_abs_err(vals)