import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from load_data import read_mha, get_paths_csv
import SimpleITK as sitk
import os

# planning_paths = get_paths_csv("/exports/lkeb-hpc/tlandman/Thesis/MultiTask/data/thomas/planning_dose_dir.csv")
# dose_paths = get_paths_csv("/exports/lkeb-hpc/tlandman/Thesis/MultiTask/data/thomas/fixed_Dose.csv")
# CT_paths = get_paths_csv("/exports/lkeb-hpc/tlandman/Thesis/MultiTask/data/thomas/fixed_CTImage.csv")
# reader = sitk.ImageFileReader()
# reader.SetImageIO("MetaImageIO")
# writer = sitk.ImageFileWriter()
# for i in range(len(planning_paths)):
#     reader.SetFileName(os.path.join(planning_paths[i],'Dose.mha'))
#     planning_Dose = reader.Execute()
#     reader.SetFileName(os.path.join(planning_paths[i],'CTImage.mha'))
#     planning_CT = reader.Execute()
#
#     #     dose_paths[i])
#     # daily_Dose = reader.Execute()
#     # reader.SetFileName(CT_paths[i])
#     # daily_CT = reader.Execute()
#
#     planning_Dose.SetOrigin(planning_CT.GetOrigin())
#     planning_Dose.SetDirection(planning_CT.GetDirection())
#     planning_Dose.SetSpacing(planning_CT.GetSpacing())
#     # writer.SetFileName(os.path.join(dose_paths[i], 'Dose.mha'))
#     # writer.Execute(daily_Dose)
#     print(planning_paths[i])
#     print(planning_Dose.GetSpacing())
#     print(planning_Dose.GetOrigin())
#     # print(dose_paths[i])
#     print(planning_CT.GetSpacing())
#     print(planning_CT.GetOrigin())
#     print(' ')
#
# dose_moving_path = "/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_02/visit_20070529/Dose.mha"
planning_CT_path = "/exports/lkeb-hpc/mseelmahdy/HaukelandAffine/Patient_04/visit_20070605/Images/CTImage.mha"
# dose_fixed_path = "/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_02/visit_20070601/Dose.mha"
# dose_fixed_path = "/exports/lkeb-hpc/mseelmahdy/HaukelandAffine/Patient_02/visit_20070601/Images_Affine/CTImage.mha"
daily_CT_path = "/exports/lkeb-hpc/mseelmahdy/HaukelandAffine/Patient_04/visit_20070612/Images/CTImage.mha"
reader = sitk.ImageFileReader()
reader.SetImageIO("MetaImageIO")
reader.SetFileName(planning_CT_path)
planning_CT = reader.Execute()
reader.SetFileName(daily_CT_path)
daily_CT = reader.Execute()
euler3d = sitk.Euler3DTransform()
inv_euler3d = euler3d.GetInverse()
resampled_image = sitk.Resample(planning_CT, daily_CT.GetSize(), sitk.AffineTransform(3), sitk.sitkLinear, daily_CT.GetOrigin(), daily_CT.GetSpacing(), daily_CT.GetDirection())
# resampled_image_2 = sitk.Resample(planning_CT, daily_CT.GetSize(), sitk.AffineTransform(3), sitk.sitkLinear, daily_CT.GetOrigin(), daily_CT.GetSpacing(), daily_CT.GetDirection())
np_resam = sitk.GetArrayFromImage(resampled_image)
# np_resam_2 = sitk.GetArrayFromImage(resampled_image)
reader.SetFileName("/exports/lkeb-hpc/mseelmahdy/HaukelandAffine/Patient_04/visit_20070612/Images_Affine/CTImage.mha")
Affine_CT = sitk.GetArrayFromImage(reader.Execute())
diff = np_resam-Affine_CT
print(planning_CT.GetSpacing())
print(daily_CT.GetSpacing())
print(planning_CT.GetOrigin())
print(daily_CT.GetOrigin())
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(sitk.GetArrayFromImage(daily_CT)[65,:,:])
axs[0, 1].imshow(sitk.GetArrayFromImage(planning_CT)[65,:,:])
axs[1, 0].imshow(Affine_CT[65,:,:])
axs[1, 1].imshow(np_resam[65,:,:])
plt.show()
# plt.imshow(sitk.GetArrayViewFromImage(resampled_image)[55,:,:])
# plt.axis('off')
# plt.show()
#
# print(planning_CT.GetSpacing())
# print(daily_CT.GetSpacing())
# print(resampled_image.GetSpacing())
# print(planning_CT.GetSize())
# print(daily_CT.GetSize())
# print(resampled_image.GetSize())
# print(planning_CT.GetOrigin())
# print(daily_CT.GetOrigin())
# print(resampled_image.GetOrigin())
# #
# #
# # euler2d = sitk.Euler2DTransform()
# # # Why do we set the center?
# euler2d.SetCenter(logo.TransformContinuousIndexToPhysicalPoint(np.array(logo.GetSize())/2.0))
#
# tx = 64
# ty = 32
# euler2d.SetTranslation((tx, ty))
#
# extreme_points = [logo.TransformIndexToPhysicalPoint((0,0)),
#                   logo.TransformIndexToPhysicalPoint((logo.GetWidth(),0)),
#                   logo.TransformIndexToPhysicalPoint((logo.GetWidth(),logo.GetHeight())),
#                   logo.TransformIndexToPhysicalPoint((0,logo.GetHeight()))]
# inv_euler2d = euler2d.GetInverse()
#
# extreme_points_transformed = [inv_euler2d.TransformPoint(pnt) for pnt in extreme_points]
# min_x = min(extreme_points_transformed)[0]
# min_y = min(extreme_points_transformed, key=lambda p: p[1])[1]
# max_x = max(extreme_points_transformed)[0]
# max_y = max(extreme_points_transformed, key=lambda p: p[1])[1]
#
# # Use the original spacing (arbitrary decision).
# output_spacing = logo.GetSpacing()
# # Identity cosine matrix (arbitrary decision).
# output_direction = [1.0, 0.0, 0.0, 1.0]
# # Minimal x,y coordinates are the new origin.
# output_origin = [min_x, min_y]
# # Compute grid size based on the physical size and spacing.
# output_size = [int((max_x-min_x)/output_spacing[0]), int((max_y-min_y)/output_spacing[1])]
#
# resampled_image = sitk.Resample(logo, output_size, euler2d, sitk.sitkLinear, output_origin, output_spacing, output_direction)
# plt.imshow(sitk.GetArrayViewFromImage(resampled_image))
# plt.axis('off')
# plt.show()