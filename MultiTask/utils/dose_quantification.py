import numpy as np
import sys
sys.path.append('/exports/lkeb-hpc/tlandman/tf.14/lib/python3.6/site-packages/')
import SimpleITK as sitk
from pymedphys import gamma

def gamma_pass(groundtruth_dose, predicted_dose, groundtruth_contours, distance=2, threshold=2):

    assert groundtruth_dose.GetSize() == predicted_dose.GetSize(), \
                 "`sample` and `reference` must have the same shape"

    dims = groundtruth_dose.GetSize()
    origin = groundtruth_dose.GetOrigin()
    spacing = groundtruth_dose.GetSpacing()
    x = np.linspace(origin[0], origin[0] + spacing[0] * (dims[0] - 1), dims[0])
    y = np.linspace(origin[1], origin[0] + spacing[1] * (dims[1] - 1), dims[1])
    z = np.linspace(origin[2], origin[2] + spacing[2] * (dims[2] - 1), dims[2])
    axes = (z, y, x)

    dose = sitk.GetArrayFromImage(groundtruth_dose)
    dose_pred = sitk.GetArrayFromImage(predicted_dose)
    contours = sitk.GetArrayFromImage(groundtruth_contours)

    gamma_map = gamma(axes, dose, axes, dose_pred, threshold, distance, lower_percent_dose_cutoff=0.01,
                      interp_fraction=2, local_gamma=False, max_gamma=1, skip_once_passed=True, quiet=True)

    gamma_pass = np.where(gamma_map < 1, True, False)

    dose_mask = np.invert(np.isnan(gamma_map))
    gtv_mask = np.where(contours == 4, True, False)
    sv_mask = np.where(contours == 3, True, False)
    rectum_mask = np.where(contours == 2, True, False)
    bladder_mask = np.where(contours == 1, True, False)

    dose_gamma = np.count_nonzero(gamma_pass[dose_mask]) / np.size(gamma_pass[dose_mask])
    gtv_gamma = np.count_nonzero(gamma_pass[gtv_mask]) / np.size(gamma_pass[gtv_mask & dose_mask])
    sv_gamma = np.count_nonzero(gamma_pass[sv_mask]) / np.size(gamma_pass[sv_mask & dose_mask])
    rectum_gamma = np.count_nonzero(gamma_pass[rectum_mask]) / np.size(gamma_pass[rectum_mask & dose_mask])
    bladder_gamma = np.count_nonzero(gamma_pass[bladder_mask]) / np.size(gamma_pass[bladder_mask & dose_mask])

    return dose_gamma, gtv_gamma, sv_gamma, rectum_gamma, bladder_gamma

def Vx_target(groundtruth_dose, predicted_dose, groundtruth_contours):
    dose = sitk.GetArrayFromImage(groundtruth_dose)
    dose_pred = sitk.GetArrayFromImage(predicted_dose)
    contours = sitk.GetArrayFromImage(groundtruth_contours)

    gtv_dose_ref = dose[np.where(contours == 4, True, False)]
    sv_dose_ref = dose[np.where(contours == 3, True, False)]
    gtv_dose_pred = dose_pred[np.where(contours == 4, True, False)]
    sv_dose_pred = dose_pred[np.where(contours == 3, True, False)]

    V95_gtv_ref = np.sum(gtv_dose_ref >= 74 * 0.95) / np.size(gtv_dose_ref)
    V110_gtv_ref = np.sum(gtv_dose_ref <= 74 * 1.10) / np.size(gtv_dose_ref)
    V95_sv_ref = np.sum(sv_dose_ref >= 55 * 0.95) / np.size(sv_dose_ref)
    V110_sv_ref = np.sum(sv_dose_ref <= 55 * 1.10) / np.size(sv_dose_ref)

    V95_gtv_pred = np.sum(gtv_dose_pred >= 74 * 0.95) / np.size(gtv_dose_pred)
    V110_gtv_pred = np.sum(gtv_dose_pred <= 74 * 1.10) / np.size(gtv_dose_pred)
    V95_sv_pred = np.sum(sv_dose_pred >= 55 * 0.95) / np.size(sv_dose_pred)
    V110_sv_pred = np.sum(sv_dose_pred <= 55 * 1.10) / np.size(sv_dose_pred)

    V95_gtv = (V95_gtv_ref - V95_gtv_pred) / V95_gtv_ref
    V110_gtv = (V110_gtv_ref - V110_gtv_pred) / V110_gtv_ref
    V95_sv = (V95_sv_ref - V95_sv_pred) / V95_sv_ref
    V110_sv = (V110_sv_ref - V110_sv_pred) / V110_sv_ref

    Dmean_gtv_ref = np.mean(gtv_dose_ref)
    D95_gtv_ref = np.sort(gtv_dose_ref.flatten())[int(-np.size(gtv_dose_ref) * 0.95)]
    Dmean_sv_ref = np.mean(sv_dose_ref)
    D95_sv_ref = np.sort(sv_dose_ref.flatten())[int(-np.size(sv_dose_ref) * 0.95)]

    Dmean_gtv_pred = np.mean(gtv_dose_pred)
    D95_gtv_pred = np.sort(gtv_dose_pred.flatten())[int(-np.size(gtv_dose_pred) * 0.95)]
    Dmean_sv_pred = np.mean(sv_dose_pred)
    D95_sv_pred = np.sort(sv_dose_pred.flatten())[int(-np.size(sv_dose_pred) * 0.95)]

    Dmean_gtv = (Dmean_gtv_ref - Dmean_gtv_pred) / Dmean_gtv_ref
    D95_gtv = (D95_gtv_ref - D95_gtv_pred) / D95_gtv_ref
    Dmean_sv = (Dmean_sv_ref - Dmean_sv_pred) / Dmean_sv_ref
    D95_sv = (D95_sv_ref - D95_sv_pred) / D95_sv_ref

    return V95_gtv, V110_gtv, Dmean_gtv, D95_gtv, V95_sv, V110_sv, Dmean_sv, D95_sv

def Dx_oar(groundtruth_dose, predicted_dose, groundtruth_contours):
    dose = sitk.GetArrayFromImage(groundtruth_dose)
    dose_pred = sitk.GetArrayFromImage(predicted_dose)
    contours = sitk.GetArrayFromImage(groundtruth_contours)

    rectum_dose_ref = dose[np.where(contours == 2, True, False)]
    bladder_dose_ref = dose[np.where(contours == 1, True, False)]
    rectum_dose_pred = dose_pred[np.where(contours == 2, True, False)]
    bladder_dose_pred = dose_pred[np.where(contours == 1, True, False)]

    Dmean_rectum_ref = np.mean(rectum_dose_ref)
    D2_rectum_ref = np.sort(rectum_dose_ref.flatten())[int(-np.size(rectum_dose_ref) * 0.02)]
    Dmean_bladder_ref = np.mean(bladder_dose_ref)
    D2_bladder_ref = np.sort(bladder_dose_ref.flatten())[int(-np.size(bladder_dose_ref) * 0.02)]

    Dmean_rectum_pred = np.mean(rectum_dose_pred)
    D2_rectum_pred = np.sort(rectum_dose_pred.flatten())[int(-np.size(rectum_dose_pred) * 0.02)]
    Dmean_bladder_pred = np.mean(bladder_dose_pred)
    D2_bladder_pred = np.sort(bladder_dose_pred.flatten())[int(-np.size(bladder_dose_pred) * 0.02)]

    Dmean_rectum = (Dmean_rectum_ref - Dmean_rectum_pred) / Dmean_rectum_ref
    D2_rectum = (D2_rectum_ref - D2_rectum_pred) / D2_rectum_ref
    Dmean_bladder = (Dmean_bladder_ref - Dmean_bladder_pred) / Dmean_bladder_ref
    D2_bladder = (D2_bladder_ref - D2_bladder_pred) / D2_bladder_ref

    V45_rectum = 77
    V45_bladder = 77

    return Dmean_rectum, D2_rectum, V45_rectum, Dmean_bladder, D2_bladder, V45_bladder





pred_dose_path = '/exports/lkeb-hpc/tlandman/Thesis/MultiTask/experiments/Single-Task/Dose_input_Sf_If/output/HMC/Patient_22/visit_20071102/Dose.mha'
dose_path = '/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_22/visit_20071102/Dose.mha'
cont_path = '/exports/lkeb-hpc/tlandman/Data/Patient_MHA/Patient_22/visit_20071102/Segmentation.mha'
reader = sitk.ImageFileReader()
reader.SetImageIO("MetaImageIO")
reader.SetFileName(pred_dose_path)
pred_dose = reader.Execute()
reader.SetFileName(dose_path)
dose = reader.Execute()
reader.SetFileName(cont_path)
cont = reader.Execute()
# gamma_pass(dose, pred_dose, cont)
# Vx_target(dose, pred_dose, cont)
# Dx_oar(dose, pred_dose, cont)


# def gamma_pass_rate(groundtruth_dose, predicted_dose, groundtruth_contours, distance=2, threshold=2):
#     """
#         Distance to Agreement between a sample and reference using gamma evaluation.
#
#         Parameters
#         ----------
#         sample : ndarray
#             Sample dataset, simulation output for example
#         reference : ndarray
#             Reference dataset, what the `sample` dataset is expected to be
#         distance : int
#             Search window limit in the same units as `resolution`
#         threshold : float
#             The maximum passable deviation in `sample` and `reference`
#         resolution : tuple
#             The resolution of each axis of `sample` and `reference`
#         signed : bool
#             Returns signed gamma for identifying hot/cold fails
#
#         Returns
#         -------
#         gamma_map : ndarray
#             g == 0     (pass) the sample and reference pixels are equal
#             0 < g <= 1 (pass) agreement within distance and threshold
#             g > 1      (fail) no agreement
#         """
#     assert groundtruth_dose.GetSize() == predicted_dose.GetSize(), \
#         "`sample` and `reference` must have the same shape"
#
#     resolution = groundtruth_dose.GetSpacing()
#     groundtruth_dose = sitk.GetArrayFromImage(groundtruth_dose)
#     predicted_dose = sitk.GetArrayFromImage(predicted_dose)
#     groundtruth_contours = sitk.GetArrayFromImage(groundtruth_contours)
#
#     # First we need to construct the distance penalty kernel, for this we use
#     # a meshgrid. The trick is creating the appropriate slices.
#
#     # We require one slice per dimension and around the current point
#     # between -distance/resolution and +distance/resolution. We transpose (.T)
#     # so we have a colum vector.
#     resolution = np.array(resolution)[tuple([np.newaxis for i in range(3)])].T
#     slices = [slice(-np.ceil(distance / r), np.ceil(distance / r) + 1) for r in resolution]
#
#     # Scale the meshgide to the resolution of the images, google "numpy.mgrid"
#     kernel = np.mgrid[slices] * resolution
#
#     # Distance squared from the central voxel
#     kernel = np.sum(kernel ** 2, axis=0).astype(np.float)
#
#     # If the distance from the central voxel is greater than the distance
#     # threshold, set to infinity so that it will not be selected as the minimum
#     kernel[np.where(np.sqrt(kernel) > distance)] = np.inf
#
#     # Divide by the square of the distance threshold, this is the cost penalty
#     kernel = kernel / distance ** 2
#
#     # ndimag.generic_filter needs to know the footprint of the
#     # kernel which must be flat
#     footprint = np.ones_like(kernel)
#     kernel = kernel.flatten()
#
#     # Apply the dose threshold penalty (see gamma evaluation equation), here
#     # we are still under the sqrt.
#     values = (groundtruth_dose - predicted_dose) ** 2 / (threshold) ** 2
#
#     # Move the distance penalty kernel over the dose penalised values and search
#     # for the minimum of the sum between the kernel and the values under it. This
#     # is the point of closest agreement.
#     gamma_map = generic_filter(values, \
#                                lambda vals: np.minimum.reduce(vals + kernel), footprint=footprint)
#
#     # Euclidean distance
#     gamma_map = np.sqrt(gamma_map)
#
#     gamma_pass = np.where(gamma_map <= 1, True, False)
#
#     dose_mask = np.where((groundtruth_dose > 0.1) | (predicted_dose > 0.1), True, False)
#     gamma_dose = gamma_pass[dose_mask]
#     gamma_dose = np.count_nonzero(gamma_dose) / np.size(gamma_dose)
#     print(gamma_dose)
#
#     gtv_mask = np.where(groundtruth_contours == 4, True, False)
#     gamma_gtv = gamma_pass[gtv_mask & dose_mask]
#     gamma_gtv = np.count_nonzero(gamma_gtv) / np.size(gamma_gtv)
#     print(gamma_gtv)
#
#     sv_mask = np.where(groundtruth_contours == 3, True, False)
#     gamma_sv = gamma_pass[sv_mask & dose_mask]
#     gamma_sv = np.count_nonzero(gamma_sv) / np.size(gamma_sv)
#     print(gamma_sv)
#
#     rectum_mask = np.where(groundtruth_contours == 2, True, False)
#     gamma_rectum = gamma_pass[rectum_mask & dose_mask]
#     gamma_rectum = np.count_nonzero(gamma_rectum) / np.size(gamma_rectum)
#     print(gamma_rectum)
#
#     bladder_mask = np.where(groundtruth_contours == 1, True, False)
#     gamma_bladder = gamma_pass[bladder_mask & dose_mask]
#     gamma_bladder = np.count_nonzero(gamma_bladder) / np.size(gamma_bladder)
#     print(gamma_bladder)
#
#     return gamma_dose, gamma_gtv, gamma_sv, gamma_rectum, gamma_bladder
#
#
# def gamma_evaluation(sample, reference, distance, threshold, resolution, signed=False):
#     """
#     Distance to Agreement between a sample and reference using gamma evaluation.
#
#     Parameters
#     ----------
#     sample : ndarray
#         Sample dataset, simulation output for example
#     reference : ndarray
#         Reference dataset, what the `sample` dataset is expected to be
#     distance : int
#         Search window limit in the same units as `resolution`
#     threshold : float
#         The maximum passable deviation in `sample` and `reference`
#     resolution : tuple
#         The resolution of each axis of `sample` and `reference`
#     signed : bool
#         Returns signed gamma for identifying hot/cold fails
#
#     Returns
#     -------
#     gamma_map : ndarray
#         g == 0     (pass) the sample and reference pixels are equal
#         0 < g <= 1 (pass) agreement within distance and threshold
#         g > 1      (fail) no agreement
#     """
#
#     ndim = len(resolution)
#     assert sample.ndim == reference.ndim == ndim, \
#         "`sample` and `reference` dimensions must equal `resolution` length"
#     assert sample.shape == reference.shape, \
#         "`sample` and `reference` must have the same shape"
#
#     # First we need to construct the distance penalty kernel, for this we use
#     # a meshgrid. The trick is creating the appropriate slices.
#
#     # We require one slice per dimension and around the current point
#     # between -distance/resolution and +distance/resolution. We transpose (.T)
#     # so we have a colum vector.
#     resolution = numpy.array(resolution)[tuple([numpy.newaxis for i in range(ndim)])].T
#     slices = [slice(-ceil(distance / r), ceil(distance / r) + 1) for r in resolution]
#
#     # Scale the meshgide to the resolution of the images, google "numpy.mgrid"
#     kernel = numpy.mgrid[slices] * resolution
#
#     # Distance squared from the central voxel
#     kernel = numpy.sum(kernel ** 2, axis=0).astype(numpy.float)
#
#     # If the distance from the central voxel is greater than the distance
#     # threshold, set to infinity so that it will not be selected as the minimum
#     kernel[numpy.where(numpy.sqrt(kernel) > distance)] = numpy.inf
#
#     # Divide by the square of the distance threshold, this is the cost penalty
#     kernel = kernel / distance ** 2
#
#     # ndimag.generic_filter needs to know the footprint of the
#     # kernel which must be flat
#     footprint = numpy.ones_like(kernel)
#     kernel = kernel.flatten()
#
#     # Apply the dose threshold penalty (see gamma evaluation equation), here
#     # we are still under the sqrt.
#     values = (reference - sample) ** 2 / (threshold) ** 2
#
#     # Move the distance penalty kernel over the dose penalised values and search
#     # for the minimum of the sum between the kernel and the values under it. This
#     # is the point of closest agreement.
#     gamma_map = generic_filter(values, \
#                                lambda vals: numpy.minimum.reduce(vals + kernel), footprint=footprint)
#
#     # Euclidean distance
#     gamma_map = numpy.sqrt(gamma_map)
#
#     # Rough signed gamma evaluation.
#     if (signed):
#         return gamma_map * numpy.sign(sample - reference)
#     else:
#         return gamma_map
#
# def pass_rate(gamma_map):
#     """
#     Gives the fraction of points that pass the gamma test
#
#     Parameters
#     ----------
#     gamma_map : ndarray
#         Gamma map with gamma indexes for all the points of the image
#
#     Returns
#     -------
#     pass_rate : fraction of points that pass the gamma test
#
#     """
#     gamma_pass = numpy.where(gamma_map <= 1, True, False)
#     pass_rate = numpy.count_nonzero(gamma_pass) / numpy.size(gamma_pass)
#     return pass_rate