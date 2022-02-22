import numpy as np
import SimpleITK as sitk
from scipy.ndimage.filters import generic_filter
from pymedphys import gamma

def gamma_pass(groundtruth_dose, predicted_dose, groundtruth_contours, distance=2, threshold=2):

    assert groundtruth_dose.GetSize() == predicted_dose.GetSize(), \
                 "`sample` and `reference` must have the same shape"

    dims = groundtruth_dose.GetSize()
    origin = groundtruth_dose.GetOrigin()
    spacing = groundtruth_dose.GetSpacing()
    print(dims)
    print(origin)
    print(spacing)
    x = np.linspace(origin[0], origin[0] + spacing[0] * (dims[0] - 1), dims[0])
    y = np.linspace(origin[1], origin[0] + spacing[1] * (dims[1] - 1), dims[1])
    z = np.linspace(origin[2], origin[2] + spacing[2] * (dims[2] - 1), dims[2])
    axes = (z, y, x)

gamma_pass()



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