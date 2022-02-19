import os

import SimpleITK as sitk
import pandas as pd
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner

from utils.dataset_niftynet import set_dataParam
from utils.dose_quantification import gamma_pass_rate


class evaluation_dose(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.partitioner = ImageSetsPartitioner().initialise(data_param=set_dataParam(self.args, self.config),
                                                             data_split_file=self.config.csv_split_file,
                                                             new_partition=False)
        self.run()

    def run(self):

        for partition in self.args.split_set:
            if partition == 'validation':
                dataset = 'HMC'
            elif partition == 'inference':
                dataset = 'EMC'

            files_list = self.partitioner.get_file_list(partition, 'fixed_dose')['fixed_dose'].values.tolist()

            writer = pd.ExcelWriter(os.path.join(self.args.output_dir, dataset, "Evaluation-Dose.xlsx"),
                                    engine='xlsxwriter')

            if os.path.exists(os.path.join(self.args.output_dir, dataset, "Evaluation-Dose.xlsx")):
                print("There is already an excel with this name")

            else:

                # refresh the values
                self.gamma_2_2 = []
                self.gamma_2_2_gtv = []
                self.gamma_2_2_sv = []
                self.gamma_2_2_rectum = []
                self.gamma_2_2_bladder = []

                self.V_95_gtv = []
                self.V_110_gtv = []

                self.V_95_sv = []
                self.V_110_sv = []

                self.D_mean_rectum = []
                self.D_2_rectum = []

                self.D_mean_bladder = []
                self.D_2_bladder= []

                self.patient_arr = []
                self.scan_arr = []

                for file in files_list:
                    patient_name = file.split('/')[-3]
                    visit_name = file.split('/')[-2]

                    groundtruth_dose = sitk.ReadImage(file)
                    groundtruth_contours = sitk.ReadImage(os.path.join(os.path.split(file)[0], 'Segmentation.mha'))
                    predicted_dose = sitk.ReadImage(os.path.join(self.args.output_dir, dataset, patient_name, visit_name, 'Dose.mha'))

                    # groundtruth_bladder = sitk.BinaryThreshold(groundtruth_segmentation, lowerThreshold=1,
                    #                                            upperThreshold=1, insideValue=1, outsideValue=0)
                    # groundtruth_rectum = sitk.BinaryThreshold(groundtruth_segmentation, lowerThreshold=2,
                    #                                           upperThreshold=2, insideValue=1, outsideValue=0)
                    # groundtruth_sv = sitk.BinaryThreshold(groundtruth_segmentation, lowerThreshold=3,
                    #                                       upperThreshold=3, insideValue=1, outsideValue=0)
                    # groundtruth_prostate = sitk.BinaryThreshold(groundtruth_segmentation, lowerThreshold=4,
                    #                                             upperThreshold=4, insideValue=1, outsideValue=0)


                    gamma_2_2, gamma_2_2_gtv, gamma_2_2_sv, gamma_2_2_rectum, gamma_2_2_bladder = gamma_pass_rate(groundtruth_dose, predicted_dose, groundtruth_contours)
                    # gamma_2_2 = gamma_pass_rate(groundtruth_dose, predicted_dose, groundtruth_contours, distance=2, threshold=2)

                    # dsc_bladder, msd_bladder, hd_bladder, _, _ = DSC_MSD_HD95_Seg(groundtruth_bladder,
                    #                                                               predicted_bladder,
                    #                                                               self.args.num_components[1],
                    #                                                               resample_spacing=self.args.voxel_dim)
                    # dsc_rectum, msd_rectum, hd_rectum, _, _ = DSC_MSD_HD95_Seg(groundtruth_rectum,
                    #                                                            predicted_rectum,
                    #                                                            self.args.num_components[1],
                    #                                                            resample_spacing=self.args.voxel_dim)
                    # dsc_sv, msd_sv, hd_sv, _, _ = DSC_MSD_HD95_Seg(groundtruth_sv, predicted_sv,
                    #                                                self.args.num_components[0],
                    #                                                resample_spacing=self.args.voxel_dim)
                    # dsc_prostate, msd_prostate, hd_prostate, _, _ = DSC_MSD_HD95_Seg(groundtruth_prostate,
                    #                                                                  predicted_prostate,
                    #                                                                  self.args.num_components[
                    #                                                                      1],
                    #                                                                  resample_spacing=self.args.voxel_dim)

                    self.patient_arr.append(patient_name)
                    self.scan_arr.append(visit_name)

                    self.gamma_2_2.append(gamma_2_2)
                    self.gamma_2_2_gtv.append(gamma_2_2_gtv)
                    self.gamma_2_2_sv.append(gamma_2_2_sv)
                    self.gamma_2_2_rectum.append(gamma_2_2_rectum)
                    self.gamma_2_2_bladder.append(gamma_2_2_bladder)
                    #
                    # self.V_95_gtv.append()
                    # self.V_110_gtv.append()
                    # self.V_95_sv.append()
                    # self.V_110_sv.append()
                    #
                    # self.D_mean_rectum.append()
                    # self.D_2_rectum.append()
                    # self.D_mean_bladder.append()
                    # self.D_2_bladder.append()


                    print(dataset, patient_name, visit_name)


                data = {'Patient': self.patient_arr, 'Scan': self.scan_arr, 'gamma_2_2': self.gamma_2_2, 'gamma_2_2_gtv': self.gamma_2_2_gtv,
                        'gamma_2_2_sv': self.gamma_2_2_sv, 'gamma_2_2_rectum': self.gamma_2_2_rectum, 'gamma_2_2_bladder': self.gamma_2_2_bladder}

                df = pd.DataFrame(data, dtype=float)

                df = df.reindex(['Patient', 'Scan', 'gamma_2_2', 'gamma_2_2_gtv', 'gamma_2_2_sv', 'gamma_2_2_rectum', 'gamma_2_2_bladder'], axis=1)

                df.loc['Median'] = df.median()
                df.loc['Min'] = df.min()
                df.loc['Max'] = df.max()
                df.loc['Q75'] = df.quantile(.75)
                df.loc['Q25'] = df.quantile(.25)
                df.loc['IQR'] = df.loc['Q75'] - df.loc['Q25']
                df.loc['LowerOutlierLimit'] = df.loc['Q25'] - (df.loc['IQR'] * 1.5)
                df.loc['UpperOutlierLimit'] = df.loc['Q75'] + (df.loc['IQR'] * 1.5)
                df.loc['Mean'] = df.mean()
                df.loc['Std'] = df.std()

                df.to_excel(writer, sheet_name='eval')
                writer.save()