import os

import SimpleITK as sitk
import pandas as pd
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner

from utils.dataset_niftynet import set_dataParam
from utils.dose_quantification import gamma_pass, Vx_target, Dx_oar


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

                self.gamma_1_1 = []
                self.gamma_1_1_gtv = []
                self.gamma_1_1_sv = []
                self.gamma_1_1_rectum = []
                self.gamma_1_1_bladder = []

                self.V95_gtv = []
                self.V110_gtv = []
                self.V95_sv = []
                self.V110_sv = []

                self.Dmean_rectum = []
                self.D2_rectum = []
                self.Dmean_bladder = []
                self.D2_bladder= []

                self.patient_arr = []
                self.scan_arr = []

                for file in files_list:
                    patient_name = file.split('/')[-3]
                    visit_name = file.split('/')[-2]

                    groundtruth_dose = sitk.ReadImage(file)
                    groundtruth_contours = sitk.ReadImage(os.path.join(os.path.split(file)[0], 'Segmentation.mha'))
                    predicted_dose = sitk.ReadImage(os.path.join(self.args.output_dir, dataset, patient_name,
                                                                 visit_name, 'Dose.mha'))

                    gamma_2_2, gamma_2_2_gtv, gamma_2_2_sv, gamma_2_2_rectum, gamma_2_2_bladder = gamma_pass(
                        groundtruth_dose, predicted_dose, groundtruth_contours, distance=2, threshold=2)

                    gamma_1_1, gamma_1_1_gtv, gamma_1_1_sv, gamma_1_1_rectum, gamma_1_1_bladder = gamma_pass(
                        groundtruth_dose, predicted_dose, groundtruth_contours, distance=1, threshold=1)

                    V95_gtv, V110_gtv, V95_sv, V110_sv = Vx_target(groundtruth_dose, predicted_dose,
                                                                   groundtruth_contours)

                    Dmean_rectum, D2_rectum, Dmean_bladder, D2_bladder = Dx_oar(groundtruth_dose, predicted_dose,
                                                                   groundtruth_contours)
                    self.patient_arr.append(patient_name)
                    self.scan_arr.append(visit_name)

                    self.gamma_2_2.append(gamma_2_2)
                    self.gamma_2_2_gtv.append(gamma_2_2_gtv)
                    self.gamma_2_2_sv.append(gamma_2_2_sv)
                    self.gamma_2_2_rectum.append(gamma_2_2_rectum)
                    self.gamma_2_2_bladder.append(gamma_2_2_bladder)

                    self.gamma_1_1.append(gamma_1_1)
                    self.gamma_1_1_gtv.append(gamma_1_1_gtv)
                    self.gamma_1_1_sv.append(gamma_1_1_sv)
                    self.gamma_1_1_rectum.append(gamma_1_1_rectum)
                    self.gamma_1_1_bladder.append(gamma_1_1_bladder)

                    self.V95_gtv.append(V95_gtv)
                    self.V110_gtv.append(V110_gtv)
                    self.V95_sv.append(V95_sv)
                    self.V110_sv.append(V110_sv)

                    self.Dmean_rectum.append(Dmean_rectum)
                    self.D2_rectum.append(D2_rectum)
                    self.Dmean_bladder.append(Dmean_bladder)
                    self.D2_bladder.append(D2_bladder)

                    print(dataset, patient_name, visit_name)


                data = {'Patient': self.patient_arr, 'Scan': self.scan_arr, 'gamma_2_2': self.gamma_2_2, 'gamma_2_2_gtv': self.gamma_2_2_gtv,
                        'gamma_2_2_sv': self.gamma_2_2_sv, 'gamma_2_2_rectum': self.gamma_2_2_rectum, 'gamma_2_2_bladder': self.gamma_2_2_bladder,
                        'gamma_1_1': self.gamma_1_1, 'gamma_1_1_gtv': self.gamma_1_1_gtv, 'gamma_1_1_sv': self.gamma_1_1_sv,
                        'gamma_1_1_rectum': self.gamma_1_1_rectum, 'gamma_1_1_bladder': self.gamma_1_1_bladder, 'V95_gtv': self.V95_gtv,
                        'V110_gtv': self.V95_sv, 'V95_sv': self.V110_gtv, 'V110_sv': self.V110_sv, 'Dmean_rectum': self.Dmean_rectum,
                        'D2_rectum':self.D2_rectum, 'Dmean_bladder':self.Dmean_bladder, 'D2_bladder':self.D2_bladder}

                df = pd.DataFrame(data, dtype=float)

                df = df.reindex(['Patient', 'Scan', 'gamma_2_2', 'gamma_2_2_gtv', 'gamma_2_2_sv', 'gamma_2_2_rectum',
                                 'gamma_2_2_bladder', 'gamma_1_1', 'gamma_1_1_gtv', 'gamma_1_1_sv', 'gamma_1_1_rectum',
                                 'gamma_1_1_bladder', 'V95_gtv', 'V110_gtv', 'V95_sv', 'V110_sv', 'Dmean_rectum',
                                 'D2_rectum', 'Dmean_bladder', 'D2_bladder'], axis=1)

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