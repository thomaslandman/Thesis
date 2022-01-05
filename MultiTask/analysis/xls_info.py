import matplotlib
matplotlib.use('TkAgg')
from pylab import *
import pandas as pd
import os
from generate_tables import printTables
from generate_boxplots import GenerateBoxplots
from boxplot_temp import GenerateBoxplots_MSD

class ExpInfo():
    def __init__(self, tag, title):
        self.tag = tag
        self.title = title

class XlsxInfo():
    def __init__(self, tag, title, fileName, path, isReg, rawData, expIdx):
        self.tag = tag
        self.title = title
        self.fileName = fileName
        self.path = path
        self.isReg = isReg
        self.rawData = rawData
        self.expIdx = expIdx

    def __str__(self):
        return "\n" + repr(self.tag) + " " + repr(self.title) + " " + repr(self.fileName) + " " + repr(self.path) + " " + repr(self.isReg)

    def __repr__(self):
        return self.__str__()

    def __gt__(self, other):
        # The second half of the condition orders the segmentation file before the registration file.
        return self.expIdx > other.expIdx or (self.tag == other.tag and self.isReg)

    def isSimilar(self, otherXlsxInfo):
        return self.tag == otherXlsxInfo.tag or self.title == otherXlsxInfo.title

class XlsxLoader():
    def __init__(self):
        self.infos = []
        self.exp_bladder = {'DSC': [], 'MSD': [], 'HD': [], 'DSC_outliers': [], 'MSD_outliers': [], 'HD_outliers': []}
        self.exp_rectum = {'DSC': [], 'MSD': [], 'HD': [], 'DSC_outliers': [], 'MSD_outliers': [], 'HD_outliers': []}
        self.exp_sv = {'DSC': [], 'MSD': [], 'HD': [], 'DSC_outliers': [], 'MSD_outliers': [], 'HD_outliers': []}
        self.exp_gtv = {'DSC': [], 'MSD': [], 'HD': [], 'DSC_outliers': [], 'MSD_outliers': [], 'HD_outliers': []}

    def parseColumn(self, exp, metric, col, rawData, numPatients):
        exp[metric].append(np.array(rawData.iloc[:, col].values[:numPatients]))

    # Removes and counts situations in which an organ (likely SV) was not predicted.
    def removeOutliers(self, exp, shouldRemoveOutliers):
        exp['DSC' + '_outliers'].append([])
        exp['MSD' + '_outliers'].append([])
        exp['HD' + '_outliers'].append([])
        if not shouldRemoveOutliers:
            return
        indices = []
        for i in range(len(exp['MSD'][-1])):
            if exp['MSD'][-1][i] >= 100 or exp['HD'][-1][i] >= 100:
                # print(">=100! " + repr(exp['MSD'][-1][i]) + "," + repr(exp['HD'][-1][i]))
                exp['DSC_outliers'][-1].append(exp['DSC'][-1][i])
                exp['MSD_outliers'][-1].append(exp['MSD'][-1][i])
                exp['HD_outliers'][-1].append(exp['HD'][-1][i])
                indices.append(i)
        exp['DSC'][-1] = np.delete(exp['DSC'][len(exp['DSC'])-1], indices)
        exp['MSD'][-1] = np.delete(exp['MSD'][len(exp['MSD'])-1], indices)
        exp['HD'][-1] = np.delete(exp['HD'][len(exp['HD'])-1], indices)

    def parseData(self, shouldRemoveOutliers):
        for info in self.infos:
            if 'HMC' in info.path:
                numPatients = 50
            elif 'EMC' in info.path:
                numPatients = 42
            rawData = info.rawData
            arr = np.array(rawData.iloc[:, 1].values[:numPatients])
            for i in range(arr.shape[0]):
                if not str(arr[i]).startswith("Patient"):
                    numPatients = i
                    break
            self.parseColumn(self.exp_bladder, 'DSC', 3, rawData, numPatients)
            self.parseColumn(self.exp_bladder, 'MSD', 4, rawData, numPatients)
            self.parseColumn(self.exp_bladder, 'HD', 5, rawData, numPatients)
            self.removeOutliers(self.exp_bladder, shouldRemoveOutliers)

            self.parseColumn(self.exp_rectum, 'DSC', 6, rawData, numPatients)
            self.parseColumn(self.exp_rectum, 'MSD', 7, rawData, numPatients)
            self.parseColumn(self.exp_rectum, 'HD', 8, rawData, numPatients)
            self.removeOutliers(self.exp_rectum, shouldRemoveOutliers)

            self.parseColumn(self.exp_sv, 'DSC', 9, rawData, numPatients)
            self.parseColumn(self.exp_sv, 'MSD', 10, rawData, numPatients)
            self.parseColumn(self.exp_sv, 'HD', 11, rawData, numPatients)
            self.removeOutliers(self.exp_sv, shouldRemoveOutliers)

            self.parseColumn(self.exp_gtv, 'DSC', 12, rawData, numPatients)
            self.parseColumn(self.exp_gtv, 'MSD', 13, rawData, numPatients)
            self.parseColumn(self.exp_gtv, 'HD', 14, rawData, numPatients)
            self.removeOutliers(self.exp_gtv, shouldRemoveOutliers)

    def load(self, xlsx_files_folder, exp_infos, printTags=False, shouldRemoveOutliers=True):
        if printTags:
            print("All Tags: { ")
        for file in os.listdir(xlsx_files_folder):
            if not file.endswith(".xlsx"):
                continue
            if file.startswith('.'):
                continue
            if file.startswith('~'):
                continue

            path = os.path.join(xlsx_files_folder, file)

            tag = file
            tag = re.sub('\d ', '', tag) # Remove numbers with a space after it
            tag = tag.replace("Evaluation - ", "")
            tag = tag.replace("Evaluation-", "")
            tag = tag.replace("Reg ", "")
            tag = tag.replace("reg ", "")
            tag = tag.replace("Seg ", "")
            tag = tag.replace("seg ", "")
            tag = tag.replace("joint_unet_", "")
            tag = tag.replace(" (merged)", "")
            tag = tag.replace("_SV", "")
            tag = tag.replace(".xlsx", "")
            tag = tag.replace("Reg-", "")
            tag = tag.replace("reg-", "")
            tag = tag.replace("Seg-", "")
            tag = tag.replace("seg-", "")
            if printTags and ("Evaluation-Seg" not in file or "separate" in file):  # To print tags only once instead of twice.
                print(tag)

            expIdx = -1
            for i in range(len(exp_infos)):
                if exp_infos[i].tag == tag:
                    expIdx = i
                    break
            if expIdx == -1:
                continue

            rawData = pd.read_excel(path, engine='openpyxl')
            if "Evaluation-Reg" in file:
                isReg = True
            else:
                isReg = False

            temp = np.mean(rawData.iloc[:, 3])
            if temp <= 0.01:
                print("Warning! Zero value detected: " + tag + ". This file might be empty!")
            else:
                self.infos.append(XlsxInfo(tag=tag, title=exp_infos[expIdx].title, fileName=file, path=path, isReg=isReg, rawData=rawData, expIdx=expIdx))
        if printTags:
            print("} (All Tags)\n")

        self.infos.sort()
        self.parseData(shouldRemoveOutliers)

experiments = {}

exp_infos = [

    ExpInfo('Single-Task-Seg_input_If', 'SEGa'),
    ExpInfo('Single-Task-Seg_input_If_Sm', 'SEGb'),
    ExpInfo('Single-Task-Seg_input_If_Im_Sm', 'SEGc'),
    ExpInfo('Single-Task-Reg_input_If_Im', 'REGa'),
    ExpInfo('Single-Task-Reg_input_If_Im_Sm', 'REGb'),
]
experiments["./all_xlsx_files/HMC"] = exp_infos

# exp_infos = [
#
#     ExpInfo('Single-Task-Seg_input_If', 'SEGa'),
# ]
# experiments["./all_xlsx_files/HMC"] = exp_infos

only_experiment = None  # Fill in experiment name, or make this NONE to do ALL experiments.
print_tags = True  # True: Print all experiment tags at the start.
tablesAreDebug = False  # True: Readable tables. False: LaTeX tables.
shouldRemoveOutliers = True  # True: Removes and counts cases in which organs (usually SV) are not predicted.
printMeanAllFourOrgansDebug = True  # True: If debug, show also sum of mean over all four organs * 0.25.
generateBoxplots = True  # True: Generate boxplots, but could be slow.
makeWorstPathGray = True  # True: If not debug, worst path between seg and reg will have gray textcolor per organ.
significanceComparisonName = ''  # Do not forget to change this between experiments!

for out_file, exp_infos in experiments.items():
    if only_experiment is not None and only_experiment not in out_file:
        continue
    print("-----------------------------------------------------------------------------------------------------------")
    print("Experiment: " + out_file)
    xlsxLoader = XlsxLoader()
    xlsxLoader.load(out_file, exp_infos, print_tags, shouldRemoveOutliers)
    print_tags = False
    printTables(xlsxLoader, tablesAreDebug, printMeanAllFourOrgansDebug, makeWorstPathGray, out_file.split('/')[-1], significanceComparisonName)
    if generateBoxplots:
        GenerateBoxplots(xlsxLoader, out_file)
        GenerateBoxplots_MSD(xlsxLoader, out_file)
