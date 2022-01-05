import matplotlib
matplotlib.use('TkAgg')
from pylab import *
from scipy.stats import wilcoxon

def Wilcoxon_test(group1, group2):
    z_statistic, p_value_z = wilcoxon(group1, group2)
    if p_value_z < 0.05:
        return True
    else:
        return False

def sigTest(list, metric, i, significanceComparisonIdx, hasOtherPathBefore, hasOtherPathAfter):
    if (hasOtherPathBefore and i - 1 == significanceComparisonIdx) or (hasOtherPathAfter and i + 1 == significanceComparisonIdx):
        return False
    if len(list[metric][i])+len(list[metric+'_outliers'][i]) == 44:
        if significanceComparisonIdx != i and significanceComparisonIdx != -1:
            # Medpys has only 44 values for HMC
            prunedList = [np.concatenate([list[metric][significanceComparisonIdx], list[metric+'_outliers'][significanceComparisonIdx]])[i] for i in [0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,18,19,20,21,22,23,24,26,27,28,29,30,31,32,34,35,36,37,38,39,41,42,43,44,45,46,47,48]]
            return Wilcoxon_test(prunedList, np.concatenate([list[metric][i], list[metric+'_outliers'][i]]))
    # print("significanceComparisonIdx = " + repr(significanceComparisonIdx))
    return significanceComparisonIdx != i and significanceComparisonIdx != -1 and Wilcoxon_test(np.concatenate([list[metric][significanceComparisonIdx], list[metric+'_outliers'][significanceComparisonIdx]]), np.concatenate([list[metric][i], list[metric+'_outliers'][i]]))

def significanceComparisonGetBestPath(expAndMetric, significanceComparisonIdxPath1, significanceComparisonIdxPath2, higherIsBetter):
    if significanceComparisonIdxPath2 == -1:
        return significanceComparisonIdxPath1
    if np.mean(expAndMetric[significanceComparisonIdxPath1]) > np.mean(expAndMetric[significanceComparisonIdxPath2]):
        if higherIsBetter:
            return significanceComparisonIdxPath1
        else:
            return significanceComparisonIdxPath2
    else:
        if higherIsBetter:
            return significanceComparisonIdxPath2
        else:
            return significanceComparisonIdxPath1

def makeGray(s, gray):
    if gray:
        return "\\textcolor{gray}{" + s + "}"
    return s

def makeBold(s, bold):
    if bold:
        return "\\textbf{" + s + "}"
    return s

def makeTableEntry(mean, std, median, idx, higherIsBetter, meanPrecision, stdPrecision, numOutliers, hasOtherPathBefore, hasOtherPathAfter, makeWorstPathGray, isSignificant):
    medianPrecision = meanPrecision
    #       higherIsBetter and np.argmax(mean) == idx or not higherIsBetter and np.argmin(mean) == idx:
    boldMean = (higherIsBetter and max(mean) == mean[idx] or not higherIsBetter and min(mean) == mean[idx])
    grayMean = makeWorstPathGray and ((hasOtherPathBefore and (higherIsBetter and mean[idx] < mean[idx-1] or not higherIsBetter and mean[idx] > mean[idx-1])) \
                               or (hasOtherPathAfter and (higherIsBetter and mean[idx] < mean[idx+1] or not higherIsBetter and mean[idx] > mean[idx+1])))
    boldMedian = (higherIsBetter and max(median) == median[idx] or not higherIsBetter and min(median) == median[idx])
    grayMedian = makeWorstPathGray and ((hasOtherPathBefore and (higherIsBetter and median[idx] < median[idx-1] or not higherIsBetter and median[idx] > median[idx-1])) \
                               or (hasOtherPathAfter and (higherIsBetter and median[idx] < median[idx+1] or not higherIsBetter and median[idx] > median[idx+1])))
    s = "$"
    #if numOutliers > 0:#TODO: TEMP DISABLED
    #    s += "~~~"
    s += makeBold("%0.*f" % (meanPrecision, mean[idx]), boldMean)
    s += " \\pm "
    s += makeBold("%0.*f" % (stdPrecision, std[idx]), boldMean)
    if isSignificant:
        s += "^{\\dagger}"
    else:
        s += "~"
    #if numOutliers > 0:
        #s += "\\hphantom{}^{(" + repr(numOutliers) + ")}" #TODO: TEMP DISABLED
    s += "$"
    m = "& " + makeGray(makeBold("%0.*f" % (medianPrecision, median[idx]), boldMedian), grayMedian)
    return "& " + makeGray(s, grayMean) + " " + m + " "

def printTable(xlsxLoader, metric, higherIsBetter, makeWorstPathGray, label, significanceComparisonName, meanPrecision=1, stdPrecision=1):
    medianPrecision = meanPrecision

    mean_gtv = []
    std_gtv = []
    median_gtv = []
    mean_sv = []
    std_sv = []
    median_sv = []
    mean_rectum = []
    std_rectum = []
    median_rectum = []
    mean_bladder = []
    std_bladder = []
    median_bladder = []

    n = len(xlsxLoader.infos)

    significanceComparisonIdxPath1 = -1
    significanceComparisonIdxPath2 = -1
    for i in range(n):
        if xlsxLoader.infos[i].tag == significanceComparisonName:
            if significanceComparisonIdxPath1 == -1:
                significanceComparisonIdxPath1 = i
            elif significanceComparisonIdxPath2 == -1:
                significanceComparisonIdxPath2 = i
            else:
                assert False

    significanceComparisonIdx_gtv = significanceComparisonGetBestPath(xlsxLoader.exp_gtv[metric], significanceComparisonIdxPath1, significanceComparisonIdxPath2, higherIsBetter)
    significanceComparisonIdx_sv = significanceComparisonGetBestPath(xlsxLoader.exp_sv[metric], significanceComparisonIdxPath1, significanceComparisonIdxPath2, higherIsBetter)
    significanceComparisonIdx_rectum = significanceComparisonGetBestPath(xlsxLoader.exp_rectum[metric], significanceComparisonIdxPath1, significanceComparisonIdxPath2, higherIsBetter)
    significanceComparisonIdx_bladder = significanceComparisonGetBestPath(xlsxLoader.exp_bladder[metric], significanceComparisonIdxPath1, significanceComparisonIdxPath2, higherIsBetter)

    for i in range(n):
        mean_gtv.append(round(np.mean(xlsxLoader.exp_gtv[metric][i]), meanPrecision))
        std_gtv.append(round(np.std(xlsxLoader.exp_gtv[metric][i], ddof=1), stdPrecision))
        median_gtv.append(round(np.median(xlsxLoader.exp_gtv[metric][i]), medianPrecision))

        mean_sv.append(round(np.mean(xlsxLoader.exp_sv[metric][i]), meanPrecision))
        std_sv.append(round(np.std(xlsxLoader.exp_sv[metric][i], ddof=1), stdPrecision))
        median_sv.append(round(np.median(xlsxLoader.exp_sv[metric][i]), medianPrecision))

        mean_rectum.append(round(np.mean(xlsxLoader.exp_rectum[metric][i]), meanPrecision))
        std_rectum.append(round(np.std(xlsxLoader.exp_rectum[metric][i], ddof=1), stdPrecision))
        median_rectum.append(round(np.median(xlsxLoader.exp_rectum[metric][i]), medianPrecision))

        mean_bladder.append(round(np.mean(xlsxLoader.exp_bladder[metric][i]), meanPrecision))
        std_bladder.append(round(np.std(xlsxLoader.exp_bladder[metric][i], ddof=1), stdPrecision))
        median_bladder.append(round(np.median(xlsxLoader.exp_bladder[metric][i]), medianPrecision))

    # if metric is not None:
    s = "\n\\begin{table*}[!htb]" +\
        "\n	\\centering" +\
        "\n	\\setlength{\\tabcolsep}{3pt}" +\
        "\n	\\caption[Table caption text]{"
    if metric == 'DSC':
        s += label + ' DSC values for the different approaches. Higher values are better.}'
    elif metric == 'MSD':
        s += label + ' MSD (mm) values for the different approaches. Lower values are better.}'
    elif metric == 'HD':
        s += label + ' \\%95HD (mm) values for the different approaches. Lower values are better.}'
    s += "\n	\\resizebox{\\textwidth}{!}{" +\
         "\n		\\begin{tabular}{lrcccccccc} " +\
         "\n			&&\multicolumn{2}{c}{Prostate}&\multicolumn{2}{c}{Seminal vesicles}&\multicolumn{2}{c}{Rectum}& \multicolumn{2}{c}{Bladder} \\\\ \\hline" +\
         "\n			& Output Path & $\\mu \\pm \\sigma$ & Median & $\\mu \\pm \\sigma$ & Median & $\\mu \\pm \\sigma$ & Median & $\\mu \\pm \\sigma$ & Median \\\\ \\hline"
    print(s)

    for i in range(n):
        hasOtherPathBefore = (i > 0 and xlsxLoader.infos[i-1].isSimilar(xlsxLoader.infos[i]))
        hasOtherPathAfter = (i < n-1 and xlsxLoader.infos[i].isSimilar(xlsxLoader.infos[i+1]))
        shouldMakeWorstPathGray = makeWorstPathGray and "Separate" not in xlsxLoader.infos[i].title
        s = ""
        if not hasOtherPathBefore and not hasOtherPathAfter:
            s += "\\multicolumn{2}{l}{ "
        if not hasOtherPathBefore:  # No title for reg if title of this method was already printed for seg.
            s += xlsxLoader.infos[i].title
        if not hasOtherPathBefore and not hasOtherPathAfter:
            s += " }"
        else:
            s += " & "
        if hasOtherPathBefore or hasOtherPathAfter:
            if xlsxLoader.infos[i].isReg:
                s += "\\textit{Registration} "
            else:
                s += "\\textit{Segmentation} "
        s += makeTableEntry(mean_gtv, std_gtv, median_gtv, i, higherIsBetter, meanPrecision, stdPrecision, len(xlsxLoader.exp_gtv[metric+'_outliers'][i]), hasOtherPathBefore, hasOtherPathAfter, shouldMakeWorstPathGray, sigTest(xlsxLoader.exp_gtv, metric, i, significanceComparisonIdx_gtv, hasOtherPathBefore, hasOtherPathAfter))
        s += makeTableEntry(mean_sv, std_sv, median_sv, i, higherIsBetter, meanPrecision, stdPrecision, len(xlsxLoader.exp_sv[metric+'_outliers'][i]), hasOtherPathBefore, hasOtherPathAfter, shouldMakeWorstPathGray,  sigTest(xlsxLoader.exp_sv, metric, i, significanceComparisonIdx_sv, hasOtherPathBefore, hasOtherPathAfter))
        s += makeTableEntry(mean_rectum, std_rectum, median_rectum, i, higherIsBetter, meanPrecision, stdPrecision, len(xlsxLoader.exp_rectum[metric+'_outliers'][i]), hasOtherPathBefore, hasOtherPathAfter, shouldMakeWorstPathGray, sigTest(xlsxLoader.exp_rectum, metric, i, significanceComparisonIdx_rectum, hasOtherPathBefore, hasOtherPathAfter))
        s += makeTableEntry(mean_bladder, std_bladder, median_bladder, i, higherIsBetter, meanPrecision, stdPrecision, len(xlsxLoader.exp_bladder[metric+'_outliers'][i]), hasOtherPathBefore, hasOtherPathAfter, shouldMakeWorstPathGray, sigTest(xlsxLoader.exp_bladder, metric, i, significanceComparisonIdx_bladder, hasOtherPathBefore, hasOtherPathAfter))
        s += "\\\\"
        if not hasOtherPathAfter:  # No hline between seg and reg of the same method
            s += " \\hline"
        print(s)

    # if metric is not None:
    print("		\\end{tabular}" +\
          "\n	}" +\
          "\n	\\label{table:" + label + "_" + metric + "}" +\
          "\n\\end{table*}")

def printTables(xlsxLoader, tablesAreDebug, printMeanAllFourOrgansDebug, makeWorstPathGray, label, significanceComparisonName):
    if tablesAreDebug:
        print("\n%---------------- DSC ----------------%")
        printDebugTable(xlsxLoader, 'DSC', True, printMeanAllFourOrgansDebug, label, meanPrecision=2, stdPrecision=2)
        print("\n%---------------- MSD ----------------%")
        printDebugTable(xlsxLoader, 'MSD', False, printMeanAllFourOrgansDebug, label, meanPrecision=2)
        print("\n%---------------- HD ----------------%")
        printDebugTable(xlsxLoader, 'HD', False, printMeanAllFourOrgansDebug, label)
        print("")  # Newline
    else:
        print("\n%---------------- DSC ----------------%")
        printTable(xlsxLoader, 'DSC', True, makeWorstPathGray, label, significanceComparisonName, meanPrecision=2, stdPrecision=2)
        print("\n%---------------- MSD ----------------%")
        printTable(xlsxLoader, 'MSD', False, makeWorstPathGray, label, significanceComparisonName, meanPrecision=2)
        print("\n%---------------- HD ----------------%")
        printTable(xlsxLoader, 'HD', False, makeWorstPathGray, label, significanceComparisonName)
        print("")  # Newline

def printDebugTable(xlsxLoader, metric, higherIsBetter, printMeanAllFourOrgansDebug, label, meanPrecision=1, stdPrecision=1):
    mean_gtv = []
    std_gtv = []
    mean_sv = []
    std_sv = []
    mean_rectum = []
    std_rectum = []
    mean_bladder = []
    std_bladder = []

    n = len(xlsxLoader.infos)
    for i in range(n):
        mean_gtv.append(round(np.mean(xlsxLoader.exp_gtv[metric][i]), meanPrecision))
        std_gtv.append(round(np.std(xlsxLoader.exp_gtv[metric][i], ddof=1), stdPrecision))

        mean_sv.append(round(np.mean(xlsxLoader.exp_sv[metric][i]), meanPrecision))
        std_sv.append(round(np.std(xlsxLoader.exp_sv[metric][i], ddof=1), stdPrecision))

        mean_rectum.append(round(np.mean(xlsxLoader.exp_rectum[metric][i]), meanPrecision))
        std_rectum.append(round(np.std(xlsxLoader.exp_rectum[metric][i], ddof=1), stdPrecision))

        mean_bladder.append(round(np.mean(xlsxLoader.exp_bladder[metric][i]), meanPrecision))
        std_bladder.append(round(np.std(xlsxLoader.exp_bladder[metric][i], ddof=1), stdPrecision))


    maxNameSpace = 35 #includes "Segmentation" and "Registration"
    s = "--" + label
    while (len(s) < maxNameSpace):
        s += "-"
    s += "| Prostate |    SV    |  Rectum  |  Bladder |"
    print(s)

    for i in range(n):
        if not (i > 0 and xlsxLoader.infos[i-1].isSimilar(xlsxLoader.infos[i])): # No title for reg if title of this method was already printed for seg.
            s = xlsxLoader.infos[i].title
        else:
            s = ""
        while len(s) < maxNameSpace - 14:
            s += " "
        if xlsxLoader.infos[i].isReg:
            s += " Registration "
        else:
            s += " Segmentation "
        s += "| %0.*f" % (meanPrecision, mean_gtv[i])
        if len(xlsxLoader.exp_gtv[metric+'_outliers'][i]) > 0:
            s += " (" + repr(len(xlsxLoader.exp_gtv[metric+'_outliers'][i])) + ")"
        if higherIsBetter and max(mean_gtv) == mean_gtv[i] or not higherIsBetter and min(mean_gtv) == mean_gtv[i]:
            s += " !!!"
        while len(s) < maxNameSpace + 11:
            s += " "
        s += "| %0.*f" % (meanPrecision, mean_sv[i])
        if len(xlsxLoader.exp_sv[metric+'_outliers'][i]) > 0:
            s += " (" + repr(len(xlsxLoader.exp_sv[metric+'_outliers'][i])) + ")"
        if higherIsBetter and max(mean_sv) == mean_sv[i] or not higherIsBetter and min(mean_sv) == mean_sv[i]:
            s += " !!!"
        while len(s) < maxNameSpace + 22:
            s += " "
        s += "| %0.*f" % (meanPrecision, mean_rectum[i])
        if len(xlsxLoader.exp_rectum[metric+'_outliers'][i]) > 0:
            s += " (" + repr(len(xlsxLoader.exp_rectum[metric+'_outliers'][i])) + ")"
        if higherIsBetter and max(mean_rectum) == mean_rectum[i] or not higherIsBetter and min(mean_rectum) == mean_rectum[i]:
            s += " !!!"
        while len(s) < maxNameSpace + 33:
            s += " "
        s += "| %0.*f" % (meanPrecision, mean_bladder[i])
        if len(xlsxLoader.exp_bladder[metric+'_outliers'][i]) > 0:
            s += " (" + repr(len(xlsxLoader.exp_bladder[metric+'_outliers'][i])) + ")"
        if higherIsBetter and max(mean_bladder) == mean_bladder[i] or not higherIsBetter and min(mean_bladder) == mean_bladder[i]:
            s += " !!!"
        while len(s) < maxNameSpace + 44:
            s += " "
        s += "|"
        if printMeanAllFourOrgansDebug:
            mean_fourOrgans = mean_gtv[i] * 0.25 + mean_sv[i] * 0.25 + mean_rectum[i] * 0.25 + mean_bladder[i] * 0.25
            s += " (%0.*f)" % (meanPrecision, mean_fourOrgans)
            if i > 0 and xlsxLoader.infos[i - 1].isSimilar(xlsxLoader.infos[i]): # best over seg and reg
                if higherIsBetter:
                    mean_bestFourOrgans = max(mean_gtv[i] * 0.25, mean_gtv[i-1] * 0.25) + max(mean_sv[i] * 0.25, mean_sv[i-1] * 0.25)\
                                          + max(mean_rectum[i] * 0.25, mean_rectum[i-1] * 0.25) + max(mean_bladder[i] * 0.25, mean_bladder[i-1] * 0.25)
                else:
                    mean_bestFourOrgans = min(mean_gtv[i] * 0.25, mean_gtv[i-1] * 0.25) + min(mean_sv[i] * 0.25, mean_sv[i-1] * 0.25)\
                                          + min(mean_rectum[i] * 0.25, mean_rectum[i-1] * 0.25) + min(mean_bladder[i] * 0.25, mean_bladder[i-1] * 0.25)
                s += " (%0.*f)" % (meanPrecision, mean_bestFourOrgans)
        print(s)
        if not (i < n-1 and xlsxLoader.infos[i].isSimilar(xlsxLoader.infos[i+1])): # No hline between reg and seg of the same method
            s = ""
            while len(s) < maxNameSpace + 45:
                s += "-"
            print(s)
