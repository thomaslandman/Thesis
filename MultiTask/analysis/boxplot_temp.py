import matplotlib
matplotlib.use('TkAgg')
from pylab import *
import os

class BoxplotSettings():
    def __init__(self, positions, splitPositions, pairPositions, margin, n, flierprops, colors, labels):
        self.positions = positions
        self.splitPositions = splitPositions
        self.pairPositions = pairPositions
        self.margin = margin
        self.n = n
        self.flierprops = flierprops
        self.colors = colors
        self.labels = labels

def setBoxColors(bp, colors):
    for patch, color in zip(bp['boxes'], colors):
        patch.set(facecolor=color, alpha=0.65)

def doPlot(xlsxLoader, settings, ax, exp, yticks, ylimLow, ylimHigh, boxplotLoc='upper left',):
    p1 = ax.boxplot(exp, positions=settings.positions, widths=(0.85-settings.margin)/settings.n, patch_artist=True,
                    flierprops=settings.flierprops)
    setBoxColors(p1, settings.colors)
    plt.xticks(settings.pairPositions[0:-1], settings.labels, rotation=0)
    plt.tick_params(axis='x', direction='in')
    plt.xlim(0, 1)
    plt.sca(ax)
    plt.yticks(yticks)
    plt.ylim(ylimLow, ylimHigh)
    plt.tick_params(axis='y', direction='in')
    ax.grid(axis='y', color="0.9", linestyle='--', linewidth=1)
    ax.xaxis.set_label_position('top')
    ax.set_axisbelow(True)
    for i in range (1,len(settings.splitPositions)):
        if i % 2 == 0:
            ax.axvspan(settings.splitPositions[i-1], settings.splitPositions[i], facecolor='grey', alpha=0.25)

    # Pick one seg box and one reg box for the legend.
    segBox = p1["boxes"][0]
    regBox = p1["boxes"][0]
    for i in range(len(settings.splitPositions)-1):
        if xlsxLoader.infos[i].isReg:
            regBox = p1["boxes"][i]
        else:
            segBox = p1["boxes"][i]
    ax.legend([segBox, regBox], ['Segmentation', 'Registration'], loc=boxplotLoc, fontsize='small', framealpha=0.75)

def GenerateBoxplots_MSD(xlsxLoader, out_file):

    params = {
        'axes.labelsize': 10,
        'legend.fontsize': 4,
        'xtick.labelsize': 6,
        'ytick.labelsize': 8,
        'text.usetex': False,
        'figure.figsize': [10, 10]
    }
    rcParams.update(params)
    # fig, axes = plt.subplots(nrows=4, ncols=3)
    fig, axes = plt.subplots(nrows=4, ncols=1)
    flierprops = {'markeredgecolor': 'black', 'marker': '+', 'markersize': 5, 'linestyle': 'none', 'linewidth': 1,
                  'markeredgewidth': 1, 'fillstyle': 'full'}

    left = 0.08  # the left side of the subplots of the figure
    right = 0.99  # the right side of the subplots of the figure
    bottom = 0.05  # the bottom of the subplots of the figure
    top = 0.95  # the top of the subplots of the figure
    wspace = 0.1  # the amount of width reserved for blank space between subplots
    hspace = 0.15  # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    n = len(xlsxLoader.infos)

    segColor = 'darkgreen'
    regColor = 'darkblue'
    colors = []
    for i in range(n):
        if xlsxLoader.infos[i].isReg or xlsxLoader.infos[i].tag == "ElastixEMC" or xlsxLoader.infos[i].tag == "MedphysEMC":
            colors.append(regColor)
        else:
            colors.append(segColor)

    margin =  0.7 / n
    positions = []
    pairPositions = [0]
    splitPositions = [0]
    labels = [""]

    if n==1:
        den = 1
    else:
        den = n-1
    for i in range(n):
        positions.append(margin + (1 - 2 * margin) / den * i)
        if i > 0 and xlsxLoader.infos[i-1].isSimilar(xlsxLoader.infos[i]):
            pairPositions.append((positions[i - 1] + positions[i]) / 2)
            if i < n - 1:
                nextPos = margin + (1 - 2 * margin) / den * (i + 1)
            else:
                nextPos = 1
            splitPositions.append((positions[i] + nextPos) / 2)
        else:
            if len(labels) % 2 == 0 and n > 6:
                labels.append("\n" + xlsxLoader.infos[i].title)
            else:
                labels.append(xlsxLoader.infos[i].title)
            if (i == 0 or not xlsxLoader.infos[i-1].isSimilar(xlsxLoader.infos[i])) \
                    and (i == n - 1 or not xlsxLoader.infos[i].isSimilar(xlsxLoader.infos[i+1])):
                pairPositions.append(positions[i])
                if i < n - 1:
                    nextPos = margin + (1 - 2 * margin) / den * (i + 1)
                else:
                    nextPos = 1
                splitPositions.append((positions[i] + nextPos) / 2)

    pairPositions.append(1)

    settings = BoxplotSettings(positions=positions, splitPositions=splitPositions, pairPositions=pairPositions,
                               margin=margin, n=n, flierprops=flierprops, colors=colors, labels=labels)

    ax = axes[0]
    ax.set_ylabel('Prostate')
    ax.set_xlabel('MSD (mm)')
    doPlot(xlsxLoader, settings, ax, xlsxLoader.exp_gtv['MSD'], np.arange(0, 11, 1), 0, 10)

    ax = axes[1]
    ax.set_ylabel('Seminal Vesicles')
    doPlot(xlsxLoader, settings, ax, xlsxLoader.exp_sv['MSD'], np.arange(0, 11, 1), 0, 10)

    ax = axes[2]
    ax.set_ylabel('Rectum')
    doPlot(xlsxLoader, settings, ax, xlsxLoader.exp_rectum['MSD'], np.arange(0, 11, 1), 0, 10)

    ax = axes[3]
    ax.set_ylabel('Bladder')
    doPlot(xlsxLoader, settings, ax, xlsxLoader.exp_bladder['MSD'], np.arange(0, 11, 1), 0, 10)


    # Flush or something
    plt.xticks(pairPositions[0:-1], labels, rotation=0)
    plt.tick_params(axis='x', direction='in')
    plt.xlim(0, 1)
    plt.sca(ax)
    plt.tick_params(axis='y', direction='in')
    ax.grid(axis='y', color="0.9", linestyle='--', linewidth=1)
    ax.xaxis.set_label_position('top')
    ax.set_axisbelow(True)

    fig.savefig(os.path.join(out_file, 'boxplot_MSD.png'), dpi=1000)
    fig.savefig(os.path.join(out_file, 'boxplot_MSD.pdf'), dpi=1000)
    print("Generated Boxplot ", os.path.join(out_file, 'boxplot_MSD.pdf'))







