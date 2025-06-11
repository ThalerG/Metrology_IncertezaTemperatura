import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def plot_test_pdf(axis, files, labels, title):
        
    for nfile,file in enumerate(files):
    
        dfFile = f"Resultados/{file}.feather"

        data = pd.read_feather(dfFile)

        data = data.sort_values(by='T2')

        cm = 1/2.54  # centimeters in inches
        plt.rcParams['font.family'] = 'Times New Roman'
        # Create a figure and axis for temperature subplot

        # Create a figure and axis for cdf subplot
        T2_all = data['T2'].values
         
        # Plot the PDF of Temperature
        n, bins, patches = axis.hist(T2_all, bins=1000, edgecolor=f'C0', alpha=0, density=True, label='_nolegend_', facecolor="none")
        n_filt = 10
        n = np.convolve(n, np.ones(n_filt)/n_filt, mode='valid')
        bins = bins[n_filt//2:-n_filt//2+1]
        axis.plot((bins[:-1] + bins[1:]) / 2, n, color=f'C{nfile}', label = labels[nfile])

    # axis.set_xlabel('Winding temperature at the end of the test [°C]', fontname = 'Times New Roman')
    # axis.set_ylabel('Probability density function', fontname = 'Times New Roman')
    # Add a legend to the subplot
    # axis.legend()
    axis.grid()
    fig.tight_layout()

    plt.rcParams['font.family'] = 'Times New Roman'

    for label in axis.get_xticklabels():
        label.set_fontproperties('Times New Roman')

    for label in axis.get_yticklabels():
        label.set_fontproperties('Times New Roman')

    axis.set_title(title, fontname='Times New Roman')

    axis.set_xlim(85, 100)

    return fig

if __name__ == "__main__":

    testFiles = [["widePoints", "widePoints", "narrowPointsBestCase", "narrowPoints"],
                ["difTamb_allPoints", "difTamb_widePoints", "difTamb_earlyPoints", "difTamb_latePoints"],
                 ["normC_allPoints", "normC_widePoints", "normC_earlyPoints", "normC_latePoints"],
                ["exponentialTf_allPoints", "exponentialTf_widePoints", "exponentialTf_earlyPoints", "exponentialTf_latePoints"]]
    
    labels = ["All points",
              "10 s, 20 s, 30 s",
              "4 s, 6 s, 8 s",
              "20 s, 22 s, 24 s"]

    titles = ["Base case",
        "Variation 1.1",
        "Variation 1.2",
        "Variation 1.3"]
    
    cm = 1/2.54 
    fig, ax = plt.subplots(4, 1, figsize=(10*cm, 12*cm))

    for files, ax, title in zip(testFiles, ax, titles):
        plot_test_pdf(ax, files, labels, title)

    plt.tight_layout()
    # plt.show()
    plt.savefig("Gráficos/extraTests.pdf")
    plt.close(fig)


        