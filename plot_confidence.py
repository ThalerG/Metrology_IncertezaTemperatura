import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_montecarlos import extract_analysis_parameters
from fcn import generate_estimation_models, final_temperature

def filter_data(data: pd.DataFrame, fixedVars: dict):
    """
    Filter the data based on the fixed variables.
    Parameters:
    - data (pd.DataFrame): The data to be filtered.
    - fixedVars (dict): A dictionary containing the fixed variables.
    Returns:
    - filtered_data (pd.DataFrame): The filtered data.
    """
    filtered_data = data.copy()
    for key, value in fixedVars.items():
        filtered_data = filtered_data[filtered_data[key] == value]
    return filtered_data

def plot_confidence(data, fixedVars, labels):
    cm = 1/2.54 
    fig, ax = plt.subplots(figsize=(14*cm, 10*cm))
    
    for i, fixedVar in enumerate(fixedVars):
        filtered_data = filter_data(data, fixedVar)
        
        ax.plot(filtered_data['s_t0'], filtered_data['u95_T2'], label=f'Upper 95% ({labels[i]})', color=f'C{i}', linestyle='--', marker='o'), 
        ax.plot(filtered_data['s_t0'], filtered_data['l95_T2'], label=f'Lower 95% ({labels[i]})', color=f'C{i}', linestyle='--', marker='x') 
    
    plt.rcParams['font.family'] = 'Times New Roman'

    for label in ax.get_xticklabels():
        label.set_fontproperties('Times New Roman')

    for label in ax.get_yticklabels():
        label.set_fontproperties('Times New Roman')

    ax.set_xscale('log')
    ax.set_xlabel('De-energization time uncertainty [s]', fontname = 'Times New Roman')
    ax.set_ylabel('Winding temperature [°C]', fontname = 'Times New Roman')
    ax.grid()
    ax.legend()
    plt.tight_layout()

    return fig, ax

def plot_distribution(fname,labels):
    cm = 1/2.54 
    fig, (ax_top, ax_bottom) = plt.subplots(2, len(fname), figsize=(22*cm, 11*cm))

    for ax in ax_bottom:
        fig.delaxes(ax)

    ax_bottom = plt.subplot(2, 1, 2)
    
    for nfile, file in enumerate(fnames):
        dfFile = f"Resultados/{file}.feather"
        dfMCVectors = f"Resultados/{file}_MCvectors.feather"
        dfInfo = f"Resultados/{file}.txt"
        df_og = pd.read_csv("Dados/data.csv")

        x_og = df_og['Time'].values
        y_og = df_og['Resistance'].values

        data = pd.read_feather(dfFile)
        data = pd.merge(data, pd.read_feather(dfMCVectors), left_index=True, right_index=True)

        data = data.sort_values(by='T2')

        analysis_param = extract_analysis_parameters(dfInfo)

        cm = 1/2.54  # centimeters in inches
        plt.rcParams['font.family'] = 'Times New Roman'
        # Create a figure and axis for temperature subplot
        
        nCurves = 2000

        ind = np.linspace(0, len(data)-1, nCurves, dtype=int)
        varList = {
            "params": data['parameters'].values,
            "R1": data['R1'].values,
            "Tamb_1": data['Tamb_1'].values,
            "Tamb_2": data['Tamb_2'].values,
            "k": data['k'].values
        }

        x = np.linspace(0, max(x_og), 100)
        R2_all = np.array([generate_estimation_models(type='exp', degree=0, params=varList['params'][i])(x) for i in ind])
        T2_all = np.array([final_temperature(varList['R1'][i], R2_all[i], varList['Tamb_1'][i], varList['Tamb_2'][i], varList['k'][i]) for i in range(len(ind))])
        for i in range(len(ind)):
            ax_top[nfile].plot(x, T2_all[i], alpha=0.2, color='C0')
        ax_top[nfile].set_title(labels[nfile], fontname = 'Times New Roman')

        # Add labels and title for temperature subplot
        ax_top[nfile].set_xlabel('Time [s]', fontname = 'Times New Roman')
        ax_top[nfile].set_ylabel('Temperature [°C]', fontname = 'Times New Roman')

        # Set the x-axis limits for both subplots
        ax_top[nfile].set_xlim(0, max(x_og))
        ax_top[nfile].set_ylim(65, 130)
        ax_top[nfile].grid()

        for label in ax_top[nfile].get_xticklabels():
            label.set_fontproperties('Times New Roman')

        for label in ax_top[nfile].get_yticklabels():
            label.set_fontproperties('Times New Roman')

        # Create a figure and axis for cdf subplot
        T2_all = data['T2'].values
         
        # Plot the PDF of Temperature
        n, bins, patches = ax_bottom.hist(T2_all, bins=1000, edgecolor='C0', alpha=0, density=True, label='_nolegend_', facecolor="none")
        n_filt = 10
        n = np.convolve(n, np.ones(n_filt)/n_filt, mode='valid')
        bins = bins[n_filt//2:-n_filt//2+1]
        ax_bottom.plot((bins[:-1] + bins[1:]) / 2, n, color=f'C{nfile}', label = labels[nfile])

    ax_bottom.set_xlabel('Winding temperature at the end of the test [°C]', fontname = 'Times New Roman')
    ax_bottom.set_ylabel('Probability density function', fontname = 'Times New Roman')
    # Add a legend to the subplot
    ax_bottom.legend()
    ax_bottom.grid()
    fig.tight_layout()

    plt.rcParams['font.family'] = 'Times New Roman'

    for label in ax_bottom.get_xticklabels():
        label.set_fontproperties('Times New Roman')

    for label in ax_bottom.get_yticklabels():
        label.set_fontproperties('Times New Roman')

    return fig

if __name__ == '__main__':
    path = "Resultados\\map_results.csv"
    # Load the data
    data = pd.read_csv(path)

    # Define the fixed variables
    fixedVars = [{'Npoints': 19, 'dt': 2, 't1': 4},
                 {'Npoints': 3, 'dt': 10, 't1': 10},
                 {'Npoints': 3, 'dt': 2, 't1': 4},
                 {'Npoints': 3, 'dt': 2, 't1': 20},
                 {'Npoints': 3, 'dt': 10, 't1': 4}]
    
    labels = ["all",
              "10s, 20s, 30s",
              "4s, 6s, 8s",
              "20s, 22s, 24s",
              "4s, 14s, 24s"]
    
    fig, ax = plot_confidence(data, fixedVars, labels)
    
    fig.savefig('Gráficos\\confidence_st0_more.pdf')

    # fnames = ["allPoints_st05s",
    #           "early_st05s",
    #           "narrowLater_st05s"]
    
    # labels = ["All points",
    #           "4s, 6s, 8s",
    #           "20s, 22s, 24s"]
    
    # fig = plot_distribution(fnames, labels)
    # fig.savefig('Gráficos\\distribution_st05s.pdf')
    

