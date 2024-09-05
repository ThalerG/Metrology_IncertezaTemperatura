import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from fcn import *
import os
from scipy.signal import savgol_filter

def extract_analysis_parameters(file_path):
    with open(file_path, 'r') as file:
        contents = file.read()

    # Define a regex pattern to extract the parameters
    pattern = re.compile(r"""
        s_dt\s*=\s*(?P<s_dt>[\d.]+)\s*
        s_dR\s*=\s*(?P<s_dR>[\d.]+)\s*

        R1\s*=\s*(?P<R1>[\d.]+)\s*
        Tamb_1\s*=\s*(?P<Tamb_1>[\d.]+)\s*
        Tamb_2\s*=\s*(?P<Tamb_2>[\d.]+)\s*
        k\s*=\s*(?P<k>[\d.]+)\s*
        s_R1\s*=\s*(?P<s_R1>[\d.]+)\s*
        s_Tamb1\s*=\s*(?P<s_Tamb1>[\d.]+)\s*
        s_Tamb2\s*=\s*(?P<s_Tamb2>[\d.]+)\s*
        s_t0\s*=\s*(?P<s_t0>[\d.]+)\s*

        N_points\s*=\s*(?P<Npoints>[\d.]+)\s*
        dt\s*=\s*(?P<dt>[\d.]+)\s*
        t1\s*=\s*(?P<t1>[\d.]+)
    """, re.VERBOSE)

    match = pattern.search(contents)
    if not match:
        raise ValueError("The file does not contain the expected analysis parameters.")

    # Convert the matched groups to a dictionary
    parameters = {key: float(value) for key, value in match.groupdict().items()}
    parameters['Npoints'] = int(parameters['Npoints'])
    parameters['dt'] = int(parameters['dt'])
    parameters['t1'] = int(parameters['t1'])
    return parameters
    

def plot_singleIteration(fname):
    """
    Plots the results of a single iteration of Monte Carlo simulation.
    Parameters:
    - fname (str): The name of the file containing the simulation results. The dataframe file should be in the 'Resultados' folder and have the extension '.feather'. 
                   The information file should have the same name and be in the same folder, but with the extension '.txt'.
    Returns:
    - fig (matplotlib.figure.Figure): The generated figure object.
    """

    # Check if both the feather and txt files exist
    if not os.path.exists(f"Resultados/{fname}.feather") or not os.path.exists(f"Resultados/{fname}.txt"):
        raise FileNotFoundError("The feather or txt file does not exist.")

    dfFile = f"Resultados/{fname}.feather"
    dfInfo = f"Resultados/{fname}.txt"
    df_og = pd.read_csv("Dados/data.csv")

    x_og = df_og['Time'].values
    y_og = df_og['Resistance'].values

    data = pd.read_feather(dfFile)

    analysis_param = extract_analysis_parameters(dfInfo)
    parameters = data['parameters'].values

    Tamb_1 = analysis_param['Tamb_1']
    Tamb_2 = analysis_param['Tamb_2']
    R1 = analysis_param['R1']
    k = analysis_param['k']

    plt.rcParams['font.family'] = 'P052'
    # Create a figure and axis for temperature subplot
    fig, ((ax1, ax2), ax) = plt.subplots(2, 2, figsize=(8, 6))

    T2_all = data['T2'].values

    parameters = parameters[0:1000]

    # Plot each exponential line for resistance
    for params in parameters:
        x = np.linspace(0, max(x_og), 100)
        R2 = generate_estimation_models(type='exp', degree=0, params=params)(x)
        ax1.plot(x, R2, alpha=0.1, color='C0', label='_nolegend_')
        T2 = final_temperature(R1, R2, Tamb_1, Tamb_2, k)
        ax2.plot(x, T2, alpha=0.1, color='C0', label='_nolegend_')

    x_tot = np.linspace(analysis_param['t1'], analysis_param['t1'] + (analysis_param['Npoints']-1)*analysis_param['dt'], analysis_param['Npoints']) 
    ind = np.isin(x_og, x_tot)

    x_tot = x_og[ind]
    y_tot = y_og[ind]

    # Scatter the original points as filled circles over the first subplot
    ax1.scatter(x_tot, y_tot, color='black', marker='o', facecolors='none')

    # Add a legend entry for the monte carlo plots
    ax1.plot([], [], alpha=0.1, label='Monte Carlo iteration', color='C0')
    # Add a legend entry for the scatter plot with the original data
    ax1.scatter([], [], color='black', marker='o', facecolors='none', label='Original Data')
    # Add a legend to the first subplot
    ax1.legend()

    # Add labels and title for resistance subplot
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Resistance [Ω]')

    # Add labels and title for temperature subplot
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Temperature [°C]')

    # Set the x-axis limits for both subplots
    ax1.set_xlim(0, max(x_og))
    ax2.set_xlim(0, max(x_og))

    # Create a figure and axis for cdf subplot
    ax3 = plt.subplot(2, 1, 2)
    n, bins, patches = ax3.hist(T2_all, bins=50, edgecolor='C0', alpha=1, density=True, label='Frequency', facecolor="none")
    # Create a second y-axis for frequency
    ax3_freq = ax3.twinx()
    ax3_freq.hist(T2_all, bins=100, color='C1', alpha=0, density=False)
    ax3_freq.set_ylabel('Frequency')

    # Plot the PDF of Temperature
    n, bins, patches = ax3.hist(T2_all, bins=1000, edgecolor='C0', alpha=0, density=True, label='_nolegend_', facecolor="none")
    n_filt = 10
    n = np.convolve(n, np.ones(n_filt)/n_filt, mode='valid')
    bins = bins[n_filt//2:-n_filt//2+1]
    ax3.plot((bins[:-1] + bins[1:]) / 2, n, color='C2', label = 'Probability Density Function')
    ax3.set_xlabel('Temperature [°C]')
    ax3.set_ylabel('Probability Density Function')
    ax3.set_title('PDF of Temperature')

    # Calculate the coverage interval
    alpha = 0.05
    lower_bound = np.percentile(T2_all, alpha/2 * 100)
    upper_bound = np.percentile(T2_all, (1 - alpha/2) * 100)

    # Color the area of the coverage interval
    ax3.fill_between((bins[:-1] + bins[1:]) / 2, 0, n, where=((bins[:-1] + bins[1:]) / 2 >= lower_bound) & ((bins[:-1] + bins[1:]) / 2 <= upper_bound), color='C1', alpha=0.5, label='95% Coverage Interval')

    # Add a legend to the subplot
    ax3.legend()

    plt.tight_layout()

    return fig


if __name__ == '__main__':
    fname = "analise3"
    fsave = "MC_narrowPoints"

    fig = plot_singleIteration(fname)

    if not os.path.exists("Gráficos"):
        os.makedirs("Gráficos")

    fig.savefig(f"Gráficos/{fsave}.pdf")