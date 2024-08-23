from fcn import *
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
import os
import warnings
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Script for performing Monte Carlo simulations and analysis of the winding temperature of a motor.')

# Add the arguments
parser.add_argument('--fsave', type=str, default='Resultados', help='Folder to save the results')
parser.add_argument('--N_montecarlo', type=int, default=200, help='Number of Monte Carlo simulations')

PLOT = False

# Incertezas de medição:

s_t0 = 0 # Incerteza do tempo inicial

s_dt = 0.001 # Incerteza do tempo de aquisição

s_dR = 0.001 # Incerteza da medição de resistência

# Condições de teste

R1 = 15.39 # Resistência no início do teste
Tamb_1 = 24 # Temperatura ambiente no início do teste
Tamb_2 = 24 # Temperatura ambiente no início do teste

k = 234.5 # Recíproco do coeficiente de temperatura do resistor
alpha = 1/(k+Tamb_1) # Coeficiente de temperatura do resistor

s_R1 = s_dR # Incerteza da medição de resistência no início do teste
s_Tamb1 = 0.1 # Incerteza da medição de temperatura no início do teste
s_Tamb2 = 0.1 # Incerteza da medição de temperatura no final do teste

# s_x = np.sqrt(s_t0**2 + s_dt**2)
s_x = s_dt
s_y = s_dR

###### Análise 
analysis_param = {
    'dt': 2,
    'Npoints': 3,
    't1': 20,
    's_t0': 1e-2
}

def process_montecarlo(xy, s_x, s_y, model):

    if callable(model):
        initial_params = [17.472,2.06,-0.0197]
    else:
        if model[0] == 'poly':
            if model[1] == 1:
                initial_params = [19.425,0.0266]
            elif model[1] == 2:
                initial_params = [19.52,-0.0381,0.00026]
            elif model[1] == 3:
                initial_params = [19.534,-0.041,0.000413,-2.31e-6]
            else:
                initial_params = [1]*(model[1]+1)
        elif model[0] == 'exp':
            initial_params = [17.472,2.06,-0.0197]
        model = generate_estimation_models(type = model[0], degree=model[1], params=params)

    x = xy[0]
    y = xy[1]

    params, _, _ = estimate_model_with_uncertainty(x, y, s_x, s_y, model=model, initial_params= initial_params,maxit = 1000000)
    
    s_x0 = np.sqrt(s_t0**2 + s_dt**2)

    R2 = model(params,0)

    return {'params': params, 'R2': R2}

def generate_montecarlo_matrix(x_og, y_og, s_x, s_y, s_t0 = s_t0, t1 = 4, dt = 2, n_x = 19, N_montecarlo = 200):
    x_tot = np.linspace(t1, t1 + (n_x-1)*dt, n_x)
    ind = np.isin(x_og, x_tot)
    
    if np.sum(ind) == 0:
        raise ValueError("None of the provided time data fits the desired configuration.")
    elif np.sum(ind) < n_x:
        warnings.warn("Not all elements of the desired configuration are present in the provided time data. Proceeding with the available data.")

    x_tot = x_og[ind]
    y_tot = y_og[ind]

    # Monte Carlo simulation for deviation of sample time
    montecarlo_matrix_x = np.random.normal(0, s_x, (len(x_tot), N_montecarlo))
    montecarlo_matrix_x = x_tot[:, np.newaxis] + montecarlo_matrix_x

    # Monte Carlo simulation for deviation of initial time
    montecarlo_t0 = np.random.normal(0, s_t0, (1, N_montecarlo))
    montecarlo_matrix_x = montecarlo_matrix_x + montecarlo_t0

    # Monte Carlo simulation for deviation of resistance measurement
    montecarlo_matrix_y = np.random.normal(0, s_y, (len(y_tot), N_montecarlo))
    montecarlo_matrix_y = y_tot[:, np.newaxis] + montecarlo_matrix_y

    montecarlo_matrix_xy = list(zip(montecarlo_matrix_x.T, montecarlo_matrix_y.T))

    return montecarlo_matrix_xy

def montecarlo_analysis(analysis_params: dict, x_og: np.ndarray, y_og: np.ndarray, model = ('exp',0), N_montecarlo = 200):

    montecarlo_matrix_xy = generate_montecarlo_matrix(x_og, y_og, s_x, s_y, s_t0 = analysis_params['s_t0'], t1 = int(analysis_params['t1']), dt = int(analysis_params['dt']), n_x = int(analysis_params['Npoints']), N_montecarlo = N_montecarlo)
    
    estimation_model = generate_estimation_models(type = model[0], degree=model[1])

    results_model = []

    for xy in tqdm(montecarlo_matrix_xy, desc='Monte Carlo Simulation', total=N_montecarlo):
        results_model.append(process_montecarlo(xy, s_x, s_y, estimation_model))

    # Calculate the mean values of R2, s_R2, T2, and s_T2 from results_model
    mean_R2 = np.mean([result['R2'] for result in results_model])
    T2 = final_temperature(R1, mean_R2, Tamb_1, Tamb_2, k)

    # Calculate the standard deviation of R2, s_R2, T2, and s_T2 from results_model
    std_R2 = np.std([result['R2'] for result in results_model])
    s_T2 = final_temperature_uncertainty(R1, mean_R2, Tamb_1, Tamb_2, k, s_R1, std_R2, s_Tamb1, s_Tamb2)

    # Append results
    results = {'parameters': [result['params'] for result in results_model],
               'mean_R2': mean_R2, 'std_R2': std_R2,
               'mean_T2': T2, 'std_T2': s_T2}

    return results

def res_montecarlo_temp_calc(N_montecarlo = 200, model = ('exp',0)):
    file_path = "Dados/data.csv"
    df = pd.read_csv(file_path)

    x_og = df['Time'].values
    y_og = df['Resistance'].values

    results = montecarlo_analysis(analysis_param, x_og, y_og, model, N_montecarlo)

    return results, x_og, y_og

if __name__ == '__main__':
    # Parse the arguments
    args = parser.parse_args()

    # Assign the parsed values to variables
    fsave = args.fsave
    N_montecarlo = args.N_montecarlo

    # Check if the folder exists
    if not os.path.exists(fsave):
        # Create the folder
        os.makedirs(fsave)

    model = ('exp',0)

    results, x_og, y_og = res_montecarlo_temp_calc(N_montecarlo, model)

    # Extract the parameters from the results
    parameters = results['parameters']

    if PLOT:
        # Create a figure and axis for temperature subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

        # Plot each exponential line for resistance
        for params in parameters:
            x = np.linspace(0, max(x_og), 100)
            R2 = generate_estimation_models(type='exp', degree=0, params=params)(x)
            ax1.plot(x, R2, color='blue', alpha=0.01)
            T2 = final_temperature(R1, R2, Tamb_1, Tamb_2, k)
            ax2.plot(x, T2, color='red', alpha=0.01)

        x_tot = np.linspace(analysis_param['t1'], analysis_param['t1'] + (analysis_param['Npoints']-1)*analysis_param['dt'], analysis_param['Npoints']) 
        ind = np.isin(x_og, x_tot)

        x_tot = x_og[ind]
        y_tot = y_og[ind]

        # Scatter the original points as filled circles over the first subplot
        ax1.scatter(x_tot, y_tot, color='black', marker='o', facecolors='none')

        # Add labels and title for resistance subplot
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Resistance [Ω]')

        # Add labels and title for temperature subplot
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Temperature [°C]')

        # Show the plot
        plt.show()