from fcn import *
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import warnings
from multiprocessing import Pool
import functools
import sys
import argparse

PLOT = False
CALC_R2 = False

parser = argparse.ArgumentParser()
parser.add_argument('--N_montecarlo', type=int, default=int(1e6), help='Number of Monte Carlo simulations')
parser.add_argument('--fname', type=str, default='montecarlo_results', help='Name of file to save the results')

args = parser.parse_args()

N_montecarlo = args.N_montecarlo
fname = args.fname

# Incertezas de medição:

s_dt = 0.001 # Incerteza do tempo de aquisição

s_dR = 0.001 # Incerteza da medição de resistência

# Condições de teste

R1 = 15.39 # Resistência no início do teste
Tamb_1 = 24 # Temperatura ambiente no início do teste
Tamb_2 = 24 # Temperatura ambiente no início do teste

k = 234.5 # Recíproco do coeficiente de temperatura do resistor
alpha = 1/(k+Tamb_1) # Coeficiente de temperatura do resistor

s_R1 = s_dR # Incerteza da medição de resistência no início do teste
s_Tamb1 = 0.2 # Incerteza da medição de temperatura no início do teste
s_Tamb2 = 0.2 # Incerteza da medição de temperatura no final do teste

# s_x = np.sqrt(s_t0**2 + s_dt**2)
s_x = s_dt
s_y = s_dR

initial_params = [17.472,2.06,-0.0197]

###### Análise 
analysis_param = {
    'dt': 2,
    'Npoints': 19,
    't1': 4,
    's_t0': 1e-2
}

def exp_model(params, x):
    return params[0] + params[1]*np.exp(params[2]*x)

def fast_process_montecarlo(xy):

    x_batch = xy[0]
    y_batch = xy[1]
    Tamb_1_batch = xy[2]
    Tamb_2_batch = xy[3]
    R1_batch = xy[4]

    results = []
    for x, y, Tamb_1, Tamb_2, R1 in zip(x_batch, y_batch, Tamb_1_batch, Tamb_2_batch, R1_batch):

        params, _, _ = estimate_model_with_uncertainty(x, y, s_x, s_y, model=exp_model, initial_params= initial_params,maxit = 1000000)

        R2 = exp_model(params,0)
        T2 = final_temperature(R1, R2, Tamb_1, Tamb_2, k)

        results.append({'params': params, 'R2': R2, 'T2': T2})
    return results

def generate_montecarlo_matrix(x_og, y_og, s_x, s_y, s_t0 = 0.01, t1 = 4, dt = 2, n_x = 19, N_montecarlo = 200):
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

    # Monte Carlo simulation for deviation of ambient temperature
    montecarlo_matrix_Tamb_1 = Tamb_1 + np.random.normal(0, s_Tamb1, (1,N_montecarlo))
    montecarlo_matrix_Tamb_2 = Tamb_2 + np.random.normal(0, s_Tamb2, (1,N_montecarlo))

    montecarlo_matrix_R1 = R1 + np.random.normal(0, s_R1, (1,N_montecarlo))

    n_per_batch = 100
    n_batches = np.ceil(N_montecarlo/n_per_batch)

    montecarlo_matrix_x = np.array_split(montecarlo_matrix_x.T, n_batches, axis=0)
    montecarlo_matrix_y = np.array_split(montecarlo_matrix_y.T, n_batches, axis=0)
    montecarlo_matrix_Tamb_1 = np.array_split(montecarlo_matrix_Tamb_1.T, n_batches, axis=0)
    montecarlo_matrix_Tamb_2 = np.array_split(montecarlo_matrix_Tamb_2.T, n_batches, axis=0)
    montecarlo_matrix_R1 = np.array_split(montecarlo_matrix_R1.T, n_batches, axis=0)

    montecarlo_matrix_xy = list(zip(montecarlo_matrix_x, montecarlo_matrix_y, montecarlo_matrix_Tamb_1, montecarlo_matrix_Tamb_2, montecarlo_matrix_R1)) 

    return montecarlo_matrix_xy

def res_montecarlo_temp_montecarlo(N_montecarlo = 200, parallel = True):
    file_path = "Dados/data.csv"
    df = pd.read_csv(file_path)

    x_og = df['Time'].values
    y_og = df['Resistance'].values

    montecarlo_matrix = generate_montecarlo_matrix(x_og, y_og, s_x, s_y, s_t0 = analysis_param['s_t0'], t1 = int(analysis_param['t1']), dt = int(analysis_param['dt']), n_x = int(analysis_param['Npoints']), N_montecarlo = N_montecarlo)

    if parallel:
        n_jobs = os.cpu_count()
        with Pool(n_jobs) as p:
            results_model = list(tqdm(p.imap(functools.partial(fast_process_montecarlo), montecarlo_matrix), desc='Monte Carlo Simulation', total=len(montecarlo_matrix), leave = False))
    else:
        results_model = []

        for xy in tqdm(montecarlo_matrix, desc='Monte Carlo Simulation', total=len(montecarlo_matrix), leave = False):
            results_model.append(fast_process_montecarlo(xy))

    results = [item for sublist in results_model for item in sublist] # Flatten the list

    results = {'params': [result['params'] for result in results],
               'R2': [result['R2'] for result in results],
                'T2': [result['T2'] for result in results]}
    
    return results, x_og, y_og


if __name__ == '__main__':
    # Assign the parsed values to variables
    fsave = 'Resultados'

    # Check if the folder exists
    if not os.path.exists(fsave):
        # Create the folder
        os.makedirs(fsave)

    model = ('exp',0)

    results, x_og, y_og = res_montecarlo_temp_montecarlo(N_montecarlo)

    # Extract the parameters from the results
    parameters = results['params']

    R2_all =[]
    T2_all = []

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
            R2_all.append(R2[0])
            T2_all.append(T2[0])

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

        # Create a histogram of T2 values
        plt.hist(T2_all, bins=100, density=True, color='blue', edgecolor='black')

        # Add labels and title to the histogram
        plt.xlabel('Temperature [°C]')
        plt.ylabel('Frequency')
        plt.title('Histogram of T2')

        # Show the histogram
        plt.show()
    
    else:
        for params in parameters:
            R2 = generate_estimation_models(type='exp', degree=0, params=params)(0)
            T2 = final_temperature(R1, R2, Tamb_1, Tamb_2, k)
            R2_all.append(R2)
            T2_all.append(T2)

    mean_R2 = np.mean(R2_all)
    std_R2 = np.std(R2_all)
    mean_T2 = np.mean(T2_all)
    std_T2 = np.std(T2_all)

    # Calculate the coverage interval of 95% of T2
    alpha = 0.05
    lower_bound = np.percentile(T2_all, alpha/2 * 100)
    upper_bound = np.percentile(T2_all, (1 - alpha/2) * 100)

    print(f"R2 Monte Carlo, T2 Monte Carlo")
    print(f"R2: {mean_R2} ± {std_R2}")
    print(f"T2: {mean_T2} ± {std_T2}")
    print(f"Coverage Interval of 95% of T2: [{lower_bound}, {upper_bound}]")

    # Save the results to a feather file
    results = pd.DataFrame({'R2': R2_all, 'T2': T2_all, 'parameters': parameters})
    results.to_feather(f'{fsave}/{fname}.feather')

    # Save the conditions to a text file
    conditions = f"""
    s_dt = {s_dt}
    s_dR = {s_dR}

    R1 = {R1}
    Tamb_1 = {Tamb_1}
    Tamb_2 = {Tamb_2}
    k = {k}
    s_R1 = {s_R1}
    s_Tamb1 = {s_Tamb1}
    s_Tamb2 = {s_Tamb2}
    s_t0 = {analysis_param['s_t0']}

    N_points = {analysis_param['Npoints']}
    dt = {analysis_param['dt']}
    t1 = {analysis_param['t1']}
    """

    with open(f'{fsave}/{fname}.txt', 'w') as file:
        file.write(conditions)