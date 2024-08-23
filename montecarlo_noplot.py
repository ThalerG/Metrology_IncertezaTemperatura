from fcn import *
import numpy as np
import pandas as pd
import tqdm
import functools
from multiprocessing import Pool
from itertools import product
import os
import warnings
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Script for performing Monte Carlo simulations and analysis of the winding temperature of a motor.')

# Add the arguments
parser.add_argument('--fsave', type=str, default='Resultados', help='Folder to save the results')
parser.add_argument('--N_montecarlo', type=int, default=200, help='Number of Monte Carlo simulations')

# Incertezas de medição:

s_t0 = 0.1 # Incerteza do tempo inicial

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
analyses = {
    'dt': [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40],
    'Npoints': [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
    't1': [4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40],
    's_t0': [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1e0, 2e0, 5e0, 1e1]
}

def process_montecarlo(xy, s_x, s_y, model):
    estimation_model = generate_estimation_models(type = model[0], degree=model[1])

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

    x = xy[0]
    y = xy[1]

    params, uncertainty, result = estimate_model_with_uncertainty(x, y, s_x, s_y, model=estimation_model, initial_params= initial_params,maxit = 1000000)

    estimated_model = generate_estimation_models(type = model[0], degree=model[1], params=params)
    
    s_x0 = np.sqrt(s_t0**2 + s_dt**2)
    uncertainty_model = generate_estimation_uncertainty_models(params=params, s_params=uncertainty, s_x = s_x0, type=model[0], degree=model[1])

    R2 = estimated_model(0)
    s_R2 = uncertainty_model(0)

    DT = delta_temperature(R1, R2, Tamb_1, Tamb_2, k)
    s_DT = delta_temperature_uncertainty(R1, R2, Tamb_1, Tamb_2, k, s_R1, s_R2, s_Tamb1, s_Tamb2)

    T2 = final_temperature(R1, R2, Tamb_1, Tamb_2, k)
    s_T2 = final_temperature_uncertainty(R1, R2, Tamb_1, Tamb_2, k, s_R1, s_R2, s_Tamb1, s_Tamb2)
    return {'params': params, 'uncertainty': uncertainty, 'result': result, 'R2': R2, 's_R2': s_R2, 'T2': T2, 's_T2': s_T2, 'DT': DT, 's_DT': s_DT}

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

def montecarlo_analysis(analysis_params: dict, x_og: np.ndarray, y_og: np.ndarray):    
    conditions = pd.DataFrame(columns=analysis_params.keys(), data=product(*analysis_params.values()))

    ind = (conditions['t1'] >= x_og[0]) & ((conditions['t1'] + (conditions['Npoints']-1)*conditions['dt']) <= x_og[-1])
    conditions = conditions.loc[ind]

    results_data = []

    for k,row in tqdm.tqdm(conditions.iterrows(), desc = 'Condition', position=1, leave = True,  total=conditions.shape[0]):
        montecarlo_matrix_xy = generate_montecarlo_matrix(x_og, y_og, s_x, s_y, s_t0 = row['s_t0'], t1 = int(row['t1']), dt = int(row['dt']), n_x = int(row['Npoints']), N_montecarlo = N_montecarlo)
        
        with Pool(n_jobs) as p:
            results_model = p.map(functools.partial(process_montecarlo, model=model, s_x=s_x, s_y=s_y), montecarlo_matrix_xy)

        # Calculate the mean values of R2, s_R2, T2, and s_T2 from results_model
        mean_R2 = np.mean([result['R2'] for result in results_model])
        mean_s_R2 = np.mean([result['s_R2'] for result in results_model])
        mean_sum_square = np.mean([result['result'].sum_square for result in results_model])

        DT = delta_temperature(R1, mean_R2, Tamb_1, Tamb_2, k)
        T2 = final_temperature(R1, mean_R2, Tamb_1, Tamb_2, k)

        # Calculate the standard deviation of R2, s_R2, T2, and s_T2 from results_model
        std_R2 = np.std([result['R2'] for result in results_model])
        std_s_R2 = np.std([result['s_R2'] for result in results_model])
        std_sum_square = np.std([result['result'].sum_square for result in results_model])

        s_DT = delta_temperature_uncertainty(R1, mean_R2, Tamb_1, Tamb_2, k, s_R1, std_R2, s_Tamb1, s_Tamb2)
        s_T2 = final_temperature_uncertainty(R1, mean_R2, Tamb_1, Tamb_2, k, s_R1, std_R2, s_Tamb1, s_Tamb2)

        # Append results
        results_data.append([mean_sum_square, mean_R2, mean_s_R2, DT, T2,
                             std_sum_square, std_R2, std_s_R2, s_DT, s_T2])
        
    results_labels = ['mean_SSE', 'mean_Resistance', 'mean_EstimationUncertainty', 'mean_DeltaTemperature', 'mean_Temperature',
                      'std_SSE', 'std_Resistance', 'std_EstimationUncertainty', 'std_DeltaTemperature', 'std_Temperature']
    
    conditions.to_csv('Resultados/conditions.csv', index=False)
    results = pd.DataFrame(columns=results_labels, data=results_data)
    results.to_csv('Resultados/results_noconditions.csv', index=False)
    results = pd.concat([conditions, results], axis=1)

    return results

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
    file_path = "Dados/data.csv"
    df = pd.read_csv(file_path)

    model = ('exp',0)

    x_og = df['Time'].values
    y_og = df['Resistance'].values

    n_jobs = os.cpu_count()

    results = montecarlo_analysis(analyses, x_og, y_og)

    results.to_csv(fsave + '/results.csv', index=False)