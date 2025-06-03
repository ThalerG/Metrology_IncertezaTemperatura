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

# initial_params = [1.20486213, 0.0115704,  -0.03001399] # Mensagem 1 do Pacheco
# initial_params = [6.58958550e+01,  4.23401967e-01, -3.34059000e-02] # Mensagem 2 do Pacheco
initial_params = [ 3.73139056e+01,  6.43352733e-01, -6.21373849e-03] # Ponto 13 do Pacheco

def exp_model(params, x):
    # TODO: Add docstring

    return params[0] + params[1]*np.exp(params[2]*x)

def fast_process_montecarlo(xy, s_dt = 0.01, s_dR = 0.001):
    # TODO: Add docstring

    x = xy[0]
    y = xy[1]
    Tamb_1 = xy[2]
    Tamb_2 = xy[3]
    R1 = xy[4]
    k = xy[5]

    params, _, out = estimate_model_with_uncertainty(x, y, None, None, model=exp_model, initial_params= initial_params,maxit = 10000000)

    if np.linalg.norm(params) >= 100:
        # x_fit = np.linspace(0, 60, 100)
        # plt.figure(figsize=(8, 6))
        # plt.plot(x, y, 'o', label='Original Data')
        # plt.plot(x_fit, exp_model(params, x_fit), '-', label='Fitted Curve')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.title('Fitted Exponential Model')
        # plt.legend()
        # plt.show()
        warnings.warn("Parameters are too large, check the model fit")

    R2 = exp_model(params,0)
    T2 = final_temperature(R1, R2, Tamb_1, Tamb_2, k)
    DT = T2 - Tamb_1

    return {'params': initial_params, 'R2': R2, 'T2': T2, 'DT': DT}

def generate_montecarlo_matrix(x_og, y_og, 
                               s_dt = 0.01, s_dR = 0.001, s_t0 = 0.01, s_R1 = 0.001, s_Tamb1 = 0.2, s_Tamb2 = 0.2, s_cvol = 1,
                               N1 = 4, dN = 2, n_x = 19, N_montecarlo = 200, 
                               Tamb_1 = 24, Tamb_2 = 24, R1 = 15.39, cvol = 100):
    # TODO: Add docstring

    # TODO: Alter the function to work for floats

    N1 = int(N1)
    dN = int(dN)
    n_x = int(n_x)

    ind = range(max(N1,0), min(N1 + n_x * dN,len(x_og)), dN)

    x_tot = x_og[ind]
    y_tot = y_og[ind]

    # Monte Carlo simulation for deviation of sample time
    montecarlo_matrix_x = np.random.uniform(x_tot-s_dt, x_tot+s_dt, (N_montecarlo,len(x_tot)))

    # Monte Carlo simulation for deviation of initial time
    montecarlo_t0 = np.random.normal(0, s_t0, (N_montecarlo, 1))
    montecarlo_matrix_x = montecarlo_matrix_x + montecarlo_t0

    s_dR_FS = 0.00004*100
    s_dR_Relative = y_tot*0.0001
    # Monte Carlo simulation for deviation of resistance measurement
    montecarlo_matrix_y = np.random.uniform(y_tot-s_dR_FS-s_dR_Relative, y_tot+s_dR_FS+s_dR_Relative, (N_montecarlo, len(y_tot)))
    # montecarlo_matrix_y = y_tot+np.random.normal(0, s_dR_FS, (N_montecarlo, len(y_tot))) + np.random.normal(0, s_dR_Relative, (N_montecarlo, len(y_tot)))

    # Monte Carlo simulation for deviation of ambient temperature
    montecarlo_matrix_Tamb_1 =  np.random.normal(Tamb_1, s_Tamb1, N_montecarlo)
    montecarlo_matrix_Tamb_2 =  np.random.normal(Tamb_2, s_Tamb2, N_montecarlo)

    montecarlo_matrix_R1 = np.random.normal(R1, s_R1, N_montecarlo)

    montecarlo_matrix_k = 25450/np.random.normal(cvol, s_cvol, N_montecarlo) - 20

    montecarlo_matrix_xy = list(zip(montecarlo_matrix_x, montecarlo_matrix_y, montecarlo_matrix_Tamb_1, montecarlo_matrix_Tamb_2, montecarlo_matrix_R1, montecarlo_matrix_k)) 

    return montecarlo_matrix_xy

def res_montecarlo_temp_montecarlo(parallel = True, 
                                   s_dt = 0.01, s_dR = 0.001, s_t0 = 0.01, s_R1 = 0.001, s_Tamb1 = 0.2, s_Tamb2 = 0.2, s_cvol = 1,
                                   N1 = 4, dN = 2, n_x = 19, N_montecarlo = 200, 
                                   Tamb_1 = 24, Tamb_2 = 24, R1 = 15.39, cvol = 100, t0 = 0.5):
    # TODO: Add docstring

    # x_og = np.array([15, 30, 45])
    # y_og = np.array([1.2122381926, 1.2095643282, 1.2078597546]) # Mensagem 1 do Pacheco
    # y_og = np.array([66.1523818970, 66.0512771606, 65.9900207520]) # Mensagem 2 do Pacheco

    x_og = np.array([60, 120, 180]) # Ponto 13 do Pacheco
    y_og = np.array([37.7570381165, 37.6191291809, 37.5241394043]) # Ponto 13 do Pacheco


    montecarlo_matrix = generate_montecarlo_matrix(x_og, y_og, 
                                                   s_dt = s_dt, s_dR = s_dR, s_t0 = s_t0, s_R1 = s_R1, s_Tamb1 = s_Tamb1, s_Tamb2 = s_Tamb2, s_cvol = s_cvol,
                                                   N1 = N1, dN = dN, n_x = n_x, N_montecarlo = N_montecarlo,
                                                   Tamb_1 = Tamb_1, Tamb_2 = Tamb_2, R1 = R1, cvol = cvol)

    if parallel:
        n_jobs = os.cpu_count()
        with Pool(n_jobs) as p:
            results_model = list(tqdm(p.imap(functools.partial(fast_process_montecarlo, s_dt = s_dt, s_dR = s_dR), montecarlo_matrix), desc='Monte Carlo Simulation', total=N_montecarlo, leave = False))
    else:
        results_model = []

        for xy in tqdm(montecarlo_matrix, desc='Monte Carlo Simulation', total=N_montecarlo, leave = False):
            results_model.append(fast_process_montecarlo(xy))
    
    results = {'params': [result['params'] for result in results_model],
               'R2': [result['R2'] for result in results_model],
                'T2': [result['T2'] for result in results_model],
                'DT': [result['DT'] for result in results_model]}
    
    montecarlo_vectors = {'t': [xy[0] for xy in montecarlo_matrix],
                          'R': [xy[1] for xy in montecarlo_matrix],
                          'Tamb_1': [xy[2] for xy in montecarlo_matrix],
                          'Tamb_2': [xy[3] for xy in montecarlo_matrix],
                          'R1': [xy[4] for xy in montecarlo_matrix],
                          'k': [xy[5] for xy in montecarlo_matrix]}

    return results, x_og, y_og, montecarlo_vectors

if __name__ == '__main__':
    ###### Valores default:

    analysis_param = {
        's_t0': 1e-1
    }

    ### Parser para valores da linha de comando
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, default='montecarlo_results', help='Name of file to save the results')
    parser.add_argument('--N1', type=int, default=int(analysis_param['N1']), help='Initial time')
    parser.add_argument('--s_t0', type=float, default=analysis_param['s_t0'], help='Uncertainty of the initial time')
    parser.add_argument('--Plot', action='store_true', help='Enable plotting')
    parser.add_argument('--NoSave', action='store_false', help='Disable saving results')
    parser.add_argument('--execTime', action='store_true', help='Enable execution time measurement')
    parser.add_argument('--s_dt', type=float, default=0.01, help='Uncertainty of the time acquisition')
    parser.add_argument('--s_dR', type=float, default=0.001, help='Uncertainty of the resistance measurement')
    parser.add_argument('--R1', type=float, default=15.39, help='Initial resistance')
    parser.add_argument('--Tamb_1', type=float, default=24, help='Initial ambient temperature')
    parser.add_argument('--Tamb_2', type=float, default=24, help='Final ambient temperature')
    parser.add_argument('--s_R1', type=float, default=0.001, help='Uncertainty of the initial resistance')
    parser.add_argument('--s_Tamb1', type=float, default=0.2, help='Uncertainty of the initial ambient temperature')
    parser.add_argument('--s_Tamb2', type=float, default=0.2, help='Uncertainty of the final ambient temperature')
    parser.add_argument('--cvol', type=float, default=100, help='Copper volumetric conductivity')
    parser.add_argument('--s_cvol', type=float, default=1, help='Uncertainty of the copper volumetric conductivity')

    args = parser.parse_args()

    analysis_param['s_t0'] = args.s_t0

    s_dt = args.s_dt
    s_dR = args.s_dR
    R1 = args.R1
    Tamb_1 = args.Tamb_1
    Tamb_2 = args.Tamb_2
    s_R1 = args.s_R1
    s_Tamb1 = args.s_Tamb1
    s_Tamb2 = args.s_Tamb2
    cvol = args.cvol
    s_cvol = args.s_cvol

    k = 25450/cvol - 20 # Recíproca do coeficiente de temperatura do cobre a 0°C

    SAVE = args.NoSave
    TIMEIT = args.execTime
    PLOT = args.Plot
    N_montecarlo = args.N_montecarlo
    fname = args.fname


    if TIMEIT:
        import time
        start_time = time.time()
    

    results, x_og, y_og, montecarlo_vectors = res_montecarlo_temp_montecarlo(N_montecarlo = N_montecarlo, 
                                                                              s_dt = s_dt, s_dR = s_dR, s_t0 = analysis_param['s_t0'], s_R1 = s_R1, s_Tamb1 = s_Tamb1, s_Tamb2 = s_Tamb2, s_cvol = s_cvol,
                                                                              N1 = analysis_param['N1'], dN = analysis_param['dN'], n_x = analysis_param['Npoints'], 
                                                                              Tamb_1 = Tamb_1, Tamb_2 = Tamb_2, R1 = R1, cvol = cvol)

    if TIMEIT:
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.5f} s")

    # Extract the parameters from the results
    parameters = results['params']

    R2_all =[]
    T2_all = []

    if PLOT:
        # Create a figure and axis for temperature subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

        # Plot each exponential line for resistance
        for ind,params in enumerate(parameters):
            x = np.linspace(0, max(x_og), 100)
            R2 = generate_estimation_models(type='exp', degree=0, params=params)(x)
            ax1.plot(x, R2, color='blue', alpha=0.01)
            T2 = final_temperature(montecarlo_vectors['R1'][ind], R2, montecarlo_vectors['Tamb_1'][ind], montecarlo_vectors['Tamb_2'][ind], montecarlo_vectors['k'][ind])
            ax2.plot(x, T2, color='red', alpha=0.01)

        x_tot = np.linspace(analysis_param['N1'], analysis_param['N1'] + (analysis_param['Npoints']-1)*analysis_param['dN'], analysis_param['Npoints']) 
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
    
    R2_all = results['R2']
    T2_all = results['T2']

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

    if SAVE:
        # Assign the parsed values to variables
        fsave = 'Resultados'
        
        # Check if the folder exists
        if not os.path.exists(fsave):
            # Create the folder
            os.makedirs(fsave)
        
        # Save the results to a feather file
        results = pd.DataFrame({'R2': results['R2'], 'T2': results['T2'], 'DT': results['DT'],'parameters': results['params']})
        results.to_feather(f'{fsave}/{fname}.feather')
        montecarlo_vectors = pd.DataFrame.from_dict(montecarlo_vectors)
        montecarlo_vectors.to_feather(f'{fsave}/{fname}_MCvectors.feather')

        # Save the conditions to a text file
        conditions = f"""s_dt = {s_dt}
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
    dN = {analysis_param['dN']}
    N1 = {analysis_param['N1']}

    mean_R2 = {mean_R2:.{sys.float_info.dig}g}
    std_R2 = {std_R2:.{sys.float_info.dig}g}
    mean_T2 = {mean_T2:.{sys.float_info.dig}g}
    std_T2 = {std_T2:.{sys.float_info.dig}g}
    l95_T2 = {lower_bound:.{sys.float_info.dig}g}
    u95_T2 = {upper_bound:.{sys.float_info.dig}g}"""

        with open(f'{fsave}/{fname}.txt', 'w') as file:
            file.write(conditions)