from fcn import *
from fast_montecarlo_embraco import res_montecarlo_temp_montecarlo
import numpy as np
import pandas as pd
import tqdm
from itertools import product, chain
from scipy.stats import kurtosis, skew
import argparse
import os

def generateConditions(baseValues, analysesAlter, x_og, analysisSkip = None):
    allAnalysis = [{**baseValues, **an} for an in analysesAlter]

    conditions = pd.DataFrame()

    for params in allAnalysis:
        keys, values = zip(*params.items())
        conditions = pd.concat([conditions, pd.DataFrame(list(product(*values)), columns=keys)], axis=0)

    conditions = conditions.drop_duplicates().reset_index(drop=True)

    conditions['indexes'] = conditions.apply(lambda row: list(range(int(row['N1']), int(row['N1']) + int(row['Npoints']) * int(row['dN']), int(row['dN']))), axis=1)

    if analysisSkip is not None:
        analysisSkip = analysisSkip[analysisSkip.columns.intersection(conditions.columns)]
        conditions = pd.merge(conditions, analysisSkip, how='outer', indicator=True).query("_merge == 'left_only'").drop('_merge', axis=1).reset_index(drop=True)

    return conditions

def montecarlo_analysis(analysis_params: list[dict], x_og: np.ndarray, y_og: np.ndarray):
    # TODO: Add docstring

    results_data = {"mean_T2": [], "std_T2": [], "l95_T2": [], "u95_T2": [], "kur_T2": [], "skew_T2": [],
                    "mean_R2": [], "std_R2": [], "l95_R2": [], "u95_R2": [], "kur_R2": [], "skew_R2": [],
                    "mean_DT": [], "std_DT": [], "l95_DT": [], "u95_DT": [], "kur_DT": [], "skew_DT": []}

    for k,row in tqdm.tqdm(conditions.iterrows(), desc = 'Condition', position=1, leave = True,  total=conditions.shape[0]):

        results, _, _, montecarlo_vectors = res_montecarlo_temp_montecarlo(N_montecarlo = N_montecarlo, 
                                                                              s_dt = row['s_dt'], s_dR = row['s_dR'], s_t0 = row['s_t0'], s_R1 = row['s_R1'], s_Tamb1 = row['s_Tamb1'], s_Tamb2 = row['s_Tamb2'], s_cvol = row['s_cvol'],
                                                                              N1 = row['N1'], dN = row['dN'], n_x = row['Npoints'], 
                                                                              R1 = row['R1'], Tamb_1 = row['Tamb1'], Tamb_2 = row['Tamb2'], cvol = row['cvol'], t0 = row['t0'])
        
        results_data["mean_T2"].append(np.mean(results['T2']))
        results_data["std_T2"].append(np.std(results['T2']))
        results_data["l95_T2"].append(np.percentile(results['T2'], 2.5))
        results_data["u95_T2"].append(np.percentile(results['T2'], 97.5))
        results_data["kur_T2"].append(kurtosis(results['T2']))
        results_data["skew_T2"].append(skew(results['T2']))

        results_data["mean_R2"].append(np.mean(results['R2']))
        results_data["std_R2"].append(np.std(results['R2']))
        results_data["l95_R2"].append(np.percentile(results['R2'], 2.5))
        results_data["u95_R2"].append(np.percentile(results['R2'], 97.5))
        results_data["kur_R2"].append(kurtosis(results['R2']))
        results_data["skew_R2"].append(skew(results['R2']))

        results_data["mean_DT"].append(np.mean(results['DT']))
        results_data["std_DT"].append(np.std(results['DT']))
        results_data["l95_DT"].append(np.percentile(results['DT'], 2.5))
        results_data["u95_DT"].append(np.percentile(results['DT'], 97.5))
        results_data["kur_DT"].append(kurtosis(results['DT']))
        results_data["skew_DT"].append(skew(results['DT']))

        results = pd.DataFrame({'R2': results['R2'], 'T2': results['T2'], 'DT': results['DT'],'parameters': results['params']})
        results.to_feather(f'{fsave}/embraco3.feather')
        montecarlo_vectors = pd.DataFrame.from_dict(montecarlo_vectors)
        montecarlo_vectors.to_feather(f'{fsave}/embraco3_MCvectors.feather')

    results = pd.DataFrame(results_data)   
    results = pd.concat([conditions.reset_index(drop=True), results.reset_index(drop=True)], axis=1)   
        
    return results

if __name__ == '__main__':
    APPEND = False
    # Create the parser
    parser = argparse.ArgumentParser(description='Script for performing Monte Carlo simulations and analysis of the winding temperature of a motor.')

    # Add the arguments
    parser.add_argument('--fsave', type=str, default='Resultados', help='Folder to save the results')
    parser.add_argument('--N_montecarlo', type=int, default=1000000, help='Number of Monte Carlo simulations')

    baseValues = {'dN': [1],
                'Npoints': [3],
                'N1': [0],
                'R1': [31.6],
                't0': [0],
                'Tamb1': [25],
                'Tamb2': [25],
                'cvol': [100],
                's_t0': [0.1],
                's_dt': [0.001],
                's_dR': [0.001],
                's_R1': [0.001],
                's_Tamb1': [0.2],
                's_Tamb2': [0.2],
                's_cvol': [1]}

    analysesAlter = [
                {'Npoints': [3], 'N1': [0], 'dN': [1]},
                ]
    
    # Parse the arguments
    args = parser.parse_args()

    # Assign the parsed values to variables
    fsave = args.fsave
    N_montecarlo = args.N_montecarlo

    # Check if the folder exists
    if not os.path.exists(fsave):
        # Create the folder
        os.makedirs(fsave)
    file_path = "Dados/comp1.csv"
    df = pd.read_csv(file_path)

    # x_og = np.array([15, 30, 45])
    # y_og = np.array([1.2122381926, 1.2095643282, 1.2078597546]) # Mensagem 1 do Pacheco
    # y_og = np.array([66.1523818970, 66.0512771606, 65.9900207520]) # Mensagem 2 do Pacheco

    x_og = np.array([60, 120, 180]) # Ponto 13 do Pacheco
    y_og = np.array([37.7570381165, 37.6191291809, 37.5241394043]) # Ponto 13 do Pacheco

    conditionsDone = pd.read_csv('Resultados/map_results_embraco3.csv') if APPEND else None

    conditions = generateConditions(baseValues, analysesAlter, x_og, conditionsDone)

    results = montecarlo_analysis(conditions, x_og, y_og)

    if APPEND:
        results.to_csv(fsave + '/map_results_embraco3.csv', mode='a', header=False, index=False)
    else:
        results.to_csv(fsave + '/map_results_embraco3.csv', index=False)