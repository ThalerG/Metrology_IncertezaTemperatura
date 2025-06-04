from fast_montecarlo import res_montecarlo_temp_montecarlo
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    
    n_tests = 5
    fsave = 'Resultados'

    # Check if the folder exists
    if not os.path.exists(fsave):
        # Create the folder
        os.makedirs(fsave)

    saveFile = f'{fsave}/convergence.csv'

    n_montecarlo = [1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6]

    for N in tqdm(n_montecarlo, desc='Monte Carlo Convergence', position=0, leave=False):
        for i in tqdm(range(n_tests), desc='Test', position=1, leave=False):
            tAll, _, _, _ = res_montecarlo_temp_montecarlo(N_montecarlo = int(N))
            tmonte = {'mean_R2': np.mean(tAll['R2']), 'mean_T2': np.mean(tAll['T2']),
                      'std_R2': np.std(tAll['R2']), 'std_T2': np.std(tAll['T2']),
                      'time_elapsed': tAll['time_elapsed'], 'N': int(N)} 
            
            pd.DataFrame([tmonte]).to_csv(saveFile, index=False, mode='a', header=not os.path.exists(saveFile))
            
