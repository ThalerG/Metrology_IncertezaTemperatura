from montecarlo_temperature import res_montecarlo_temp_montecarlo
from tqdm import tqdm
import pandas as pd
import os

if __name__ == '__main__':

    n_tests = 100
    fsave = 'Resultados'

    # Check if the folder exists
    if not os.path.exists(fsave):
        # Create the folder
        os.makedirs(fsave)

    n_montecarlo = [1e1, 2e1, 3e1, 4e1, 5e1, 6e1, 7e1, 8e1, 9e1, 1e2, 2e2, 3e2, 4e2, 5e2, 6e2, 7e2, 8e2, 9e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5]

    results = []

    for N in tqdm(n_montecarlo, desc='Monte Carlo Convergence', position=0, leave=False):
        for i in tqdm(range(n_tests), desc='Test', position=1, leave=False):
            tmonte, _,_ = res_montecarlo_temp_montecarlo(int(N), parallel=True)
            tmonte['N'] = N
            tmonte['test'] = i
            del tmonte['parameters']
            results.append(tmonte)

    df = pd.DataFrame(results)
    df.to_csv(f'{fsave}/convergence.csv', index=False)
            