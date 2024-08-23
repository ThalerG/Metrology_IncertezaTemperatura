from montecarlo_resistance import res_montecarlo_temp_calc
from montecarlo_temperature_full import res_montecarlo_temp_montecarlo
from tqdm import tqdm
import pandas as pd
import os

if __name__ == '__main__':

    n_tests = 10
    fsave = 'Results'

    # Check if the folder exists
    if not os.path.exists(fsave):
        # Create the folder
        os.makedirs(fsave)

    n_montecarlo = [1e1, 2e1, 5e1, 1e2, 2e2, 5e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5]

    results = []

    for N in tqdm(n_montecarlo, desc='Monte Carlo Convergence', position=0, leave=True):
        for i in tqdm(range(n_tests), desc='Test', position=0, leave=True):
            tcalc, _,_ = res_montecarlo_temp_calc(int(N))
            tcalc['N'] = N
            tcalc['type'] = 'calc'
            tcalc['test'] = i
            results.append(tcalc)

            tmonte, _,_ = res_montecarlo_temp_montecarlo(int(N))
            tmonte['N'] = N
            tmonte['type'] = 'monte carlo'
            tmonte['test'] = i
            results.append(tmonte)

    df = pd.DataFrame(results)
    df.to_csv(f'{fsave}/convergence.csv', index=False)
            