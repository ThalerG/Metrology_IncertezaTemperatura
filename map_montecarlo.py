from fcn import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import tqdm
import functools
from multiprocessing import Pool
from itertools import product
import os

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

# Número de análises comparativas

n_analyses = 3

###### Análise 0: Nº de pontos x Distância entre pontos ######

an0_dT = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]
an0_Npoints = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
an0_t1 = 4

###### Análise 1: Distância entre pontos x tempo inicial ######

an1_t1 = [2,4,6,8,10]
an1_dT = [2,4,6,8,10]
an1_Npoints = 4

###### Análise 2: Tempo inicial x incerteza t0 ######

an2_Npoints = 4
an2_dt = 8
an2_t1 = [2, 4, 6, 8, 10, 12, 14, 16]
an2_s_t0 = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]

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
        raise Warning("Not all elements of the desired configuration are present in the provided time data. Proceeding with the available data.")

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

if __name__ == '__main__':
    file_path = "Dados/data.csv"
    df = pd.read_csv(file_path)

    model = ('exp',0)

    N_montecarlo = 200 # Número de simulações de Monte Carlo 

    x_og = df['Time'].values
    y_og = df['Resistance'].values

    # Generate HTML report
    html_report = "<html>\n<head>\n<title>Analysis Report</title>\n</head>\n<body>\n"

    # Test Variables
    html_report += "<h2>Test Variables</h2>\n"
    html_report += "<table>\n"
    html_report += "<tr>\n"
    html_report += "<th>Variable</th>\n"
    html_report += "<th>Value</th>\n"
    html_report += "<th>Unit</th>\n"
    html_report += "<th>Uncertainty</th>\n"
    html_report += "<th>Symbol</th>\n"  # Add a column for the symbol
    html_report += "</tr>\n"
    html_report += f"<tr><td>Resistance at the beginning of the test</td><td>{R1}</td><td>Ω</td><td>{s_dR}</td><td>R<sub>1</sub></td></tr>\n"
    html_report += f"<tr><td>Resistance at the end of the test</td><td>Monte Carlo mean</td><td>Ω</td><td>Monte Carlo deviation</td><td>R<sub>2</sub></td></tr>\n"
    html_report += f"<tr><td>Ambient temperature at the beginning of the test</td><td>{Tamb_1}</td><td>°C</td><td>{s_Tamb1}</td><td>T<sub>amb,1</sub></td></tr>\n"
    html_report += f"<tr><td>Ambient temperature at the end of the test</td><td>{Tamb_2}</td><td>°C</td><td>{s_Tamb2}</td><td>T<sub>amb,2</sub></td></tr>\n"
    html_report += f"<tr><td>Reciprocal of the coefficient of temperature</td><td>{k}</td><td>°C</td><td>-</td><td>k</td></tr>\n"
    html_report += f"<tr><td>Test temperature difference</td><td> Calculated </td><td>°C</td><td> Calculated </td><td>ΔT</sub></td></tr>\n"
    html_report += f"<tr><td>Winding temperature at the end of the test</td><td> Calculated </td><td>°C</td><td> Calculated </td><td>T<sub>2</sub></td></tr>\n"
    html_report += "</table>\n"

    html_report += "<p>Other variables:</p>\n"

    html_report += f"<li><td>Uncertainty of initial time μ<sub>t<sub>0</sub></sub> (s): {s_t0}</li>\n"
    html_report += f"<li><td>Uncertainty of acquisition time μ<sub>dt</sub> (s): {s_dt}</li>\n"
    html_report += f"<li><td>Uncertainty of resistance measurement μ<sub>dR</sub> (Ω): {s_dR}</li>\n"
    html_report += f"<li>Number of Monte Carlo simulations: {N_montecarlo}</li>\n"
    html_report += "</ul>\n\n"

    # Equations
    html_report += "<h2>Original equation</h2>\n"
    html_report += r'<img src="https://latex.codecogs.com/svg.image?\Delta&space;t=\frac{(R_2-R_1)}{R_1}(k&plus;T_{amb,1})-(T_{amb,2}-T_{amb,1})" title="\Delta T=\frac{(R_2-R_1)}{R_1}(k+T_{amb,1})-(T_{amb,2}-T_{amb,1})" />' + "<br>\n"
    html_report += r'<img src="https://latex.codecogs.com/svg.image?T_2=T_{amb,1}&plus;\Delta&space;T&space;" title="T_2=T_{amb,1}+\Delta T" />' + "<br>\n"

    html_report += "<h2>Individual uncertainty</h2>\n"
    html_report += '<div style="display: flex;">\n'
    html_report += '<div style="flex: 50%; padding: 5px;">\n'
    html_report += r'<img src="https://latex.codecogs.com/svg.image?\frac{\partial\Delta&space;T}{\partial&space;T_{amb,1}}=\frac{(R_2-R1)}{R_1}&plus;1&space;" title="\frac{\partial \Delta T}{\partial T_{amb,1}}=\frac{(R_2-R1)}{R_1}+1"/>' + "<br>\n"
    html_report += r'<img src="https://latex.codecogs.com/svg.image?\frac{\partial\Delta&space;T}{\partial&space;T_{amb,2}}=-1&space;" title="\frac{\partial\Delta T}{\partial T_{amb,2}}=-1"/>' + "<br>\n"
    html_report += '</div>\n'
    html_report += '<div style="flex: 50%; padding: 5px;">\n'
    html_report += r'<img src="https://latex.codecogs.com/svg.image?\frac{\partial\Delta&space;T}{\partial&space;R_1}=\frac{-R_2}{R_1^2}(k&plus;T_{amb,1})" title="\frac{\partial\Delta T}{\partial R_1}=\frac{-R_2}{R_1^2}(k+T_{amb,1})"/>' + "<br>\n"
    html_report += r'<img src="https://latex.codecogs.com/svg.image?\frac{\partial\Delta&space;T}{\partial&space;R_2}=\frac{(k&plus;T_{amb,1})}{R_1}" title="\frac{\partial\Delta T}{\partial R_2}=\frac{(k+T_{amb,1})}{R_1}"/>'+ "<br>\n"
    html_report += '</div>\n'
    html_report += '</div>\n'

    html_report += "<p>For T<sub>2</sub>, the only difference is the partial derivative relative to T<sub>amb,1</sub>:</p>\n"
    html_report += r'<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;T_2}{\partial&space;T_{amb,1}}=\frac{(R_2-R_1)}{R_1}&plus;2&space;" title="\frac{\partial T_2}{\partial T_{amb,1}}=\frac{(R_2-R_1)}{R_1}+2 " />' + "<br>\n"

    html_report += f"<h2>Model comparison</h2>\n"

    n_jobs = os.cpu_count()

    for an in tqdm.tqdm(range(n_analyses), desc = 'Model', position=0):
        if an == 0: ###### Análise 0: Nº de pontos x Distância entre pontos ######
            conditions = product(an0_dT, an0_Npoints)
            conditions = [condition for condition in conditions if condition[0]*condition[1] <= x_og[-1]]

            df_values = pd.DataFrame(columns=['dT','N_points','SSE', 'Resistance', 'Estimation uncertainty', 'Delta Temperature', 'Temperature'])
            df_stdvalues = pd.DataFrame(columns=['dT','N_points','SSE', 'Resistance', 'Estimation uncertainty', 'Delta Temperature','Temperature'])
            html_report += f"<h3>Analysis 0: Number of measurements x Time between measurements</h3>\n"

            for (ind,condition) in enumerate(tqdm.tqdm(conditions, desc = 'Condition', position=1)):

                dT = condition[0]
                Npoints = condition[1]

                montecarlo_matrix_xy = generate_montecarlo_matrix(x_og, y_og, s_x, s_y, s_t0 = s_t0, t1 = an0_t1, dt = dT, n_x = Npoints, N_montecarlo = N_montecarlo)

                with Pool(n_jobs) as p:
                    results_model = p.map(functools.partial(process_montecarlo, model=model, s_x=s_x, s_y=s_y), montecarlo_matrix_xy)

                # Calculate the mean values of R2, s_R2, T2, and s_T2 from results_model
                mean_R2 = np.mean([result['R2'] for result in results_model])
                mean_s_R2 = np.mean([result['s_R2'] for result in results_model])
                sum_square = np.mean([result['result'].sum_square for result in results_model])

                DT = delta_temperature(R1, mean_R2, Tamb_1, Tamb_2, k)
                T2 = final_temperature(R1, mean_R2, Tamb_1, Tamb_2, k)

                # Calculate the standard deviation of R2, s_R2, T2, and s_T2 from results_model
                std_R2 = np.std([result['R2'] for result in results_model])
                std_s_R2 = np.std([result['s_R2'] for result in results_model])
                std_sum_square = np.std([result['result'].sum_square for result in results_model])

                s_DT = delta_temperature_uncertainty(R1, mean_R2, Tamb_1, Tamb_2, k, s_R1, std_R2, s_Tamb1, s_Tamb2)
                s_T2 = final_temperature_uncertainty(R1, mean_R2, Tamb_1, Tamb_2, k, s_R1, std_R2, s_Tamb1, s_Tamb2)

                # Add the mean values to the dataframe
                df_values.loc[ind] = [dT, Npoints, sum_square, mean_R2, mean_s_R2, DT, T2]

                # Add the standard deviation values to the dataframe
                df_stdvalues.loc[ind] = [dT, Npoints, std_sum_square, std_R2, std_s_R2, s_DT, s_T2]

            x = df_values['dT'].unique()
            y = df_values['N_points'].unique()

            z_sse = np.empty((len(y), len(x)))
            z_r2 = np.empty((len(y), len(x)))
            z_t2 = np.empty((len(y), len(x)))
            z_s_r2 = np.empty((len(y), len(x)))
            z_s_t2 = np.empty((len(y), len(x)))

            for i, dT in enumerate(x):
                for j, Npoints in enumerate(y):
                    row = df_values[(df_values['dT'] == dT) & (df_values['N_points'] == Npoints)]
                    row_s = df_stdvalues[(df_stdvalues['dT'] == dT) & (df_stdvalues['N_points'] == Npoints)]
                    if len(row) > 0:
                        z_sse[j, i] = row['SSE'].values[0]
                        z_r2[j, i] = row['Resistance'].values[0]
                        z_t2[j, i] = row['Temperature'].values[0]
                        z_s_r2[j, i] = row_s['Resistance'].values[0]
                        z_s_t2[j, i] = row_s['Temperature'].values[0]
                    else:
                        z_sse[j, i] = np.nan
                        z_r2[j, i] = np.nan
                        z_t2[j, i] = np.nan
                        z_s_r2[j, i] = np.nan
                        z_s_t2[j, i] = np.nan

            # Create the figure object
            fig = make_subplots(rows=1, cols=3, 
                                specs=[[{"type": "surface"},{"type": "surface"},{"type": "surface"}]],
                                subplot_titles=['SSE [Ω²]', 'Resistance [Ω]', 'Temperature [°C]'])

            fig.add_trace(go.Surface(x=x, y=y, z=z_sse, name='SSE'), row=1, col=1)
            fig.update_scenes(xaxis_title='dT [s]', 
                              yaxis_title='N_points', 
                              zaxis_title='Value')

            fig.add_trace(go.Surface(x=x, y=y, z=z_r2, name='R2'), row=1, col=2)

            fig.add_trace(go.Surface(x=x, y=y, z=z_t2, name='T2'), row=1, col=3)

            # Add the plot to the HTML report
            html_report += fig.to_html(full_html=False)


            # Create the figure object
            fig = make_subplots(rows=1, cols=2, 
                                specs=[[{"type": "surface"},{"type": "surface"}]],
                                subplot_titles=("Resistance uncertainty [Ω]", "Temperature uncertainty [°C]"))

            fig.add_trace(go.Surface(x=x, y=y, z=z_s_r2, name='s_R2'), row=1, col=1)
            fig.update_scenes(xaxis_title='dT [s]', 
                              yaxis_title='N_points', 
                              zaxis_title='Value')
            fig.add_trace(go.Surface(x=x, y=y, z=z_s_t2, name='s_T2'), row=1, col=2)

            # Add the plot to the HTML report
            html_report += fig.to_html(full_html=False)
            
        elif an == 1: ###### Análise 1: Distância entre pontos x tempo inicial ######
            conditions = product(an1_t1, an1_dT)
            conditions = [condition for condition in conditions if condition[0] + condition[1]*(an1_Npoints-1) <= x_og[-1]]

            df_values = pd.DataFrame(columns=['t1','dT','SSE', 'Resistance', 'Estimation uncertainty', 'Delta Temperature', 'Temperature'])
            df_stdvalues = pd.DataFrame(columns=['t1','dT','SSE', 'Resistance', 'Estimation uncertainty', 'Delta Temperature','Temperature'])

            for (ind,condition) in enumerate(tqdm.tqdm(conditions, desc = 'Condition', position=1)):
                html_report += f"<h3>Analysis 1: Time between measurements x Initial time</h3>\n"

                t1 = condition[0]
                dT = condition[1]

                montecarlo_matrix_xy = generate_montecarlo_matrix(x_og, y_og, s_x, s_y, s_t0 = s_t0, t1 = t1, dt = dT, n_x = an1_Npoints, N_montecarlo = N_montecarlo)

                with Pool(n_jobs) as p:
                    results_model = p.map(functools.partial(process_montecarlo, model=model, s_x=s_x, s_y=s_y), montecarlo_matrix_xy)

                # Calculate the mean values of R2, s_R2, T2, and s_T2 from results_model
                mean_R2 = np.mean([result['R2'] for result in results_model])
                mean_s_R2 = np.mean([result['s_R2'] for result in results_model])
                sum_square = np.mean([result['result'].sum_square for result in results_model])

                DT = delta_temperature(R1, mean_R2, Tamb_1, Tamb_2, k)
                T2 = final_temperature(R1, mean_R2, Tamb_1, Tamb_2, k)

                # Calculate the standard deviation of R2, s_R2, T2, and s_T2 from results_model
                std_R2 = np.std([result['R2'] for result in results_model])
                std_s_R2 = np.std([result['s_R2'] for result in results_model])
                std_sum_square = np.std([result['result'].sum_square for result in results_model])

                s_DT = delta_temperature_uncertainty(R1, mean_R2, Tamb_1, Tamb_2, k, s_R1, std_R2, s_Tamb1, s_Tamb2)
                s_T2 = final_temperature_uncertainty(R1, mean_R2, Tamb_1, Tamb_2, k, s_R1, std_R2, s_Tamb1, s_Tamb2)

                # Add the mean values to the dataframe
                df_values.loc[ind] = [t1, dT, sum_square, mean_R2, mean_s_R2, DT, T2]

                # Add the standard deviation
                df_stdvalues.loc[ind] = [t1, dT, std_sum_square, std_R2, std_s_R2, s_DT, s_T2]

            x = df_values['dT'].unique()
            y = df_values['t1'].unique()

            z_sse = np.empty((len(y), len(x)))
            z_r2 = np.empty((len(y), len(x)))
            z_t2 = np.empty((len(y), len(x)))
            z_s_r2 = np.empty((len(y), len(x)))
            z_s_t2 = np.empty((len(y), len(x)))

            for i, dT in enumerate(x):
                for j, t1 in enumerate(y):
                    row = df_values[(df_values['dT'] == dT) & (df_values['t1'] == t1)]
                    row_s = df_stdvalues[(df_stdvalues['dT'] == dT) & (df_stdvalues['t1'] == t1)]
                    if len(row) > 0:
                        z_sse[j, i] = row['SSE'].values[0]
                        z_r2[j, i] = row['Resistance'].values[0]
                        z_t2[j, i] = row['Temperature'].values[0]
                        z_s_r2[j, i] = row_s['Resistance'].values[0]
                        z_s_t2[j, i] = row_s['Temperature'].values[0]
                    else:
                        z_sse[j, i] = np.nan
                        z_r2[j, i] = np.nan
                        z_t2[j, i] = np.nan
                        z_s_r2[j, i] = np.nan
                        z_s_t2[j, i] = np.nan

            # Create the figure object
            fig = make_subplots(rows=1, cols=3, 
                                specs=[[{"type": "surface"},{"type": "surface"},{"type": "surface"}]],
                                subplot_titles=['SSE [Ω²]', 'Resistance [Ω]', 'Temperature [°C]'])

            fig.add_trace(go.Surface(x=x, y=y, z=z_sse, name='SSE'), row=1, col=1)
            fig.update_scenes(xaxis_title='dT [s]', 
                              yaxis_title='t1 [s]', 
                              zaxis_title='Value')

            fig.add_trace(go.Surface(x=x, y=y, z=z_r2, name='R2'), row=1, col=2)

            fig.add_trace(go.Surface(x=x, y=y, z=z_t2, name='T2'), row=1, col=3)

            # Add the plot to the HTML report
            html_report += fig.to_html(full_html=False)

            # Create the figure object
            fig = make_subplots(rows=1, cols=2, 
                                specs=[[{"type": "surface"},{"type": "surface"}]],
                                subplot_titles=("Resistance uncertainty [Ω]", "Temperature uncertainty [°C]"))

            fig.add_trace(go.Surface(x=x, y=y, z=z_s_r2, name='s_R2'), row=1, col=1)
            fig.update_scenes(xaxis_title='dT [s]', 
                              yaxis_title='t1 [s]', 
                              zaxis_title='Value')
            fig.add_trace(go.Surface(x=x, y=y, z=z_s_t2, name='s_T2'), row=1, col=2)

            # Add the plot to the HTML report
            html_report += fig.to_html(full_html=False)

        elif an == 2: ###### Análise 2: Tempo inicial x incerteza t0 ######
            conditions = product(an2_t1, an2_s_t0)
            conditions = [condition for condition in conditions if condition[0]+an2_dt*(an2_Npoints -1) <= x_og[-1]]

            df_values = pd.DataFrame(columns=['t1','s_t0','SSE', 'Resistance', 'Estimation uncertainty', 'Delta Temperature', 'Temperature'])
            df_stdvalues = pd.DataFrame(columns=['t1','s_t0','SSE', 'Resistance', 'Estimation uncertainty', 'Delta Temperature','Temperature'])

            for (ind,condition) in enumerate(tqdm.tqdm(conditions, desc = 'Condition', position=1)):
                html_report += f"<h3>Analysis 2: Initial time x Uncertainty of initial time</h3>\n"

                t1 = condition[0]
                s_t0 = condition[1]

                montecarlo_matrix_xy = generate_montecarlo_matrix(x_og, y_og, s_x, s_y, s_t0 = s_t0, t1 = t1, dt = an2_dt, n_x = an2_Npoints, N_montecarlo = N_montecarlo)

                with Pool(n_jobs) as p:
                    results_model = p.map(functools.partial(process_montecarlo, model=model, s_x=s_x, s_y=s_y), montecarlo_matrix_xy)

                # Calculate the mean values of R2, s_R2, T2, and s_T2 from results_model
                mean_R2 = np.mean([result['R2'] for result in results_model])
                mean_s_R2 = np.mean([result['s_R2'] for result in results_model])
                sum_square = np.mean([result['result'].sum_square for result in results_model])

                DT = delta_temperature(R1, mean_R2, Tamb_1, Tamb_2, k)
                T2 = final_temperature(R1, mean_R2, Tamb_1, Tamb_2, k)

                # Calculate the standard deviation of R2, s_R2, T2, and s_T2 from results_model
                std_R2 = np.std([result['R2'] for result in results_model])
                std_s_R2 = np.std([result['s_R2'] for result in results_model])
                std_sum_square = np.std([result['result'].sum_square for result in results_model])

                s_DT = delta_temperature_uncertainty(R1, mean_R2, Tamb_1, Tamb_2, k, s_R1, std_R2, s_Tamb1, s_Tamb2)
                s_T2 = final_temperature_uncertainty(R1, mean_R2, Tamb_1, Tamb_2, k, s_R1, std_R2, s_Tamb1, s_Tamb2)

                # Add the mean values to the dataframe
                df_values.loc[ind] = [t1, s_t0, sum_square, mean_R2, mean_s_R2, DT, T2]

                # Add the standard deviation
                df_stdvalues.loc[ind] = [t1, s_t0, std_sum_square, std_R2, std_s_R2, s_DT, s_T2]

            x = df_values['dT'].unique()
            y = df_values['s_t0'].unique()

            z_sse = np.empty((len(y), len(x)))
            z_r2 = np.empty((len(y), len(x)))
            z_t2 = np.empty((len(y), len(x)))
            z_s_r2 = np.empty((len(y), len(x)))
            z_s_t2 = np.empty((len(y), len(x)))

            for i, dT in enumerate(x):
                for j, s_t0 in enumerate(y):
                    row = df_values[(df_values['dT'] == dT) & (df_values['s_t0'] == s_t0)]
                    row_s = df_stdvalues[(df_stdvalues['dT'] == dT) & (df_stdvalues['s_t0'] == s_t0)]
                    if len(row) > 0:
                        z_sse[j, i] = row['SSE'].values[0]
                        z_r2[j, i] = row['Resistance'].values[0]
                        z_t2[j, i] = row['Temperature'].values[0]
                        z_s_r2[j, i] = row_s['Resistance'].values[0]
                        z_s_t2[j, i] = row_s['Temperature'].values[0]
                    else:
                        z_sse[j, i] = np.nan
                        z_r2[j, i] = np.nan
                        z_t2[j, i] = np.nan
                        z_s_r2[j, i] = np.nan
                        z_s_t2[j, i] = np.nan

            # Create the figure object
            fig = make_subplots(rows=1, cols=3, 
                                specs=[[{"type": "surface"},{"type": "surface"},{"type": "surface"}]],
                                subplot_titles=['SSE [Ω²]', 'Resistance [Ω]', 'Temperature [°C]'])

            fig.add_trace(go.Surface(x=x, y=y, z=z_sse, name='SSE'), row=1, col=1)
            fig.update_scenes(xaxis_title='dT [s]', 
                              yaxis_title='s_t0 [s]', 
                              zaxis_title='Value')

            fig.add_trace(go.Surface(x=x, y=y, z=z_r2, name='R2'), row=1, col=2)

            fig.add_trace(go.Surface(x=x, y=y, z=z_t2, name='T2'), row=1, col=3)

            # Add the plot to the HTML report
            html_report += fig.to_html(full_html=False)

            # Create the figure object
            fig = make_subplots(rows=1, cols=2, 
                                specs=[[{"type": "surface"},{"type": "surface"}]],
                                subplot_titles=("Resistance uncertainty [Ω]", "Temperature uncertainty [°C]"))

            fig.add_trace(go.Surface(x=x, y=y, z=z_s_r2, name='s_R2'), row=1, col=1)
            fig.update_scenes(xaxis_title='dT [s]', 
                              yaxis_title='s_t0 [s]', 
                              zaxis_title='Value')
            fig.add_trace(go.Surface(x=x, y=y, z=z_s_t2, name='s_T2'), row=1, col=2)

            # Add the plot to the HTML report
            html_report += fig.to_html(full_html=False)

    # Save the HTML report to a file
    with open("report_Map_Montecarlo.html", "w", encoding="utf-16") as file:
        file.write(html_report)