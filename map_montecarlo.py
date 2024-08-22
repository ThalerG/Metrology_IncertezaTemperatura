from fcn import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.express as px
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
parser.add_argument('--PLOTRMSE', action='store_false', help='Flag to plot RMSE')
parser.add_argument('--PLOTSAVE', action='store_false', help='Flag to save plots')
parser.add_argument('--HTMLSAVE', action='store_false', help='Flag to save HTML report')

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

analyses = []

###### Análise 0: Nº de pontos x Tempo entre pontos ######

analyses.append({
    'dt': [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40],
    'Npoints': [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
    't1': [4],
    's_t0': [0.1]
    })

###### Análise 1: Tempo entre pontos x tempo inicial ######

analyses.append({
    'dt': [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40],
    'Npoints': [3],
    't1': [4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40],
    's_t0': [0.1]
    })

###### Análise 2: Tempo inicial x incerteza t0 ######

analyses.append({
    'dt': [8],
    'Npoints': [3],
    't1': [4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40],
    's_t0': [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1e0, 2e0, 5e0, 1e1]
    })

###### Análise 3: Nº de pontos x incerteza t0 ######

analyses.append({
    'dt': [8],
    'Npoints': [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
    't1': [4],
    's_t0': [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1e0, 2e0, 5e0, 1e1]
    })

###### Análise 4: Tempo entre pontos x incerteza t0 ######

analyses.append({
    'dt': [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40],
    'Npoints': [3],
    't1': [4],
    's_t0': [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1e0, 2e0, 5e0, 1e1]
})

def get_log_ticks(min, max, step = 0.5):
    logMin = np.floor(np.log10(min))
    logMax = np.ceil(np.log10(max))
    logticks = np.arange(logMin, logMax + step, step)
    ticks = np.power(10, logticks)
    return ticks, logticks

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
    var_params = [key for key in analysis_params if len(analysis_params[key]) > 1]

    if len(var_params) > 2:
        raise ValueError("Too many variable parameters. Please provide a max of two variable parameter.")
    
    conditions = pd.DataFrame(columns=var_params, data=product(*[analysis_params[key] for key in var_params]))
    for key in analysis_params:
        if key not in var_params:
            conditions[key] = analysis_params[key][0]

    ind = (conditions['t1'] >= x_og[0]) & ((conditions['t1'] + (conditions['Npoints']-1)*conditions['dt']) <= x_og[-1])
    conditions = conditions.loc[ind]

    results_labels = ['mean_SSE', 'mean_Resistance', 'mean_EstimationUncertainty', 'mean_DeltaTemperature', 'mean_Temperature',
                      'std_SSE', 'std_Resistance', 'std_EstimationUncertainty', 'std_DeltaTemperature', 'std_Temperature']
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
        
    results = pd.DataFrame(columns=results_labels, data=results_data)
    results = pd.concat([conditions, results], axis=1)

    return results

def plot_html_results(results:pd.DataFrame, varX:str, varY:str, plot_RMSE:bool = True, log_x = False, log_y = False, log_z = None):
    varNames = {'t1': 'Initial time [s]', 
                'dt': 'Time between measurements [s]', 
                'Npoints': 'Number of measurements', 
                's_t0': 'Initial time uncertainty [s]'}

    html_report = f"Analysis: {varNames[varX]} x {varNames[varY]}</h3>\n"

    html_report += f"<p>Initial time: {results['t1'].unique()}</p>\n"
    html_report += f"<p>Time between measurements: {results['dt'].unique()}</p>\n"
    html_report += f"<p>Number of measurements: {results['Npoints'].unique()}</p>\n"
    html_report += f"<p>Initial time uncertainty: {results['s_t0'].unique()}</p>\n"

    x = results[varX].unique()
    x.sort()
    y = results[varY].unique()
    y.sort()

    z_rmse = np.empty((len(x), len(y)))
    z_r2 = np.empty((len(x), len(y)))
    z_t2 = np.empty((len(x), len(y)))
    z_s_r2 = np.empty((len(x), len(y)))
    z_s_t2 = np.empty((len(x), len(y)))

    for i, x_val in enumerate(x):
        for j, y_val in enumerate(y):
            ind = (results[varX] == x_val) & (results[varY] == y_val)
            if np.sum(ind) == 0:
                z_rmse[i,j] = np.nan
                z_r2[i,j] = np.nan
                z_t2[i,j] = np.nan
                z_s_r2[i,j] = np.nan
                z_s_t2[i,j] = np.nan
            else:
                z_rmse[i,j] = np.sqrt(results.loc[ind, 'mean_SSE'].values[0]/(results.loc[ind, 'Npoints'].values[0])) # SSE to RMSE
                z_r2[i,j] = results.loc[ind, 'mean_Resistance'].values[0]
                z_t2[i,j] = results.loc[ind, 'mean_Temperature'].values[0]
                z_s_r2[i,j] = results.loc[ind, 'std_Resistance'].values[0]
                z_s_t2[i,j] = results.loc[ind, 'std_Temperature'].values[0]

    if plot_RMSE:
        # RMSE plot
        fig = go.Figure(data=[go.Surface(x=x, y=y, z=z_rmse, name='RMSE',showscale=False)])
        fig.update_layout(title='RMSE [Ω]', scene = dict(xaxis_title = varNames[varX], yaxis_title= varNames[varY], zaxis_title='RMSE [Ω]'))
        html_report += fig.to_html(full_html=False)

        if log_x:
            fig.update_xaxes(type='log')
        if log_y:
            fig.update_yaxes(type='log')

    # Create the figure object
    fig = make_subplots(rows=1, cols=2, 
                        specs=[[{"type": "surface"},{"type": "surface"}]],
                        subplot_titles=['Resistance [Ω]', 'Temperature [°C]'])

    fig.add_trace(go.Surface(x=x, y=y, z=z_r2, name='R2',showscale=False), row=1, col=1)
    
    fig.add_trace(go.Surface(x=x, y=y, z=z_t2, name='T2',showscale=False), row=1, col=2)

    fig.update_scenes(xaxis_title=varNames[varX], 
                        yaxis_title=varNames[varY], 
                        zaxis_title='Value')
    
    if log_x:
        fig.update_xaxes(type='log')
    if log_y:
        fig.update_yaxes(type='log')

    # Add the plot to the HTML report
    html_report += fig.to_html(full_html=False)


    # Create the figure object
    fig = make_subplots(rows=1, cols=2, 
                        specs=[[{"type": "surface"},{"type": "surface"}]],
                        subplot_titles=("Resistance uncertainty [Ω]", "Temperature uncertainty [°C]"))

    fig.add_trace(go.Surface(x=x, y=y, z=z_s_r2, name='s_R2', showscale=False), row=1, col=1)
    fig.update_scenes(xaxis_title=varNames[varX], 
                        yaxis_title=varNames[varY],
                        zaxis_title='Value')
    fig.add_trace(go.Surface(x=x, y=y, z=z_s_t2, name='s_T2', showscale=False), row=1, col=2)

    if log_x:
        fig.update_xaxes(type='log')
    if log_y:
        fig.update_yaxes(type='log')

    # Add the plot to the HTML report
    html_report += fig.to_html(full_html=False)

    return html_report
    
def save_heatmap_plots(results:pd.DataFrame, varX:str, varY:str, plot_RMSE:bool = True, fsave:str = "", prefix:str = "", log_x = False, log_y = False, log_z = None):
    varNames = {'t1': 'Initial time [s]',
                'dt': 'Time between measurements [s]',
                'Npoints': 'Number of measurements',
                's_t0': 'Initial time uncertainty [s]'}
    
    if log_z is None:
        log_z = [False, False, False, False, False]
    if not plot_RMSE and len(log_z) == 4:
        log_z = [False] + log_z
    
    x = results[varX].unique()
    x.sort()
    y = results[varY].unique()
    y.sort()

    z_rmse = np.empty((len(x), len(y)))
    z_r2 = np.empty((len(x), len(y)))
    z_t2 = np.empty((len(x), len(y)))
    z_s_r2 = np.empty((len(x), len(y)))
    z_s_t2 = np.empty((len(x), len(y)))

    for i, x_val in enumerate(x):
        for j, y_val in enumerate(y):
            ind = (results[varX] == x_val) & (results[varY] == y_val)
            if np.sum(ind) == 0:
                z_rmse[i,j] = np.nan
                z_r2[i,j] = np.nan
                z_t2[i,j] = np.nan
                z_s_r2[i,j] = np.nan
                z_s_t2[i,j] = np.nan
            else:
                z_rmse[i,j] = np.sqrt(results.loc[ind, 'mean_SSE'].values[0]/(results.loc[ind, 'Npoints'].values[0])) # SSE to RMSE
                z_r2[i,j] = results.loc[ind, 'mean_Resistance'].values[0]
                z_t2[i,j] = results.loc[ind, 'mean_Temperature'].values[0]
                z_s_r2[i,j] = results.loc[ind, 'std_Resistance'].values[0]
                z_s_t2[i,j] = results.loc[ind, 'std_Temperature'].values[0]

    if plot_RMSE:
        if log_z[0]:
            ticks, logticks = get_log_ticks(np.nanmin(z_rmse), np.nanmax(z_rmse))
            text = np.vectorize("{:.1e}".format)(z_rmse).tolist()
            z_rmse = np.log(z_rmse)
            colorbar = dict(title='RMSE [Ω]',
                            tickmode="array",
                            tickvals=ticks,
                            ticktext=logticks.astype(str),
                            ticks="outside")
        else:
            text = np.vectorize("{:.2g}".format)(z_rmse).tolist()
            colorbar = dict(title='RMSE [Ω]')
        heatmap_fig = go.Figure(data=go.Heatmap(x=y, y=x, z=z_rmse.tolist(), colorbar=colorbar, text=text, texttemplate="%{text}"))
        heatmap_fig.update_layout(title='RMSE [Ω]', xaxis_title=varNames[varX], yaxis_title=varNames[varY], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_family = 'Times New Roman', font_color = 'black')
        heatmap_fig.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
        heatmap_fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
        if log_x:
            heatmap_fig.update_xaxes(type='log')
        if log_y:
            heatmap_fig.update_yaxes(type='log')
        heatmap_fig.update_coloraxes(colorscale='Turbo', cmin=np.nanmin(z_rmse), cmax=np.nanmax(z_rmse))
        heatmap_fig.write_image(fsave + '/' + prefix + '_RMSE.pdf')

    if log_z[1]:
        ticks, logticks = get_log_ticks(np.nanmin(z_r2), np.nanmax(z_r2))
        text = np.vectorize("{:.1e}".format)(z_r2).tolist()
        z_r2 = np.log(z_r2)
        colorbar = dict(title='Resistance [Ω]',
                        tickmode="array",
                        tickvals=ticks,
                        ticktext=logticks.astype(str),
                        ticks="outside")
    else:    
        colorbar = dict(title='Resistance [Ω]')
        text = np.vectorize("{:.2g}".format)(z_r2).tolist()
    heatmap_fig = go.Figure(data=go.Heatmap(x=y, y=x, z=z_s_t2, colorbar=colorbar, text=text, texttemplate="%{text}", aspect="auto"))
    heatmap_fig.update_layout(title='Resistance [Ω]', xaxis_title=varNames[varX], yaxis_title=varNames[varY], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_family = 'Times New Roman', font_color = 'black')
    heatmap_fig.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    heatmap_fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    if log_x:
        heatmap_fig.update_xaxes(type='log')
    if log_y:
        heatmap_fig.update_yaxes(type='log')
    heatmap_fig.update_coloraxes(colorscale='Turbo', cmin=np.nanmin(z_r2), cmax=np.nanmax(z_r2))
    heatmap_fig.write_image(fsave + '/' + prefix + '_R2.pdf')

    if log_z[2]:
        ticks, logticks = get_log_ticks(np.nanmin(z_t2), np.nanmax(z_t2))
        text = np.vectorize("{:.1e}".format)(z_t2).tolist()
        z_t2 = np.log(z_t2)
        colorbar = dict(title='Temperature [°C]',
                        tickmode="array",
                        tickvals=ticks,
                        ticktext=logticks.astype(str),
                        ticks="outside")
    else:
        colorbar = dict(title='Temperature [°C]')
        text = np.vectorize("{:.2g}".format)(z_t2).tolist()
    heatmap_fig = go.Figure(data=go.Heatmap(x=y, y=x, z=z_s_t2, colorbar=colorbar, text=text, texttemplate="%{text}", aspect="auto"))
    heatmap_fig.update_layout(title='Temperature [°C]', xaxis_title=varNames[varX], yaxis_title=varNames[varY], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_family = 'Times New Roman', font_color = 'black')
    heatmap_fig.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    heatmap_fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    if log_x:
        heatmap_fig.update_xaxes(type='log')
    if log_y:
        heatmap_fig.update_yaxes(type='log')
    heatmap_fig.update_coloraxes(colorscale='Turbo', cmin=np.nanmin(z_t2), cmax=np.nanmax(z_t2))
    heatmap_fig.write_image(fsave + '/' + prefix + '_T2.pdf')

    if log_z[3]:
        ticks, logticks = get_log_ticks(np.nanmin(z_s_r2), np.nanmax(z_s_r2))
        text = np.vectorize("{:.1e}".format)(z_s_r2).tolist()
        z_s_r2 = np.log(z_s_r2)
        colorbar = dict(title='Resistance uncertainty [Ω]',
                        tickmode="array",
                        tickvals=ticks,
                        ticktext=logticks.astype(str),
                        ticks="outside")
    else:
        colorbar = dict(title='Resistance uncertainty [Ω]')
        text = np.vectorize("{:.2g}".format)(z_s_r2).tolist()
    heatmap_fig = go.Figure(data=go.Heatmap(x=y, y=x, z=z_s_t2, colorbar=colorbar, text=text, texttemplate="%{text}", aspect="auto"))
    heatmap_fig.update_layout(title='Resistance uncertainty [Ω]', xaxis_title=varNames[varX], yaxis_title=varNames[varY], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_family = 'Times New Roman', font_color = 'black')
    heatmap_fig.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    heatmap_fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    if log_x:
        heatmap_fig.update_xaxes(type='log')
    if log_y:
        heatmap_fig.update_yaxes(type='log')
    heatmap_fig.update_coloraxes(colorscale='Turbo', cmin=np.nanmin(z_s_r2), cmax=np.nanmax(z_s_r2))
    heatmap_fig.write_image(fsave + '/' + prefix + '_sR2.pdf')
    
    if log_z[4]:
        ticks, logticks = get_log_ticks(np.nanmin(z_s_t2), np.nanmax(z_s_t2))
        text = np.vectorize("{:.1e}".format)(z_s_t2).tolist()
        z_s_t2 = np.log10(z_s_t2)
        colorbar = dict(title='Temperature uncertainty [°C]',
                        tickmode="array",
                        tickvals=ticks,
                        ticktext=logticks.astype(str),
                        ticks="outside")
    else:
        colorbar = dict(title='Temperature uncertainty [°C]')
        text = np.vectorize("{:.2g}".format)(z_s_t2).tolist()
    heatmap_fig = go.Figure(data=go.Heatmap(x=y, y=x, z=z_s_t2, colorbar=colorbar, text=text, texttemplate="%{text}", aspect="auto"))
    heatmap_fig.update_layout(title='Temperature uncertainty [°C]', xaxis_title=varNames[varX], yaxis_title=varNames[varY], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_family = 'Times New Roman', font_color = 'black')
    heatmap_fig.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    heatmap_fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    if log_x:
        heatmap_fig.update_xaxes(type='log')
    if log_y:
        heatmap_fig.update_yaxes(type='log')
    heatmap_fig.update_coloraxes(colorscale='Turbo', cmin=np.nanmin(z_s_t2), cmax=np.nanmax(z_s_t2))
    heatmap_fig.write_image(fsave + '/' + prefix + '_sT2.pdf')

if __name__ == '__main__':
    # Parse the arguments
    args = parser.parse_args()

    # Assign the parsed values to variables
    fsave = args.fsave
    N_montecarlo = args.N_montecarlo
    PLOTRMSE = args.PLOTRMSE
    PLOTSAVE = args.PLOTSAVE
    HTMLSAVE = args.HTMLSAVE

    # Check if the folder exists
    if not os.path.exists(fsave):
        # Create the folder
        os.makedirs(fsave)
    file_path = "Dados/data.csv"
    df = pd.read_csv(file_path)

    model = ('exp',0)

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

    allresults = []
    for an in tqdm.tqdm(analyses, desc = 'Analysis', position=0, leave = True):
        results = montecarlo_analysis(an, x_og, y_og)
        var_params = [key for key in an if len(an[key]) > 1]

        html_report += plot_html_results(results, var_params[0], var_params[1], plot_RMSE=PLOTRMSE)
        
        if PLOTSAVE:
            log_x = True if var_params[0] == 's_t0' else False
            log_y = True if var_params[1] == 's_t0' else False
            log_z = [False, False, False, True, True] if var_params[0] == 's_t0' or var_params[1] == 's_t0' else None
            save_heatmap_plots(results, var_params[0], var_params[1], plot_RMSE=PLOTRMSE, fsave=fsave, prefix=f"An_{var_params[0]}_{var_params[1]}", log_x = False, log_y = False, log_z = None)
        
        allresults.append(results)
        
    if HTMLSAVE:
        # Save the HTML report to a file
        with open(fsave + "/report_Map_Montecarlo.html", "w", encoding="utf-16") as file:
            file.write(html_report)

    allresults = pd.concat(allresults)
    allresults.to_csv(fsave + '/results.csv', index=False)