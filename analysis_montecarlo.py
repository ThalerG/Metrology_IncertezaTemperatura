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
import os

# Incertezas de medição:

s_t0 = 1 # Incerteza do tempo inicial

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

if __name__ == '__main__':
    file_path = "Dados/data.csv"
    df = pd.read_csv(file_path)

    models = [('poly',1),
            ('poly',2),
            ('poly',3),
            ('poly',4),
            ('exp',0)
            ]

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

    df_values = pd.DataFrame(columns=['Model Type', 'Degree', 'SSE', 'Resistance', 'Estimation uncertainty', 'Delta Temperature', 'Temperature'])
    df_stdvalues = pd.DataFrame(columns=['Model Type', 'Degree', 'SSE', 'Resistance', 'Estimation uncertainty', 'Delta Temperature','Temperature'])

    results = []
    results_allModels = []

    # Monte Carlo simulation for deviation of sample time
    montecarlo_matrix_x = np.random.normal(0, s_x, (len(x_og), N_montecarlo))
    montecarlo_matrix_x = x_og[:, np.newaxis] + montecarlo_matrix_x

    # Monte Carlo simulation for deviation of initial time
    montecarlo_t0 = np.random.normal(0, s_t0, (1, N_montecarlo))
    montecarlo_matrix_x = montecarlo_matrix_x + montecarlo_t0

    # Monte Carlo simulation for deviation of resistance measurement
    montecarlo_matrix_y = np.random.normal(0, s_y, (len(y_og), N_montecarlo))
    montecarlo_matrix_y = y_og[:, np.newaxis] + montecarlo_matrix_y

    montecarlo_matrix_xy = list(zip(montecarlo_matrix_x.T, montecarlo_matrix_y.T))

    n_jobs = os.cpu_count()

    for (k,model) in enumerate(tqdm.tqdm(models, desc = 'Model', position=0)):
        with Pool(n_jobs) as p:
            results_model = p.map(functools.partial(process_montecarlo, model=model, s_x=s_x, s_y=s_y), montecarlo_matrix_xy)

        results_model = {'type': model[0], 'degree':model[1], 'results': results_model}
        results_allModels.append(results_model)


        # Calculate the mean values of R2, s_R2, T2, and s_T2 from results_model
        mean_R2 = np.mean([result['R2'] for result in results_model['results']])
        mean_s_R2 = np.mean([result['s_R2'] for result in results_model['results']])
        sum_square = np.mean([result['result'].sum_square for result in results_model['results']])

        DT = delta_temperature(R1, mean_R2, Tamb_1, Tamb_2, k)
        T2 = final_temperature(R1, mean_R2, Tamb_1, Tamb_2, k)

        # Calculate the standard deviation of R2, s_R2, T2, and s_T2 from results_model
        std_R2 = np.std([result['R2'] for result in results_model['results']])
        std_s_R2 = np.std([result['s_R2'] for result in results_model['results']])
        std_sum_square = np.std([result['result'].sum_square for result in results_model['results']])

        s_DT = delta_temperature_uncertainty(R1, mean_R2, Tamb_1, Tamb_2, k, s_R1, std_R2, s_Tamb1, s_Tamb2)
        s_T2 = final_temperature_uncertainty(R1, mean_R2, Tamb_1, Tamb_2, k, s_R1, std_R2, s_Tamb1, s_Tamb2)

        # Add the mean values to the dataframe
        df_values.loc[k] = [model[0], model[1], sum_square, mean_R2, mean_s_R2, DT, T2]

        # Add the standard deviation values to the dataframe
        df_stdvalues.loc[k] = [model[0], model[1], std_sum_square, std_R2, std_s_R2, s_DT, s_T2]

    html_report += f"<h2>Model comparison</h2>\n"

    # Create a dataframe with R2 values and corresponding model type and degree
    df_R2 = pd.DataFrame({'R2': [result['R2'] for model_result in results_allModels for result in model_result['results']], 'Model Type': [model_result['type'] for model_result in results_allModels for _ in model_result['results']], 'Degree': [model_result['degree'] for model_result in results_allModels for _ in model_result['results']]})

    for k in range(len(df_R2)):
        df_R2.loc[k, 'Model Type'] = f"{df_R2.loc[k, 'Model Type']} {df_R2.loc[k, 'Degree']}" if df_R2.loc[k, 'Model Type'] == 'poly' else df_R2.loc[k, 'Model Type']

    # Create a violin plot
    fig = px.violin(df_R2, x='Model Type', y='R2', color='Model Type')

    # Update layout
    fig.update_layout(xaxis_title='Model Type and Degree', yaxis_title='Resistance [Ω]', title='Distribution of estimated resistance at the end of the test')

    # Add the violin plot to the HTML report
    html_report += fig.to_html(full_html=False)

    # Create the figure object
    fig = go.Figure()
    
    # Plot the Gaussian curve for each model
    for i, row in df_values.iterrows():
        model_type = row['Model Type']
        degree = row['Degree']
        mean_DT = row['Delta Temperature']
        std_DT = df_stdvalues.loc[i, 'Delta Temperature']

        # Create the x-axis values for the Gaussian curve
        x = np.linspace(mean_DT - 3 * std_DT, mean_DT + 3 * std_DT, 100)

        # Calculate the y-axis values for the Gaussian curve
        y = 1 / (std_DT * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean_DT) / std_DT) ** 2)

        # Add the trace to the figure
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'{model_type} {degree}'))

    # Update layout
    fig.update_layout(xaxis_title='Temperature [°C]', yaxis_title='Probability Density', showlegend=True, title='Probability distribution for ΔT')

    # Add the plot to the HTML report
    html_report += fig.to_html(full_html=False)

    # Create the figure object
    fig = go.Figure()

    # Plot the Gaussian curve for each model
    for i, row in df_values.iterrows():
        model_type = row['Model Type']
        degree = row['Degree']
        mean_T2 = row['Temperature']
        std_T2 = df_stdvalues.loc[i, 'Temperature']

        # Create the x-axis values for the Gaussian curve
        x = np.linspace(mean_T2 - 3 * std_T2, mean_T2 + 3 * std_T2, 100)

        # Calculate the y-axis values for the Gaussian curve
        y = 1 / (std_T2 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean_T2) / std_T2) ** 2)

        # Add the trace to the figure
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'{model_type} {degree}'))

    # Update layout
    fig.update_layout(xaxis_title='Temperature [°C]', yaxis_title='Probability Density', showlegend=True, title='Probability distribution for T2')

    # Add the plot to the HTML report
    html_report += fig.to_html(full_html=False)

    # Add the dataframe as a table in the html file
    html_report += "<h2>Mean and Std values</h2>\n"
    html_report += "<table class='centered'>\n"
    html_report += "<tr>\n"
    html_report += "<th>Model Type</th>\n"
    html_report += "<th>Degree</th>\n"
    html_report += "<th>SSE [Ω²]</th>\n"
    html_report += "<th>Resistance [Ω]</th>\n"
    html_report += "<th>Resistance estimation uncertainty [Ω]</th>\n"
    html_report += "<th>Delta temperature [°C]</th>\n"
    html_report += "<th>Final temperature [°C]</th>\n"
    html_report += "</tr>\n"

    for i in range(len(df_values)):
        html_report += "<tr>\n"
        html_report += f"<td>{df_values.loc[i, 'Model Type']}</td>\n"
        html_report += f"<td>{df_values.loc[i, 'Degree']}</td>\n"
        html_report += f"<td>{df_values.loc[i, 'SSE']:.5g} ± {df_stdvalues.loc[i, 'SSE']:.5g}</td>\n"
        html_report += f"<td>{df_values.loc[i, 'Resistance']:.5g} ± {df_stdvalues.loc[i, 'Resistance']:.5g}</td>\n"
        html_report += f"<td>{df_values.loc[i, 'Estimation uncertainty']:.5g} ± {df_stdvalues.loc[i, 'Estimation uncertainty']:.5g}</td>\n"
        html_report += f"<td>{df_values.loc[i, 'Delta Temperature']:.5g} ± {df_stdvalues.loc[i, 'Delta Temperature']:.5g}</td>\n"
        html_report += f"<td>{df_values.loc[i, 'Temperature']:.5g} ± {df_stdvalues.loc[i, 'Temperature']:.5g}</td>\n"
        html_report += "</tr>\n"

    html_report += "</table>\n"
    html_report += "<style>\n"
    html_report += "table {border-collapse: collapse; width: 100%;}\n"
    html_report += "th, td {text-align: center; padding: 8px;}\n"
    html_report += "tr:nth-child(even) {background-color: #f2f2f2;}\n"
    html_report += "th {background-color: #4CAF50; color: white;}\n"
    html_report += "</style>\n"

    html_report += "</table>\n"

    # Save the HTML report to a file
    with open("report_Montecarlo.html", "w", encoding="utf-16") as file:
        file.write(html_report)