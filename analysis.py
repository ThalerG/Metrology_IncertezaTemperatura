from fcn import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

file_path = "Dados/data.csv"
df = pd.read_csv(file_path)

models = [('poly',1),
          ('poly',2),
          ('poly',3),
          ('exp',0)
        ]

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

s_R1 = 0.01 # Incerteza da medição de resistência no início do teste
s_Tamb1 = 0.1 # Incerteza da medição de temperatura no início do teste
s_Tamb2 = 0.1 # Incerteza da medição de temperatura no final do teste

x = df['Time'].values
y = df['Resistance'].values

s_x = np.sqrt(s_t0**2 + s_dt**2)
s_y = s_dR

# Generate HTML report
html_report = "<html>\n<head>\n<title>Analysis Report</title>\n</head>\n<body>\n"

df_values = pd.DataFrame(columns=['Model Type', 'Degree', 'SSE', 'Resistance', 'Uncertainty', 'Temperature', 'Temperature Uncertainty'])

results = []

for (k,model) in enumerate(models):
    estimation_model = generate_estimation_models(type = model[0], degree=model[1])

    if model[0] == 'poly':
        initial_params = [1]*(model[1]+1)
    elif model[0] == 'exp':
        initial_params = [18,2,-0.01]

    params, uncertainty, result = estimate_model_with_uncertainty(x, y, s_x, s_y, model=estimation_model, initial_params= initial_params,maxit = 10000)

    estimated_model = generate_estimation_models(type = model[0], degree=model[1], params=params)
    uncertainty_model = generate_estimation_uncertainty_models(params=params, s_params=uncertainty, s_x = s_x, type=model[0], degree=model[1])

    R2 = estimated_model(0)
    s_R2 = uncertainty_model(0)

    T2 = final_temperature(R1, R2, Tamb_1, alpha)
    s_T2 = final_temperature_uncertainty(R1, R2, Tamb_1,alpha, s_R1, s_R2, s_Tamb1)

    x_plot = np.insert(x, 0, 0)
    s_R= uncertainty_model(x_plot)

    html_report += f"<h2>Model {k+1}</h2>\n"

    fig = make_subplots()
    fig.add_trace(go.Scatter(x=x_plot, y=estimated_model(x_plot), name='ODR Estimation', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=x_plot, y=estimated_model(x_plot) - s_R, name='Estimation uncertainty', line=dict(color='red', dash = 'dash', width = 1), fillcolor='rgba(255,0,0,0.3)', showlegend=False))
    fig.add_trace(go.Scatter(x=x_plot, y=estimated_model(x_plot) + s_R, name='Estimation uncertainty', fill='tonexty', line=dict(color='red', dash = 'dash', width = 1), fillcolor='rgba(255,0,0,0.3)'))
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', error_y=dict(type='data', array=[s_y]*len(y)), error_x=dict(type='data', array=[s_x]*len(x)), name='Original data with uncertainty'))
    fig.update_layout(xaxis_title='Time [s]', yaxis_title='Resistance [Ω]', showlegend=True)
    html_report += fig.to_html(full_html=False)

    results.append(estimated_model(x_plot))

    df_values.loc[k] = [model[0], model[1], result.sum_square, R2, s_R2, T2, s_T2]

html_report += f"<h2>Comparacao</h2>\n"

fig = make_subplots()
for k in range(len(results)):
    fig.add_trace(go.Scatter(x=x_plot, y=results[k], name=f'Model {k+1}'))

fig.add_trace(go.Scatter(x=x, y=y, mode='markers', error_y=dict(type='data', array=[s_y]*len(y)), error_x=dict(type='data', array=[s_x]*len(x)), name='Original data with uncertainty'))
fig.update_layout(xaxis_title='Time [s]', yaxis_title='Resistance [Ω]', showlegend=True)
html_report += fig.to_html(full_html=False)

# Add the dataframe as a table in the html file
html_report += df_values.to_html(index=False, classes='centered')

html_report += "</body>\n</html>"

# Save the HTML report to a file
with open("report.html", "w", encoding="utf-8") as file:
    file.write(html_report)