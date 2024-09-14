import pandas as pd
from plot_montecarlos import extract_analysis_parameters
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors


# Path to the CSV file
csv_file = 'Resultados/beges.csv'

# Read the CSV file into a DataFrame
df_beges = pd.read_csv(csv_file)

analysis_files = ['Resultados/beges1_allPoints', 'Resultados/beges2', 'Resultados/beges3']

df_montecarlo = pd.DataFrame()

for k,file in enumerate(analysis_files):
    params = extract_analysis_parameters(file + '.txt')
    selected_params = {key: params[key] for key in ['Npoints', 'dt', 't1']}
    selected_params["begesLin"] = df_beges.iloc[0, k+1]
    selected_params["begesPoly2"] = df_beges.iloc[1, k+1]
    selected_params["begesPoly4"] = df_beges.iloc[2, k+1]
    selected_params["begesStd"] = df_beges.iloc[3, k+1]
    df_full = pd.read_feather(file + '.feather')
    selected_params["mean_DT"] = df_full['DT'].mean()
    selected_params["std_DT"] = df_full['DT'].std()

    df_montecarlo = pd.concat([df_montecarlo, pd.DataFrame([selected_params])], ignore_index=True)

    

# Plotting
fig, ax = plt.subplots(figsize=(8, 4))

# Define labels for each index
index_labels = [f'{Npoints} measurements\n Start at {t1} s, measure every {dt} s' for Npoints, dt, t1 in df_montecarlo[['Npoints', 'dt', 't1']].values]

# Plot begesLin
ax.errorbar(df_montecarlo.index - 0.15, df_montecarlo['begesLin'], yerr=df_montecarlo['begesStd'], fmt='o', label='Linear extrapolation [12]', color='C0', capsize=5)

# Plot begesPoly2
ax.errorbar(df_montecarlo.index - 0.05, df_montecarlo['begesPoly2'], yerr=df_montecarlo['begesStd'], fmt='o', label=r'Polynomial extrapolation, 2$^{\text{nd}}$ order [12]', color='C1', capsize=5)

# Plot begesPoly4
ax.errorbar(df_montecarlo.index + 0.05, df_montecarlo['begesPoly4'], yerr=df_montecarlo['begesStd'], fmt='o', label=r'Polynomial extrapolation, 4$^{\text{th}}$ order [12]', color='C2', capsize=5)

# Plot mean_DT
ax.errorbar(df_montecarlo.index + 0.15, df_montecarlo['mean_DT'], yerr=df_montecarlo['std_DT'], fmt='o', label='Proposed method', color='C3', capsize=5)

# Set custom x-ticks
ax.set_xticks(df_montecarlo.index)
ax.set_xticklabels(index_labels[:len(df_montecarlo.index)])

for label in ax.get_xticklabels():
    label.set_fontproperties('P052')

for label in ax.get_yticklabels():
    label.set_fontproperties('P052')

plt.rcParams['font.family'] = 'P052'

# Labels and legend
ax.set_ylabel('Temperature rise [°C]', fontname='P052')
ax.legend()

plt.tight_layout()

if not os.path.exists("Gráficos"):
    os.makedirs("Gráficos")

fig.savefig(f"Gráficos/begesCompare.pdf")
