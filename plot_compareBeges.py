import pandas as pd
from plot_montecarlos import extract_analysis_parameters
import matplotlib.pyplot as plt
import os
import warnings
import matplotlib.colors as mcolors
import numpy as np
from fcn import estimate_model_with_uncertainty

def get_unc_GUM(baseValues):
    
    initial_params = [17.472,2.06,-0.0197]

    def exp_model(params, x):
        return params[0] + params[1]*np.exp(params[2]*x)

    dt = baseValues['dt']
    t1 = baseValues['t1']
    N_points = baseValues['Npoints']

    s_t0 = baseValues['s_t0']
    s_dt = baseValues['s_dt']
    s_dR = baseValues['s_dR']

    s_R1 = baseValues['s_R1']
    s_Tamb1 = baseValues['s_Tamb1']
    s_Tamb2 = baseValues['s_Tamb2']
    s_cvol = 1

    file_path = "Dados/data.csv"
    df = pd.read_csv(file_path)

    x_og = df['Time'].values
    y_og = df['Resistance'].values

    x_tot = np.linspace(t1, t1 + (N_points-1)*dt, N_points)
    ind = np.isin(x_og, x_tot)

    if np.sum(ind) == 0:
        raise ValueError("None of the provided time data fits the desired configuration.")
    elif np.sum(ind) < N_points:
        warnings.warn("Not all elements of the desired configuration are present in the provided time data. Proceeding with the available data.")

    x_tot = x_og[ind]
    y_tot = y_og[ind]

    params, _, res = estimate_model_with_uncertainty(x_tot, y_tot, s_dt, s_dR, model=exp_model, initial_params= initial_params,maxit = 1000000)

    B0 = params[0]
    B1 = params[1]
    B2 = params[2]

    s_B0 = np.sqrt(res.cov_beta[0][0])
    s_B1 = np.sqrt(res.cov_beta[1][1])
    s_B2 = np.sqrt(res.cov_beta[2][2])

    Tamb_1 = baseValues['Tamb_1']
    Tamb_2 = baseValues['Tamb_2']
    R1 = baseValues['R1']
    cvol = 100
    t0 = 0

    k = 25450/cvol-20

    R2 = B0+B1*np.exp(B2*t0)

    s_DT_cov = [((R2-R1)/R1 + 1)*s_Tamb1, # Uncertainty of initial ambient temperature
            -1*s_Tamb2, # Uncertainty of final ambient temperature
            -R2*(k+Tamb_1)/(R1**2)*s_R1, # Uncertainty of initial resistance
            -(R2-R1)/R1*25450/cvol**2*s_cvol, # Uncertainty of copper purity
            -R2*(k+Tamb_1)/(R1**2)*s_B0, # Uncertainty of beta0
            -np.exp(B2*t0)*R2*(k+Tamb_1)/(R1**2)*s_B1, # Uncertainty of beta1
            -t0*B1*np.exp(B2*t0)*R2*(k+Tamb_1)/(R1**2)*s_B2, # Uncertainty of beta2
            -B2*B1*np.exp(B2*t0)*R2*(k+Tamb_1)/(R1**2)*s_t0 # Uncertainty of t0
            ]
    
    s = []
    for sign0 in [True,False]:
        b0 = (B0+s_B0) if sign0 else (B0-s_B0)
        for sign1 in [True,False]:
            b1 = (B1+s_B1) if sign1 else (B1-s_B1)
            for sign2 in [True,False]:
                b2 = (B2+s_B2) if sign2 else (B2-s_B2)
                for signt0 in [True,False]:
                    t0 = (t0+s_t0) if signt0 else (t0-s_t0)
                    R = b0+b1*np.exp(b2*t0)-R2
                    s.append(abs(R))
    sR2_bound = np.max(s)

    s_DT_bound = np.linalg.norm(s_DT_cov[0:4] + [sR2_bound])
    
    DT = (R2-R1)/R1*(k+Tamb_1)-(Tamb_2-Tamb_1)

    s_noR2 = np.linalg.norm(s_DT_cov[0:4])
    return DT, np.linalg.norm(s_DT_cov), s_DT_cov, s_DT_bound, s_noR2

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

    selected_params["DTGum"], selected_params["sDTGum"], _, selected_params["DTGum_bound"], selected_params["sDTGum_noR2"] = get_unc_GUM(params)

    df_montecarlo = pd.concat([df_montecarlo, pd.DataFrame([selected_params])], ignore_index=True)

    

# Plotting
cm = 1/2.54  # centimeters in inches
fig, ax = plt.subplots(figsize=(14*cm, 8*cm))

# Define labels for each index
index_labels = [f'{Npoints} measurements\n Start at {t1} s\n Measure every {dt} s' for Npoints, dt, t1 in df_montecarlo[['Npoints', 'dt', 't1']].values]

# Plot begesLin
ax.errorbar(df_montecarlo.index - 0.3, df_montecarlo['begesLin'], yerr=df_montecarlo['begesStd'], fmt='o', label='Linear [12]', color='C0', capsize=5)

# Plot begesPoly2
ax.errorbar(df_montecarlo.index - 0.2, df_montecarlo['begesPoly2'], yerr=df_montecarlo['begesStd'], fmt='o', label=r'2$^{\text{nd}}$ order polynomial [12]', color='C1', capsize=5)

# Plot begesPoly4
ax.errorbar(df_montecarlo.index - 0.1, df_montecarlo['begesPoly4'], yerr=df_montecarlo['begesStd'], fmt='o', label=r'4$^{\text{nd}}$ order polynomial [12]', color='C2', capsize=5)

# Plot GUM
ax.errorbar(df_montecarlo.index + 0, df_montecarlo['DTGum'], yerr=df_montecarlo['sDTGum_noR2'], fmt='o', label='GUM [] (no uR2)', color='C3', capsize=5)

# Plot GUM
ax.errorbar(df_montecarlo.index + 0.1, df_montecarlo['DTGum'], yerr=df_montecarlo['sDTGum'], fmt='o', label='GUM [] + covariance (prop.) []', color='C4', capsize=5)

# Plot GUM
ax.errorbar(df_montecarlo.index + 0.2, df_montecarlo['DTGum'], yerr=df_montecarlo['DTGum_bound'], fmt='o', label='GUM [] + covariance (bound.) [32]', color='C5', capsize=5)

# Plot mean_DT
ax.errorbar(df_montecarlo.index + 0.3, df_montecarlo['mean_DT'], yerr=df_montecarlo['std_DT'], fmt='o', label='Proposed method', color='C6', capsize=5)

# Set custom x-ticks
ax.set_xticks(df_montecarlo.index)
ax.set_xticklabels(index_labels[:len(df_montecarlo.index)])

for label in ax.get_xticklabels():
    label.set_fontproperties('Times New Roman')

for label in ax.get_yticklabels():
    label.set_fontproperties('Times New Roman')

plt.rcParams['font.family'] = 'Times New Roman'

# Labels and legend
ax.set_ylabel('Temperature rise [°C]', fontname='Times New Roman')
ax.legend()

ax.grid()

plt.tight_layout()

if not os.path.exists("Gráficos"):
    os.makedirs("Gráficos")

fig.savefig(f"Gráficos/begesCompareB.pdf")
