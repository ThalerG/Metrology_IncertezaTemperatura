import pandas as pd
from plot_montecarlos import extract_analysis_parameters

# Path to the CSV file
csv_file = 'Resultados/beges.csv'

# Read the CSV file into a DataFrame
df_beges = pd.read_csv(csv_file)

analysis_files = ['Resultados/beges1_allPoints', 'Resultados/beges2', 'Resultados/beges3']

df_montecarlo = pd.DataFrame()

for k,file in enumerate(analysis_files):
    params = extract_analysis_parameters(file + '.txt')
    selected_params = {key: params[key] for key in ['Npoints', 'dt', 'mean_T2', 'std_T2', 'l95_T2', 'u95_T2']}
    selected_params["begesLin"] = df_beges.iloc[0, k+1]
    selected_params["begesPoly2"] = df_beges.iloc[1, k+1]
    selected_params["begesPoly4"] = df_beges.iloc[2, k+1]
    selected_params["begesStd"] = df_beges.iloc[3, k+1]
    df_montecarlo = pd.concat([df_montecarlo, pd.DataFrame([selected_params])], ignore_index=True)

print(df_montecarlo)
