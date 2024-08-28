import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Import the data
df = pd.read_csv('Resultados/convergence.csv')

# Step 2: Group by 'type' and 'N' and calculate mean and std for 'mean_T2' and 'std_T2'
grouped = df.groupby(['N']).agg({
    'mean_T2_MC': ['mean', 'std'],
    'std_T2_MC': ['mean', 'std'],
    'mean_T2_calc': ['mean', 'std'],
    'std_T2_calc': ['mean', 'std']
}).reset_index()

# Flatten the MultiIndex columns
grouped.columns = ['N', 'mean_T2_MC_mean', 'mean_T2_MC_std', 'std_T2_MC_mean', 'std_T2_MC_std','mean_T2_calc_mean', 'mean_T2_calc_std', 'std_T2_calc_mean', 'std_T2_calc_std']

# Step 3: Plotting
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

# Plot mean_T2 with shaded area for std
# calculated T2
axes[0].plot(grouped['N'], grouped['mean_T2_calc_mean'], label = "Calculado")
axes[0].fill_between(grouped['N'], grouped['mean_T2_calc_mean'] - grouped['mean_T2_calc_std'], grouped['mean_T2_calc_mean'] + grouped['mean_T2_calc_std'], alpha=0.2, label='_nolegend_')

# monte carlo T2
axes[0].plot(grouped['N'], grouped['mean_T2_MC_mean'], label = "Monte Carlo")
axes[0].fill_between(grouped['N'], grouped['mean_T2_MC_mean'] - grouped['mean_T2_MC_std'], grouped['mean_T2_MC_mean'] + grouped['mean_T2_MC_std'], alpha=0.2, label='_nolegend_')

axes[0].set_xscale('log')
axes[0].set_xlabel('Number of Monte Carlo simulations')
axes[0].set_ylabel('Temperature [°C]')
axes[0].set_title('Temperature at end of test')
axes[0].legend(["Monte Carlo resistance, calculated temperature", "Monte Carlo temperature"])

# Plot std_T2 with shaded area for std
axes[1].plot(grouped['N'], grouped['std_T2_calc_mean'], label = "Calculado")
axes[1].fill_between(grouped['N'], grouped['std_T2_calc_mean'] - grouped['std_T2_calc_std'], grouped['std_T2_calc_mean'] + grouped['std_T2_calc_std'], alpha=0.2, label='_nolegend_')

# monte carlo T2
axes[1].plot(grouped['N'], grouped['std_T2_MC_mean'], label = "Monte Carlo")
axes[1].fill_between(grouped['N'], grouped['std_T2_MC_mean'] - grouped['std_T2_MC_std'], grouped['std_T2_MC_mean'] + grouped['std_T2_MC_std'], alpha=0.2, label='_nolegend_')

axes[1].set_xscale('log')
axes[1].set_xlabel('Number of Monte Carlo simulations')
axes[1].set_ylabel('Temperature uncertainty [°C]')
axes[1].set_title('Temperature uncertainty at end of test')
axes[1].legend(["Monte Carlo resistance, calculated temperature", "Monte Carlo temperature"])

plt.tight_layout()
plt.show()