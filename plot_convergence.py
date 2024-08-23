import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Import the data
df = pd.read_csv('Resultados/convergence.csv')

# Step 2: Group by 'type' and 'N' and calculate mean and std for 'mean_T2' and 'std_T2'
grouped = df.groupby(['type', 'N']).agg({
    'mean_T2': ['mean', 'std'],
    'std_T2': ['mean', 'std']
}).reset_index()

# Flatten the MultiIndex columns
grouped.columns = ['type', 'N', 'mean_T2_mean', 'mean_T2_std', 'std_T2_mean', 'std_T2_std']

# Step 3: Plotting
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

# Plot mean_T2 with shaded area for std
for t in grouped['type'].unique():
    subset = grouped[grouped['type'] == t]
    axes[0].plot(subset['N'], subset['mean_T2_mean'], label=t)
    axes[0].fill_between(subset['N'], subset['mean_T2_mean'] - subset['mean_T2_std'], subset['mean_T2_mean'] + subset['mean_T2_std'], alpha=0.2, label='_nolegend_')

axes[0].set_xscale('log')
axes[0].set_xlabel('Number of Monte Carlo simulations')
axes[0].set_ylabel('Temperature [°C]')
axes[0].set_title('Temperature at end of test')
axes[0].legend(["Monte Carlo resistance, calculated temperature", "Monte Carlo temperature"])

# Plot std_T2 with shaded area for std
for t in grouped['type'].unique():
    subset = grouped[grouped['type'] == t]
    axes[1].plot(subset['N'], subset['std_T2_mean'], label=t)
    axes[1].fill_between(subset['N'], subset['std_T2_mean'] - subset['std_T2_std'], subset['std_T2_mean'] + subset['std_T2_std'], alpha=0.2, label='_nolegend_')

axes[1].set_xscale('log')
axes[1].set_xlabel('Number of Monte Carlo simulations')
axes[1].set_ylabel('Temperature uncertainty [°C]')
axes[1].set_title('Temperature uncertainty at end of test')
axes[1].legend(["Monte Carlo resistance, calculated temperature", "Monte Carlo temperature"])

plt.tight_layout()
plt.show()