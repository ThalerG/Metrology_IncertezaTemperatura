import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Import the data
df = pd.read_csv('Resultados/convergence.csv')

# Step 2: Group by 'type' and 'N' and calculate mean and std for 'mean_T2' and 'std_T2'
grouped = df.groupby(['N']).agg({
    'mean_T2': ['mean', 'std'],
    'std_T2': ['mean', 'std'],
}).reset_index()

# Flatten the MultiIndex columns
grouped.columns = ['N', 'mean_T2_mean', 'mean_T2_std', 'std_T2_mean', 'std_T2_std']

cm = 1/2.54  # centimeters in inches

# Step 3: Plotting
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14*cm, 12*cm))

plt.rcParams["font.family"] = "Times New Roman"

# Plot mean_T2 with shaded area for std
# monte carlo T2
axes[0].plot(grouped['N'], grouped['mean_T2_mean'], label = "Monte Carlo")
axes[0].fill_between(grouped['N'], grouped['mean_T2_mean'] - grouped['mean_T2_std'], grouped['mean_T2_mean'] + grouped['mean_T2_std'], alpha=0.2)

axes[0].set_xscale('log')
axes[0].set_xlabel('Number of Monte Carlo simulations', fontname='Times New Roman')
axes[0].set_ylabel('Temperature [°C]', fontname='Times New Roman')
axes[0].legend(["Monte Carlo mean", "Monte Carlo standard deviation"])
axes[0].grid(True)

for label in axes[0].get_xticklabels():
    label.set_fontproperties('Times New Roman')

for label in axes[0].get_yticklabels():
    label.set_fontproperties('Times New Roman')

# monte carlo T2
axes[1].plot(grouped['N'], grouped['std_T2_mean'], label = "Monte Carlo")
axes[1].fill_between(grouped['N'], grouped['std_T2_mean'] - grouped['std_T2_std'], grouped['std_T2_mean'] + grouped['std_T2_std'], alpha=0.2)

axes[1].set_xscale('log')
axes[1].set_xlabel('Number of Monte Carlo simulations', fontname='Times New Roman')
axes[1].set_ylabel('Uncertainty [°C]', fontname='Times New Roman')

axes[1].grid(True)

for label in axes[1].get_xticklabels():
    label.set_fontproperties('Times New Roman')

for label in axes[1].get_yticklabels():
    label.set_fontproperties('Times New Roman')

plt.tight_layout()
plt.savefig('Gráficos/convergence_plot.pdf')
plt.show()