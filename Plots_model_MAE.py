import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.savefig("plot.png", dpi=300)

htest = np.array(pd.read_csv('H_testvalues_n100.csv',header=None),dtype='float')
hnn = np.array(pd.read_csv('H_NNestimated_n100.csv',header=None),dtype='float')

htest = htest[:,0]
hnn = hnn[:,0]

overall_mae = np.mean(np.abs(htest - hnn))
print("Overall MAE:", overall_mae)
# -----------------------------------------------------------------------------
# Define the bin size and the bins
bin_size = 0.02
bins = np.arange(0.0, 1.0 + bin_size, bin_size)

# Calculate the mean predicted Hurst value for each bin
bin_centers = (bins[:-1] + bins[1:]) / 2.0
mean_values = []
for i in range(len(bins) - 1):
    lower_bound = bins[i]
    upper_bound = bins[i + 1]
    mask = np.logical_and(htest >= lower_bound, htest < upper_bound)
    mean_value = np.mean(hnn[mask])
    mean_values.append(mean_value)

# Plot the scatter plot and the mean predictions line
fig, ax = plt.subplots(figsize=(6, 8))

# Plot the scatter plot
ax.plot(htest, hnn, '.', label='Predicted Values')
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Ideal Predictions')

# Plot the mean predictions line
ax.plot(bin_centers, mean_values, 'k-', linewidth=2.0, label='Mean Predictions')
ax.tick_params(axis='both', which='major', labelsize=18, width=2, length=6)
# Set the axis labels and title
ax.set_xlabel('H simulated', fontsize=26)
ax.set_ylabel('H estimated', fontsize=26)
# Set the axis limits
ax.set_xlim([0.0, 1.0])

#---------------------------------------------------------
# Gaussian Kernel Density Estimation
minv = 0.
maxv = 1.
X, Y = np.mgrid[minv:maxv:200j, minv:maxv:200j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([htest, hnn])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)

# Calculate MAE for each bin
bin_size = 0.025
bins = np.arange(0, 1 + bin_size, bin_size)
mae_values = []

for i in range(len(bins) - 1):
    lower_bound = bins[i]
    upper_bound = bins[i + 1]
    mask = np.logical_and(htest >= lower_bound, htest < upper_bound)
    mae = np.mean(np.abs(htest[mask] - hnn[mask]))
    mae_values.append(mae)
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 8), gridspec_kw={'height_ratios': [7, 1], 'hspace': 0.0})

# Plot imshow plot
im = ax1.imshow(np.rot90(Z), extent=[minv, maxv, minv, maxv], aspect='auto')
ax1.set_xlim([minv, maxv])
ax1.set_ylim([minv, maxv])
ax1.set_xticklabels([])  # Remove x-axis tick labels from the imshow plot
ax1.set_ylabel('H estimated', fontsize=28)
ax1.tick_params(axis='both', which='major', labelsize=18, width=2, length=6)
# Plot the MAE bar chart
bar_positions = bins[:-1] + bin_size / 2  # Shift bar positions to the right by half the bin size
ax2.bar(bar_positions, mae_values, width=bin_size * 0.9)  # Use the new bar_positions
ax2.tick_params(axis='both', which='major', labelsize=18, width=2, length=6)
ax2.set_xlim(0, 1)
ax2.set_xlabel('H simulated', fontsize=28)
ax2.set_ylabel('MAE', fontsize=26)


fig.tight_layout()
plt.show()  