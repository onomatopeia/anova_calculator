import numpy as np
import scipy.stats as stats
from scipy.stats import kruskal

# Example data for three groups
group1 = [0.97996, 1.352739,
          2.299865,
          0.644973,
          0.953261
          ]
group2 = [1.513466,
          0.820838,
          1.040415,
          1.278243,
          1.49005,
          2.410586,
          2.332986,
          1.621659
          ]
group3 = [1.746329,
          1.291876,
          2.050322,
          1.039545
          ]

# Perform one-way ANOVA
f_statistic, p_value = stats.f_oneway(group1, group2, group3)

# Calculate the sum of squares
data = [group1, group2, group3]
grand_mean = np.mean(np.concatenate(data))

# Between-group sum of squares (SS_between)
ss_between = sum(len(group) * (np.mean(group) - grand_mean) ** 2 for group in data)

# Total sum of squares (SS_total)
ss_total = sum((x - grand_mean) ** 2 for group in data for x in group)

# Calculate eta squared (η²)
eta_squared = ss_between / ss_total

# Calculate Cohen's f
cohen_f = np.sqrt(eta_squared / (1 - eta_squared))

print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")
print(f"Eta squared (η²): {eta_squared}")
print(f"Cohen's f: {cohen_f}")

# Example data
group1 = [0.97996, 1.352739,
          2.299865,
          0.644973,
          0.953261
          ]
group2 = [1.513466,
          0.820838,
          1.040415,
          1.278243,
          1.49005,
          2.410586,
          2.332986,
          1.621659
          ]
group3 = [1.746329,
          1.291876,
          2.050322,
          1.039545
          ]

# Perform Kruskal-Wallis test
h_statistic, p_value = kruskal(group1, group2, group3)

# Calculate epsilon-squared (effect size)
N = len(group1) + len(group2) + len(group3)  # Total number of observations
k = 3  # Number of groups
epsilon_squared = (h_statistic - k + 1) / (N - k)

print(f"Kruskal-Wallis H-statistic: {h_statistic:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Epsilon-squared (effect size): {epsilon_squared:.4f}")
