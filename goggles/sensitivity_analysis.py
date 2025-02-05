from statsmodels.stats.power import FTestAnovaPower

# Sensitivity analysis: What effect size can be detected with a smaller sample?
sample_size_per_group = 67  # Adjust based on your feasible sample size
alpha = 0.05
power = 0.80
num_groups = 3

# Solve for effect size
anova_power = FTestAnovaPower()
detectable_effect_size = anova_power.solve_power(nobs=sample_size_per_group, effect_size=None, alpha=alpha, power=power, k_groups=num_groups)

print(f"Detectable effect size (f): {detectable_effect_size:.2f}")
