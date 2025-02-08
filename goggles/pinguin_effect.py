import numpy as np
import pingouin as pg
import pandas as pd
from statsmodels.stats.oneway import effectsize_oneway
from statsmodels.stats.power import FTestAnovaPower

df = pd.read_excel(
    r'D:\pwr\kolor\moc_20250130\TRY plus KOLOR 2025 DLA AGATY.xlsx',
    sheet_name='Sheet2',
    skiprows=1
)

dvs = [
    'R bucket',
    'R helmet + face',
    'R jacket',
    'Y bag',
    'Y bucket',
    'Y helmet + face'
]

pinguin_np2 = []
statsmodels_f2 = []
statsmodels_f = []
statsmodels_anova = []
power_anova = []
group = 'Total fixation duration in AOI [s]'
pinguin_f2 = []
power = []
for dv in dvs:
    welch_res = pg.welch_anova(dv=dv, between=group, data=df)
    np2 = welch_res.iloc[0]['np2']
    pinguin_np2.append(np2)
    pinguin_f2.append(np2/(1-np2))

    means_alt = []
    vars_ = []
    nobs = []
    for key in df[group].unique():
        means_alt.append(df.loc[df[group] == key, dv].mean())
        vars_.append(df.loc[df[group] == key, dv].var())
        nobs.append(len(df[df[group] == key]))
    print(means_alt)
    print(vars_)
    statsmodels_anova.append(np.sqrt(effectsize_oneway(means_alt, vars_, np.array(nobs), use_var='equal')))
    statsmodels_f2.append(effectsize_oneway(means_alt, vars_, np.array(nobs), use_var="unequal"))
    statsmodels_f.append(np.sqrt(statsmodels_f2[-1]))
    power.append(FTestAnovaPower().power(
        effect_size=statsmodels_f[-1],
        nobs=169,
        alpha=0.05,
        k_groups=3,
    ))
    power_anova.append(FTestAnovaPower().solve_power(
        power=0.8,
        effect_size=0.25,
        alpha=0.05,
        k_groups=3,
    ))


df = pd.DataFrame(data={
    'dv': dvs,
    'pinguin_np2': pinguin_np2,
    'pinguin_f2': pinguin_f2,
    'statsmodels_f2': statsmodels_f2,
    'statsmodels_f': statsmodels_f,
    'statsmodels_anova': statsmodels_anova,
    'power': power,
    'power_anova': power_anova,
})
print(df.to_markdown())

