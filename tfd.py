from pathlib import Path

import pandas as pd

from goggles import anova_hsd, kruskal_wallis_dunn

data_file_path = Path(__file__).parent.joinpath('data', '20241217', 'data.ods')

sheet_transparent = "TFD_T_Total_Fixation_Duration"
sheet_red = "TFD_R_Total_Fixation_Duration"
sheet_yellow = "TFD_Y_Total_Fixation_Duration"

df_t = pd.read_excel(
    data_file_path,
    sheet_name=sheet_transparent,
    usecols=range(1, 8),
    engine="odf",
    skiprows=1,
)
df_r = pd.read_excel(
    data_file_path,
    engine='odf',
    sheet_name=sheet_red,
    skiprows=1,
    usecols=range(1, 8),
)
df_y = pd.read_excel(
    data_file_path,
    engine='odf',
    sheet_name=sheet_yellow,
    skiprows=1,
    usecols=range(1, 8),
)

red_bucket_column = "R bucket"
red_helmet_column = "R helmet + face"
red_jacket_column = "R jacket"
yellow_bucket_column = "Y bucket"
yellow_helmet_column = "Y helmet + face"
yellow_bag_column = "Y bag"

columns = [
    red_jacket_column,
    red_helmet_column,
    red_bucket_column,
    yellow_bucket_column,
    yellow_bag_column,
    yellow_helmet_column,
]

dfs = {'Transparent': df_t, 'Yellow': df_y, 'Red': df_r}
results_folder = Path(__file__).parent.joinpath('results')

col = yellow_bag_column
output_folder = results_folder.joinpath(col)
output_folder.mkdir(exist_ok=True)

samples = {df_name: df[col] for df_name, df in dfs.items()}
print('ANOVA Assumptions')
assumptions_passed = True
print('\n1. Equal cell sizes')
assumptions_passed &= anova_hsd.equal_size_samples(*samples.values())
print('\n2. Normality')
assumptions_passed &= anova_hsd.normality(samples, output_folder)
print('\n3. Homoscedasticity')
assumptions_passed &= anova_hsd.equal_variances(*samples.values())

if assumptions_passed:
    print('\nAll ANOVA assumptions have passed.')

    if anova_hsd.one_way_anova(*samples.values()):
        result = anova_hsd.pairwise_comparisons(samples)
        if result:
            print('At least one pair has significantly different means.')
        else:
            print('No significant differences in pairs\' means.')
else:
    print('\nNot all ANOVA assumptions have passed. Switching to Kruskal-Wallis Test.')
    print('4. Similarity of shape')
    kruskal_wallis_dunn.similarity_of_shape(samples, output_folder)
