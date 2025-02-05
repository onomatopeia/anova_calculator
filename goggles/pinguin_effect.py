import pingouin as pg
import pandas as pd

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

for dv in dvs:
    welch_res = pg.welch_anova(dv=dv, between='Total fixation duration in AOI [s]', data=df)
    print(dv, welch_res.iloc[0]['np2'])
print(welch_res.iloc[0])
